"""Fine-tuning de reconnaissance — RECETTE B1 (alignée sur le code de référence
old_code_paper_classB, audité le 2026-06-27 : `features` gelée, schedule LR
warmup+cosine, augmentation flip horizontal, images chargées ALIGNÉES par
landmarks ArcFace plutôt que simplement redimensionnées).

Trois scénarios pour la source des images de surveillance :
- real      : uniquement les vraies images de surveillance du Bloc B
- synthetic : uniquement les images générées (paths.synth_dataset)
- mixed     : TOUTES les vraies images + un pourcentage CONFIGURABLE (ablation,
              recognition.synthetic_ratio) du volume synthétique disponible — pour
              étudier le volume optimal de données synthétiques en complément du réel.

Scope=layer3+4 (dégèle layer3/layer4/bn2/fc, gèle le reste — `features`, la
BatchNorm1d finale, reste TOUJOURS gelée, convention arcface_torch confirmée dans le
code de référence), AdamW, ancrage 50/50 mugshot/surveillance, reprenable
(checkpoint), multi-seed (cf. cfg['seeds']).

Identités d'entraînement = Bloc B (cf. CLAUDE.md : B sert à valider+générer le
synthétique ET à fine-tuner le recognizer ; C reste vierge pour l'évaluation finale).
"""
from __future__ import annotations
import math
import random
from collections import defaultdict
from pathlib import Path

from src.data.pairs import list_pairs
from src.generator.face_detect import load_face_app, load_aligned_face_tensor
from src.recognition.arcmargin import ArcMarginProduct
from src.utils.arcface_backbone import EMBEDDING_SIZE, iresnet50, preprocess_for_arcface
from src.utils.checkpoint import latest_checkpoint, load_checkpoint, resume_step, save_checkpoint
from src.utils.logging import get_logger

log = get_logger()

# 'features' (BatchNorm1d finale) volontairement EXCLUE : reste gelée même en
# scope layer3_4, comme dans le code de référence ("design arcface_torch").
SCOPE_TRAINABLE_PREFIXES = {
    "layer3_4": ("layer3", "layer4", "bn2", "fc"),
}


def _apply_scope(net, scope: str) -> list:
    if scope not in SCOPE_TRAINABLE_PREFIXES:
        raise NotImplementedError(f"TODO(claude): scope de dégel non défini pour '{scope}'.")
    prefixes = SCOPE_TRAINABLE_PREFIXES[scope]
    net.requires_grad_(False)
    trainable = []
    for name, p in net.named_parameters():
        if any(name.startswith(pre) for pre in prefixes):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def _get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float) -> float:
    """Warmup linéaire puis décroissance cosine — port direct du code de référence B1."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def _build_pools(cfg: dict, condition: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Pools mugshot/surveillance par identité (Bloc B), selon la source demandée.

    'mixed' garde TOUJOURS 100% des vraies images, et ajoute une fraction
    (recognition.synthetic_ratio, défaut 1.0 = tout) du volume synthétique
    disponible par identité — c'est le levier de l'étude d'ablation."""
    if condition not in ("real", "synthetic", "mixed"):
        raise ValueError(f"Condition inconnue : {condition} (attendu real|synthetic|mixed).")

    pairs_b = list_pairs(cfg, block="B")
    mugshot_of: dict[str, list[str]] = {}
    surveillance_of: dict[str, list[str]] = defaultdict(list)
    for p in pairs_b:
        mugshot_of.setdefault(p.identity, [p.mugshot_path])
        if condition in ("real", "mixed"):
            surveillance_of[p.identity].append(p.target_path)  # 100% du réel, jamais réduit

    if condition in ("synthetic", "mixed"):
        synth_root = Path(cfg["paths"]["synth_dataset"])
        ratio = cfg["recognition"].get("synthetic_ratio", 1.0) if condition == "mixed" else 1.0
        for identity in mugshot_of:
            files = sorted(str(f) for f in (synth_root / identity).glob("*.png"))
            if not files:
                raise RuntimeError(
                    f"Aucune image synthétique pour l'identité {identity} sous {synth_root} "
                    f"(condition={condition}). Lancer l'étage 'generate' avant 'train_recognition'.")
            n_keep = round(len(files) * ratio)
            surveillance_of[identity].extend(files[:n_keep])
    return mugshot_of, surveillance_of


def _build_aligned_cache(face_app, mugshot_of: dict, surveillance_of: dict, cache_dir: str | None = None):
    """Pré-aligne (5 points, ArcFace) et met en cache CHAQUE image distincte une seule
    fois (au lieu de re-détecter/ré-aligner à chaque tirage aléatoire, sur des centaines
    de pas). load_aligned_face_tensor ne lève jamais d'erreur (repli sur un simple
    resize si aucun visage détecté, cf. code de référence) : aucune identité/image
    n'est exclue ici, juste mise en cache.

    cache_dir (paths.aligned_cache) : persiste aussi sur disque -- une image déjà
    alignée lors d'un seed/condition précédent (ou d'une évaluation) n'est plus
    jamais re-détectée, même dans un nouveau processus."""
    cache: dict[str, "object"] = {}
    for identity, paths in mugshot_of.items():
        for p in [paths[0], *surveillance_of.get(identity, [])]:
            if p not in cache:  # setdefault évaluerait load_aligned_face_tensor à chaque
                cache[p] = load_aligned_face_tensor(p, face_app, cache_dir=cache_dir)  # fois -> pas de cache réel
    return cache


def train(cfg: dict, condition: str, seed: int) -> str:
    import torch
    import torch.nn.functional as F
    from src.utils.seed import set_seed

    set_seed(seed)
    rec_cfg = cfg["recognition"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mugshot_of, surveillance_of = _build_pools(cfg, condition)
    face_app = load_face_app(cfg)
    aligned_cache = _build_aligned_cache(face_app, mugshot_of, surveillance_of,
                                          cache_dir=cfg["paths"].get("aligned_cache"))
    identities = sorted(mugshot_of)
    label_of = {identity: i for i, identity in enumerate(identities)}

    net = iresnet50()
    weights_path = cfg["paths"].get("arcface_weights")
    if weights_path and Path(weights_path).exists():
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    trainable = _apply_scope(net, rec_cfg["scope"])
    head = ArcMarginProduct(EMBEDDING_SIZE, num_classes=len(identities),
                             scale=rec_cfg["arcmargin"]["scale"], margin=rec_cfg["arcmargin"]["margin"])
    net, head = net.to(device), head.to(device)

    optimizer = torch.optim.AdamW(trainable + list(head.parameters()),
                                   lr=rec_cfg["lr"], weight_decay=rec_cfg["weight_decay"])

    ckpt_dir = cfg["paths"]["checkpoints"]
    tag = f"recognition_{condition}_seed{seed}"
    step = resume_step(ckpt_dir, tag)
    ckpt_path = latest_checkpoint(ckpt_dir, tag)
    if ckpt_path is not None:
        state = load_checkpoint(ckpt_path)
        net.load_state_dict(state["net"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        log.info("[%s seed=%d] reprise depuis step=%d (%s)", condition, seed, step, ckpt_path)

    batch_size = rec_cfg["batch_size"]
    steps_per_epoch = max(1, len(identities) // max(1, batch_size // 2))
    total_epochs = rec_cfg["epochs"]
    max_steps = total_epochs * steps_per_epoch
    hflip_prob = rec_cfg.get("hflip_prob", 0.0)
    warmup_epochs = rec_cfg.get("warmup_epochs", 0)

    rng = random.Random(seed)
    saved_path = ckpt_path
    while step < max_steps:
        epoch = step // steps_per_epoch
        lr = _get_lr(epoch, warmup_epochs, total_epochs, rec_cfg["lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        imgs, labels = [], []
        for _ in range(batch_size):
            identity = rng.choice(identities)
            pool = mugshot_of if rng.random() < rec_cfg["anchor_ratio"] else surveillance_of
            tensor = aligned_cache[rng.choice(pool[identity])]
            if rng.random() < hflip_prob:
                tensor = tensor.flip(-1)  # flip horizontal (augmentation, cf. code B1)
            imgs.append(tensor)
            labels.append(label_of[identity])

        batch = torch.stack(imgs).to(device)
        labels_t = torch.tensor(labels, device=device)

        embeddings = net(preprocess_for_arcface(batch))
        logits = head(embeddings, labels_t)
        loss = F.cross_entropy(logits, labels_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % rec_cfg["log_every"] == 0 or step == max_steps:
            log.info("[%s seed=%d] step=%d/%d epoch=%d lr=%.2e loss=%.4f",
                      condition, seed, step, max_steps, epoch, lr, loss.item())
        if step % rec_cfg["ckpt_every"] == 0 or step == max_steps:
            saved_path = save_checkpoint(
                {"net": net.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_dir, tag, step)

    return str(saved_path)
