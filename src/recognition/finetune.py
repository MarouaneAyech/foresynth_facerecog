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
from src.recognition.arcmargin import ArcFaceHead, ArcMarginProduct
from src.recognition.lora import freeze_all_batchnorm, inject_lora
from src.recognition.naming import recognition_tag
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


def _steps_per_epoch(rec_cfg: dict, batch_size: int, n_identities: int, surveillance_of: dict) -> int:
    """legacy : ancienne definition (archive full_finetune, ~3 pas/epoque).
    dataset : comme bari/papier ch.3 -- une epoque = 2 x (nb d'images de surveillance
    du pool) tirages, soit autant de pas que le volume de donnees l'impose (le
    synthetique ajoute au reel allonge donc l'entrainement, comme dans le papier)."""
    if rec_cfg.get("sampler", "legacy") == "bari":
        n_surv = sum(len(v) for v in surveillance_of.values())
        return max(1, (2 * n_surv) // batch_size)  # drop_last, comme le DataLoader de bari
    return max(1, n_identities // max(1, batch_size // 2))


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
    mechanism = rec_cfg.get("mechanism", "full_finetune")
    if mechanism == "lora":
        lora_cfg = rec_cfg["lora"]
        net.requires_grad_(False)
        n_conv, n_fc = inject_lora(net, r=lora_cfg["rank"], alpha=lora_cfg["alpha"],
                                    target_layers=tuple(lora_cfg["target_layers"]),
                                    include_fc=lora_cfg["include_fc"])
        trainable = [p for p in net.parameters() if p.requires_grad]
        log.info("LoRA r=%d alpha=%d : %d conv + %d fc adaptes, %d parametres entrainables",
                  lora_cfg["rank"], lora_cfg["alpha"], n_conv, n_fc, sum(p.numel() for p in trainable))
    elif mechanism == "full_finetune":
        trainable = _apply_scope(net, rec_cfg["scope"])
    else:
        raise ValueError(f"recognition.mechanism inconnu : {mechanism} (full_finetune|lora)")
    head_cls = ArcFaceHead if mechanism == "lora" else ArcMarginProduct
    head = head_cls(EMBEDDING_SIZE, num_classes=len(identities),
                    scale=rec_cfg["arcmargin"]["scale"], margin=rec_cfg["arcmargin"]["margin"])
    net, head = net.to(device), head.to(device)

    optimizer = torch.optim.AdamW(trainable + list(head.parameters()),
                                   lr=rec_cfg["lr"], weight_decay=rec_cfg["weight_decay"])

    ckpt_dir = cfg["paths"]["checkpoints"]
    tag = recognition_tag(cfg, condition, seed)
    step = resume_step(ckpt_dir, tag)
    ckpt_path = latest_checkpoint(ckpt_dir, tag)
    if ckpt_path is not None:
        state = load_checkpoint(ckpt_path)
        net.load_state_dict(state["net"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        log.info("[%s seed=%d] reprise depuis step=%d (%s)", condition, seed, step, ckpt_path)

    batch_size = rec_cfg["batch_size"]
    steps_per_epoch = _steps_per_epoch(rec_cfg, batch_size, len(identities), surveillance_of)
    total_epochs = rec_cfg["epochs"]
    max_steps = total_epochs * steps_per_epoch
    hflip_prob = rec_cfg.get("hflip_prob", 0.0)
    warmup_epochs = rec_cfg.get("warmup_epochs", 0)
    sampler = rec_cfg.get("sampler", "legacy")
    log.info("[%s seed=%d] mecanisme=%s sampler=%s : %d pas/epoque, %d pas au total",
              condition, seed, mechanism, sampler, steps_per_epoch, max_steps)

    if sampler == "bari":
        # Echantillonneur du ch.3 (bari/data/train_loader.py) : tirage AVEC remise sur
        # l'ensemble des images, mugshots surponderes pour une proportion anchor_ratio
        # (50/50 en esperance), 2 x n_surveillance tirages par epoque.
        samples = []
        for identity in identities:
            samples.append((mugshot_of[identity][0], label_of[identity], True))
            samples += [(p, label_of[identity], False) for p in surveillance_of[identity]]
        n_surv = sum(1 for s in samples if not s[2])
        anchor = rec_cfg["anchor_ratio"]
        w_mug = n_surv * anchor / (1.0 - anchor) / len(identities)
        sample_weights = torch.tensor([w_mug if s[2] else 1.0 for s in samples], dtype=torch.double)
        epoch_idx = {"epoch": -1, "idx": None}

    def next_batch(step: int, epoch: int):
        if sampler == "bari":
            if epoch_idx["epoch"] != epoch:
                g = torch.Generator().manual_seed(seed * 1000 + epoch)
                epoch_idx["idx"] = torch.multinomial(sample_weights, 2 * n_surv, replacement=True, generator=g)
                epoch_idx["epoch"] = epoch
            pos = step % steps_per_epoch
            chosen = [samples[i] for i in epoch_idx["idx"][pos * batch_size:(pos + 1) * batch_size].tolist()]
            step_rng = random.Random(seed * 10_000_000 + step)
            tensors = [aligned_cache[p].flip(-1) if step_rng.random() < hflip_prob else aligned_cache[p]
                       for p, _, _ in chosen]
            return tensors, [lab for _, lab, _ in chosen]
        imgs, labels = [], []
        for _ in range(batch_size):
            identity = rng.choice(identities)
            pool = mugshot_of if rng.random() < rec_cfg["anchor_ratio"] else surveillance_of
            tensor = aligned_cache[rng.choice(pool[identity])]
            if rng.random() < hflip_prob:
                tensor = tensor.flip(-1)  # flip horizontal (augmentation, cf. code B1)
            imgs.append(tensor)
            labels.append(label_of[identity])
        return imgs, labels

    # LoRA : BN figees au sens des statistiques (papier ch.3, §3.3) -- requires_grad=False
    # ne suffit pas. Aucun net.train() n'est rappele plus bas, un seul appel suffit.
    net.train()
    if mechanism == "lora":
        log.info("BatchNorm gelees (eval) : %d modules", freeze_all_batchnorm(net))

    rng = random.Random(seed)
    saved_path = ckpt_path
    while step < max_steps:
        epoch = step // steps_per_epoch
        lr = _get_lr(epoch, warmup_epochs, total_epochs, rec_cfg["lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        imgs, labels = next_batch(step, epoch)

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
                {"net": net.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                 "mechanism": mechanism,
                 "lora_rank": rec_cfg["lora"]["rank"] if mechanism == "lora" else 0,
                 "lora_alpha": rec_cfg["lora"]["alpha"] if mechanism == "lora" else 0,
                 "lora_target_layers": ",".join(rec_cfg["lora"]["target_layers"]) if mechanism == "lora" else "",
                 "lora_include_fc": bool(rec_cfg["lora"]["include_fc"]) if mechanism == "lora" else False},
                ckpt_dir, tag, step)

    return str(saved_path)
