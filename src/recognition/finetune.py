"""Fine-tuning de reconnaissance — RECETTE B1 INCHANGÉE.

Seule variable : la SOURCE des images de surveillance (real | synthetic | mixed).
Scope=layer3+4 (dégèle layer3/layer4/bn2/fc/features, gèle le reste), AdamW,
ancrage 50/50 mugshot/surveillance, reprenable (checkpoint), multi-seed (cf. cfg['seeds']).

Identités d'entraînement = Bloc B (cf. CLAUDE.md : B sert à valider+générer le
synthétique ET à fine-tuner le recognizer ; C reste vierge pour l'évaluation finale).
"""
from __future__ import annotations
import random
from collections import defaultdict
from pathlib import Path

from src.data.pairs import list_pairs
from src.recognition.arcmargin import ArcMarginProduct
from src.utils.arcface_backbone import EMBEDDING_SIZE, iresnet50, load_image_as_tensor, preprocess_for_arcface
from src.utils.checkpoint import latest_checkpoint, load_checkpoint, resume_step, save_checkpoint
from src.utils.logging import get_logger

log = get_logger()

SCOPE_TRAINABLE_PREFIXES = {
    "layer3_4": ("layer3", "layer4", "bn2", "fc", "features"),
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


def _build_pools(cfg: dict, condition: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Pools mugshot/surveillance par identité (Bloc B), selon la source demandée."""
    if condition not in ("real", "synthetic", "mixed"):
        raise ValueError(f"Condition inconnue : {condition} (attendu real|synthetic|mixed).")

    pairs_b = list_pairs(cfg, block="B")
    mugshot_of: dict[str, list[str]] = {}
    surveillance_of: dict[str, list[str]] = defaultdict(list)
    for p in pairs_b:
        mugshot_of.setdefault(p.identity, [p.mugshot_path])
        if condition in ("real", "mixed"):
            surveillance_of[p.identity].append(p.target_path)

    if condition in ("synthetic", "mixed"):
        synth_root = Path(cfg["paths"]["synth_dataset"])
        for identity in mugshot_of:
            files = sorted(str(f) for f in (synth_root / identity).glob("*.png"))
            if not files:
                raise RuntimeError(
                    f"Aucune image synthétique pour l'identité {identity} sous {synth_root} "
                    f"(condition={condition}). Lancer l'étage 'generate' avant 'train_recognition'.")
            surveillance_of[identity].extend(files)
    return mugshot_of, surveillance_of


def train(cfg: dict, condition: str, seed: int) -> str:
    import torch
    import torch.nn.functional as F
    from src.utils.seed import set_seed

    set_seed(seed)
    rec_cfg = cfg["recognition"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mugshot_of, surveillance_of = _build_pools(cfg, condition)
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
    max_steps = rec_cfg["epochs"] * steps_per_epoch

    rng = random.Random(seed)
    saved_path = ckpt_path
    while step < max_steps:
        imgs, labels = [], []
        for _ in range(batch_size):
            identity = rng.choice(identities)
            pool = mugshot_of if rng.random() < rec_cfg["anchor_ratio"] else surveillance_of
            imgs.append(load_image_as_tensor(rng.choice(pool[identity])))
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
            log.info("[%s seed=%d] step=%d/%d loss=%.4f", condition, seed, step, max_steps, loss.item())
        if step % rec_cfg["ckpt_every"] == 0 or step == max_steps:
            saved_path = save_checkpoint(
                {"net": net.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_dir, tag, step)

    return str(saved_path)
