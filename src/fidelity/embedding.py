"""Proximité d'identité : cosinus ArcFace synthétique vs vrai (même id, Bloc B).

Réutilise l'embedder ArcFace partagé (src.utils.arcface_backbone) : c'est la même
mesure d'identité que la perte de generator/identity_loss.py, pour rester cohérent
entre ce qui guide l'entraînement et ce qui sert de garde-fou go/no-go.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from src.data.pairs import list_pairs
from src.utils.arcface_backbone import load_arcface_embedder, load_image_as_tensor


def _embed_paths(paths: list[str], embedder, device, batch_size: int = 32):
    import torch
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([load_image_as_tensor(p) for p in paths[i:i + batch_size]]).to(device)
            embs.append(embedder(batch).cpu())
    return torch.cat(embs, dim=0)


def mean_identity_cosine(cfg: dict) -> float:
    """Moyenne, par identité du Bloc B, du cosinus entre embeddings ArcFace des
    échantillons synthétiques (paths.synth_dataset) et des vraies cibles de
    surveillance (modality/distance courants)."""
    real_by_id: dict[str, list[str]] = defaultdict(list)
    for p in list_pairs(cfg, block="B"):
        real_by_id[p.identity].append(p.target_path)

    synth_root = Path(cfg["paths"]["synth_dataset"])
    embedder = load_arcface_embedder(cfg)
    device = next(embedder.parameters()).device

    per_identity_cos = []
    for identity, real_paths in real_by_id.items():
        synth_paths = sorted(str(p) for p in (synth_root / identity).glob("*.png"))
        if not synth_paths:
            continue  # pas encore généré pour cette identité
        real_emb = _embed_paths(real_paths, embedder, device)
        synth_emb = _embed_paths(synth_paths, embedder, device)
        per_identity_cos.append((synth_emb @ real_emb.T).mean().item())

    if not per_identity_cos:
        raise RuntimeError(
            "Aucune image synthétique sous paths.synth_dataset pour le Bloc B : "
            "lancer l'étage 'generate' avant 'fidelity'.")
    return sum(per_identity_cos) / len(per_identity_cos)
