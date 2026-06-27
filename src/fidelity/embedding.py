"""Proximité d'identité : cosinus ArcFace synthétique vs vrai (même id, Bloc B).

Réutilise l'embedder ArcFace partagé (src.utils.arcface_backbone) : c'est la même
mesure d'identité que la perte de generator/identity_loss.py, pour rester cohérent
entre ce qui guide l'entraînement et ce qui sert de garde-fou go/no-go.

Images chargées ALIGNÉES par landmarks (cf. CLAUDE.md : "même pipeline que le
backbone ArcFace"), pas simplement redimensionnées -- cohérent avec recognition/
(audité contre old_code_paper_classB le 2026-06-27).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from src.data.pairs import list_pairs
from src.generator.face_detect import load_aligned_face_tensor, load_face_app
from src.utils.arcface_backbone import load_arcface_embedder
from src.utils.logging import get_logger

log = get_logger()


def _embed_paths(paths: list[str], embedder, device, face_app, batch_size: int = 32,
                  cache_dir: str | None = None):
    """load_aligned_face_tensor ne lève jamais d'erreur (repli sur un simple resize si
    aucun visage détecté, cf. code de référence) : aucune exclusion ici."""
    import torch

    tensors = [load_aligned_face_tensor(p, face_app, cache_dir=cache_dir) for p in paths]
    if not tensors:
        return None
    embs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(device)
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
    face_app = load_face_app(cfg)
    cache_dir = cfg["paths"].get("aligned_cache")

    per_identity_cos = []
    for identity, real_paths in real_by_id.items():
        synth_paths = sorted(str(p) for p in (synth_root / identity).glob("*.png"))
        if not synth_paths:
            continue  # pas encore généré pour cette identité
        real_emb = _embed_paths(real_paths, embedder, device, face_app, cache_dir=cache_dir)
        synth_emb = _embed_paths(synth_paths, embedder, device, face_app, cache_dir=cache_dir)
        if real_emb is None or synth_emb is None:
            continue
        per_identity_cos.append((synth_emb @ real_emb.T).mean().item())

    if not per_identity_cos:
        raise RuntimeError(
            "Aucune image synthétique sous paths.synth_dataset pour le Bloc B : "
            "lancer l'étage 'generate' avant 'fidelity'.")
    return sum(per_identity_cos) / len(per_identity_cos)
