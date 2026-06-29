"""Filtre PAR IMAGE des échantillons synthétiques, avant l'entraînement de
reconnaissance — complémentaire du garde-fou global (fidelity/gate.py).

Le FID est une distance entre DISTRIBUTIONS (moyenne + covariance des features
Inception sur tout un lot) : mathématiquement indéfini pour une seule image
(une covariance ne s'estime pas sur n=1). Seul le cosinus ArcFace est définissable
par image — ce filtre ne fait QUE ça : retirer du pool d'entraînement les
échantillons dont l'identité n'est pas assez préservée (probablement du bruit
pur, cf. avertissements "aucun visage détecté").

Limite connue (cf. discussion 2026-06-29) : un cosinus élevé favorise les images
NETTES (plus faciles à reconnaître pour ArcFace), pas les mieux dégradées. Ce
filtre élimine le pire (bruit), mais ne sélectionne pas une "bonne dégradation"
-- il n'a aucune notion de fidélité de dégradation, seulement de fidélité
d'identité.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from src.data.pairs import list_pairs
from src.generator.face_detect import load_aligned_face_tensor, load_face_app
from src.utils.arcface_backbone import load_arcface_embedder
from src.utils.logging import get_logger

log = get_logger()


def filter_synthetic(cfg: dict) -> dict[str, dict[str, int]]:
    """Déplace (ne supprime jamais) les images synthétiques sous le seuil de
    cosinus vers un sous-dossier 'rejected/' par identité -- réversible, et
    invisible pour le glob('*.png') de _build_pools() (recognition/finetune.py),
    qui ne regarde que le niveau racine de chaque dossier d'identité.

    Idempotent : relancer ce stage ne réévalue que les images encore au niveau
    racine (les déjà-rejetées restent dans 'rejected/', non reconsidérées)."""
    import torch

    threshold = cfg["fidelity"]["filter_cos_min"]
    real_by_id: dict[str, list[str]] = defaultdict(list)
    for p in list_pairs(cfg, block="B"):
        real_by_id[p.identity].append(p.target_path)

    synth_root = Path(cfg["paths"]["synth_dataset"])
    embedder = load_arcface_embedder(cfg)
    device = next(embedder.parameters()).device
    face_app = load_face_app(cfg)
    cache_dir = cfg["paths"].get("aligned_cache")

    counts: dict[str, dict[str, int]] = {}
    for identity, real_paths in sorted(real_by_id.items()):
        identity_dir = synth_root / identity
        synth_paths = sorted(identity_dir.glob("*.png"))
        if not synth_paths:
            continue

        real_tensors = [load_aligned_face_tensor(p, face_app, cache_dir=cache_dir) for p in real_paths]
        with torch.no_grad():
            real_emb = embedder(torch.stack(real_tensors).to(device)).mean(dim=0, keepdim=True)
            real_emb = real_emb / real_emb.norm(dim=-1, keepdim=True)

        rejected_dir = identity_dir / "rejected"
        kept, rejected = 0, 0
        for synth_path in synth_paths:
            tensor = load_aligned_face_tensor(str(synth_path), face_app, cache_dir=cache_dir)
            with torch.no_grad():
                emb = embedder(tensor.unsqueeze(0).to(device))
                emb = emb / emb.norm(dim=-1, keepdim=True)
            cos = (emb @ real_emb.T).item()
            if cos < threshold:
                rejected_dir.mkdir(exist_ok=True)
                synth_path.rename(rejected_dir / synth_path.name)
                rejected += 1
            else:
                kept += 1
        counts[identity] = {"kept": kept, "rejected": rejected}
        log.info("Identité %s : %d gardée(s), %d rejetée(s) (seuil cos>=%.2f)",
                  identity, kept, rejected, threshold)

    total_kept = sum(c["kept"] for c in counts.values())
    total_rejected = sum(c["rejected"] for c in counts.values())
    total = max(1, total_kept + total_rejected)
    log.info("Filtre synthétique terminé : %d gardée(s), %d rejetée(s) (%.1f%% retenu)",
              total_kept, total_rejected, 100 * total_kept / total)
    return counts
