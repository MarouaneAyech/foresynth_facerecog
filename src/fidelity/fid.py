"""FID synthétique <-> vraie surveillance (Bloc B), par (modality, distance).

Restreint volontairement aux vraies cibles du Bloc B pour le (modality, distance)
courant (pas tout scface_root, qui mélangerait mugshots/IR/autres distances).
"""
from __future__ import annotations
from pathlib import Path

from src.data.pairs import list_pairs


def compute_fid(cfg: dict) -> float:
    import torch
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    real_paths = sorted({p.target_path for p in list_pairs(cfg, block="B")})
    synth_paths = sorted(str(p) for p in Path(cfg["paths"]["synth_dataset"]).rglob("*.png"))
    if not synth_paths:
        raise RuntimeError(
            "Aucune image synthétique sous paths.synth_dataset : lancer l'étage "
            "'generate' avant 'fidelity'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dims = 2048
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)

    mu_real, sigma_real = calculate_activation_statistics(real_paths, model, device=device, dims=dims)
    mu_synth, sigma_synth = calculate_activation_statistics(synth_paths, model, device=device, dims=dims)
    return float(calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth))
