"""Perte d'identité : cosinus entre embeddings ArcFace (verrou anti-dérive).

arcface est un src.utils.arcface_backbone.ArcFaceEmbedder GELÉ : le gradient remonte
par gen_img (pixels), pas par les poids ArcFace.
"""
from __future__ import annotations
import torch


def identity_cosine_loss(gen_img: torch.Tensor, ref_embedding: torch.Tensor, arcface,
                          weights: torch.Tensor | None = None) -> torch.Tensor:
    """gen_img: (B,3,H,W) en [0,1]. ref_embedding: (B,512) déjà L2-normalisé (mugshot).

    weights (B,), optionnel : pondère la contribution de chaque échantillon (ex. par
    alphas_cumprod[timesteps]). gen_img provient ici d'une estimation x0_pred en un
    seul pas — fiable à faible bruit, quasi inexploitable à fort bruit ; sans
    pondération, le gradient utile (peu de pas à faible bruit) est noyé par le bruit
    des pas à fort bruit (tirés aléatoirement, majoritaires)."""
    gen_embedding = arcface(gen_img)
    cos = (gen_embedding * ref_embedding).sum(dim=-1)
    per_sample = 1.0 - cos
    if weights is None:
        return per_sample.mean()
    weights = weights.to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
