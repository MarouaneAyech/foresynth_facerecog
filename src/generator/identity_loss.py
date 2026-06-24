"""Perte d'identité : cosinus entre embeddings ArcFace (verrou anti-dérive).

arcface est un src.utils.arcface_backbone.ArcFaceEmbedder GELÉ : le gradient remonte
par gen_img (pixels), pas par les poids ArcFace.
"""
from __future__ import annotations
import torch


def identity_cosine_loss(gen_img: torch.Tensor, ref_embedding: torch.Tensor, arcface) -> torch.Tensor:
    """gen_img: (B,3,H,W) en [0,1]. ref_embedding: (B,512) déjà L2-normalisé (mugshot)."""
    gen_embedding = arcface(gen_img)
    cos = (gen_embedding * ref_embedding).sum(dim=-1)
    return (1.0 - cos).mean()
