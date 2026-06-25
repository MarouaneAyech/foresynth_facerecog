"""Amorce de dégradation haute-ordre (style Real-ESRGAN) : aide la LoRA à atteindre d1.

Pipeline CLASSIQUE (pas de réseau pré-entraîné, juste des transformations
paramétriques) : flou gaussien -> resize down/up -> bruit gaussien -> compression
JPEG, empilé deux fois ("haut-ordre"). Sévérité paramétrée par distance (d1 = la plus
loin = la plus dégradée). Sert d'augmentation des cibles réelles pendant l'entraînement
de la LoRA (cfg['generator']['degrade_prior']) ET de baseline non-apprise comparable.
"""
from __future__ import annotations
import io
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# (sigma_flou, échelle_resize, std_bruit, qualité_jpeg) — plus dégradé pour d1 (4.20m).
_SEVERITY = {
    "d1": dict(blur_sigma=(1.0, 3.0), resize_scale=(0.2, 0.5), noise_std=(0.01, 0.05), jpeg_quality=(20, 50)),
    "d2": dict(blur_sigma=(0.5, 2.0), resize_scale=(0.4, 0.7), noise_std=(0.005, 0.03), jpeg_quality=(30, 60)),
    "d3": dict(blur_sigma=(0.2, 1.0), resize_scale=(0.6, 0.9), noise_std=(0.0, 0.02), jpeg_quality=(40, 80)),
}
N_PASSES = 2  # "haut-ordre" = dégradation empilée deux fois


def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = max(3, int(sigma * 3) | 1)  # impair
    return TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


def _resize_down_up(img: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = img.shape[-2:]
    small = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def _add_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    return (img + torch.randn_like(img) * std).clamp(0, 1)


def _jpeg_roundtrip(img: torch.Tensor, quality: int) -> torch.Tensor:
    # TF.to_pil_image/to_tensor passent par PIL (CPU) : le tenseur revient toujours sur
    # CPU, peu importe le device d'entrée. Rapatrié explicitement plus bas (high_order_degrade).
    pil = TF.to_pil_image(img.detach().cpu().clamp(0, 1))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return TF.to_tensor(Image.open(buf).convert("RGB"))


def high_order_degrade(img: torch.Tensor, distance: str, rng_seed: int | None = None) -> torch.Tensor:
    """img: (3,H,W) en [0,1]. distance: 'd1'|'d2'|'d3'. Retourne une version dégradée."""
    if distance not in _SEVERITY:
        raise NotImplementedError(f"TODO(claude): sévérité de dégradation non définie pour distance={distance}")
    device = img.device
    rng = random.Random(rng_seed)
    sev = _SEVERITY[distance]
    out = img
    for _ in range(N_PASSES):
        out = _gaussian_blur(out, rng.uniform(*sev["blur_sigma"]))
        out = _resize_down_up(out, rng.uniform(*sev["resize_scale"]))
        out = _add_noise(out, rng.uniform(*sev["noise_std"]))
        out = _jpeg_roundtrip(out, rng.randint(*sev["jpeg_quality"]))
    return out.to(device)  # _jpeg_roundtrip repasse toujours par CPU (PIL)
