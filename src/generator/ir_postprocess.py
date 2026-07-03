"""Post-traitement déterministe des images synthétiques IR.

Arc2Face (SD1.5) génère toujours du RGB. Avec une LoRA seule, le résultat est
incohérent (certaines images grises, d'autres fade-couleur) : le prior SD1.5
résiste à la transformation spectrale IR de haut rang (cf. CLAUDE.md "Crochet IR").
Ce module normalise toutes les images générées pour correspondre visuellement aux
caméras IR SCface (cam_6/cam_7) :

  1. Niveaux de gris (luminance rec.601) — caractéristique fondamentale de l'IR
  2. Flou gaussien — optique caméra IR + distance d1 (4.20m)
  3. Bruit additif gaussien — approximation du shot noise du capteur IR

Appliqué après 'generate', avant 'fidelity'. Modifie les PNG in-place dans
synth_dataset. Idempotent : relancer avec les mêmes paramètres donne un résultat
stable (flou+bruit sur une image déjà grise). Le bruit est déterministe par image
(graine = hash du nom de fichier) pour assurer la reproductibilité entre relances.

Paramètres calibrables dans configs/ir_d1.yaml (section ir_postprocess) :
  sigma_blur : rayon du flou gaussien en pixels (espace 512x512 généré)
  noise_std  : écart-type du bruit additif (espace [0,1])
"""
from __future__ import annotations
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

from src.utils.logging import get_logger

log = get_logger()


def _apply_ir(img: Image.Image, sigma_blur: float, noise_std: float,
              rng: np.random.Generator) -> Image.Image:
    # 1. Niveaux de gris (luminance rec.601 : 0.299R + 0.587G + 0.114B)
    gray = img.convert("L")

    # 2. Flou gaussien (optique caméra IR + propagation à distance)
    if sigma_blur > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma_blur))

    # 3. Bruit additif (shot noise capteur IR, approximé par gaussienne)
    if noise_std > 0:
        arr = np.array(gray, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, noise_std, arr.shape).astype(np.float32), 0.0, 1.0)
        gray = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    # Retourne en RGB (3 canaux identiques) : reste compatible avec tout le pipeline
    # aval (ArcFace, filter_synthetic, fidelity — tous attendent 3 canaux).
    return gray.convert("RGB")


def _seed_from_stem(stem: str) -> int:
    """Graine déterministe depuis le nom de fichier (ex. '001_007').
    Portable entre machines (pas de dépendance à PYTHONHASHSEED ou au chemin absolu)."""
    return int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % (2 ** 32)


def ir_postprocess_dataset(cfg: dict) -> None:
    """Applique le post-traitement IR in-place sur toutes les images du synth_dataset.

    Lève ValueError si modality != 'ir' (évite d'appliquer par erreur sur le visible).
    Lève FileNotFoundError si synth_dataset n'existe pas encore (generate doit tourner avant).
    """
    if cfg["modality"] != "ir":
        raise ValueError(
            f"ir_postprocess est réservé à modality=ir (reçu : {cfg['modality']}). "
            "Ce stage n'a pas de sens sur le visible.")

    pp_cfg = cfg.get("ir_postprocess", {})
    sigma_blur: float = pp_cfg.get("sigma_blur", 1.5)
    noise_std: float = pp_cfg.get("noise_std", 0.02)

    synth_root = Path(cfg["paths"]["synth_dataset"])
    if not synth_root.is_dir():
        raise FileNotFoundError(
            f"synth_dataset introuvable : {synth_root}. Lancer d'abord le stage 'generate'.")

    total = 0
    for identity_dir in sorted(synth_root.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = sorted(identity_dir.glob("*.png"))
        if not images:
            continue
        for img_path in images:
            rng = np.random.default_rng(_seed_from_stem(img_path.stem))
            processed = _apply_ir(Image.open(img_path).convert("RGB"), sigma_blur, noise_std, rng)
            processed.save(img_path)
        total += len(images)
        log.info("ir_postprocess : %s → %d images", identity_dir.name, len(images))

    log.info("ir_postprocess terminé : %d images (sigma_blur=%.2f, noise_std=%.4f)",
             total, sigma_blur, noise_std)
