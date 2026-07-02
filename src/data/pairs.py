"""Chargement des paires (mugshot, cible) pour une (modality, distance) donnée.

La 'cible' = vraie image de surveillance correspondante (Bloc A pour entraîner la LoRA).
Une identité a plusieurs cibles (une par caméra) : list_pairs retourne une FacePair par
(identité, caméra), pas une seule par identité (cf. CLAUDE.md "diversité intra-classe").

ALIGNEMENT : aligner mugshot ET cible avec le MÊME pipeline que le backbone ArcFace
(112x112, 5 points). Source d'échec n°1 si négligé. Pas fait ici : list_pairs ne fait
que résoudre des chemins valides, l'alignement est la responsabilité du chargeur
d'images (generator/), pour garder cette fonction pure et testable sans I/O image.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from src.data.partition import run as load_partition_validated

# SCface (Grgic et al.) : 5 caméras VISIBLES (1-5) + 2 caméras à capacité infrarouge/
# vision nocturne (6-7) -- confirmé visuellement le 2026-06-27 (cam_6/cam_7 sont en
# niveaux de gris, éclairage IR caractéristique, alors que cam_1..5 sont en couleur).
# cam_{6,7} sous "surveillance_cameras_distance_N" sont les captures IR à distance
# (distinctes de surveillance_cameras_IR_cam8, qui sont des MUGSHOTS IR, pas des
# captures à distance). Les inclure comme cibles "visible" mélangeait silencieusement
# ~28,6% d'images infrarouges dans tout le pipeline (génération, fidelity, reconnaissance).
N_CAMS_VISIBLE = 5  # caméras 1 à 5 SEULEMENT pour la modalité 'visible'
N_CAMS_IR = 2       # caméras 6 et 7 SEULEMENT pour la modalité 'ir'
                    # (surveillance_cameras_IR_cam8 = mugshots IR, pas des captures à distance)


@dataclass
class FacePair:
    identity: str
    mugshot_path: str
    target_path: str        # surveillance réelle (modality, distance)
    modality: str
    distance: str


def _mugshot_path(scface_root: Path, identity: str) -> Path:
    return scface_root / "mugshot_frontal_cropped_all" / f"{identity}_frontal.JPG"


def _visible_target_paths(scface_root: Path, identity: str, distance: str) -> list[Path]:
    distance_num = distance[1]  # "d1" -> "1"
    dist_dir = scface_root / f"surveillance_cameras_distance_{distance_num}"
    return [dist_dir / f"cam_{cam}" / f"{identity}_cam{cam}_1.jpg"
            for cam in range(1, N_CAMS_VISIBLE + 1)]


def _ir_target_paths(scface_root: Path, identity: str, distance: str) -> list[Path]:
    """Caméras IR 6 et 7, même arborescence que le visible
    (surveillance_cameras_distance_N/cam_6|7) mais images en niveaux de gris
    éclairage IR. surveillance_cameras_IR_cam8 = mugshots IR (pas des cibles
    de distance), non utilisé ici."""
    distance_num = distance[1]  # "d1" -> "1"
    dist_dir = scface_root / f"surveillance_cameras_distance_{distance_num}"
    first_cam_ir = N_CAMS_VISIBLE + 1  # 6
    return [dist_dir / f"cam_{cam}" / f"{identity}_cam{cam}_1.jpg"
            for cam in range(first_cam_ir, first_cam_ir + N_CAMS_IR)]


def list_pairs(cfg: dict, block: str) -> list[FacePair]:
    """Retourne les paires (mugshot, cible) d'un bloc ("A"/"B"/"C") pour
    cfg['modality']/cfg['distance'].

    Implémenté : visible/d1, ir/d1.
    Stubs (NotImplementedError) : visible/d2, visible/d3, ir/d2, ir/d3."""
    modality, distance = cfg["modality"], cfg["distance"]
    if distance != "d1":
        raise NotImplementedError(
            f"TODO(claude): list_pairs non implémenté pour distance={distance} "
            f"(d2/d3 saturés en visible, cf. CLAUDE.md ; hors périmètre actuel pour IR)."
        )
    if modality not in ("visible", "ir"):
        raise NotImplementedError(
            f"TODO(claude): list_pairs non implémenté pour modality={modality}."
        )

    scface_root = Path(cfg["paths"]["scface_root"])
    parts = load_partition_validated(cfg)
    identities = parts[block]

    target_fn = _visible_target_paths if modality == "visible" else _ir_target_paths

    pairs: list[FacePair] = []
    for identity in identities:
        mugshot_path = _mugshot_path(scface_root, identity)
        if not mugshot_path.exists():
            raise FileNotFoundError(f"Mugshot manquant pour l'identité {identity}: {mugshot_path}")
        for target_path in target_fn(scface_root, identity, distance):
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Cible surveillance manquante pour l'identité {identity}: {target_path}")
            pairs.append(FacePair(
                identity=identity,
                mugshot_path=str(mugshot_path),
                target_path=str(target_path),
                modality=modality,
                distance=distance,
            ))
    return pairs
