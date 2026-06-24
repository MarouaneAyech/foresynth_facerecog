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

N_CAMS_VISIBLE = 7  # caméras 1 à 7, identiques pour d1/d2/d3 (cf. SCface)


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


def list_pairs(cfg: dict, block: str) -> list[FacePair]:
    """Retourne les paires (mugshot, cible) d'un bloc ("A"/"B"/"C") pour
    cfg['modality']/cfg['distance']. Seul visible/d1 est implémenté dans cette itération
    (cf. CLAUDE.md, périmètre actuel) ; les autres combinaisons restent des stubs."""
    modality, distance = cfg["modality"], cfg["distance"]
    if modality != "visible" or distance != "d1":
        raise NotImplementedError(
            f"TODO(claude): list_pairs non implémenté pour modality={modality} "
            f"distance={distance} (hors périmètre de l'itération actuelle)."
        )

    scface_root = Path(cfg["paths"]["scface_root"])
    parts = load_partition_validated(cfg)
    identities = parts[block]

    pairs: list[FacePair] = []
    for identity in identities:
        mugshot_path = _mugshot_path(scface_root, identity)
        if not mugshot_path.exists():
            raise FileNotFoundError(f"Mugshot manquant pour l'identité {identity}: {mugshot_path}")
        for target_path in _visible_target_paths(scface_root, identity, distance):
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
