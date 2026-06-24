"""Interface unique de génération + fabrique (factory) indexée par (modality, distance).

C'EST le point de flexibilité IR : un seul contrat, quelle que soit la modalité.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Sequence
from src.data.pairs import FacePair


@runtime_checkable
class Generator(Protocol):
    def fit(self, pairs: Sequence[FacePair]) -> None:
        """Entraîne l'adaptateur (LoRA) sur des paires RÉELLES (Bloc A)."""
        ...

    def sample(self, mugshot_path: str, k: int) -> list[str]:
        """Génère k images synthétiques (modality, distance) ; renvoie les chemins."""
        ...


def build_generator(cfg: dict) -> Generator:
    """Sélectionne le générateur selon la config. Étend ici pour l'IR sans toucher au reste."""
    backbone = cfg["generator"]["backbone"]
    adapter = cfg["generator"]["adapter"]
    if backbone == "arc2face":
        from src.generator.base_arc2face import Arc2FaceGenerator
        return Arc2FaceGenerator(cfg, adapter=adapter)
    raise ValueError(f"Backbone générateur inconnu : {backbone}")
