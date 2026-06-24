"""Chemins résolus depuis la config (racine = Drive en Colab). Crée les dossiers de sortie."""
from pathlib import Path
def ensure_dirs(cfg: dict) -> None:
    for key in ("outputs", "checkpoints", "synth_dataset"):
        p = cfg["paths"].get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
