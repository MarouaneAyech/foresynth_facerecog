"""Logger minimal, sûr en Colab.

get_logger() seul n'écrit que sur la console (stdout) -- perdu si la session Colab
coupe sans que le notebook ait sauvé la sortie de la cellule. attach_file_handler()
ajoute en plus une copie persistante sur Drive (cf. CLAUDE.md : logs au même endroit
que checkpoints/données), appelée une fois cfg connu (experiments.run.main).
"""
from __future__ import annotations
import logging
from pathlib import Path
import sys


def get_logger(name: str = "forensic-synth") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h); log.setLevel(logging.INFO)
    return log


def attach_file_handler(log: logging.Logger, log_dir: str | Path, filename: str) -> None:
    """Ajoute une copie des logs vers un fichier (append) sous log_dir, en plus de la
    console. Idempotent : n'ajoute pas de doublon si déjà attaché pour ce fichier."""
    path = Path(log_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path for h in log.handlers):
        return
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
