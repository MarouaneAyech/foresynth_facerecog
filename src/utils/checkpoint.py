"""Sauvegarde/reprise générique pour survivre aux coupures Colab.

torch importé en local (lazy) pour rester utilisable (résolution de chemins,
métadonnées) sans dépendance torch dans les environnements qui n'en ont pas besoin.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


def latest_checkpoint(ckpt_dir: str | Path, tag: str) -> Optional[Path]:
    d = Path(ckpt_dir)
    cands = sorted(d.glob(f"{tag}_step*.ckpt"))
    return cands[-1] if cands else None


def save_checkpoint(state: dict[str, Any], ckpt_dir: str | Path, tag: str, step: int) -> Path:
    import torch
    d = Path(ckpt_dir); d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tag}_step{step:07d}.ckpt"
    # Écriture ATOMIQUE (tmp + rename) : une coupure Colab en plein torch.save() sur
    # le chemin final laisserait un .ckpt tronqué qui se recharge silencieusement
    # avec des poids garbage (pas d'erreur claire) -> LoRA corrompue indétectée,
    # génération en bruit pur (incident constaté le 2026-06-28, coupure pile au pas
    # 3800 = multiple exact de ckpt_every). os.replace est atomique sur Windows/POSIX
    # une fois l'écriture du fichier temporaire terminée.
    tmp_path = path.with_suffix(".ckpt.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    meta = {k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))}
    meta_path = d / f"{tag}_latest.json"
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps({"step": step, "path": str(path), "meta": meta}, indent=2))
    tmp_meta.replace(meta_path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu")


def resume_step(ckpt_dir: str | Path, tag: str) -> int:
    meta = Path(ckpt_dir) / f"{tag}_latest.json"
    if meta.exists():
        return int(json.loads(meta.read_text()).get("step", 0))
    return 0
