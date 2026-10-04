"""Chargement de config : base.yaml + override de scénario, avec résolution ${...}.

La racine des artefacts (data SCface, checkpoints, outputs) est `paths.drive_root`.
Sur Colab, c'est le chemin Drive du YAML. En local (ou tout autre environnement),
définir la variable d'environnement FORENSIC_SYNTH_ROOT pour la surcharger : c'est
le SEUL changement nécessaire pour faire tourner l'expérimentation ailleurs.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any
import yaml

_ROOT_ENV_VAR = "FORENSIC_SYNTH_ROOT"

_VAR = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _resolve(cfg: dict) -> dict:
    """Résout récursivement les ${a.b.c} dans les chaînes."""
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            prev = None
            cur = node
            while prev != cur and "${" in cur:
                prev = cur
                cur = _VAR.sub(lambda m: str(_get(cfg, m.group(1))), cur)
            return cur
        return node
    # plusieurs passes pour les références imbriquées
    out = cfg
    for _ in range(5):
        out = walk(out)
    return out


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """Applique des surcharges 'a.b.c=valeur' (valeur lue en YAML : 0.5, true, [7,42]).
    Utilisé par le notebook (cellule Paramètres) via `run.py --set` : les choix de
    l'expérience sont explicites, tracés (runs_log.jsonl) et n'exigent pas d'éditer un YAML."""
    for item in items:
        if "=" not in item:
            raise ValueError(f"Surcharge invalide (attendu cle=valeur) : {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Cle inconnue dans la config : {dotted!r} (manque '{part}')")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Cle inconnue dans la config : {dotted!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base_name = raw.pop("defaults", None)
    if base_name:
        base_path = path.parent / f"{base_name}.yaml"
        base = yaml.safe_load(base_path.read_text())
        raw = _deep_merge(base, raw)
    env_root = os.environ.get(_ROOT_ENV_VAR)
    if env_root:
        raw["paths"]["drive_root"] = env_root
    return _resolve(raw)
