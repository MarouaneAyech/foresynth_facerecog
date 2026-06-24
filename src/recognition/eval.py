"""Évaluation rank-1 sur le Bloc C (vierge). En visible : d1 SEULEMENT (cf. CLAUDE.md).

Protocole identification fermée classique : galerie = mugshots du Bloc C (1/identité),
probes = images de surveillance du Bloc C ; rank-1 = la galerie la plus proche
(cosinus embedding) partage l'identité de la probe.
"""
from __future__ import annotations
from pathlib import Path

from src.data.pairs import list_pairs
from src.utils.arcface_backbone import iresnet50, load_image_as_tensor, preprocess_for_arcface
from src.utils.logging import get_logger

log = get_logger()


def _embed_all(net, paths: list[str], device):
    import torch
    import torch.nn.functional as F
    imgs = torch.stack([load_image_as_tensor(p) for p in paths]).to(device)
    with torch.no_grad():
        emb = net(preprocess_for_arcface(imgs))
    return F.normalize(emb, dim=-1)


def evaluate(cfg: dict, weights_path: str) -> dict[str, float]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = iresnet50().to(device)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state["net"] if "net" in state else state)
    net.eval()

    results: dict[str, float] = {}
    for modality, distance in cfg["eval"]["terrains"]:
        if (modality, distance) != ("visible", "d1"):
            raise NotImplementedError(
                f"TODO(claude): évaluation non implémentée pour {modality}/{distance} "
                "(hors périmètre actuel, cf. CLAUDE.md : visible -> d1 seulement).")

        eval_cfg = {**cfg, "modality": modality, "distance": distance}
        pairs_c = list_pairs(eval_cfg, block="C")

        gallery_paths, gallery_ids, probe_paths, probe_ids = [], [], [], []
        seen: set[str] = set()
        for p in pairs_c:
            if p.identity not in seen:
                gallery_paths.append(p.mugshot_path)
                gallery_ids.append(p.identity)
                seen.add(p.identity)
            probe_paths.append(p.target_path)
            probe_ids.append(p.identity)

        gallery_emb = _embed_all(net, gallery_paths, device)
        probe_emb = _embed_all(net, probe_paths, device)
        nearest = (probe_emb @ gallery_emb.T).argmax(dim=1).tolist()
        correct = sum(gallery_ids[i] == pid for i, pid in zip(nearest, probe_ids))
        rank1 = correct / len(probe_ids)

        terrain = f"{modality}_{distance}"
        results[terrain] = rank1
        log.info("EVAL %s rank1=%.4f (%d probes, %d galerie)", terrain, rank1, len(probe_ids), len(gallery_paths))
    return results
