"""Évaluation rank-1 sur le Bloc C (vierge). En visible : d1 SEULEMENT (cf. CLAUDE.md).

Protocole identification fermée classique : galerie = mugshots du Bloc C (1/identité),
probes = images de surveillance du Bloc C ; rank-1 = la galerie la plus proche
(cosinus embedding) partage l'identité de la probe.

Images chargées ALIGNÉES par landmarks (cf. src.generator.face_detect), pas
simplement redimensionnées : un ArcFace gelé est très sensible au désalignement,
audité contre le code de référence (old_code_paper_classB, cache pré-aligné via
antelopev2) le 2026-06-27 -- explique une baseline sans fine-tuning anormalement
basse avant ce correctif.
"""
from __future__ import annotations

from src.data.pairs import list_pairs
from src.generator.face_detect import load_aligned_face_tensor, load_face_app
from src.utils.arcface_backbone import iresnet50, preprocess_for_arcface
from src.utils.logging import get_logger

log = get_logger()


def _load_aligned_batch(paths: list[str], ids: list[str], face_app) -> tuple[list, list[str]]:
    """Charge en aligné, EXCLUT silencieusement (log warning) les images sans visage
    détecté plutôt que de planter toute l'évaluation pour un cas atypique."""
    tensors, kept_ids = [], []
    for path, identity in zip(paths, ids):
        try:
            tensors.append(load_aligned_face_tensor(path, face_app))
            kept_ids.append(identity)
        except ValueError as e:
            log.warning("Exclu de l'évaluation (visage non détecté) : %s", e)
    return tensors, kept_ids


def _embed(tensors: list, net, device):
    import torch
    import torch.nn.functional as F
    imgs = torch.stack(tensors).to(device)
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
    face_app = load_face_app(cfg)

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

        gallery_tensors, gallery_ids = _load_aligned_batch(gallery_paths, gallery_ids, face_app)
        valid_gallery_ids = set(gallery_ids)
        # Une probe dont la galerie de référence est introuvable (mugshot non détecté)
        # ne peut de toute façon jamais être correctement appariée : exclue aussi.
        kept = [(p, i) for p, i in zip(probe_paths, probe_ids) if i in valid_gallery_ids]
        probe_paths, probe_ids = [p for p, _ in kept], [i for _, i in kept]
        probe_tensors, probe_ids = _load_aligned_batch(probe_paths, probe_ids, face_app)

        gallery_emb = _embed(gallery_tensors, net, device)
        probe_emb = _embed(probe_tensors, net, device)
        nearest = (probe_emb @ gallery_emb.T).argmax(dim=1).tolist()
        correct = sum(gallery_ids[i] == pid for i, pid in zip(nearest, probe_ids))
        rank1 = correct / len(probe_ids)

        terrain = f"{modality}_{distance}"
        results[terrain] = rank1
        log.info("EVAL %s rank1=%.4f (%d probes, %d galerie)", terrain, rank1, len(probe_ids), len(gallery_ids))
    return results
