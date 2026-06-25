"""Détection de visage partagée (insightface/antelopev2).

Utilisée par generator/base_arc2face.py (extraction de l'embedding ID, conditionnement)
ET par le diagnostic check_faces() (vérifier la détectabilité AVANT de lancer un
entraînement de plusieurs heures, plutôt que de le découvrir au milieu d'un run —
cf. incident identité 058 : visage présent mais cadrage trop serré pour le détecteur).
"""
from __future__ import annotations
from pathlib import Path


def link_insightface_pack(cfg: dict, pack_name: str) -> None:
    """Lie (symlink) un pack insightface déjà déposé (Drive) vers l'emplacement par
    défaut attendu par FaceAnalysis (~/.insightface/models/<pack>), sans dupliquer
    les fichiers. Sans effet si déjà présent ou si rien n'est déposé."""
    local_pack_dir = Path(cfg["paths"].get("insightface_models_dir", "")) / pack_name
    target = Path.home() / ".insightface" / "models" / pack_name
    if target.exists() or not local_pack_dir.is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(local_pack_dir, target_is_directory=True)


def load_face_app(cfg: dict):
    """Charge FaceAnalysis (détection + landmarks + embedding) sans rien d'autre
    (pas Arc2Face/diffusers) : utilisable seul pour un diagnostic rapide."""
    import torch
    from insightface.app import FaceAnalysis

    pack_name = cfg["generator"]["pretrained"]["insightface_pack"]
    link_insightface_pack(cfg, pack_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = FaceAnalysis(
        name=pack_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda" else ["CPUExecutionProvider"])
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def detect_largest_face(face_app, image_path: str):
    """Retourne le plus grand visage détecté (objet insightface Face) ou None."""
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]  # RGB->BGR (insightface)
    faces = face_app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def check_faces(cfg: dict, block: str) -> dict[str, bool]:
    """Diagnostic AVANT entraînement : détectabilité du mugshot de chaque identité
    distincte du bloc donné. Rapide (insightface seul, pas Arc2Face/diffusers)."""
    from src.data.pairs import list_pairs
    from src.utils.logging import get_logger

    log = get_logger()
    pairs = list_pairs(cfg, block=block)
    mugshot_by_identity = {p.identity: p.mugshot_path for p in pairs}
    face_app = load_face_app(cfg)

    results: dict[str, bool] = {}
    for identity, path in sorted(mugshot_by_identity.items()):
        ok = detect_largest_face(face_app, path) is not None
        results[identity] = ok
        if not ok:
            log.warning("Bloc %s, identité %s : AUCUN visage détecté (%s)", block, identity, path)

    n_fail = sum(not ok for ok in results.values())
    log.info("Bloc %s : %d/%d identités avec visage détecté (%d échec(s))",
              block, len(results) - n_fail, len(results), n_fail)
    return results
