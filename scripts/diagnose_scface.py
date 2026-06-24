"""Diagnostic rapide de la base SCface avant expérimentation.

Vérifie, pour le périmètre visible/d1 (mugshots + surveillance distance 1) :
- couverture des 130 identités (mugshots, et par caméra distance 1)
- validité des images (ouverture PIL, taille non nulle, résolution, mode couleur)
- cohérence avec les listes officielles SCface (distance1_cam*.txt)

Fait aussi un comptage léger (présence/taille) sur distance 2/3, IR et rotation,
pour confirmer que rien ne manque pour les itérations futures (sans validation profonde).

Usage : python scripts/diagnose_scface.py [chemin_vers_SCface_database]
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, UnidentifiedImageError

N_IDENTITIES = 130
CAMS_D1 = range(1, 8)


def validate_image(path: Path) -> dict:
    info = {"path": str(path), "ok": True, "error": None, "size_bytes": 0,
            "width": None, "height": None, "mode": None}
    try:
        info["size_bytes"] = path.stat().st_size
        if info["size_bytes"] == 0:
            info["ok"] = False
            info["error"] = "fichier vide (0 octet)"
            return info
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() invalide l'objet, ré-ouvrir pour lire les infos
            info["width"], info["height"] = img.size
            info["mode"] = img.mode
    except (UnidentifiedImageError, OSError) as e:
        info["ok"] = False
        info["error"] = str(e)
    return info


def scan_dir(d: Path) -> list[dict]:
    return [validate_image(p) for p in sorted(d.glob("*")) if p.is_file()]


def summarize(label: str, results: list[dict]) -> None:
    n = len(results)
    bad = [r for r in results if not r["ok"]]
    sizes = [(r["width"], r["height"]) for r in results if r["ok"]]
    distinct_res = sorted(set(sizes))
    print(f"\n--- {label} ---")
    print(f"  fichiers: {n} | corrompus/vides: {len(bad)}")
    if distinct_res:
        if len(distinct_res) <= 5:
            print(f"  résolutions: {distinct_res}")
        else:
            ws = [w for w, h in sizes]
            hs = [h for w, h in sizes]
            print(f"  résolutions: {len(distinct_res)} distinctes "
                  f"(largeur {min(ws)}-{max(ws)}, hauteur {min(hs)}-{max(hs)})")
    for r in bad:
        print(f"  [INVALIDE] {r['path']}: {r['error']}")


def expected_ids() -> list[str]:
    return [f"{i:03d}" for i in range(1, N_IDENTITIES + 1)]


def check_mugshots(root: Path) -> None:
    for sub in ("mugshot_frontal_cropped_all", "mugshot_frontal_original_all"):
        d = root / sub
        if not d.exists():
            print(f"\n--- {sub} ---\n  [MANQUANT] dossier absent: {d}")
            continue
        results = scan_dir(d)
        summarize(sub, results)
        ids_present = {p.name[:3] for p in d.glob("*") if p.is_file()}
        missing = sorted(set(expected_ids()) - ids_present)
        if missing:
            print(f"  [MANQUANT] {len(missing)} identité(s) sans mugshot: {missing}")


def check_distance1(root: Path) -> None:
    d1_root = root / "surveillance_cameras_distance_1"
    if not d1_root.exists():
        print(f"\n--- surveillance_cameras_distance_1 ---\n  [MANQUANT] dossier absent: {d1_root}")
        return
    all_results = []
    per_id_cam = defaultdict(set)
    for cam in CAMS_D1:
        cam_dir = d1_root / f"cam_{cam}"
        if not cam_dir.exists():
            print(f"\n--- distance_1/cam_{cam} ---\n  [MANQUANT] dossier absent: {cam_dir}")
            continue
        results = scan_dir(cam_dir)
        summarize(f"surveillance_cameras_distance_1/cam_{cam}", results)
        all_results.extend(results)
        for p in cam_dir.glob("*"):
            if p.is_file():
                per_id_cam[p.name[:3]].add(cam)

        # cohérence avec la liste officielle distance1_camN.txt
        list_file = root / f"distance1_cam{cam}.txt"
        if list_file.exists():
            expected = {line.split()[0].split("_")[0] for line in
                        list_file.read_text().splitlines() if line.strip()}
            present = {p.name[:3] for p in cam_dir.glob("*") if p.is_file()}
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            if missing:
                print(f"  [MANQUANT vs liste officielle] {len(missing)} id manquante(s): {missing}")
            if extra:
                print(f"  [INATTENDU vs liste officielle] {len(extra)} id en trop: {extra}")
        else:
            print(f"  [INFO] liste officielle absente: {list_file.name}")

    missing_ids = [i for i in expected_ids() if len(per_id_cam.get(i, set())) < len(list(CAMS_D1))]
    if missing_ids:
        print(f"\n  [RESUME] {len(missing_ids)} identité(s) sans les 7 caméras d1 complètes:")
        for i in missing_ids:
            have = sorted(per_id_cam.get(i, set()))
            print(f"    id {i}: caméras présentes = {have}")
    else:
        print(f"\n  [RESUME] les {N_IDENTITIES} identités ont leurs 7 caméras distance 1 complètes.")


def check_future_scope_lightweight(root: Path) -> None:
    print("\n=== Comptage léger (hors périmètre actuel visible/d1) ===")
    for sub in ("mugshot_rotation_all", "surveillance_cameras_distance_2",
                "surveillance_cameras_distance_3", "surveillance_cameras_IR_cam8",
                "surveillance_cameras_all"):
        d = root / sub
        if not d.exists():
            print(f"  [MANQUANT] {sub}: dossier absent")
            continue
        n = sum(1 for p in d.rglob("*") if p.is_file())
        sizes = [p.stat().st_size for p in d.rglob("*") if p.is_file()]
        n_empty = sum(1 for s in sizes if s == 0)
        print(f"  {sub}: {n} fichier(s), {n_empty} vide(s)")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "data" / "SCface_database"
    if not root.exists():
        print(f"[ERREUR] chemin introuvable: {root}")
        sys.exit(1)
    print(f"Racine SCface: {root}")

    check_mugshots(root)
    check_distance1(root)
    check_future_scope_lightweight(root)


if __name__ == "__main__":
    main()
