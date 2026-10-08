"""À LANCER SUR COLAB (insightface + antelopev2 requis) : régénère les entrées du cache
d'alignement nécessaires à l'évaluation d'un terrain, puis zippe les nouveaux fichiers.

Utilise le code d'alignement du projet (src.generator.face_detect), donc le même format et
les mêmes clés (chemin Drive de l'image + taille) que les entraînements et évaluations. Le
zip est à télécharger puis à décompresser dans `aligned_cache/` du projet local.

Seules les images du bloc demandé (par défaut C : mugshots + sondes de surveillance) sont
traitées ; les entrées déjà présentes dans le cache ne sont pas recalculées.

La clé de chaque entrée dépend du chemin de l'IMAGE source (et non du dossier du cache) :
choisir un autre dossier avec --cache-dir ne change donc pas les clés.

Avec --synthetic, les images synthétiques des identités du bloc sont aussi traitées (utile
pour les figures de fidélité : bloc B + synthétique).

Exemple (cellule Colab, Drive monté, FORENSIC_SYNTH_ROOT défini, depuis la racine du dépôt) :
    !python analysis/build_aligned_cache.py --config configs/ir_d1.yaml --block C \
        --cache-dir /content/drive/MyDrive/forensic-synth/aligned_cache_ir
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.pairs import list_pairs  # noqa: E402
from src.generator.face_detect import _cache_path, load_aligned_face_tensor, load_face_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ir_d1.yaml")
    ap.add_argument("--block", default="C", choices=["A", "B", "C"])
    ap.add_argument("--cache-dir", default=None,
                    help="dossier du cache à remplir (défaut : paths.aligned_cache de la config)")
    ap.add_argument("--synthetic", action="store_true",
                    help="ajoute les images synthétiques (paths.synth_dataset) des identités du bloc")
    ap.add_argument("--zip", default="/content/aligned_cache_nouveaux.zip")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    cache_dir = Path(args.cache_dir or cfg["paths"]["aligned_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    pairs = list_pairs(cfg, block=args.block)
    for p in pairs:
        for q in (p.mugshot_path, p.target_path):
            if q not in paths:
                paths.append(q)
    if args.synthetic:
        synth_root = Path(cfg["paths"]["synth_dataset"])
        for identity in sorted({p.identity for p in pairs}):
            paths += [str(f) for f in sorted((synth_root / identity).glob("*.png"))]
    todo = [p for p in paths if not _cache_path(str(cache_dir), p, 112).exists()]
    print(f"{args.config} bloc {args.block} : {len(paths)} images, {len(todo)} absentes du cache")
    if not todo:
        return

    app = load_face_app(cfg)
    created = []
    for p in todo:
        load_aligned_face_tensor(p, app, cache_dir=str(cache_dir))
        created.append(_cache_path(str(cache_dir), p, 112))
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in created:
            z.write(f, arcname=f.name)
    size = os.path.getsize(args.zip) / 1e6
    print(f"{len(created)} entrées créées -> {args.zip} ({size:.1f} Mo). "
          "À télécharger puis décompresser dans aligned_cache/ du projet local.")


if __name__ == "__main__":
    main()
