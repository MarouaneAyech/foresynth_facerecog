"""Exemples illustrant l'écart d'identité du synthétique, pour une identité du bloc B.

Reproduit la mesure du stage `fidelity` (embedding.identity_cosine_distribution) : cosinus
ArcFace entre CHAQUE image synthétique et la moyenne normalisée des embeddings des vraies
images de surveillance de la même identité. Le modèle ArcFace est le modèle pré-entraîné
(reconstruit depuis un checkpoint LoRA en annulant les matrices B, cf. extract_embeddings.py).

Sorties : analysis/outputs/<terrain>/identity_<id>_cosines.csv et figures/ch4-identite-<id>.pdf
(mugshot, vraies images de surveillance, meilleures et moins bonnes images synthétiques).

Usage : .venv-analysis/Scripts/python analysis/identity_examples.py --identity 001
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FORENSIC_SYNTH_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from extract_embeddings import build_net, cache_file, find_checkpoints  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.pairs import list_pairs  # noqa: E402
from src.utils.arcface_backbone import preprocess_for_arcface  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "pdf.fonttype": 42})
THRESHOLD = 0.45


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", default="001")
    ap.add_argument("--terrain", default="visible_d1", choices=["visible_d1", "ir_d1"])
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--n", type=int, default=4, help="nombre d'exemples par extrême")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / f"{args.terrain}.yaml")
    ckpt_dir = Path(args.ckpt_dir or ROOT / "checkpoints" / args.terrain.replace("_", " "))
    ref_ckpt = next(iter(find_checkpoints(ckpt_dir, [], False).values()))
    net = build_net(ref_ckpt, zero_lora=True)  # modèle pré-entraîné non adapté
    caches = [Path(cfg["paths"]["aligned_cache"]), ROOT / "aligned_cache_ir"]

    def tensor_of(path) -> torch.Tensor:
        f = next(c for d in caches if (c := cache_file(path, d)).exists())
        return torch.load(f)

    @torch.no_grad()
    def embed(tensors) -> torch.Tensor:
        return F.normalize(net(preprocess_for_arcface(torch.stack(tensors))), dim=-1)

    pairs = [p for p in list_pairs(cfg, block="B") if p.identity == args.identity]
    if not pairs:
        raise SystemExit(f"Identité {args.identity} absente du bloc B")
    real_paths = [p.target_path for p in pairs]
    mugshot = pairs[0].mugshot_path
    real_emb = embed([tensor_of(p) for p in real_paths]).mean(0, keepdim=True)
    real_emb = real_emb / real_emb.norm(dim=-1, keepdim=True)

    synth_dir = Path(cfg["paths"]["synth_dataset"]) / args.identity
    synth = sorted(synth_dir.glob("*.png"))
    cos = [(float((embed([tensor_of(p)]) @ real_emb.T).item()), p) for p in synth]
    cos_real = [float((embed([tensor_of(p)]) @ real_emb.T).item()) for p in real_paths]
    cos.sort(key=lambda t: t[0], reverse=True)

    out_csv = ROOT / "analysis" / "outputs" / args.terrain / f"identity_{args.identity}_cosines.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "cosinus_avec_identite_reelle"])
        w.writerows([[p.name, f"{c:.4f}"] for c, p in cos])
    vals = [c for c, _ in cos]
    print(f"identité {args.identity} : {len(vals)} images synthétiques, cosinus moyen {sum(vals)/len(vals):.3f}, "
          f"min {min(vals):.3f}, max {max(vals):.3f}, {sum(v >= THRESHOLD for v in vals)} >= {THRESHOLD}")
    for c, p in cos:
        print(f"  {p.name}  {c:.3f}")
    print("vraies images (cosinus avec la moyenne, y compris elles-mêmes) :",
          ", ".join(f"{Path(p).name}:{c:.2f}" for p, c in zip(real_paths, cos_real)))

    # --- figure : références, meilleures, moins bonnes ---
    best, worst = cos[: args.n], cos[-args.n:][::-1]
    n = args.n
    fig, axes = plt.subplots(3, max(n, 1 + len(real_paths)), figsize=(2.0 * max(n, 1 + len(real_paths)), 6.9))
    for ax in axes.ravel():
        ax.axis("off")
    ax = axes[0, 0]
    ax.imshow(Image.open(mugshot).convert("RGB")); ax.set_title("Mugshot", fontsize=9)
    for j, p in enumerate(real_paths):
        ax = axes[0, 1 + j]
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(f"Réel, cam. {Path(p).name.split('_cam')[1][0]}", fontsize=9)
    for row, (label, items) in zip((1, 2), (("Meilleures", best), ("Moins bonnes", worst))):
        for j, (c, p) in enumerate(items):
            ax = axes[row, j]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(f"{label[:-1] if False else ''}{c:.2f}", fontsize=10,
                         color="#1c5aa6" if c >= THRESHOLD else "#b8420f")
        axes[row, 0].text(-0.05, 0.5, label, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=10)
    fig.tight_layout()
    out_pdf = ROOT / "figures" / f"ch4-identite-{args.identity}.pdf"
    fig.savefig(out_pdf, dpi=200)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=110)
    print("figure écrite :", out_pdf)


if __name__ == "__main__":
    main()
