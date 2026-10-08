"""Figure d'exemples de fidélité d'identité : une identité bien préservée (cosinus >= 0,45)
face à une identité mal préservée, avec le cosinus de chaque image synthétique.

Le cosinus est celui du stage `fidelity` (cf. identity_examples.py) : image synthétique
contre la moyenne normalisée des embeddings des vraies images de surveillance de la même
identité, avec le modèle ArcFace pré-entraîné.

Sortie : figures/ch4-fidelite-exemples.pdf (+ .png de contrôle).
Usage : .venv-analysis/Scripts/python analysis/fidelity_exemples.py --good 023 --bad 001
"""
from __future__ import annotations

import argparse
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
GOOD_COLOR, BAD_COLOR = "#1c5aa6", "#b8420f"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good", default="023")
    ap.add_argument("--bad", default="001")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "visible d1"))
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "visible_d1.yaml")
    net = build_net(next(iter(find_checkpoints(Path(args.ckpt_dir), [], False).values())), zero_lora=True)
    cache = Path(cfg["paths"]["aligned_cache"])

    def tensor_of(p) -> torch.Tensor:
        return torch.load(cache_file(p, cache))

    @torch.no_grad()
    def embed(ts) -> torch.Tensor:
        return F.normalize(net(preprocess_for_arcface(torch.stack(ts))), dim=-1)

    pairs_b = list_pairs(cfg, block="B")
    rows = []
    for ident, mode in ((args.good, "best"), (args.bad, "worst")):
        pairs = [p for p in pairs_b if p.identity == ident]
        real = [p.target_path for p in pairs]
        r = embed([tensor_of(p) for p in real]).mean(0, keepdim=True)
        r = r / r.norm(dim=-1, keepdim=True)
        synth = sorted((Path(cfg["paths"]["synth_dataset"]) / ident).glob("*.png"))
        cos = sorted(((float((embed([tensor_of(p)]) @ r.T).item()), p) for p in synth),
                     key=lambda t: t[0], reverse=(mode == "best"))
        print(f"identité {ident} ({mode}) : " + ", ".join(f"{p.name}={c:.3f}" for c, p in cos[:args.n]))
        rows.append((ident, pairs[0].mugshot_path, real[2], cos[:args.n]))  # cam 3 comme vue réelle

    n = args.n
    fig, axes = plt.subplots(2, 2 + n, figsize=(2.05 * (2 + n), 4.7))
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ident, mug, real, items) in enumerate(rows):
        axes[i, 0].imshow(Image.open(mug).convert("RGB"))
        axes[i, 0].set_title(f"Mugshot (id. {ident})", fontsize=9)
        axes[i, 1].imshow(Image.open(real).convert("RGB"))
        axes[i, 1].set_title("Réel, cam. 3", fontsize=9)
        for j, (c, p) in enumerate(items):
            axes[i, 2 + j].imshow(Image.open(p).convert("RGB"))
            axes[i, 2 + j].set_title(f"cos = {c:.2f}".replace(".", ","), fontsize=10,
                                     color=GOOD_COLOR if c >= THRESHOLD else BAD_COLOR)
    fig.tight_layout()
    out = ROOT / "figures" / "ch4-fidelite-exemples.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print("écrit", out)


if __name__ == "__main__":
    main()
