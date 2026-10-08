"""Figures de fidélité qui ne demandent pas de calcul d'embeddings :
  - ch4-fidelite-comparaison.pdf : cosinus moyen (± écart-type par image) du synthétique et de la
    référence réel contre réel, en visible et en infrarouge (valeurs du stage `fidelity`) ;
  - ch4-fidelite-ir-exemples.pdf : illustration qualitative IR (mugshot, vraies captures IR,
    images synthétiques post-traitées), SANS sélection par cosinus (images 000 à 002).

Valeurs saisies depuis les logs du stage `fidelity` (Colab) :
  visible : synthétique 0,3069 ± 0,0972 (n=1000) ; réel/réel 0,6028 ± 0,0872 (n=250) ; FID 212,9
  infrarouge : synthétique 0,1652 ± 0,0775 (n=1000) ; réel/réel 0,3922 ± 0,1340 (n=100) ; FID 259,6
Usage : .venv-analysis/Scripts/python analysis/fidelity_ir_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ACCENT, INK, MUTED, GRID, LIGHT = "#2a78d6", "#0b0b0b", "#52514e", "#dcdbd6", "#b9b8b2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
})
fr = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
STATS = {  # (moyenne, écart-type) par image
    "Visible d1": {"synth": (0.3069, 0.0972), "real": (0.6028, 0.0872)},
    "Infrarouge d1": {"synth": (0.1652, 0.0775), "real": (0.3922, 0.1340)},
}


def comparison() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    w = 0.34
    for i, (name, d) in enumerate(STATS.items()):
        for j, (key, color, label) in enumerate((("real", LIGHT, "Réel contre réel"),
                                                 ("synth", ACCENT, "Synthétique"))):
            m, s = d[key]
            x = i + (j - 0.5) * w
            ax.bar(x, m, width=w * 0.92, color=color, edgecolor="white", lw=1.0, zorder=2,
                   label=label if i == 0 else None)
            ax.errorbar([x], [m], yerr=[s], color=INK, capsize=3, elinewidth=1.1, zorder=3)
            ax.text(x, 0.03, f"{m:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=10.5,
                    color="white" if key == "synth" else INK, zorder=4)
        ratio = d["synth"][0] / d["real"][0]
        ax.text(i, 0.93, f"synthétique / réel = {ratio:.2f}".replace(".", ","), ha="center", va="center",
                fontsize=9.5, color=INK)
    ax.axhline(0.45, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=1)
    ax.text(1.62, 0.455, "Seuil 0,45", ha="right", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(STATS)), list(STATS))
    ax.set_xlim(-0.6, 1.65)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(fr)
    ax.set_ylabel("Cosinus ArcFace moyen avec l'identité réelle")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.86))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "ch4-fidelite-comparaison.pdf")
    plt.close(fig)


def ir_examples(ids=("001", "023"), n_synth=3) -> None:
    scf = ROOT / "data" / "SCface_database"
    d1 = scf / "surveillance_cameras_distance_1"
    ncol = 3 + n_synth
    fig, axes = plt.subplots(len(ids), ncol, figsize=(2.0 * ncol, 2.35 * len(ids)))
    for ax in axes.ravel():
        ax.axis("off")
    for r, ident in enumerate(ids):
        axes[r, 0].imshow(Image.open(scf / "mugshot_frontal_cropped_all" / f"{ident}_frontal.JPG").convert("RGB"))
        axes[r, 0].set_title(f"Mugshot (id. {ident})", fontsize=9)
        for k, cam in enumerate((6, 7)):
            axes[r, 1 + k].imshow(Image.open(d1 / f"cam_{cam}" / f"{ident}_cam{cam}_1.jpg").convert("L"),
                                  cmap="gray", vmin=0, vmax=255)
            axes[r, 1 + k].set_title(f"Réel IR, cam. {cam}", fontsize=9)
        synth = sorted((ROOT / "outputs" / "synth" / "ir_d1" / ident).glob("*.png"))[:n_synth]
        for k, p in enumerate(synth):
            axes[r, 3 + k].imshow(Image.open(p).convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[r, 3 + k].set_title(f"Synthétique {p.stem.split('_')[1]}", fontsize=9)
    fig.tight_layout()
    out = ROOT / "figures" / "ch4-fidelite-ir-exemples.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    comparison()
    ir_examples()
    print("figures écrites")
