"""Figures et statistiques de fidélité (visible d1) : distribution des cosinus par image
(synthétique contre baseline réel contre réel) et cosinus moyen par identité.

Reproduit les mesures du stage `fidelity` (src/fidelity/embedding.py) avec le modèle ArcFace
pré-entraîné reconstruit depuis un checkpoint LoRA (cf. extract_embeddings.py) :
  - synthétique : cosinus de chaque image avec la moyenne normalisée des embeddings des vraies
    images de surveillance de la même identité (bloc B) ;
  - baseline : cosinus de chaque vraie image avec la moyenne des AUTRES vraies images de son
    identité (leave-one-out).

Sorties : figures/ch4-fidelite-distribution.pdf, figures/ch4-fidelite-identites.pdf,
analysis/outputs/visible_d1/fidelity_*.csv, statistiques affichées.
Usage : .venv-analysis/Scripts/python analysis/fidelity_figures.py
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from extract_embeddings import build_net, cache_file, find_checkpoints  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.pairs import list_pairs  # noqa: E402
from src.utils.arcface_backbone import preprocess_for_arcface  # noqa: E402

THRESHOLD = 0.45
ACCENT, ACCENT2, INK, MUTED, GRID, LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdbd6", "#b9b8b2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
})
fr = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
HIGHLIGHT = {"023": ACCENT, "001": ACCENT2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", default="visible_d1", choices=["visible_d1", "ir_d1"])
    args = ap.parse_args()
    suffix = "-ir" if args.terrain == "ir_d1" else ""
    cfg = load_config(ROOT / "configs" / f"{args.terrain}.yaml")
    ckpt = next(iter(find_checkpoints(ROOT / "checkpoints" / args.terrain.replace("_", " "), [], False).values()))
    net = build_net(ckpt, zero_lora=True)
    caches = [Path(cfg["paths"]["aligned_cache"]), ROOT / "aligned_cache_ir"]

    def tensor_of(p) -> torch.Tensor:
        f = next((c for d in caches if (c := cache_file(p, d)).exists()), None)
        if f is None:
            raise FileNotFoundError(f"Image absente du cache d'alignement : {p}")
        return torch.load(f)

    @torch.no_grad()
    def embed(ts) -> torch.Tensor:
        return F.normalize(net(preprocess_for_arcface(torch.stack(ts))), dim=-1)

    real_by_id: dict[str, list[str]] = {}
    for p in list_pairs(cfg, block="B"):
        real_by_id.setdefault(p.identity, []).append(p.target_path)

    synth_rows, base_rows = [], []
    for ident, paths in sorted(real_by_id.items()):
        e = embed([tensor_of(p) for p in paths])
        mean = F.normalize(e.mean(0, keepdim=True), dim=-1)
        sp = sorted((Path(cfg["paths"]["synth_dataset"]) / ident).glob("*.png"))
        c = (embed([tensor_of(p) for p in sp]) @ mean.T).squeeze(1).tolist()
        synth_rows += [(ident, p.name, v) for p, v in zip(sp, c)]
        for i in range(len(paths)):  # leave-one-out
            others = F.normalize(torch.cat([e[:i], e[i + 1:]]).mean(0, keepdim=True), dim=-1)
            base_rows.append((ident, Path(paths[i]).name, float((e[i:i + 1] @ others.T).item())))

    out = ROOT / "analysis" / "outputs" / args.terrain
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fidelity_per_image.csv", synth_rows), ("fidelity_baseline_reel.csv", base_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["identite", "image", "cosinus"])
            w.writerows([[a, b, f"{c:.4f}"] for a, b, c in rows])

    s = np.array([r[2] for r in synth_rows])
    b = np.array([r[2] for r in base_rows])
    t, p = stats.ttest_ind(s, b, equal_var=False)
    pooled = np.sqrt((s.var(ddof=1) * (len(s) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(s) + len(b) - 2))
    print(f"synthétique : n={len(s)} moyenne {s.mean():.4f} écart-type {s.std():.4f} ; "
          f"réel/réel : n={len(b)} moyenne {b.mean():.4f} écart-type {b.std():.4f}")
    print(f"Welch t={t:.1f} p={p:.2e} ; d de Cohen = {(s.mean() - b.mean()) / pooled:.2f} ; rapport des moyennes {s.mean() / b.mean():.2f}")
    print("synthétique : percentiles 10/25/50/75/90/99 =", [round(float(np.percentile(s, q)), 3) for q in (10, 25, 50, 75, 90, 99)],
          f"; max {s.max():.3f}")
    print(f"réel/réel : percentiles 5/25/50 = {[round(float(np.percentile(b, q)), 3) for q in (5, 25, 50)]}")
    print(f"images >= {THRESHOLD} : synthétique {100 * (s >= THRESHOLD).mean():.1f} % ({int((s >= THRESHOLD).sum())}/{len(s)}), "
          f"réel/réel {100 * (b >= THRESHOLD).mean():.1f} %")
    print(f"synthétique >= quantile 5 % du réel/réel ({np.percentile(b, 5):.3f}) : {100 * (s >= np.percentile(b, 5)).mean():.1f} %")
    ident_mean = {}
    for ident, _, v in synth_rows:
        ident_mean.setdefault(ident, []).append(v)
    ident_mean = {k: float(np.mean(v)) for k, v in ident_mean.items()}
    means = np.array(sorted(ident_mean.values()))
    print(f"par identité : moyenne des moyennes {means.mean():.3f}, min {means.min():.3f}, max {means.max():.3f}, "
          f"{int((means >= THRESHOLD).sum())}/50 identités avec moyenne >= {THRESHOLD}, "
          f"{sum(any(v >= THRESHOLD for v in [r[2] for r in synth_rows if r[0] == k]) for k in ident_mean)}/50 avec au moins une image >= {THRESHOLD}")
    print("rang de 001 / 023 :", {k: sorted(ident_mean, key=ident_mean.get, reverse=True).index(k) + 1 for k in ("001", "023")})

    def style(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=3)

    # --- figure 1 : distributions des cosinus par image ---
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(min(s.min(), b.min()) - 0.02, max(s.max(), b.max()) + 0.02, 41)
    ax.hist(b, bins=bins, density=True, color=LIGHT, edgecolor="white", lw=0.6, label=f"Réel contre réel (n = {len(b)})", zorder=2)
    ax.hist(s, bins=bins, density=True, color=ACCENT, alpha=0.78, edgecolor="white", lw=0.6,
            label=f"Synthétique (n = {len(s)})", zorder=3)
    ax.axvline(THRESHOLD, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=4)
    ax.text(THRESHOLD + 0.01, ax.get_ylim()[1] * 0.97, "Seuil 0,45", ha="left", va="top", fontsize=9.5, color=INK)
    ax.axvline(b.mean(), color=MUTED, lw=1.1, zorder=4)
    ax.axvline(s.mean(), color=ACCENT, lw=1.4, zorder=4)
    ax.text(s.mean() - 0.01, ax.get_ylim()[1] * 0.80, "moy. " + f"{s.mean():.2f}".replace(".", ","), ha="right", va="top",
            fontsize=9.5, color=ACCENT)
    ax.text(b.mean() + 0.01, ax.get_ylim()[1] * 0.80, "moy. " + f"{b.mean():.2f}".replace(".", ","), ha="left", va="top",
            fontsize=9.5, color=MUTED)
    ax.set_xlabel("Cosinus ArcFace avec l'identité réelle")
    ax.set_ylabel("Densité")
    ax.xaxis.set_major_formatter(fr)
    ax.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.62))
    style(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"ch4-fidelite-distribution{suffix}.pdf")
    plt.close(fig)

    # --- figure 2 : cosinus moyen par identité (triées) ---
    order = sorted(ident_mean, key=ident_mean.get, reverse=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [ident_mean[k] for k in order]
    ax.bar(range(len(order)), vals, color=[HIGHLIGHT.get(k, LIGHT) for k in order], width=0.78, zorder=2)
    ax.axhline(b.mean(), color=MUTED, lw=1.2, zorder=3)
    ax.axhline(THRESHOLD, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=3)
    ax.text(len(order) - 0.5, b.mean() + 0.012, "Réel contre réel (" + f"{b.mean():.2f}".replace(".", ",") + ")", ha="right", va="bottom", fontsize=9.5, color=MUTED)
    ax.text(len(order) - 0.5, THRESHOLD - 0.012, "Seuil 0,45", ha="right", va="top", fontsize=9.5, color=INK)
    for k in ("023", "001"):
        i = order.index(k)
        ax.annotate(f"id. {k}", (i, ident_mean[k]), (i + (3.0 if k == "023" else -3.5), ident_mean[k] + 0.08),
                    fontsize=9.5, color=HIGHLIGHT[k], ha="center",
                    arrowprops=dict(arrowstyle="-", color=HIGHLIGHT[k], lw=0.9))
    ax.set_xlim(-1, len(order))
    ax.set_xticks([])
    ax.set_xlabel("Identités du bloc B (triées par cosinus moyen décroissant)")
    ax.set_ylabel("Cosinus moyen du synthétique")
    ax.yaxis.set_major_formatter(fr)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"ch4-fidelite-identites{suffix}.pdf")
    plt.close(fig)
    print("figures écrites")


if __name__ == "__main__":
    main()
