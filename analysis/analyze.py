"""Analyses du chapitre 4 à partir des embeddings extraits (analysis/extract_embeddings.py).

Produit, pour un terrain, dans analysis/outputs/<terrain>/ :
  - bootstrap_summary.csv : rank-1 (moyenne des graines) et IC à 95 % par rééchantillonnage
    des IDENTITÉS (les probes d'une même identité ne sont pas indépendants) ;
  - bootstrap_paired.csv  : différences appariées entre conditions (mêmes rééchantillons),
    avec IC à 95 % et part des rééchantillons où la différence est <= 0 ;
  - cmc.csv, genuine.csv, per_camera.csv, per_identity.csv ;
et dans figures/ : ch4-cmc, ch4-genuine, ch4-camera (suffixe -ir pour l'infrarouge).

Usage : .venv-analysis/Scripts/python analysis/analyze.py --terrain visible_d1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
B = 10_000
KEY_COMPARISONS = [  # (a, b) : différence a - b
    ("real", "baseline"), ("synthetic", "baseline"), ("synthetic", "real"),
    ("mixed_0.1", "baseline"), ("mixed_0.1", "real"), ("mixed_0.5", "baseline"),
    ("mixed_0.5", "real"), ("mixed_1.0", "baseline"), ("mixed_1.0", "real"),
    ("mixed_0.1", "mixed_1.0"), ("mixed_0.5", "mixed_1.0"), ("mixed_0.5", "mixed_0.1"),
]
CMC_SERIES = ["baseline", "real", "mixed_0.1", "mixed_0.5", "mixed_1.0", "synthetic"]
CAMERA_COMPARE = ["mixed_0.5", "mixed_1.0"]

ACCENT, ACCENT2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdbd6"
BLUES = {"mixed_0.1": "#9cc3f0", "mixed_0.5": "#4f93e0", "mixed_1.0": "#1c5aa6"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
})
fr2 = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def label(key: str) -> str:
    if key == "baseline":
        return "Baseline non adaptée"
    if key == "real":
        return "Réel seul"
    if key == "synthetic":
        return "Synthétique seul"
    return f"Réel + {round(float(key.split('_')[1]) * 100)} % de synthétique"


def load(emb_dir: Path):
    runs: dict[str, list[dict]] = {}
    ref = None
    for f in sorted(emb_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        cond = str(z["condition"])
        key = f"mixed_{float(z['ratio']):.1f}" if cond == "mixed" else cond
        scores = z["scores"]
        gids, pids = z["gallery_ids"], z["probe_ids"]
        true_col = np.array([int(np.where(gids == p)[0][0]) for p in pids])
        true_score = scores[np.arange(len(pids)), true_col]
        rank = 1 + (scores > true_score[:, None]).sum(1)
        masked = scores.copy()
        masked[np.arange(len(pids)), true_col] = -np.inf
        sig = (tuple(gids), tuple(pids), tuple(z["probe_cam"]))
        if ref is None:
            ref = sig
        assert sig == ref, f"{f.name} : galerie/probes différents des autres fichiers"
        runs.setdefault(key, []).append(dict(
            seed=int(z["seed"]), rank=rank, correct=rank == 1, genuine=true_score,
            impostor=masked.max(1)))
    return runs, np.array(ref[0]), np.array(ref[1]), np.array(ref[2])


def order(keys):
    def k(s):
        return (0, 0) if s == "baseline" else (1, 0) if s == "real" else (2, 0) if s == "synthetic" \
            else (3, float(s.split("_")[1]))
    return sorted(keys, key=k)


def write_csv(path: Path, header: list[str], rows: list[list]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", default="visible_d1", choices=["visible_d1", "ir_d1"])
    args = ap.parse_args()
    suffix = "-ir" if args.terrain == "ir_d1" else ""
    emb_dir = ROOT / "analysis" / "outputs" / "embeddings" / args.terrain
    out_dir = ROOT / "analysis" / "outputs" / args.terrain
    fig_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, gids, pids, pcam = load(emb_dir)
    keys = order(runs)
    ids = list(gids)
    id_idx = np.array([ids.index(p) for p in pids])  # identité de chaque probe
    n_id = len(ids)
    print(f"{args.terrain} : {n_id} identités, {len(pids)} probes, conditions : "
          + ", ".join(f"{k}(n={len(runs[k])})" for k in keys))

    # --- précision par identité (moyenne sur graines et probes) -> bootstrap d'identités ---
    acc = {}
    for k in keys:
        per_probe = np.mean([r["correct"] for r in runs[k]], axis=0)
        acc[k] = np.array([per_probe[id_idx == i].mean() for i in range(n_id)])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_id, size=(B, n_id))
    boot = {k: acc[k][idx].mean(1) for k in keys}

    rows = []
    for k in keys:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append([k, len(runs[k]), f"{acc[k].mean():.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    write_csv(out_dir / "bootstrap_summary.csv", ["condition", "n_graines", "rank1", "ic95_bas", "ic95_haut"], rows)
    print("\nRank-1 (moyenne des graines) et IC 95 % par rééchantillonnage des identités :")
    for r in rows:
        print(f"  {r[0]:10s} n={r[1]}  {r[2]}  [{r[3]}, {r[4]}]")

    rows = []
    for a, b in KEY_COMPARISONS:
        if a in boot and b in boot:
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append([a, b, f"{(acc[a].mean() - acc[b].mean()) * 100:+.1f}", f"{lo * 100:+.1f}",
                         f"{hi * 100:+.1f}", f"{(d <= 0).mean():.4f}"])
    write_csv(out_dir / "bootstrap_paired.csv",
              ["a", "b", "diff_points", "ic95_bas", "ic95_haut", "part_diff_<=0"], rows)
    print("\nDifférences appariées a - b (points de rank-1) [IC 95 %] :")
    for r in rows:
        print(f"  {r[0]:10s} - {r[1]:10s} {r[2]:>6s}  [{r[3]}, {r[4]}]  p(diff<=0)={r[5]}")

    # --- CMC ---
    max_k = 10
    cmc = {k: np.array([np.mean([(r["rank"] <= j).mean() for r in runs[k]]) for j in range(1, max_k + 1)])
           for k in keys}
    write_csv(out_dir / "cmc.csv", ["condition"] + [f"rank{j}" for j in range(1, max_k + 1)],
              [[k] + [f"{v:.4f}" for v in cmc[k]] for k in keys])

    # --- similarité genuine / marge ---
    rows = []
    for k in keys:
        g = np.mean([r["genuine"].mean() for r in runs[k]])
        im = np.mean([r["impostor"].mean() for r in runs[k]])
        rows.append([k, f"{g:.4f}", f"{im:.4f}", f"{g - im:.4f}"])
    write_csv(out_dir / "genuine.csv", ["condition", "cos_genuine_moyen", "cos_imposteur_max_moyen", "marge"], rows)
    gen = {r[0]: (float(r[1]), float(r[2])) for r in rows}

    # --- par caméra ---
    cams = sorted(set(pcam.tolist()))
    cam_acc = {k: {c: float(np.mean([r["correct"][pcam == c].mean() for r in runs[k]])) for c in cams}
               for k in keys}
    rows = []
    for k in keys:
        rows.append([k] + [f"{cam_acc[k][c]:.4f}" for c in cams])
    write_csv(out_dir / "per_camera.csv", ["condition"] + [f"cam{c}" for c in cams], rows)
    cam_gain_ci = {}
    for k in CAMERA_COMPARE:
        if k in runs and "baseline" in runs:
            for c in cams:
                a = np.array([np.mean([r["correct"][(pcam == c) & (id_idx == i)].mean() for r in runs[k]])
                              for i in range(n_id)])
                b0 = np.array([np.mean([r["correct"][(pcam == c) & (id_idx == i)].mean() for r in runs["baseline"]])
                               for i in range(n_id)])
                d = (a[idx] - b0[idx]).mean(1)
                cam_gain_ci[(k, c)] = ((a - b0).mean() * 100, *np.percentile(d, [2.5, 97.5]) * 100)

    # --- par identité : mixte 100 % contre baseline ---
    ref_k = "mixed_1.0" if "mixed_1.0" in acc else keys[-1]
    if "baseline" in acc:
        delta = acc[ref_k] - acc["baseline"]
        write_csv(out_dir / "per_identity.csv", ["identite", "baseline", ref_k, "gain_points"],
                  [[ids[i], f"{acc['baseline'][i]:.3f}", f"{acc[ref_k][i]:.3f}", f"{delta[i] * 100:+.1f}"]
                   for i in range(n_id)])
        print(f"\nPar identité ({ref_k} contre baseline) : {int((delta > 0).sum())} identités améliorées, "
              f"{int((delta < 0).sum())} dégradées, {int((delta == 0).sum())} inchangées")

    print("\nCMC (rank-1, rank-3, rank-5) :")
    for k in keys:
        print(f"  {k:10s} {cmc[k][0]:.4f}  {cmc[k][2]:.4f}  {cmc[k][4]:.4f}")
    print("\nSimilarité genuine moyenne / imposteur max moyen / marge :")
    for r in rows if False else [[k, *gen[k], gen[k][0] - gen[k][1]] for k in keys]:
        print(f"  {r[0]:10s} {r[1]:.4f}  {r[2]:.4f}  {r[3]:+.4f}")

    # ================= figures =================
    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=3)

    # CMC
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    xs = np.arange(1, max_k + 1)
    spec = {"baseline": (MUTED, (0, (5, 3)), "o"), "real": (ACCENT2, "-", "s"),
            "synthetic": (INK, (0, (1, 2)), "^"),
            "mixed_0.1": (BLUES["mixed_0.1"], "-", "o"), "mixed_0.5": (BLUES["mixed_0.5"], "-", "o"),
            "mixed_1.0": (BLUES["mixed_1.0"], "-", "o")}
    for k in [s for s in CMC_SERIES if s in cmc]:
        color, ls, mk = spec[k]
        ax.plot(xs, cmc[k], color=color, ls=ls, marker=mk, ms=5, lw=1.6, mec="white", mew=0.8,
                label=label(k))
    ax.set_xticks(xs)
    ax.set_xlim(0.7, max_k + 0.3)
    ax.yaxis.set_major_formatter(fr2)
    ax.set_xlabel("Rang k")
    ax.set_ylabel("Taux d'identification cumulé")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.8)
    style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / f"ch4-cmc{suffix}.pdf")
    plt.close(fig)

    # similarité genuine et imposteur en fonction de la proportion de synthétique
    mixed = [k for k in keys if k.startswith("mixed_")]
    if mixed and "real" in gen:
        fig, ax = plt.subplots(figsize=(6.4, 3.7))
        xs = [0] + [float(k.split("_")[1]) * 100 for k in mixed]
        g = [gen["real"][0]] + [gen[k][0] for k in mixed]
        im = [gen["real"][1]] + [gen[k][1] for k in mixed]
        ax.plot(xs, g, color=ACCENT, lw=1.6, marker="o", ms=6, mec="white", mew=1.0)
        ax.plot(xs, im, color=ACCENT2, lw=1.6, ls=(0, (6, 2)), marker="s", ms=6, mec="white", mew=1.0)
        if "baseline" in gen:
            ax.axhline(gen["baseline"][0], color=ACCENT, ls=(0, (1, 2)), lw=1.1, alpha=0.7)
            ax.axhline(gen["baseline"][1], color=ACCENT2, ls=(0, (1, 2)), lw=1.1, alpha=0.7)
        if "baseline" in gen:
            ax.text(-3, gen["baseline"][0] + 0.004, "Baseline (genuine)", color=ACCENT, fontsize=9,
                    ha="left", va="bottom")
            ax.text(-3, gen["baseline"][1] - 0.004, "Baseline (imposteur)", color=ACCENT2, fontsize=9,
                    ha="left", va="top")
        ax.text(xs[-1] + 2, g[-1], "Genuine", color=ACCENT, va="center", fontsize=10)
        ax.text(xs[-1] + 2, im[-1], "Imposteur (max)", color=ACCENT2, va="center", fontsize=10)
        ax.set_xlim(-4, 126)
        ax.set_xticks(range(0, 101, 10))
        ax.set_yticks(np.arange(0.10, 0.41, 0.05))
        ax.yaxis.set_major_formatter(fr2)
        ax.set_xlabel("Proportion de synthétique ajoutée au réel (%)")
        ax.set_ylabel("Similarité cosinus moyenne")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"ch4-genuine{suffix}.pdf")
        plt.close(fig)

    # gain par caméra
    if cam_gain_ci:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        w = 0.36
        for j, k in enumerate(CAMERA_COMPARE):
            if (k, cams[0]) not in cam_gain_ci:
                continue
            vals = np.array([cam_gain_ci[(k, c)] for c in cams])
            x = np.arange(len(cams)) + (j - 0.5) * w
            ax.bar(x, vals[:, 0], width=w, color=BLUES[k], edgecolor="white", lw=1.0, zorder=2,
                   label=label(k))
            ax.errorbar(x, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
                        fmt="none", ecolor=INK, capsize=3, elinewidth=1.1, zorder=3)
        ax.axhline(0, color=MUTED, lw=1.0)
        ax.set_xticks(np.arange(len(cams)), [f"Caméra {c}" for c in cams])
        ax.set_ylabel("Gain de rank-1 sur la baseline (points)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"ch4-camera{suffix}.pdf")
        plt.close(fig)
    print("\nfigures écrites dans", fig_dir)


if __name__ == "__main__":
    main()
