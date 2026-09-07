"""Garde-fou go/no-go AVANT toute reconnaissance. Logique de décision réelle."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GateResult:
    fid: float
    cos: float
    passed: bool
    reason: str


def decide(fid: float, cos: float, cfg: dict) -> GateResult:
    fmax = cfg["fidelity"]["fid_max"]
    cmin = cfg["fidelity"]["cos_min"]
    ok_fid, ok_cos = fid <= fmax, cos >= cmin
    passed = ok_fid and ok_cos
    reason = "OK" if passed else (
        f"FID {fid:.1f}>{fmax}" if not ok_fid else ""
    ) + ("" if ok_cos else f" cos {cos:.3f}<{cmin}")
    return GateResult(fid=fid, cos=cos, passed=passed, reason=reason.strip() or "OK")


@dataclass
class BaselineComparison:
    synth_mean: float
    baseline_mean: float
    p_value: float
    coherent: bool   # synthétique pas significativement en dessous du plancher réel
    reason: str


def compare_to_baseline(synth: "CosineDistribution", baseline: "CosineDistribution",
                         alpha: float = 0.05) -> BaselineComparison:
    """Tranche si le cosinus synthétique est cohérent avec le plancher réel-vs-réel
    (leave-one-out, cf. embedding.real_identity_cosine_baseline), plutôt que de le
    juger contre un seuil arbitraire (fidelity.cos_min, jamais calibré empiriquement).

    Test de Welch (variances inégales, tailles d'échantillon différentes) sur les
    cosinus bruts des deux distributions. "Cohérent" = le synthétique n'est PAS
    significativement en dessous de la baseline réelle -- un synthétique égal ou
    supérieur n'est jamais un problème, seul un écart significatif vers le bas
    signale un vrai défaut d'identité plutôt qu'une difficulté intrinsèque du domaine
    dégradé (cf. baseline rank-1 réel très inférieur au mugshot)."""
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(synth.values, baseline.values, equal_var=False)
    below = synth.mean < baseline.mean
    coherent = not (below and p_value < alpha)
    if coherent:
        reason = (f"OK : cosinus synthétique ({synth.mean:.3f}) non significativement "
                   f"sous la baseline réelle ({baseline.mean:.3f}, p={p_value:.4f})")
    else:
        reason = (f"ECART D'IDENTITE : cosinus synthétique ({synth.mean:.3f}) significativement "
                   f"sous la baseline réelle ({baseline.mean:.3f}, p={p_value:.4f} < {alpha})")
    return BaselineComparison(synth_mean=synth.mean, baseline_mean=baseline.mean,
                               p_value=p_value, coherent=coherent, reason=reason)
