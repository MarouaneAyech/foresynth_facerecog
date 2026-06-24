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
