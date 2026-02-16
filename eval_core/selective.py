from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import math


class ConformalBinary:
    """
    Inductive Conformal Prediction for binary classification using calibrated probabilities p = P(y=1).

    Nonconformity score:
      score_i = (1 - p_i) if y_i=1 else p_i

    qhat = quantile_{ceil((n+1)(1-alpha))/n}(score)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.qhat: Optional[float] = None

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalBinary":
        p_cal = np.asarray(p_cal, dtype=float).reshape(-1)
        y_cal = np.asarray(y_cal, dtype=int).reshape(-1)

        if len(p_cal) != len(y_cal):
            raise ValueError("p_cal and y_cal must have same length")

        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        n = len(scores)

        k = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        k = max(0, min(k, n - 1))

        self.qhat = float(np.sort(scores)[k])
        return self

    def predict_set(self, p: float) -> List[str]:
        """
        Returns a set among:
          ["high"], ["low"], or ["low","high"] when ambiguous; else ["mid"] when qhat not fitted.
        (You can map these to your UI labels.)
        """
        p = float(p)

        if self.qhat is None:
            # fallback heuristic only
            if p >= 0.7:
                return ["high"]
            if p <= 0.3:
                return ["low"]
            return ["mid"]

        out = []
        # include "high" if score_high = 1-p <= qhat
        if (1.0 - p) <= self.qhat:
            out.append("high")
        # include "low" if score_low = p <= qhat
        if p <= self.qhat:
            out.append("low")

        if len(out) == 2:
            return ["low", "high"]
        if len(out) == 1:
            return out
        return ["mid"]


def abstain_decision(
    epistemic: float,
    conformal_set: List[str],
    epistemic_threshold: float = 0.5,
    allow_ambiguous_set: bool = False,
) -> bool:
    """
    Abstain if:
      - epistemic uncertainty too high, OR
      - conformal set is ambiguous (size>1) and not allowed
    """
    epi = float(epistemic)
    if epi > float(epistemic_threshold):
        return True

    if not allow_ambiguous_set:
        if isinstance(conformal_set, list) and len(conformal_set) > 1:
            return True

    return False


def selective_metrics(
    p: np.ndarray,
    y: np.ndarray,
    uncertainty: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute coverage-risk curve and best operating points.
    risk here uses Brier loss on accepted samples.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    u = np.asarray(uncertainty, dtype=float).reshape(-1)

    if len(p) != len(y) or len(p) != len(u):
        raise ValueError("p, y, uncertainty must have same length")

    if thresholds is None:
        # use quantiles for stable curve
        qs = np.linspace(0.0, 1.0, 11)
        thresholds = [float(np.quantile(u, q)) for q in qs]

    curve = []
    for tau in thresholds:
        mask = u <= float(tau)
        coverage = float(np.mean(mask)) if len(mask) else 0.0
        if np.sum(mask) == 0:
            curve.append({"tau": float(tau), "coverage": coverage, "risk": None, "n": 0})
            continue
        brier = float(np.mean((p[mask] - y[mask]) ** 2))
        curve.append({"tau": float(tau), "coverage": coverage, "risk": brier, "n": int(np.sum(mask))})

    # pick min risk among points with coverage>0
    feasible = [c for c in curve if c["risk"] is not None and c["n"] > 0]
    best = min(feasible, key=lambda c: c["risk"]) if feasible else None

    return {"curve": curve, "best": best}
