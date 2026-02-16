from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def _clip01(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip01(np.asarray(p).reshape(-1))
    y = np.asarray(y).reshape(-1).astype(float)
    return float(np.mean((p - y) ** 2))


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = _clip01(np.asarray(p).reshape(-1))
    y = np.asarray(y).reshape(-1).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (p >= b0) & (p < b1) if b1 < 1.0 else (p >= b0) & (p <= b1)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        w = float(np.sum(mask) / n)
        ece += w * abs(acc - conf)

    return float(ece)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    p = _clip01(np.asarray(p).reshape(-1))
    y = np.asarray(y).reshape(-1).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, Any]] = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (p >= b0) & (p < b1) if b1 < 1.0 else (p >= b0) & (p <= b1)
        if not np.any(mask):
            out.append({"bin_lo": float(b0), "bin_hi": float(b1), "count": 0, "avg_p": None, "avg_y": None})
            continue
        out.append({
            "bin_lo": float(b0),
            "bin_hi": float(b1),
            "count": int(np.sum(mask)),
            "avg_p": float(np.mean(p[mask])),
            "avg_y": float(np.mean(y[mask])),
        })
    return out


# =========================================================
# 1) Platt scaling
# =========================================================

@dataclass
class PlattCalibrator:
    """
    Fits: p_cal = sigmoid(a * logit(p_raw) + b)
    using Newton on cross-entropy with L2 on a,b (small).
    """
    a: float = 1.0
    b: float = 0.0
    l2: float = 1e-3
    max_iter: int = 50
    tol: float = 1e-8

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = _clip01(p)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p_raw = _clip01(np.asarray(p_raw).reshape(-1))
        y = np.asarray(y).reshape(-1).astype(float)

        x = self._logit(p_raw)
        a, b = float(self.a), float(self.b)

        for _ in range(self.max_iter):
            z = a * x + b
            p = self._sigmoid(z)

            # gradients
            ga = float(np.sum((p - y) * x) + self.l2 * a)
            gb = float(np.sum(p - y) + self.l2 * b)

            # Hessian terms
            w = p * (1 - p)
            haa = float(np.sum(w * x * x) + self.l2)
            hab = float(np.sum(w * x))
            hbb = float(np.sum(w) + self.l2)

            H = np.array([[haa, hab], [hab, hbb]], dtype=float)
            g = np.array([ga, gb], dtype=float)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            a -= float(delta[0])
            b -= float(delta[1])

            if float(np.linalg.norm(delta)) < self.tol:
                break

        self.a, self.b = float(a), float(b)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = _clip01(np.asarray(p_raw).reshape(-1))
        x = self._logit(p_raw)
        z = self.a * x + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return _clip01(p)

    def metrics(self, p_raw: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        p_cal = self.transform(p_raw)
        return {
            "brier": brier_score(p_cal, y),
            "ece": ece_score(p_cal, y, n_bins=n_bins),
            "reliability": reliability_bins(p_cal, y, n_bins=n_bins),
            "params": {"a": float(self.a), "b": float(self.b)},
        }


# =========================================================
# 2) Isotonic regression (PAVA)
# =========================================================

@dataclass
class IsotonicCalibrator:
    """
    Isotonic regression using PAVA on (p_raw, y).
    Produces a piecewise-constant monotone mapping.
    """
    xs_: Optional[np.ndarray] = None  # knot x
    ys_: Optional[np.ndarray] = None  # mapped y

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        x = _clip01(np.asarray(p_raw).reshape(-1))
        y = np.asarray(y).reshape(-1).astype(float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # PAVA blocks
        v = y.copy()
        w = np.ones_like(v)

        # merge adjacent violators
        i = 0
        while i < len(v) - 1:
            if v[i] <= v[i + 1]:
                i += 1
                continue
            # merge i and i+1
            new_v = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            new_w = w[i] + w[i + 1]
            v[i] = new_v
            w[i] = new_w
            v = np.delete(v, i + 1)
            w = np.delete(w, i + 1)
            x = np.delete(x, i + 1)
            if i > 0:
                i -= 1

        self.xs_ = x
        self.ys_ = _clip01(v)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self.xs_ is None or self.ys_ is None:
            raise RuntimeError("IsotonicCalibrator not fit yet.")
        x = _clip01(np.asarray(p_raw).reshape(-1))
        # stepwise interpolation: for each x, find rightmost knot <= x
        idx = np.searchsorted(self.xs_, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.ys_) - 1)
        return _clip01(self.ys_[idx])

    def metrics(self, p_raw: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        p_cal = self.transform(p_raw)
        return {
            "brier": brier_score(p_cal, y),
            "ece": ece_score(p_cal, y, n_bins=n_bins),
            "reliability": reliability_bins(p_cal, y, n_bins=n_bins),
            "knots": int(len(self.xs_)) if self.xs_ is not None else 0,
        }
