from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import NeedInfo, QualityVector, abnormal_ratio


# =========================================================
# 1) LF definition
# =========================================================

LFOutput = int  # -1 / 0 / +1


@dataclass
class LabelingFunction:
    name: str
    fn: Callable[[NeedInfo, QualityVector, Dict[str, Any]], LFOutput]


def apply_lfs(
    need_list: Sequence[NeedInfo],
    q_list: Sequence[QualityVector],
    lfs: Sequence[LabelingFunction],
    contexts: Optional[Sequence[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Returns L matrix of shape (n_samples, n_lfs), values in {-1,0,+1}
    """
    n = len(need_list)
    m = len(lfs)
    L = np.zeros((n, m), dtype=int)
    if contexts is None:
        contexts = [{} for _ in range(n)]

    for i in range(n):
        need = need_list[i]
        Q = q_list[i]
        ctx = contexts[i] or {}
        for j, lf in enumerate(lfs):
            try:
                out = int(lf.fn(need, Q, ctx))
            except Exception:
                out = 0
            if out not in (-1, 0, 1):
                out = 0
            L[i, j] = out
    return L


# =========================================================
# 2) A small default LF set (edit to match your medical_logic)
# =========================================================

def lf_abnormal_ratio_high(threshold: float = 0.45) -> LabelingFunction:
    def _f(need: NeedInfo, Q: QualityVector, ctx: Dict[str, Any]) -> LFOutput:
        r = abnormal_ratio(need)
        if r >= threshold:
            return +1
        if r <= max(0.05, threshold * 0.3):
            return -1
        return 0
    return LabelingFunction(name=f"abnormal_ratio_high@{threshold}", fn=_f)


def lf_quality_abstain(q_threshold: float = 0.65) -> LabelingFunction:
    """
    If quality is too bad, abstain (0) — avoids letting junk OCR dominate.
    """
    def _f(need: NeedInfo, Q: QualityVector, ctx: Dict[str, Any]) -> LFOutput:
        if Q.severity() >= q_threshold:
            return 0
        return 0
    return LabelingFunction(name=f"quality_abstain@{q_threshold}", fn=_f)


def lf_image_insufficient_high() -> LabelingFunction:
    """
    If image insufficient flag is high, vote high risk OR abstain depending on your policy.
    Here: vote +1 because model might want conservative output (you can change to 0).
    """
    def _f(need: NeedInfo, Q: QualityVector, ctx: Dict[str, Any]) -> LFOutput:
        if Q.image_insufficient >= 0.5:
            return +1
        return 0
    return LabelingFunction(name="image_insufficient_high", fn=_f)


def build_default_lfs() -> List[LabelingFunction]:
    return [
        lf_abnormal_ratio_high(0.45),
        lf_image_insufficient_high(),
        lf_quality_abstain(0.65),
    ]


# =========================================================
# 3) Label Model (EM)
# =========================================================

class LabelModelEM:
    """
    A simple EM label model for LF outputs in {-1,0,+1}.
    Assumptions:
      - Conditional independence of LFs given y
      - Each LF has an accuracy a_j in (0.5, 1), symmetric:
          P(lf votes correctly | it votes) = a_j
      - LF has propensity pi_j:
          P(lf votes (non-abstain)) = pi_j
      - Abstain probability = 1 - pi_j, independent of y (simplification)

    Outputs:
      - p_y1: posterior probability P(y=1 | L)
      - estimated accuracies and propensities
    """

    def __init__(
        self,
        init_prior: float = 0.3,
        init_acc: float = 0.7,
        init_propensity: float = 0.6,
        max_iter: int = 50,
        tol: float = 1e-5,
        eps: float = 1e-9,
    ):
        self.init_prior = float(init_prior)
        self.init_acc = float(init_acc)
        self.init_prop = float(init_propensity)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)

        self.class_prior_: float = self.init_prior
        self.acc_: Optional[np.ndarray] = None   # (m,)
        self.prop_: Optional[np.ndarray] = None  # (m,)

    @staticmethod
    def _clip01(x: float, lo: float = 1e-4, hi: float = 1 - 1e-4) -> float:
        return float(min(max(x, lo), hi))

    def fit(self, L: np.ndarray) -> "LabelModelEM":
        L = np.asarray(L, dtype=int)
        n, m = L.shape

        # initialize
        prior = self._clip01(self.class_prior_, 1e-3, 1 - 1e-3)
        acc = np.full(m, self._clip01(self.init_acc, 0.51, 0.99), dtype=float)
        prop = np.full(m, self._clip01(self.init_prop, 1e-3, 1 - 1e-3), dtype=float)

        prev_ll = None

        for _ in range(self.max_iter):
            # ----- E-step: compute posterior p_i = P(y=1|L_i)
            logp1 = np.log(prior + self.eps) * np.ones(n)
            logp0 = np.log(1 - prior + self.eps) * np.ones(n)

            for j in range(m):
                lj = L[:, j]
                vote = (lj != 0)

                # propensity
                logp1[vote] += np.log(prop[j] + self.eps)
                logp0[vote] += np.log(prop[j] + self.eps)
                logp1[~vote] += np.log(1 - prop[j] + self.eps)
                logp0[~vote] += np.log(1 - prop[j] + self.eps)

                # correctness model for non-abstain votes
                # if y=1: +1 correct, -1 incorrect
                # if y=0: -1 correct, +1 incorrect
                pos = (lj == +1)
                neg = (lj == -1)

                logp1[pos] += np.log(acc[j] + self.eps)
                logp1[neg] += np.log(1 - acc[j] + self.eps)

                logp0[neg] += np.log(acc[j] + self.eps)
                logp0[pos] += np.log(1 - acc[j] + self.eps)

            # normalize
            mx = np.maximum(logp1, logp0)
            p1 = np.exp(logp1 - mx)
            p0 = np.exp(logp0 - mx)
            post = p1 / (p1 + p0 + self.eps)

            # log-likelihood (for stopping)
            ll = float(np.sum(mx + np.log(p1 + p0 + self.eps)))
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # ----- M-step
            prior = self._clip01(float(np.mean(post)), 1e-3, 1 - 1e-3)

            for j in range(m):
                lj = L[:, j]
                vote = (lj != 0)
                prop[j] = self._clip01(float(np.mean(vote)), 1e-3, 1 - 1e-3)

                if np.sum(vote) == 0:
                    acc[j] = self._clip01(self.init_acc, 0.51, 0.99)
                    continue

                # Expected correctness under posterior:
                # If lf outputs +1:
                #   correct when y=1 (prob post), incorrect when y=0 (prob 1-post)
                # If lf outputs -1:
                #   correct when y=0, incorrect when y=1
                pos = (lj == +1)
                neg = (lj == -1)

                correct = np.zeros(n, dtype=float)
                correct[pos] = post[pos]
                correct[neg] = (1 - post[neg])

                acc_hat = float(np.sum(correct[vote]) / (np.sum(vote) + self.eps))

                # keep >=0.5 to avoid degenerate flips
                acc[j] = self._clip01(acc_hat, 0.51, 0.99)

        self.class_prior_ = float(prior)
        self.acc_ = acc
        self.prop_ = prop
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.acc_ is None or self.prop_ is None:
            raise RuntimeError("LabelModelEM not fit yet.")
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        prior = self._clip01(self.class_prior_, 1e-6, 1 - 1e-6)
        acc = self.acc_
        prop = self.prop_

        logp1 = np.log(prior + self.eps) * np.ones(n)
        logp0 = np.log(1 - prior + self.eps) * np.ones(n)

        for j in range(m):
            lj = L[:, j]
            vote = (lj != 0)

            logp1[vote] += np.log(prop[j] + self.eps)
            logp0[vote] += np.log(prop[j] + self.eps)
            logp1[~vote] += np.log(1 - prop[j] + self.eps)
            logp0[~vote] += np.log(1 - prop[j] + self.eps)

            pos = (lj == +1)
            neg = (lj == -1)

            logp1[pos] += np.log(acc[j] + self.eps)
            logp1[neg] += np.log(1 - acc[j] + self.eps)

            logp0[neg] += np.log(acc[j] + self.eps)
            logp0[pos] += np.log(1 - acc[j] + self.eps)

        mx = np.maximum(logp1, logp0)
        p1 = np.exp(logp1 - mx)
        p0 = np.exp(logp0 - mx)
        post = p1 / (p1 + p0 + self.eps)
        return post

    def summary(self) -> Dict[str, Any]:
        return {
            "class_prior": float(self.class_prior_),
            "acc": None if self.acc_ is None else [float(x) for x in self.acc_],
            "propensity": None if self.prop_ is None else [float(x) for x in self.prop_],
        }
