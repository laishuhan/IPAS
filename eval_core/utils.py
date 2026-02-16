# === utils.py ===
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Callable

from models import _FEATURE_ORDER, QualityVector, NeedInfo


# =========================================================
# 1) 多分类校准：Temperature Scaling
# =========================================================

def _clip01(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)

def nll_multiclass(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))

def ece_toplabel(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Multi-class ECE using top-label confidence.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    acc = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf >= b0) & (conf < b1) if b1 < 1.0 else (conf >= b0) & (conf <= b1)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask) / n) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)

@dataclass
class TemperatureCalibrator:
    """
    多分类温度缩放：p_cal = softmax(logits / T)
    由于你当前模型输出的是概率 p，我们用 logits = log(p) 近似（论文可解释为 post-hoc scaling）。
    """
    T: float = 1.0
    max_iter: int = 200
    lr: float = 0.05

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        # logits approx
        logits = np.log(np.clip(p_raw, 1e-12, 1.0))
        T = float(self.T)

        # simple gradient descent on NLL wrt T
        for _ in range(self.max_iter):
            pT = self.transform_logits(logits, T)
            loss = nll_multiclass(pT, y)

            # gradient (approx): d/dT of NLL with logits/T
            # We do numerical gradient for stability (small cost, ok for paper experiments)
            eps = 1e-4
            pT2 = self.transform_logits(logits, T + eps)
            loss2 = nll_multiclass(pT2, y)
            g = (loss2 - loss) / eps

            T_new = max(0.05, T - self.lr * g)
            if abs(T_new - T) < 1e-6:
                T = T_new
                break
            T = T_new

        self.T = float(T)
        return self

    @staticmethod
    def transform_logits(logits: np.ndarray, T: float) -> np.ndarray:
        z = logits / float(T)
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(np.asarray(p_raw, dtype=float), 1e-12, 1.0))
        return self.transform_logits(logits, self.T)

    def report(self, p_raw: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        p_cal = self.transform(p_raw)
        return {
            "method": "temperature",
            "T": float(self.T),
            "nll": float(nll_multiclass(p_cal, y)),
            "ece_top": float(ece_toplabel(p_cal, y)),
        }


# =========================================================
# 2) 多分类共形预测：APS (Adaptive Prediction Sets)
# =========================================================

class ConformalAPS:
    """
    APS:
      score_i = sum_{k: p_k >= p_{y_i}} p_k  (rank-based cumulative mass)
    Then choose qhat as (1-alpha) quantile of scores,
    predict set = smallest set of labels whose cumulative prob >= qhat
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.qhat: Optional[float] = None

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalAPS":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int).reshape(-1)
        scores = []
        for i in range(len(y_cal)):
            pi = p_cal[i]
            yi = y_cal[i]
            thr = pi[yi]
            score = float(np.sum(pi[pi >= thr]))
            scores.append(score)
        scores = np.asarray(scores, dtype=float)
        n = len(scores)
        k = min(max(int(np.ceil((n + 1) * (1 - self.alpha))) - 1, 0), n - 1)
        self.qhat = float(np.sort(scores)[k])
        return self

    def predict_set(self, p: np.ndarray) -> List[int]:
        p = np.asarray(p, dtype=float).reshape(-1)
        if self.qhat is None:
            return [int(np.argmax(p))]

        order = np.argsort(-p)
        cum = 0.0
        out = []
        for idx in order:
            out.append(int(idx))
            cum += float(p[idx])
            if cum >= self.qhat:
                break
        return out if out else [int(np.argmax(p))]


def abstain_decision_multiclass(
    epistemic: float,
    pred_set: List[int],
    entropy: float,
    epistemic_threshold: float = 0.6,
    entropy_threshold: float = 1.0,
    allow_set_size_gt1: bool = False,
) -> bool:
    if epistemic > epistemic_threshold:
        return True
    if entropy > entropy_threshold:
        return True
    if (not allow_set_size_gt1) and (len(pred_set) > 1):
        return True
    return False


# =========================================================
# 3) 证据链生成（保留你的接口）
# =========================================================

def build_evidence_chain(
    need: NeedInfo,
    Q: QualityVector,
    p_model: Any,
    lf_names: Sequence[str],
    lf_outputs: Sequence[int],
    bayes_phi: Optional[np.ndarray] = None,
    bayes_w: Optional[np.ndarray] = None,
    top_k: int = 5,

    # --- 新增：多分类解释用 ---
    multiclass_x: Optional[np.ndarray] = None,
    multiclass_W: Optional[np.ndarray] = None,
    multiclass_p: Optional[np.ndarray] = None,
    target_labels: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fired = []
    for name, out in zip(lf_names, lf_outputs):
        if out != 0:
            fired.append({"lf": name, "vote": "high_risk" if out > 0 else "low_risk"})

    contribs = []

    # --- 原二分类贡献（保留）---
    if bayes_phi is not None and bayes_w is not None:
        raw_contrib = bayes_phi * bayes_w
        idx = np.argsort(-np.abs(raw_contrib))[:top_k]
        for i in idx:
            nm = _FEATURE_ORDER[i] if i < len(_FEATURE_ORDER) else f"feat_{i}"
            contribs.append({"feature": nm, "impact": float(raw_contrib[i])})

    # --- 新增：多分类贡献（用于 p1+p2）---
    if (multiclass_x is not None) and (multiclass_W is not None) and (multiclass_p is not None) and (target_labels is not None):
        try:
            mc = topk_contrib_softmax_target(
                x=np.asarray(multiclass_x).reshape(-1),
                W=np.asarray(multiclass_W),
                p=np.asarray(multiclass_p).reshape(-1),
                target_labels=list(target_labels),
                feature_names=feature_names,
                top_k=max(top_k, 8),
            )
            # 用更明确的字段覆盖/追加
            contribs = mc
        except Exception as e:
            contribs.append({"feature": "contrib_error", "x": 0.0, "grad": 0.0, "contribution": 0.0, "error": str(e)})

    return {
        "p_model": p_model if isinstance(p_model, (float, int, list, dict)) else str(p_model),
        "quality_severity": float(Q.severity()),
        "weak_supervision_votes": fired,
        "top_feature_contributions": contribs
    }



# =========================================================
# 4) 反事实溯因（保持你原有）
# =========================================================

def probability_constrained_recourse(
    x: np.ndarray,
    prob_fn: Callable[[np.ndarray], float],
    tau: float,
    max_steps: int = 50
) -> Dict[str, Any]:
    x_cur = (np.asarray(x).reshape(-1) > 0.5).astype(float)
    p0 = float(prob_fn(x_cur))
    if p0 <= tau:
        return {"status": "already_below_threshold", "p_before": p0, "p_after": p0, "steps": [], "x_cf": x_cur.tolist()}

    steps, p_cur = [], p0
    for _ in range(max_steps):
        best_i, best_p, best_x = None, p_cur, None
        for i in range(len(x_cur)):
            x_try = x_cur.copy()
            x_try[i] = 1.0 - x_try[i]
            p_try = float(prob_fn(x_try))
            if p_try < best_p - 1e-12:
                best_p, best_i, best_x = p_try, i, x_try

        if best_i is None:
            break
        steps.append({
            "feature_index": int(best_i),
            "action": "set_to_normal" if x_cur[best_i] > 0.5 else "activate",
            "p_after_step": best_p
        })
        x_cur, p_cur = best_x, best_p
        if p_cur <= tau:
            break

    return {"status": "optimized" if p_cur <= tau else "stopped", "p_before": p0, "p_after": p_cur, "steps": steps, "x_cf": x_cur.tolist()}

# ==============================
# Multi-class feature contributions
# Explain target: p_abn = p1 + p2
# ==============================

def topk_contrib_softmax_target(
    x: np.ndarray,
    W: np.ndarray,
    p: np.ndarray,
    target_labels: List[int],
    feature_names: Optional[List[str]] = None,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    对 softmax 输出的 target 概率（例如 p1+p2）做局部贡献解释（基于梯度的 local attribution）。

    输入：
      x: (d,) 当前样本特征
      W: (d,K) softmax 权重（BayesianSoftmax.W_map）
      p: (K,) 当前样本 softmax 概率（建议用 p_final/校准后概率也可，但解释更推荐用 raw logits 对应的 p）
      target_labels: 目标集合，比如 [1,2] 表示异常总风险
      feature_names: 可选，长度 d
    输出：
      list: TopK 特征贡献，按 |contribution| 降序
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1)

    d, K = W.shape
    assert x.shape[0] == d, f"x dim {x.shape[0]} != W dim {d}"
    assert p.shape[0] == K, f"p dim {p.shape[0]} != K {K}"

    # softmax 的局部梯度：
    # dp_k/dx_j = p_k * (W_{j,k} - sum_m p_m W_{j,m})
    # 目标概率 p_T = sum_{k in T} p_k
    # dp_T/dx_j = sum_{k in T} dp_k/dx_j
    W_bar = W @ p  # (d,)  sum_m p_m W_{:,m}
    grad = np.zeros(d, dtype=float)
    for k in target_labels:
        grad += p[k] * (W[:, k] - W_bar)

    # 局部贡献：x_j * grad_j（可理解为 local linearization）
    contrib = x * grad

    # 取 TopK
    idx = np.argsort(-np.abs(contrib))[:top_k]
    out = []
    for j in idx:
        name = feature_names[j] if (feature_names is not None and j < len(feature_names)) else f"feat_{j}"
        out.append({
            "feature": name,
            "x": float(x[j]),
            "grad": float(grad[j]),
            "contribution": float(contrib[j])
        })
    return out