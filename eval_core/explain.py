# explain.py
# -*- coding: utf-8 -*-
"""
explain.py
----------
职责：
1) softmax target 的局部贡献解释（梯度 * x）
2) evidence chain：汇总模型概率、质量/弱监督信息、top features
3) 反事实 recourse：在二值 indicator 向量上做概率阈值约束的翻转搜索

注意：
- 融合后主模型输入是 embedding z，而不是原始指标向量。
- 解释层面：
  A) 对 softmax head：使用 z 作为特征，做局部梯度贡献解释
  B) 对“异常原因”recourse：仍可在 indicator 层做 demo（取自 raw JSON 的 indicator_analysis 近似）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# =========================================================
# 1) Softmax target local contribution
# =========================================================

def topk_contrib_softmax_target(
    x: np.ndarray,
    W: np.ndarray,
    p: np.ndarray,
    target_labels: List[int],
    feature_names: Optional[List[str]] = None,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    对 softmax 输出的 target 概率（例如 p1+p2）做局部贡献解释（基于梯度 local attribution）。

    输入：
      x: (d,) 当前样本特征（融合后通常是 embedding z）
      W: (d,K) softmax 权重（BayesianSoftmax.W_map）
      p: (K,) 当前样本 softmax 概率（建议使用校准后 p_final 或 raw p）
      target_labels: 目标集合，例如 [1,2] 表示异常总风险
    输出：
      list[dict] TopK 特征贡献，按 |contribution| 降序
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1)

    d, K = W.shape
    if x.shape[0] != d:
        raise ValueError(f"x dim {x.shape[0]} != W dim {d}")
    if p.shape[0] != K:
        raise ValueError(f"p dim {p.shape[0]} != K {K}")

    # softmax 局部梯度：
    # dp_k/dx_j = p_k * (W_{j,k} - sum_m p_m W_{j,m})
    # 目标概率 p_T = sum_{k in T} p_k
    # dp_T/dx_j = sum_{k in T} dp_k/dx_j
    W_bar = W @ p  # (d,)
    grad = np.zeros(d, dtype=float)
    for k in target_labels:
        grad += p[k] * (W[:, k] - W_bar)

    contrib = x * grad
    idx = np.argsort(-np.abs(contrib))[:int(top_k)]

    out = []
    for j in idx:
        name = feature_names[j] if (feature_names is not None and j < len(feature_names)) else f"feat_{j}"
        out.append({
            "feature": name,
            "x": float(x[j]),
            "grad": float(grad[j]),
            "contribution": float(contrib[j]),
        })
    return out


# =========================================================
# 2) Evidence chain (summary)
# =========================================================

def build_evidence_chain(
    p_model: Dict[str, Any],
    lf_names: Optional[Sequence[str]] = None,
    lf_outputs: Optional[Sequence[int]] = None,
    quality_severity: Optional[float] = None,
    multiclass_x: Optional[np.ndarray] = None,
    multiclass_W: Optional[np.ndarray] = None,
    multiclass_p: Optional[np.ndarray] = None,
    target_labels: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    生成证据链（输出 dict，便于 main_eval 写入 report）。
    - p_model：模型输出与路由信息（p_raw/p_final/abstain/route_kind等）
    - weak supervision votes：如果 eval 阶段也计算 LFs，可填（可选）
    - quality_severity：如果你保留质量向量，可填（可选）
    - top features：对 softmax 目标概率做局部贡献解释（可选）
    """
    fired = []
    if lf_names is not None and lf_outputs is not None:
        for name, out in zip(lf_names, lf_outputs):
            if int(out) != -1:
                fired.append({"lf": str(name), "vote": int(out)})

    contribs: List[Dict[str, Any]] = []
    if (
        multiclass_x is not None and
        multiclass_W is not None and
        multiclass_p is not None and
        target_labels is not None
    ):
        try:
            contribs = topk_contrib_softmax_target(
                x=np.asarray(multiclass_x).reshape(-1),
                W=np.asarray(multiclass_W),
                p=np.asarray(multiclass_p).reshape(-1),
                target_labels=list(target_labels),
                feature_names=feature_names,
                top_k=int(top_k),
            )
        except Exception as e:
            contribs = [{
                "feature": "contrib_error",
                "x": 0.0,
                "grad": 0.0,
                "contribution": 0.0,
                "error": str(e),
            }]

    return {
        "p_model": p_model,
        "quality_severity": float(quality_severity) if quality_severity is not None else None,
        "weak_supervision_votes": fired,
        "top_feature_contributions": contribs,
    }


# =========================================================
# 3) Simple recourse on binary indicator vector (demo)
# =========================================================

def probability_constrained_recourse(
    x: np.ndarray,
    prob_fn: Callable[[np.ndarray], float],
    tau: float,
    max_steps: int = 50
) -> Dict[str, Any]:
    """
    在二值向量 x 上做贪心翻转，尽量降低 prob_fn(x) 到 <= tau。
    - x: (d,) binary-ish
    - prob_fn: returns probability-like scalar
    - tau: target threshold
    """
    x_cur = (np.asarray(x).reshape(-1) > 0.5).astype(float)
    p0 = float(prob_fn(x_cur))
    if p0 <= float(tau):
        return {
            "status": "already_below_threshold",
            "p_before": p0,
            "p_after": p0,
            "steps": [],
            "x_cf": x_cur.tolist(),
        }

    steps: List[Dict[str, Any]] = []
    p_cur = p0
    for _ in range(int(max_steps)):
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
            "p_after_step": float(best_p),
        })
        x_cur, p_cur = best_x, best_p
        if p_cur <= float(tau):
            break

    return {
        "status": "optimized" if p_cur <= float(tau) else "stopped",
        "p_before": p0,
        "p_after": float(p_cur),
        "steps": steps,
        "x_cf": x_cur.tolist(),
    }
