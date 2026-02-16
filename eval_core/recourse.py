from __future__ import annotations

from typing import Callable, Dict, Any, List
import numpy as np


def probability_constrained_recourse(
    x: np.ndarray,
    prob_fn: Callable[[np.ndarray], float],
    tau: float,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """
    Greedy recourse on binary indicator vector x ∈ {0,1}^d:

      min ||Δx||_0 (greedy proxy)
      s.t. prob_fn(x+Δx) <= tau

    Strategy:
      - At each step, try flipping each coordinate (1->0, also 0->1 if it decreases prob),
        choose the flip that decreases probability the most.
      - Stop once <= tau or cannot improve.

    Returns:
      - steps: list of actions
      - p_before / p_after
      - x_cf
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    tau = float(tau)

    # binarize view (keep floats but treat >0.5 as 1)
    x_cur = (x > 0.5).astype(float)

    p0 = float(prob_fn(x_cur))
    if p0 <= tau:
        return {
            "status": "already_below_threshold",
            "p_before": p0,
            "p_after": p0,
            "tau": tau,
            "steps": [],
            "x_cf": x_cur.tolist(),
        }

    steps: List[Dict[str, Any]] = []
    p_cur = p0

    for _ in range(int(max_steps)):
        best_i = None
        best_p = p_cur
        best_x = None

        for i in range(len(x_cur)):
            x_try = x_cur.copy()
            # flip
            x_try[i] = 1.0 - x_try[i]
            p_try = float(prob_fn(x_try))

            if p_try < best_p - 1e-12:
                best_p = p_try
                best_i = i
                best_x = x_try

        if best_i is None or best_x is None:
            break  # cannot improve

        action = "set_to_normal" if x_cur[best_i] > 0.5 else "activate_indicator"
        # note: action meaning depends on your indicator semantics; we keep consistent with previous file
        steps.append({
            "feature_index": int(best_i),
            "action": action,
            "p_after_step": float(best_p),
        })

        x_cur = best_x
        p_cur = best_p

        if p_cur <= tau:
            return {
                "status": "optimized",
                "p_before": p0,
                "p_after": p_cur,
                "tau": tau,
                "steps": steps,
                "x_cf": x_cur.tolist(),
            }

    return {
        "status": "stopped_no_solution" if p_cur > tau else "optimized",
        "p_before": p0,
        "p_after": p_cur,
        "tau": tau,
        "steps": steps,
        "x_cf": x_cur.tolist(),
    }
