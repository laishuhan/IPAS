from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


from core import NeedInfo, QualityVector


def _sign_vote_to_str(v: int) -> str:
    if v > 0:
        return "vote_high"
    if v < 0:
        return "vote_low"
    return "abstain"


def bayes_contributions(phi: np.ndarray, w: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    phi = np.asarray(phi).reshape(-1)
    w = np.asarray(w).reshape(-1)
    contrib = phi * w
    idx = np.argsort(-np.abs(contrib))[: max(1, int(top_k))]
    out: List[Dict[str, Any]] = []
    for i in idx:
        out.append({
            "feature_index": int(i),
            "phi": float(phi[i]),
            "weight": float(w[i]),
            "contribution": float(contrib[i]),
        })
    return out


def _sign_vote_to_str(v: int) -> str:
    return "high_risk" if v > 0 else ("low_risk" if v < 0 else "abstain")

def build_evidence_chain(
    need: NeedInfo,
    Q: QualityVector,
    p_model: float,
    lf_names: Sequence[str],
    lf_outputs: Sequence[int],
    lf_acc: Optional[Sequence[float]] = None,
    bayes_phi: Optional[np.ndarray] = None,
    bayes_w: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    fired = []
    for i, (name, out) in enumerate(zip(lf_names, lf_outputs)):
        if out != 0:
            fired.append({
                "lf": name,
                "vote": _sign_vote_to_str(out),
                "weight": float(lf_acc[i]) if lf_acc else None
            })

    contribs = []
    if bayes_phi is not None and bayes_w is not None:
        raw_contrib = bayes_phi * bayes_w
        idx = np.argsort(-np.abs(raw_contrib))[:top_k]
        from core import _FEATURE_ORDER
        for i in idx:
            contribs.append({
                "feature": _FEATURE_ORDER[i] if i < len(_FEATURE_ORDER) else f"feat_{i}",
                "impact": float(raw_contrib[i])
            })

    return {
        "p_high": float(p_model),
        "quality_severity": float(Q.severity()),
        "weak_supervision_votes": fired,
        "top_feature_contributions": contribs
    }
