from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
import numpy as np
import math


# =========================================================
# 1) Canonical Data Schema
# =========================================================

@dataclass
class GlobalInfo:
    sex: int = -1
    age: int = -1
    period_info: Any = None
    preg_info: Any = None


@dataclass
class SubReport:
    report_type: int
    status_codes: List[int]


@dataclass
class NeedInfo:
    global_info: GlobalInfo
    reports: List[SubReport] = field(default_factory=list)

    @staticmethod
    def from_legacy(legacy: Sequence[Any]) -> "NeedInfo":
        sex, age, period_info, preg_info = -1, -1, None, None
        reports: List[SubReport] = []

        if legacy:
            gi = legacy[0]
            if isinstance(gi, (list, tuple)):
                if len(gi) >= 1:
                    sex = gi[0]
                if len(gi) >= 2:
                    age = gi[1]
                if len(gi) >= 3:
                    period_info = gi[2]
                if len(gi) >= 4:
                    preg_info = gi[3]

            for sr in legacy[1:]:
                if not isinstance(sr, (list, tuple)) or len(sr) < 2:
                    continue
                r_type = int(sr[0])
                codes = list(map(int, sr[1])) if isinstance(sr[1], (list, tuple)) else []
                reports.append(SubReport(r_type, codes))

        return NeedInfo(
            global_info=GlobalInfo(int(sex), int(age), period_info, preg_info),
            reports=reports,
        )

    def to_legacy(self) -> List[Any]:
        out = [[
            self.global_info.sex,
            self.global_info.age,
            self.global_info.period_info,
            self.global_info.preg_info,
        ]]
        for r in self.reports:
            out.append([r.report_type, list(map(int, r.status_codes))])
        return out


# =========================================================
# 2) Quality Vector Q (enters likelihood)
# =========================================================

@dataclass
class QualityVector:
    ocr_missing_rate: float = 0.0
    match_abs_diff_rate: float = 0.0
    image_insufficient: float = 0.0
    plus_exist_flag: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.ocr_missing_rate,
            self.match_abs_diff_rate,
            self.image_insufficient,
            self.plus_exist_flag,
        ], dtype=float)

    def severity(self) -> float:
        # scalar severity ∈ [0,1] (heuristic weights)
        return float(
            0.40 * self.ocr_missing_rate +
            0.35 * self.match_abs_diff_rate +
            0.20 * self.image_insufficient +
            0.05 * self.plus_exist_flag
        )


def build_context_from_processed(processed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to populate Q-related and other context fields from processed json.
    Safe defaults if missing.

    Recommended keys (optional):
      - ocr_missing_rate
      - match_abs_diff_rate
      - image_insufficient (derived from img counts if possible)
      - plus_exist_flag
      - total_img_number / solvable_img_number (used to derive image_insufficient)
    """
    processed = processed or {}

    ctx: Dict[str, Any] = {}

    # direct fields if present
    for k in ("ocr_missing_rate", "match_abs_diff_rate", "image_insufficient", "plus_exist_flag"):
        if k in processed:
            ctx[k] = processed.get(k)

    # derive image_insufficient if not explicit
    if "image_insufficient" not in ctx:
        total_img = processed.get("total_img_number", None)
        solvable_img = processed.get("solvable_img_number", None)
        try:
            total_img_f = float(total_img) if total_img is not None else 0.0
            solvable_img_f = float(solvable_img) if solvable_img is not None else 0.0
            ctx["image_insufficient"] = 1.0 if (total_img_f > 0 and solvable_img_f <= 0) else 0.0
        except Exception:
            ctx["image_insufficient"] = 0.0

    return ctx


def extract_quality(need: NeedInfo, context: Dict[str, Any]) -> QualityVector:
    context = context or {}

    ocr_missing_rate = float(context.get("ocr_missing_rate", 0.0) or 0.0)
    match_abs_diff_rate = float(context.get("match_abs_diff_rate", 0.0) or 0.0)
    image_insufficient = float(context.get("image_insufficient", 0.0) or 0.0)
    plus_exist_flag = float(context.get("plus_exist_flag", 0.0) or 0.0)

    # fallback proxy: proportion of -1
    if ocr_missing_rate == 0.0:
        total = 0
        missing = 0
        for r in need.reports:
            for c in r.status_codes:
                total += 1
                if int(c) == -1:
                    missing += 1
        if total > 0:
            ocr_missing_rate = missing / total

    return QualityVector(
        ocr_missing_rate=ocr_missing_rate,
        match_abs_diff_rate=match_abs_diff_rate,
        image_insufficient=image_insufficient,
        plus_exist_flag=plus_exist_flag,
    )


# =========================================================
# 3) Indicators / statistics
# =========================================================

def extract_indicator_vector(need: NeedInfo) -> np.ndarray:
    """
    Binary abnormal indicator vector x ∈ {0,1}^d
    """
    vec = []
    for r in need.reports:
        for c in r.status_codes:
            c = int(c)
            # treat -1/0/9 as normal/unknown
            if c in (-1, 0, 9):
                vec.append(0.0)
            else:
                vec.append(1.0)
    if len(vec) == 0:
        return np.zeros(1, dtype=float)
    return np.array(vec, dtype=float)


def abnormal_ratio(need: NeedInfo) -> float:
    total = 0
    abnormal = 0
    for r in need.reports:
        for c in r.status_codes:
            c = int(c)
            if c == -1:
                continue
            total += 1
            if c not in (-1, 0, 9):
                abnormal += 1
    return float(abnormal / total) if total > 0 else 0.0


def num_reports(need: NeedInfo) -> int:
    return int(len(need.reports))


def num_codes(need: NeedInfo) -> int:
    s = 0
    for r in need.reports:
        s += len(r.status_codes)
    return int(s)


def missing_rate_codes(need: NeedInfo) -> float:
    total = 0
    miss = 0
    for r in need.reports:
        for c in r.status_codes:
            total += 1
            if int(c) == -1:
                miss += 1
    return float(miss / total) if total > 0 else 0.0


# =========================================================
# 4) Unified feature extractor
# =========================================================

_FEATURE_ORDER = [
    # generative outputs
    "gen_p_high",
    "gen_entropy",
    "gen_dimension",
    # need stats
    "abnormal_ratio",
    "num_reports",
    "num_codes",
    "missing_rate_codes",
    # quality
    "Q_severity",
    "Q_ocr_missing_rate",
    "Q_match_abs_diff_rate",
    "Q_image_insufficient",
    "Q_plus_exist_flag",
    # bias
    "bias",
]


def extract_feature_dict(
    need: NeedInfo,
    Q: QualityVector,
    gen_out: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, float]:
    """
    Returns a stable numeric feature dict used by build_model_features().
    """
    context = context or {}
    gen_p = float(gen_out.get("p_high", 0.0) or 0.0)
    gen_ent = float(gen_out.get("entropy", 0.0) or 0.0)
    gen_dim = float(gen_out.get("dimension", 0.0) or 0.0)

    f = {
        "gen_p_high": gen_p,
        "gen_entropy": gen_ent,
        "gen_dimension": gen_dim,
        "abnormal_ratio": float(abnormal_ratio(need)),
        "num_reports": float(num_reports(need)),
        "num_codes": float(num_codes(need)),
        "missing_rate_codes": float(missing_rate_codes(need)),
        "Q_severity": float(Q.severity()),
        "Q_ocr_missing_rate": float(Q.ocr_missing_rate),
        "Q_match_abs_diff_rate": float(Q.match_abs_diff_rate),
        "Q_image_insufficient": float(Q.image_insufficient),
        "Q_plus_exist_flag": float(Q.plus_exist_flag),
        "bias": 1.0,
    }
    return f


def build_model_features(feature_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert dict -> fixed-order numeric vector.
    """
    vec = []
    for k in _FEATURE_ORDER:
        vec.append(float(feature_dict.get(k, 0.0) or 0.0))
    return np.array(vec, dtype=float)


# =========================================================
# 5) Math utils
# =========================================================

def sigmoid(z):
    # supports scalar or ndarray
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def entropy_bernoulli(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))
