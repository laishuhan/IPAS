from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import NeedInfo, extract_quality, extract_indicator_vector
from generative import GenerativeRiskModel
from bayes import BayesianLogistic, build_feature_vector
from selective import ConformalBinary, selective_risk_curve
from weak_supervision import build_default_lfs, apply_lfs, LabelModelEM
from calibration import PlattCalibrator, IsotonicCalibrator, ece_score, brier_score


# -----------------------------
# I/O helpers
# -----------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_need_info_from_processed(processed: Dict[str, Any]) -> List[Any]:
    info_list = processed.get("info", [])
    if not info_list:
        raise ValueError("processed_report_info.json missing field: info")

    first = info_list[0]
    sex = first.get("sex", -1)
    age = first.get("age", -1)
    period_info = first.get("period_info", None)
    preg_info = first.get("preg_info", None)

    need_info = []
    need_info.append([sex, age, period_info, preg_info])

    for report in info_list:
        r_type = report.get("report_type", -1)
        indicators = report.get("indicator_analysis", [])
        codes = []
        for item in indicators:
            try:
                codes.append(int(item))
            except Exception:
                codes.append(-1)
        need_info.append([r_type, codes])

    return need_info


def load_dataset_from_dir(root_dir: str) -> Tuple[List[NeedInfo], List[Dict[str, Any]]]:
    """
    Expects files:
      root_dir/**/processed_report_info.json
    Returns: need_list, context_list (quality context)
    """
    need_list: List[NeedInfo] = []
    ctx_list: List[Dict[str, Any]] = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "processed_report_info.json" not in filenames:
            continue
        p = os.path.join(dirpath, "processed_report_info.json")
        processed = _read_json(p)

        legacy = _build_need_info_from_processed(processed)
        need = NeedInfo.from_legacy(legacy)

        total_img = float(processed.get("total_img_number", 0) or 0)
        solvable_img = float(processed.get("solvable_img_number", 0) or 0)
        image_insufficient = 1.0 if (total_img > 0 and solvable_img <= 0) else 0.0

        # Add more context fields here if you have them
        ctx = {
            "image_insufficient": image_insufficient,
            "ocr_missing_rate": processed.get("ocr_missing_rate", 0.0),
            "match_abs_diff_rate": processed.get("match_abs_diff_rate", 0.0),
            "plus_exist_flag": processed.get("plus_exist_flag", 0.0),
            "_path": p,
        }

        need_list.append(need)
        ctx_list.append(ctx)

    return need_list, ctx_list


def load_labels_optional(labels_path: Optional[str], ctx_list: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    Optional labels mapping.
    Format (JSON):
      { "<processed_report_info_path>": 0/1, ... }
    or:
      { "<case_id>": 0/1, ... } if you include ids in ctx_list later.
    """
    if not labels_path:
        return None
    mp = _read_json(labels_path)
    y = []
    for ctx in ctx_list:
        key = ctx.get("_path")
        if key in mp:
            y.append(int(mp[key]))
        else:
            y.append(-1)
    y = np.asarray(y, dtype=int)
    if np.all(y < 0):
        return None
    return y


# -----------------------------
# Baseline score (simple)
# -----------------------------

def baseline_rule_score(need: NeedInfo) -> float:
    """
    Very simple baseline: abnormal indicator ratio mapped to probability-like score.
    (Replace with your real rule score if you have it.)
    """
    x = extract_indicator_vector(need)
    if len(x) == 0:
        return 0.5
    r = float(np.mean(x))
    # squash
    return float(1.0 / (1.0 + np.exp(-6.0 * (r - 0.3))))


# -----------------------------
# Quality degradation / robustness
# -----------------------------

def degrade_quality_context(ctx: Dict[str, Any], severity: float = 0.3) -> Dict[str, Any]:
    """
    Synthetic degradation:
      - increase missing rate
      - increase match diff
      - optionally force image_insufficient
    """
    out = dict(ctx)
    out["ocr_missing_rate"] = float(min(1.0, float(out.get("ocr_missing_rate", 0.0) or 0.0) + severity))
    out["match_abs_diff_rate"] = float(min(1.0, float(out.get("match_abs_diff_rate", 0.0) or 0.0) + severity * 0.8))
    if severity >= 0.6:
        out["image_insufficient"] = 1.0
    return out


# -----------------------------
# Main experiment pipeline
# -----------------------------

def run_experiments(
    data_dir: str,
    out_path: str,
    labels_path: Optional[str] = None,
    alpha: float = 0.1,
    n_bins: int = 10,
) -> Dict[str, Any]:

    needs, ctxs = load_dataset_from_dir(data_dir)
    if len(needs) == 0:
        raise ValueError(f"No processed_report_info.json found under: {data_dir}")

    # quality vectors
    Qs = [extract_quality(n, c) for n, c in zip(needs, ctxs)]

    # baseline
    p_base = np.array([baseline_rule_score(n) for n in needs], dtype=float)

    # generative
    gen = GenerativeRiskModel()
    gen_outs = [gen.posterior(n, Q) for n, Q in zip(needs, Qs)]
    p_gen = np.array([g["p_high"] for g in gen_outs], dtype=float)
    ent_gen = np.array([g["entropy"] for g in gen_outs], dtype=float)

    # weak supervision
    lfs = build_default_lfs()
    L = apply_lfs(needs, Qs, lfs, ctxs)

    lm = LabelModelEM()
    lm.fit(L)
    p_soft = lm.predict_proba(L)  # soft pseudo-label

    # train Bayesian LR on soft labels
    Phi = np.vstack([build_feature_vector(g, Q.severity()) for g, Q in zip(gen_outs, Qs)])
    bayes = BayesianLogistic(lambda_reg=1.0)
    bayes.fit(Phi, p_soft)  # requires your modified bayes.py to accept soft y

    p_bayes = np.array([bayes.predict_proba(phi) for phi in Phi], dtype=float)
    epi = np.array([bayes.predictive_variance(phi) for phi in Phi], dtype=float)

    # calibration (if labels provided and available)
    y = load_labels_optional(labels_path, ctxs)
    cal_summary = {"available": False}

    p_cal_platt = p_bayes.copy()
    p_cal_iso = p_bayes.copy()

    if y is not None:
        # keep only labeled subset
        mask = (y >= 0)
        y_lab = y[mask].astype(int)
        p_lab = p_bayes[mask]

        if len(y_lab) >= 10 and len(np.unique(y_lab)) >= 2:
            platt = PlattCalibrator().fit(p_lab, y_lab)
            iso = IsotonicCalibrator().fit(p_lab, y_lab)

            p_cal_platt = p_bayes.copy()
            p_cal_platt[mask] = platt.transform(p_lab)

            p_cal_iso = p_bayes.copy()
            p_cal_iso[mask] = iso.transform(p_lab)

            cal_summary = {
                "available": True,
                "platt": platt.metrics(p_lab, y_lab, n_bins=n_bins),
                "isotonic": iso.metrics(p_lab, y_lab, n_bins=n_bins),
            }

    # conformal + selective risk (needs labels for a real evaluation)
    conf_summary = {"available": False}
    if y is not None:
        mask = (y >= 0)
        y_lab = y[mask].astype(int)
        p_lab = p_cal_platt[mask]  # use platt-calibrated on labeled subset

        if len(y_lab) >= 30 and len(np.unique(y_lab)) >= 2:
            # split calibration/eval
            idx = np.arange(len(y_lab))
            np.random.seed(0)
            np.random.shuffle(idx)
            cut = int(0.5 * len(idx))
            cal_idx, te_idx = idx[:cut], idx[cut:]

            conf = ConformalBinary(alpha=alpha).fit(p_lab[cal_idx], y_lab[cal_idx])
            sets = [conf.predict_set(float(pv)) for pv in p_lab[te_idx]]

            # uncertainty thresholds & selective curve
            # we use epistemic from original indexing; map back to labeled subset
            epi_lab = epi[mask][te_idx]
            thresholds = list(np.linspace(float(np.min(epi_lab)), float(np.max(epi_lab)), 10))

            curve = selective_risk_curve(
                p=p_lab[te_idx],
                y=y_lab[te_idx],
                uncertainty=epi_lab,
                thresholds=thresholds
            )

            conf_summary = {
                "available": True,
                "alpha": float(alpha),
                "qhat": float(conf.qhat) if conf.qhat is not None else None,
                "selective_curve": curve,
                "n_cal": int(len(cal_idx)),
                "n_test": int(len(te_idx)),
            }

    # robustness test (quality degradation)
    ctxs_bad = [degrade_quality_context(c, severity=0.4) for c in ctxs]
    Qs_bad = [extract_quality(n, c) for n, c in zip(needs, ctxs_bad)]
    gen_outs_bad = [gen.posterior(n, Q) for n, Q in zip(needs, Qs_bad)]
    Phi_bad = np.vstack([build_feature_vector(g, Q.severity()) for g, Q in zip(gen_outs_bad, Qs_bad)])
    p_bayes_bad = np.array([bayes.predict_proba(phi) for phi in Phi_bad], dtype=float)
    epi_bad = np.array([bayes.predictive_variance(phi) for phi in Phi_bad], dtype=float)

    robustness = {
        "mean_p_before": float(np.mean(p_bayes)),
        "mean_p_after": float(np.mean(p_bayes_bad)),
        "mean_epi_before": float(np.mean(epi)),
        "mean_epi_after": float(np.mean(epi_bad)),
        "delta_mean_epi": float(np.mean(epi_bad) - np.mean(epi)),
    }

    summary = {
        "counts": {"n_total": int(len(needs)), "n_labeled": int(np.sum((y >= 0))) if y is not None else 0},
        "label_model": lm.summary(),
        "lf_names": [lf.name for lf in lfs],
        "metrics_if_labeled": {
            "available": bool(y is not None),
            "brier_base": None if y is None else brier_score(p_base[y >= 0], y[y >= 0]),
            "brier_gen": None if y is None else brier_score(p_gen[y >= 0], y[y >= 0]),
            "brier_bayes": None if y is None else brier_score(p_bayes[y >= 0], y[y >= 0]),
            "ece_bayes": None if y is None else ece_score(p_bayes[y >= 0], y[y >= 0], n_bins=n_bins),
        },
        "calibration": cal_summary,
        "conformal_selective": conf_summary,
        "robustness": robustness,
        "samples_preview": [
            {
                "path": ctxs[i].get("_path"),
                "p_base": float(p_base[i]),
                "p_gen": float(p_gen[i]),
                "p_bayes": float(p_bayes[i]),
                "entropy_gen": float(ent_gen[i]),
                "q_severity": float(Qs[i].severity()),
            }
            for i in range(min(10, len(needs)))
        ],
    }

    _write_json(out_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing processed_report_info.json files")
    parser.add_argument("--out_path", type=str, required=True, help="Write JSON summary here")
    parser.add_argument("--labels_path", type=str, default=None, help="Optional JSON labels map {path:0/1}")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    summary = run_experiments(
        data_dir=args.data_dir,
        out_path=args.out_path,
        labels_path=args.labels_path,
        alpha=args.alpha,
        n_bins=args.n_bins,
    )
    print(f"[OK] wrote: {args.out_path}")
    print(f"n_total={summary['counts']['n_total']}, n_labeled={summary['counts']['n_labeled']}")


if __name__ == "__main__":
    main()
