# === main_eval.py ===
# PRISM-Risk Upgraded Pipeline: 
# Weak Supervision + Bayesian Inference + Calibration + Conformal + Evidence + Recourse
# 入口参数：--processed_report_info_path (输入JSON) --eval_info_path (输出文本)

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 导入内部模块
from core import (
    NeedInfo,
    build_context_from_processed,
    extract_quality,
    extract_indicator_vector,
    extract_feature_dict,
    build_model_features,
)
from generative import GenerativeRiskModel
from bayes import BayesianLogistic
from calibration import PlattCalibrator, IsotonicCalibrator, ece_score, brier_score, reliability_bins
from selective import ConformalBinary, abstain_decision, selective_metrics
from weak_supervision import build_default_lfs, LabelModelEM
from evidence import build_evidence_chain
from recourse import probability_constrained_recourse


# =========================================================
# 辅助工具函数
# =========================================================

def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _risk_level_cn(p: float, abstain: bool) -> str:
    if abstain:
        return "拒判（建议人工复核）"
    if p >= 0.7:
        return "高风险"
    if p <= 0.3:
        return "低风险"
    return "中风险"


def _build_need_info_from_processed(processed: Dict[str, Any]) -> List[Any]:
    """解析原始 JSON 中的指标信息"""
    info_list = processed.get("info", [])
    if not info_list:
        raise ValueError("processed_report_info.json 缺少 info 字段")

    first = info_list[0]
    sex = first.get("sex", -1)
    age = first.get("age", -1)
    period_info = first.get("period_info", None)
    preg_info = first.get("preg_info", None)

    need_info = [[sex, age, period_info, preg_info]]
    for report in info_list:
        r_type = report.get("report_type", -1)
        indicators = report.get("indicator_analysis", [])
        codes = [int(item) if str(item).isdigit() or (isinstance(item, (int, float)) and item == -1) else -1 for item in indicators]
        need_info.append([r_type, codes])

    return need_info


def _infer_splits(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """默认数据集切分：60% 训练, 20% 校准, 20% 测试"""
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = max(1, int(0.6 * n))
    n_cal = max(1, int(0.2 * n))
    return idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]


def _maybe_get_dataset(processed: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    ds = processed.get("dataset", None)
    return ds if isinstance(ds, list) and len(ds) > 0 else None


def _load_need_ctx_q(sample: Dict[str, Any]) -> Tuple[NeedInfo, Dict[str, Any], QualityVector]:
    """支持多格式样本加载"""
    if "info" in sample:
        legacy = _build_need_info_from_processed(sample)
        context = build_context_from_processed(sample)
    else:
        legacy = sample.get("legacy", None)
        context = sample.get("context", {})
    
    need = NeedInfo.from_legacy(legacy)
    Q = extract_quality(need, context)
    return need, context, Q


# =========================================================
# 主程序
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_report_info_path", type=str, required=True)
    parser.add_argument("--eval_info_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.processed_report_info_path):
        raise FileNotFoundError(f"输入文件未找到: {args.processed_report_info_path}")

    with open(args.processed_report_info_path, "r", encoding="utf-8") as f:
        processed = json.load(f)

    warnings: List[str] = []

    # 1) 单样本解析 (当前正在评估的对象)
    need, context, Q = _load_need_ctx_q(processed)

    # 2) 生成式模型后验概率 (Latent variable Z)
    gen_model = GenerativeRiskModel()
    gen_out = gen_model.posterior(need, Q)

    # 3) 弱监督投票 (Labeling Functions)
    lfs = build_default_lfs()
    # BUG FIX: 调用 lf.fn 而非直接调用对象
    lf_votes_single = np.array([lf.fn(need, Q, context) for lf in lfs], dtype=int)

    # 4) 数据集处理 (若提供 dataset 字段，则进行在线学习/校准)
    dataset = _maybe_get_dataset(processed)
    labels = processed.get("labels", None)
    
    Phi_all, y_soft, y_true = None, None, None
    train_idx, cal_idx, test_idx = None, None, None

    if dataset and len(dataset) >= 5:
        n = len(dataset)
        train_idx, cal_idx, test_idx = _infer_splits(n)
        L_all = np.zeros((n, len(lfs)), dtype=int)
        feats = []

        for i, sample in enumerate(dataset):
            ni, ci, Qi = _load_need_ctx_q(sample)
            # BUG FIX: 调用 lf.fn
            L_all[i, :] = np.array([lf.fn(ni, Qi, ci) for lf in lfs], dtype=int)
            gi = gen_model.posterior(ni, Qi)
            phi_i = build_model_features(extract_feature_dict(ni, Qi, gi, ci))
            feats.append(phi_i)

        Phi_all = np.vstack(feats)

        # 确定标签来源：真值或 LabelModel
        if isinstance(labels, list) and len(labels) == n:
            y_true = np.asarray(labels, dtype=int)
            y_soft = y_true.astype(float)
        else:
            lm = LabelModelEM()
            lm.fit(L_all[train_idx, :])
            y_soft = lm.predict_proba(L_all)
    else:
        warnings.append("未提供数据集或数据不足，将使用预置权重运行演示模式。")

    # 5) 训练/拟合贝叶斯模型 (Laplace Approximation)
    bayes = BayesianLogistic(lambda_reg=1.0)
    phi_single = build_model_features(extract_feature_dict(need, Q, gen_out, context)).reshape(1, -1)

    if Phi_all is not None and y_soft is not None:
        bayes.fit(Phi_all[train_idx, :], y_soft[train_idx])
    else:
        # 演示模式：极小规模拟合
        bayes.fit(np.vstack([phi_single, phi_single]), np.array([0.5, 0.5]))

    # 获取贝叶斯预测与不确定性分解
    pred_u = bayes.predictive_with_uncertainty(phi_single.flatten())
    p_raw = pred_u["p_mean"]
    epistemic = pred_u["epistemic"]
    aleatoric = pred_u["aleatoric"]

    # 6) 校准 (Calibration) 与 共形预测 (Conformal)
    p_final = p_raw
    calib_report = {"method": "none"}
    conformal_set = ["mid"]
    platt = PlattCalibrator()

    if Phi_all is not None and y_true is not None and cal_idx is not None:
        p_cal_raw = bayes.predict_proba(Phi_all[cal_idx, :])
        y_cal = y_true[cal_idx]
        
        # Platt Scaling
        platt.fit(p_cal_raw, y_cal)
        # BUG FIX: 使用 transform 而非 predict
        p_final = float(platt.transform(np.array([p_raw]))[0])
        p_cal_platt = platt.transform(p_cal_raw)

        calib_report = {
            "method": "platt",
            "ece": float(ece_score(p_cal_platt, y_cal)),
            "brier": float(brier_score(p_cal_platt, y_cal)),
            "params": {"a": platt.a, "b": platt.b}
        }

        # Conformal Prediction
        conf = ConformalBinary(alpha=0.1)
        conf.fit(p_cal_platt, y_cal)
        conformal_set = conf.predict_set(p_final)
    else:
        warnings.append("缺少标签或校准集，无法提供校准及共形覆盖率保证。")

    # 7) 决策逻辑：拒判与风险等级
    # 基于认识不确定性 (epistemic) 或 共形集合模糊度
    abstain = abstain_decision(epistemic, conformal_set, epistemic_threshold=0.6)
    risk_level = _risk_level_cn(p_final, abstain)

    # 8) 证据链报告 (Interpretability)
    # BUG FIX: 传递修正后的参数名及 w_map
    evidence = build_evidence_chain(
        need=need, Q=Q,
        p_model=p_final,
        lf_names=[lf.name for lf in lfs],
        lf_outputs=lf_votes_single,
        bayes_phi=phi_single.flatten(),
        bayes_w=bayes.w_map
    )

    # 9) 反事实溯因 (Recourse)
    # 目标：如果异常指标减少，概率如何变化
    def prob_oracle(x_new: np.ndarray) -> float:
        f = dict(extract_feature_dict(need, Q, gen_out, context))
        f["abnormal_ratio"] = float(np.mean(x_new)) if len(x_new) > 0 else 0.0
        p = float(bayes.predict_proba(build_model_features(f).reshape(1, -1))[0])
        if calib_report["method"] == "platt":
            p = float(platt.transform(np.array([p]))[0])
        return p

    recourse = probability_constrained_recourse(
        extract_indicator_vector(need), 
        prob_oracle, 
        tau=0.3
    )

    # 10) 输出结果保存
    res_lines = [
        "PRISM-Risk 综合评估报告",
        "================================",
        f"原始预测概率 (p_raw): {p_raw:.4f}",
        f"校准后概率 (p_final): {p_final:.4f}",
        f"风险等级 (Risk Level): {risk_level}",
        f"共形预测集 (Conformal Set): {conformal_set}",
        f"建议拒判 (Abstain): {'是' if abstain else '否'}",
        "",
        "不确定性分解 (Uncertainty)",
        "--------------------------------",
        f"偶然不确定性 (Aleatoric - 数据噪声): {aleatoric:.6f}",
        f"认识不确定性 (Epistemic - 模型认知): {epistemic:.6f}",
        "",
        "校准详情 (Calibration)",
        "--------------------------------",
        json.dumps(calib_report, ensure_ascii=False, indent=2),
        "",
        "证据链 (Evidence Chain)",
        "--------------------------------",
        json.dumps(evidence, ensure_ascii=False, indent=2),
        "",
        "溯因分析 (Recourse Path)",
        "--------------------------------",
        json.dumps(recourse, ensure_ascii=False, indent=2),
        ""
    ]

    if warnings:
        res_lines.append("警告 (Warnings)")
        res_lines.append("--------------------------------")
        for w in warnings:
            res_lines.append(f"- {w}")

    output_content = "\n".join(res_lines)
    _ensure_dir_for_file(args.eval_info_path)
    with open(args.eval_info_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"\n[Success] 评估完成。结果已写入: {args.eval_info_path}")


if __name__ == "__main__":
    main()
