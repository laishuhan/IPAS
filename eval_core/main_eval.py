# === main_eval.py ===
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from models import (
    NeedInfo,
    build_context_from_processed,
    extract_quality,
    extract_indicator_vector,
    GenerativeRiskModel,
    build_default_lfs,
    BayesianSoftmax,
)

from utils import (
    TemperatureCalibrator,
    ConformalAPS,
    abstain_decision_multiclass,
    build_evidence_chain,
    probability_constrained_recourse,
    topk_contrib_softmax_target,
)

from data_adapter import (
    build_xy_meta_evidence_from_processed,
    map_status_to_severity,
    severity_to_abnormal_binary,
)

from feature_space import featurize_sample


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _risk_level_cn_3class(y_pred: int, abstain: bool) -> str:
    if abstain:
        return "拒判（建议人工复核）"
    if y_pred == 2:
        return "重度异常"
    if y_pred == 1:
        return "轻度异常"
    return "正常"


def _build_need_info_from_processed(processed: Dict[str, Any]) -> List[Any]:
    info_list = processed.get("info", [])
    if not info_list:
        raise ValueError("processed_report_info.json 缺少 info 字段")

    first = info_list[0]
    sex = first.get("sex", -1)
    age = first.get("age", -1)
    period_info = first.get("period_info", None)
    preg_info = first.get("preg_info", None)

    legacy = [[sex, age, period_info, preg_info]]
    for report in info_list:
        r_type = report.get("report_type", -1)
        indicators = report.get("indicator_analysis", []) or []
        codes = []
        for it in indicators:
            sev = map_status_to_severity(it)
            b = severity_to_abnormal_binary(sev)
            codes.append(-1 if b is None else int(b))
        legacy.append([r_type, codes])

    return legacy


def _load_need_ctx_q(sample: Dict[str, Any]) -> Tuple[NeedInfo, Dict[str, Any], Any]:
    legacy = _build_need_info_from_processed(sample)
    context = build_context_from_processed(sample)
    need = NeedInfo.from_legacy(legacy)
    Q = extract_quality(need, context)
    return need, context, Q


def load_bundle(model_dir: str) -> Tuple[BayesianSoftmax, TemperatureCalibrator, ConformalAPS, Dict[str, Any]]:
    npz_path = os.path.join(model_dir, "softmax_model.npz")
    cal_path = os.path.join(model_dir, "calibrator.json")
    conf_path = os.path.join(model_dir, "conformal.json")
    schema_path = os.path.join(model_dir, "schema.json")

    for p in (npz_path, cal_path, conf_path, schema_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少模型文件: {p}")

    data = np.load(npz_path)
    W_map = data["W_map"]
    Sigma = data["Sigma"]
    n_classes = int(data["n_classes"])
    feature_dim = int(data["feature_dim"])

    clf = BayesianSoftmax(n_classes=n_classes)
    clf.W_map = W_map
    clf.Sigma = None if Sigma.shape == (1, 1) else Sigma

    with open(cal_path, "r", encoding="utf-8") as f:
        calj = json.load(f)
    temp = TemperatureCalibrator()
    temp.T = float(calj.get("T", 1.0))

    with open(conf_path, "r", encoding="utf-8") as f:
        confj = json.load(f)
    conf = ConformalAPS(alpha=float(confj.get("alpha", 0.1)))
    conf.qhat = confj.get("qhat", None)
    if conf.qhat is not None:
        conf.qhat = float(conf.qhat)

    with open(schema_path, "r", encoding="utf-8") as f:
        sj = json.load(f)

    schema = sj.get("schema", {})
    meta = sj.get("train_meta", {})
    extra = {"schema": schema, "train_meta": meta, "feature_dim": feature_dim}

    if clf.W_map.shape[0] != feature_dim:
        raise ValueError("模型包 feature_dim 与权重维度不一致，模型包可能损坏。")

    return clf, temp, conf, extra


def pick_model_dir(model_root: str, report_type: int) -> Tuple[str, str]:
    """
    路由优先级：type -> cluster -> global
    返回：(model_dir, route_kind)
    """
    type_dir = os.path.join(model_root, f"type_{report_type}")
    if os.path.isdir(type_dir) and os.path.exists(os.path.join(type_dir, "softmax_model.npz")):
        return type_dir, "type"

    cmap_path = os.path.join(model_root, "cluster_map.json")
    if os.path.exists(cmap_path):
        with open(cmap_path, "r", encoding="utf-8") as f:
            cj = json.load(f)
        cmap = cj.get("cluster_map", {})
        cid = cmap.get(str(report_type), None)
        if cid is not None:
            cdir = os.path.join(model_root, f"cluster_{int(cid)}")
            if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, "softmax_model.npz")):
                return cdir, f"cluster_{int(cid)}"

    global_dir = os.path.join(model_root, "global")
    return global_dir, "global"

def main():
        
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_report_info_path", type=str, required=True)
    ap.add_argument("--eval_info_path", type=str, required=True)

    default_root = os.path.join(current_dir, "model")
    ap.add_argument("--model_root", type=str, default=default_root, help="模型目录根（默认: 同级中 model）")
    args = ap.parse_args()


    # --- 增加健壮性检查 ---
    try:
        with open(args.processed_report_info_path, "r", encoding="utf-8") as f:
            processed = json.load(f)
    except Exception as e:
        print(f"[Error] 无法读取或解析 JSON 文件: {e}")
        return

    # 检查 info 字段是否存在且不为空
    infos = processed.get("info", [])
    if not isinstance(infos, list) or len(infos) == 0:
        error_msg = (
            "评估终止：输入 JSON 缺少有效的 'info' 数据。\n"
            f"文件路径: {args.processed_report_info_path}\n"
            "可能原因：OCR 未能识别出任何有效报告内容或数据清洗逻辑异常。"
        )
        _ensure_dir_for_file(args.eval_info_path)
        with open(args.eval_info_path, "w", encoding="utf-8") as f:
            f.write(error_msg)
        print(f"[Warning] {error_msg}")
        return # 优雅退出，不抛出 Crash

    # 此时可以安全访问 [0]
    info0 = infos[0]
    rt = int(info0.get("report_type", -1))
    # --- 检查结束 ---

    model_dir, route_kind = pick_model_dir(args.model_root, rt)
    clf, temp, conf, extra = load_bundle(model_dir)
    schema = extra["schema"]

    x_vec = featurize_sample(processed, schema)
    if x_vec.shape[0] != clf.W_map.shape[0]:
        raise ValueError(f"特征维度不一致：x={x_vec.shape[0]} vs model={clf.W_map.shape[0]}。请确认 train/eval 同一套 schema。")

    need, context, Q = _load_need_ctx_q(processed)
    gen_model = GenerativeRiskModel()
    _ = gen_model.posterior(need, Q)

    lfs = build_default_lfs()
    lf_votes_single = np.array([lf.fn(need, Q, context) for lf in lfs], dtype=int)

    pred_u = clf.predictive_with_uncertainty(x_vec)
    p_raw = pred_u["p"]
    entropy = pred_u["entropy"]
    epistemic = pred_u["epistemic"]
    aleatoric = pred_u["aleatoric"]

    p_final = temp.transform(p_raw.reshape(1, -1))[0]
    pred_set = conf.predict_set(p_final) if conf.qhat is not None else [int(np.argmax(p_final))]

    abstain = abstain_decision_multiclass(
        epistemic=epistemic,
        pred_set=pred_set,
        entropy=entropy,
        epistemic_threshold=0.6,
        entropy_threshold=1.0,
        allow_set_size_gt1=False,
    )

    y_pred_final = int(np.argmax(p_final))
    p_abn = float(p_final[1] + p_final[2])
    p_sev = float(p_final[2])
    risk_level = _risk_level_cn_3class(y_pred_final, abstain)

    X_single, y_single, meta_single, evidence_single, _feat_names = build_xy_meta_evidence_from_processed(processed)

    evidence_chain = build_evidence_chain(
        need=need,
        Q=Q,
        p_model={
            "p_raw": p_raw.tolist(),
            "p_final": p_final.tolist(),
            "p_abnormal": p_abn,
            "p_severe": p_sev,
            "pred_set": pred_set,
            "route_kind": route_kind,
            "model_dir_used": model_dir,
        },
        lf_names=[lf.name for lf in lfs],
        lf_outputs=lf_votes_single,
        multiclass_x=x_vec,
        multiclass_W=clf.W_map,
        multiclass_p=p_final,
        target_labels=[1, 2],   # explain p1+p2
        feature_names=None,     # 指标名不可靠：不填
        top_k=8,
    )

    try:
        severe_contribs = topk_contrib_softmax_target(
            x=x_vec, W=clf.W_map, p=p_final,
            target_labels=[2],
            feature_names=None,
            top_k=8,
        )
    except Exception as e:
        severe_contribs = [{"feature": "contrib_error", "x": 0.0, "grad": 0.0, "contribution": 0.0, "error": str(e)}]
    evidence_chain["top_feature_contributions_severe_p2"] = severe_contribs

    def prob_oracle(_: np.ndarray) -> float:
        return float(p_abn)

    recourse = probability_constrained_recourse(extract_indicator_vector(need), prob_oracle, tau=0.3)

    lines = [
        "综合评估报告（三分类 | 路由：type -> cluster -> global）",
        "================================",
        f"report_type: {rt}",
        f"route_kind: {route_kind}",
        f"model_dir_used: {model_dir}",
        "",
        "模型输出",
        "--------------------------------",
        f"p_raw: {p_raw.tolist()}",
        f"p_final: {p_final.tolist()}",
        f"p_abnormal(p1+p2): {p_abn:.6f}",
        f"p_severe(p2): {p_sev:.6f}",
        f"pred_final(argmax): {y_pred_final}",
        f"pred_set(APS): {pred_set}",
        f"risk_level: {risk_level}",
        f"abstain: {'是' if abstain else '否'}",
        "",
        "不确定性",
        "--------------------------------",
        f"entropy: {entropy:.6f}",
        f"aleatoric: {aleatoric:.6f}",
        f"epistemic: {epistemic:.6f}",
        "",
        "结构化证据",
        "--------------------------------",
        json.dumps(evidence_single[0], ensure_ascii=False, indent=2),
        "",
        "证据链（贡献：p1+p2 & p2）",
        "--------------------------------",
        json.dumps(evidence_chain, ensure_ascii=False, indent=2),
        "",
        "溯因分析（demo）",
        "--------------------------------",
        json.dumps(recourse, ensure_ascii=False, indent=2),
        "",
        "训练元信息（train_meta）",
        "--------------------------------",
        json.dumps(extra.get("train_meta", {}), ensure_ascii=False, indent=2),
        "",
    ]

    _ensure_dir_for_file(args.eval_info_path)
    with open(args.eval_info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Success] Eval finished. Written to: {args.eval_info_path}")


if __name__ == "__main__":
    main()
