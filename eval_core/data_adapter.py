# === data_adapter.py ===
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# 1) 指标状态字符串 -> 严重度(0/1/2) + 二值异常(0/1)
# -----------------------------

_NORMAL_WORDS = {"正常", "未见异常", "合格", "阴性", "无异常", "正常范围"}
_MILD_WORDS = {"偏低", "偏高", "轻度异常", "略低", "略高", "轻度"}
_SEVERE_WORDS = {"过低", "过高", "异常", "严重", "极低", "极高", "明显异常", "重度"}

def map_status_to_severity(x: Any) -> Optional[int]:
    """
    将 indicator_analysis 的元素映射为严重度:
      0=正常, 1=轻度, 2=重度
    返回 None 表示缺失/无法解析（相当于 -1）
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # 若已有数字编码：约定 0=正常, 1=轻度, 2=重度, -1=缺失
        xi = int(x)
        if xi < 0:
            return None
        if xi >= 2:
            return 2
        return xi
    s = str(x).strip()
    if not s or s == "-1":
        return None
    if s in _NORMAL_WORDS:
        return 0
    if s in _MILD_WORDS:
        return 1
    if s in _SEVERE_WORDS:
        return 2
    # 兜底：包含关键词匹配
    for w in _SEVERE_WORDS:
        if w in s:
            return 2
    for w in _MILD_WORDS:
        if w in s:
            return 1
    for w in _NORMAL_WORDS:
        if w in s:
            return 0
    return None

def severity_to_abnormal_binary(sev: Optional[int]) -> Optional[int]:
    if sev is None:
        return None
    return 0 if sev == 0 else 1


# -----------------------------
# 2) 三分类弱标签聚合：0/1/2
# -----------------------------

def build_label_3class(indicator_analysis: List[Any]) -> Tuple[int, Dict[str, Any]]:
    """
    三分类弱标签规则（可论文写法）：
      - y=0: 没有异常项
      - y=2: max_sev==2 且 (cnt_severe>=2 or cnt_bad>=3)
      - y=1: 其它异常情况
    """
    sevs = [map_status_to_severity(x) for x in (indicator_analysis or [])]
    sevs_valid = [s for s in sevs if s is not None]
    if not sevs_valid:
        # 没信息，默认轻度/或正常？这里更保守：轻度（避免把缺失当正常）
        return 1, {"max_sev": None, "cnt_bad": 0, "cnt_severe": 0, "missing": True}

    max_sev = int(max(sevs_valid))
    cnt_bad = int(sum(1 for s in sevs_valid if s >= 1))
    cnt_severe = int(sum(1 for s in sevs_valid if s == 2))

    if cnt_bad == 0:
        return 0, {"max_sev": max_sev, "cnt_bad": cnt_bad, "cnt_severe": cnt_severe, "missing": False}

    if (max_sev == 2) and (cnt_severe >= 2 or cnt_bad >= 3):
        return 2, {"max_sev": max_sev, "cnt_bad": cnt_bad, "cnt_severe": cnt_severe, "missing": False}

    return 1, {"max_sev": max_sev, "cnt_bad": cnt_bad, "cnt_severe": cnt_severe, "missing": False}


# -----------------------------
# 3) 证据抽取（用于可解释输出）
# -----------------------------

def build_evidence_from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    indicator_analysis = info.get("indicator_analysis", []) or []
    character_analysis = info.get("character_analysis", []) or []
    report_original_data = info.get("report_original_data", info.get("report_data", [])) or []

    abnormal_items = []
    for i, st in enumerate(indicator_analysis):
        sev = map_status_to_severity(st)
        if sev is None or sev == 0:
            continue
        val = report_original_data[i] if i < len(report_original_data) else None
        txt = character_analysis[i] if i < len(character_analysis) else None
        abnormal_items.append({
            "index": i,
            "status": st,
            "severity": int(sev),
            "value": val,
            "explain": txt,
        })

    return {"abnormal_items": abnormal_items}


# -----------------------------
# 4) 从 processed JSON 构建 dataset（三分类）
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_infos(processed: Dict[str, Any]) -> List[Dict[str, Any]]:
    infos = processed.get("info", [])
    return infos if isinstance(infos, list) else []

def extract_base_features_from_first_info(info0: Dict[str, Any]) -> List[float]:
    # report_type/sex/age/period_info/preg_info: 简单数值化，缺失填 -1
    def _to_float(x: Any) -> float:
        try:
            if x is None:
                return -1.0
            if isinstance(x, (int, float)):
                return float(x)
            # 字符串数字
            return float(str(x).strip())
        except Exception:
            return -1.0

    return [
        _to_float(info0.get("report_type", -1)),
        _to_float(info0.get("sex", -1)),
        _to_float(info0.get("age", -1)),
        _to_float(info0.get("period_info", -1)),
        _to_float(info0.get("preg_info", -1)),
    ]

def extract_report_data_features(info0: Dict[str, Any], use_missing_mask: bool = True) -> List[float]:
    report_data = info0.get("report_data", []) or []
    x = []
    mask = []
    for v in report_data:
        vv = float(v) if isinstance(v, (int, float)) else -1.0
        is_missing = (vv == -1.0)
        x.append(0.0 if is_missing else vv)
        mask.append(1.0 if is_missing else 0.0)
    return x + (mask if use_missing_mask else [])

def build_xy_meta_evidence_from_processed(
    processed: Dict[str, Any],
    use_missing_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    约定：
      - 一个 processed_report_info.json 对应一个样本（task_id）
      - 特征来自 info[0]（你示例就是这样）
      - 标签来自 info[0].indicator_analysis（字符串 -> 三分类）
    """
    infos = get_infos(processed)
    if not infos:
        raise ValueError("输入 JSON 缺少 info 字段或为空")

    info0 = infos[0]
    feat_report = extract_report_data_features(info0, use_missing_mask=use_missing_mask)
    feat_base = extract_base_features_from_first_info(info0)
    x = np.array(feat_report + feat_base, dtype=float).reshape(1, -1)

    y, y_stats = build_label_3class(info0.get("indicator_analysis", []) or [])
    y_arr = np.array([y], dtype=int)

    meta = [{
        "task_id": info0.get("task_id"),
        "user_name": info0.get("user_name"),
        "report_time": info0.get("report_time"),
        "label_stats": y_stats,
    }]

    evidence = [build_evidence_from_info(info0)]

    # feature names
    rd_len = len(info0.get("report_data", []) or [])
    feat_names = []
    for i in range(rd_len):
        feat_names.append(f"rd_{i}")
    if use_missing_mask:
        for i in range(rd_len):
            feat_names.append(f"rd_missing_{i}")
    feat_names += ["report_type", "sex", "age", "period_info", "preg_info"]

    return x, y_arr, meta, evidence, feat_names

def build_dataset_from_dataset_field(
    processed: Dict[str, Any],
    use_missing_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    如果 processed 里存在 dataset: [sample1, sample2,...]，则将每个 sample 视为一个 processed 结构。
    每个 sample 都要求含 info[0]。
    """
    ds = processed.get("dataset", None)
    if not isinstance(ds, list) or not ds:
        raise ValueError("dataset 字段不存在或为空")

    X_list, y_list, meta_list, ev_list = [], [], [], []
    feat_names_final: Optional[List[str]] = None

    for sample in ds:
        Xi, yi, mi, evi, fn = build_xy_meta_evidence_from_processed(sample, use_missing_mask=use_missing_mask)
        X_list.append(Xi[0])
        y_list.append(int(yi[0]))
        meta_list.extend(mi)
        ev_list.extend(evi)
        if feat_names_final is None:
            feat_names_final = fn

    X = np.vstack([np.asarray(r).reshape(1, -1) for r in X_list])
    y = np.asarray(y_list, dtype=int).reshape(-1)

    return X, y, meta_list, ev_list, (feat_names_final or [])
