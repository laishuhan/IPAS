# preprocess.py
# -*- coding: utf-8 -*-
"""
preprocess.py
-------------
职责：
1) 原始报告 JSON -> 固定维度张量 x32 (torch.Tensor[1,32])
2) 兼容 OCR/清洗后字段缺失的健壮处理
3) 提供少量数据 I/O 小工具（读取 jsonl / json）

特征协议（默认）：
[ sex(1), age_norm(1), status_scores(15), raw_values(15) ] => 32维

说明：
- sex: float(info0.get('sex', -1))
- age_norm: age/100, age<=0 则 -1
- status_scores: indicator_analysis 中的状态词映射到 {0,1,2}，未知为 -1
- raw_values: report_data 中的数值，若为 list 则取均值；不可转为 float 则 -1
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# -----------------------------
# 1) 状态词 -> 严重度分数 (0/1/2)
# -----------------------------
_SEVERITY_MAP = {
    # 正常/阴性
    "正常": 0, "阴性": 0,
    # 轻度异常
    "偏低": 1, "偏高": 1, "偏小": 1, "偏大": 1, "轻度": 1,
    # 重度异常
    "过低": 2, "过高": 2, "过小": 2, "过大": 2,
    "阳性": 2, "明显": 2, "异常": 2,
}


def map_word_to_score(word: Any) -> float:
    """
    将 OCR/解析得到的状态词映射为严重度分数 (0/1/2)。
    - 非字符串或无法解析 => -1.0
    - 支持包含匹配（关键词包含）
    """
    if not isinstance(word, str):
        return -1.0
    w = word.strip()
    if w == "" or w == "-1":
        return -1.0
    if w in _SEVERITY_MAP:
        return float(_SEVERITY_MAP[w])
    for k, v in _SEVERITY_MAP.items():
        if k in w:
            return float(v)
    return -1.0


# -----------------------------
# 2) 数值安全转换
# -----------------------------
def _safe_float(x: Any, default: float = -1.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _mean_list(x: Any, default: float = -1.0) -> float:
    """
    report_data 中有时会出现 list（如卵泡数据 [10.2, 26.4]），取均值。
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return default
        vals = [_safe_float(v, default=np.nan) for v in x]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) == 0:
            return default
        return float(np.mean(vals))
    return _safe_float(x, default=default)


def _pad_or_truncate(values: List[float], length: int, pad_value: float = -1.0) -> List[float]:
    out = list(values[:length])
    if len(out) < length:
        out.extend([pad_value] * (length - len(out)))
    return out


# -----------------------------
# 3) 核心：JSON -> x32
# -----------------------------
def preprocess_raw_sample(
    item: Dict[str, Any],
    max_indicators: int = 15,
    max_values: int = 15,
) -> torch.Tensor:
    """
    输入: 单条报告 JSON dict
    输出: torch.Tensor shape=(1, 32) float32

    向量结构：
      [ sex(1), age_norm(1), status_scores(15), raw_values(15) ]
    """
    item = item or {}
    info_list = item.get("info", [])
    info0 = info_list[0] if isinstance(info_list, list) and len(info_list) > 0 and isinstance(info_list[0], dict) else {}

    # A) demographics
    sex = _safe_float(info0.get("sex", -1), default=-1.0)
    age_raw = _safe_float(info0.get("age", -1), default=-1.0)
    age_norm = (age_raw / 100.0) if age_raw > 0 else -1.0

    # B) status words -> scores
    status_list = info0.get("indicator_analysis", []) or []
    if not isinstance(status_list, list):
        status_list = []
    status_scores = [map_word_to_score(s) for s in status_list]
    status_scores = _pad_or_truncate(status_scores, max_indicators, pad_value=-1.0)

    # C) report_data -> numeric values
    raw_list = info0.get("report_data", []) or []
    if not isinstance(raw_list, list):
        raw_list = []
    raw_values = [_mean_list(v, default=-1.0) for v in raw_list]
    raw_values = _pad_or_truncate(raw_values, max_values, pad_value=-1.0)

    final_vec = [sex, age_norm] + status_scores + raw_values
    x = torch.tensor(final_vec, dtype=torch.float32).unsqueeze(0)  # (1,32)
    return x


# -----------------------------
# 4) 常用字段提取（路由用）
# -----------------------------
def get_report_type(item: Dict[str, Any], default: int = -1) -> int:
    item = item or {}
    info_list = item.get("info", [])
    if not isinstance(info_list, list) or len(info_list) == 0 or not isinstance(info_list[0], dict):
        return int(default)
    return int(_safe_float(info_list[0].get("report_type", default), default=float(default)))


# -----------------------------
# 5) I/O Helpers（train.py 会用到）
# -----------------------------
def read_json(path: str, encoding: str = "utf-8") -> Any:
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def read_jsonl(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(text)
