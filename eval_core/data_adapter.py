# === data_adapter.py ===
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# 1) 指标状态字符串 -> 严重度(0/1/2) + 二值异常(0/1)
# -----------------------------

_NORMAL_WORDS = {"正常", "阴性"}
_MILD_WORDS = {"偏低", "偏高", "偏小", "偏大"}
_SEVERE_WORDS = {"过低", "过高", "过小", "过大", "阳性"}

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








