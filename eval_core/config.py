# config.py
# -*- coding: utf-8 -*-
"""
config.py
---------
职责：
1) 提供统一默认配置（训练 / 推理）
2) 支持从 JSON 文件加载并覆盖默认配置
3) 提供简单的 dict merge 逻辑

使用方式示例：

from config import get_default_config, load_config

cfg = get_default_config()
cfg = load_config("my_config.json", base_cfg=cfg)

然后在 train.py / main_eval.py 中使用 cfg["ssl"]["epochs"] 等。
"""

from __future__ import annotations

import json
import copy
from typing import Any, Dict


# =========================================================
# 1) Default Config
# =========================================================

def get_default_config() -> Dict[str, Any]:
    """
    返回完整默认配置。
    可在 train.py 中打印出来作为实验记录。
    """
    return {

        # -----------------------------
        # 全局
        # -----------------------------
        "global": {
            "seed": 42,
            "alpha": 0.1,          # conformal alpha
            "lambda_reg": 1.0,     # L2 for Bayesian head
        },

        # -----------------------------
        # SSL Encoder
        # -----------------------------
        "ssl": {
            "enable": True,
            "epochs": 40,
            "batch_size": 256,
            "lr": 1e-3,
            "temperature": 0.1,
            "mask_prob": 0.15,
            "jitter_std": 0.02,
            "latent_dim": 128,
            "proj_dim": 64,
        },

        # -----------------------------
        # Weak Supervision
        # -----------------------------
        "weak_label": {
            "max_iter": 120,
            "tol": 1e-6,
        },

        # -----------------------------
        # Experts
        # -----------------------------
        "experts": {
            "min_type_samples": 300,
            "enable_cluster": False,
            "cluster_k": 8,
            "min_cluster_samples": 500,
            "kmeans_restarts": 5,
            "kmeans_iters": 50,
        },

        # -----------------------------
        # Risk Control (inference)
        # -----------------------------
        "risk": {
            "tau_risk": 0.3,
            "tau_epi": 0.5,
        },

        # -----------------------------
        # Explain
        # -----------------------------
        "explain": {
            "enable": False,
            "top_k": 8,
        },
    }


# =========================================================
# 2) Recursive Dict Merge
# =========================================================

def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归更新字典：
    - 如果 value 是 dict，则递归 merge
    - 否则直接覆盖
    """
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =========================================================
# 3) Load Config From JSON
# =========================================================

def load_config(path: str, base_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    从 JSON 文件加载配置并覆盖默认值。

    参数：
      path: json 文件路径
      base_cfg: 默认配置（若为 None 则使用 get_default_config()）

    返回：
      合并后的配置 dict
    """
    if base_cfg is None:
        base_cfg = get_default_config()
    else:
        base_cfg = copy.deepcopy(base_cfg)

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    if not isinstance(user_cfg, dict):
        raise ValueError("Config file must contain a JSON object (dict).")

    merged = _deep_update(base_cfg, user_cfg)
    return merged


# =========================================================
# 4) Save Config Snapshot
# =========================================================

def save_config_snapshot(cfg: Dict[str, Any], path: str) -> None:
    """
    保存当前运行配置到文件（方便复现实验）。
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
