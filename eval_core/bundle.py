# bundle.py
# -*- coding: utf-8 -*-
"""
bundle.py
---------
职责：
1) save_bundle / load_bundle：统一模型包格式（encoder + experts heads + routing）
2) routing：推理时根据 report_type 选择 expert（type -> cluster -> global）
3) cluster_map：训练阶段可写入，推理阶段读取

Bundle 目录结构（建议）：
model_root/
  encoder/
    encoder.pth
    encoder_meta.json
  global/
    head.npz
    calibrator.json
    conformal.json
    meta.json
  type_12/ (optional)
    head.npz
    calibrator.json
    conformal.json
    meta.json
  cluster_3/ (optional)
    head.npz
    calibrator.json
    conformal.json
    meta.json
  cluster_map.json (optional)
  routing.json (optional)

本文件不做“训练数学”，只做序列化与选择逻辑。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from encoder import ReportEncoder, load_encoder
from risk_model import BayesianSoftmax, TemperatureCalibrator, ConformalAPS


# =========================================================
# 1) Small IO helpers
# =========================================================

def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _npz_save(path: str, **arrays: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    np.savez(path, **arrays)

def _npz_load(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# =========================================================
# 2) Expert model container
# =========================================================

@dataclass
class ExpertBundle:
    head: BayesianSoftmax
    calibrator: TemperatureCalibrator
    conformal: ConformalAPS
    meta: Dict[str, Any]
    kind: str           # "global" | "type" | "cluster"
    key: str            # "global" | "type_{rt}" | "cluster_{cid}"


@dataclass
class LoadedBundle:
    encoder: ReportEncoder
    encoder_meta: Dict[str, Any]
    experts: Dict[str, ExpertBundle]  # key -> ExpertBundle
    cluster_map: Dict[str, int]       # report_type(str) -> cluster_id(int)
    routing: Dict[str, Any]           # policy / thresholds etc.


# =========================================================
# 3) Save / Load Encoder
# =========================================================

def save_encoder_bundle(model_root: str, encoder_path_src: str, encoder_meta: Dict[str, Any]) -> str:
    """
    将 encoder.pth 放入 model_root/encoder/ 下，并写 meta。
    返回 encoder 目录路径。
    """
    enc_dir = os.path.join(model_root, "encoder")
    _ensure_dir(enc_dir)

    # copy weight file content by loading & saving to guarantee location
    import torch
    sd = torch.load(encoder_path_src, map_location="cpu")
    out_pth = os.path.join(enc_dir, "encoder.pth")
    torch.save(sd, out_pth)

    _write_json(os.path.join(enc_dir, "encoder_meta.json"), encoder_meta)
    return enc_dir


def load_encoder_bundle(model_root: str) -> Tuple[ReportEncoder, Dict[str, Any]]:
    enc_dir = os.path.join(model_root, "encoder")
    meta_path = os.path.join(enc_dir, "encoder_meta.json")
    pth_path = os.path.join(enc_dir, "encoder.pth")

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"encoder not found: {pth_path}")

    meta = _read_json(meta_path) if os.path.exists(meta_path) else {}
    input_dim = int(meta.get("input_dim", 32))
    latent_dim = int(meta.get("latent_dim", 128))
    output_dim = int(meta.get("output_dim", 64))

    enc = load_encoder(pth_path, input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim)
    return enc, meta


# =========================================================
# 4) Save / Load Expert
# =========================================================

def save_expert_bundle(
    expert_dir: str,
    head: BayesianSoftmax,
    calibrator: TemperatureCalibrator,
    conformal: ConformalAPS,
    meta: Dict[str, Any],
) -> None:
    """
    保存一个 expert（head + calibrator + conformal + meta）。
    """
    _ensure_dir(expert_dir)

    # head
    if head.W_map is None:
        raise ValueError("head.W_map is None, cannot save")
    W_map = head.W_map
    Sigma = head.Sigma if head.Sigma is not None else np.zeros((1, 1), dtype=float)

    _npz_save(
        os.path.join(expert_dir, "head.npz"),
        W_map=W_map,
        Sigma=Sigma,
        feature_dim=int(W_map.shape[0]),
        n_classes=int(W_map.shape[1]),
        lambda_reg=float(getattr(head, "lambda_reg", 1.0)),
    )

    # calibrator / conformal
    _write_json(os.path.join(expert_dir, "calibrator.json"), {"method": "temperature", "T": float(calibrator.T)})
    _write_json(
        os.path.join(expert_dir, "conformal.json"),
        {"method": "APS", "alpha": float(conformal.alpha), "qhat": float(conformal.qhat) if conformal.qhat is not None else None},
    )

    # meta
    _write_json(os.path.join(expert_dir, "meta.json"), meta)


def load_expert_bundle(expert_dir: str, kind: str, key: str) -> ExpertBundle:
    head_npz = os.path.join(expert_dir, "head.npz")
    cal_path = os.path.join(expert_dir, "calibrator.json")
    conf_path = os.path.join(expert_dir, "conformal.json")
    meta_path = os.path.join(expert_dir, "meta.json")

    if not os.path.exists(head_npz):
        raise FileNotFoundError(f"Missing head.npz in {expert_dir}")

    d = _npz_load(head_npz)
    W_map = np.asarray(d["W_map"], dtype=float)
    Sigma = np.asarray(d["Sigma"], dtype=float)

    n_classes = int(d.get("n_classes", W_map.shape[1]))
    lambda_reg = float(d.get("lambda_reg", 1.0))

    head = BayesianSoftmax(n_classes=n_classes, lambda_reg=lambda_reg)
    head.W_map = W_map
    head.Sigma = Sigma if Sigma.size > 1 else None

    cal_j = _read_json(cal_path) if os.path.exists(cal_path) else {"T": 1.0}
    calibrator = TemperatureCalibrator(T=float(cal_j.get("T", 1.0)))

    conf_j = _read_json(conf_path) if os.path.exists(conf_path) else {"alpha": 0.1, "qhat": None}
    conformal = ConformalAPS(alpha=float(conf_j.get("alpha", 0.1)))
    conformal.qhat = conf_j.get("qhat", None)
    if conformal.qhat is not None:
        conformal.qhat = float(conformal.qhat)

    meta = _read_json(meta_path) if os.path.exists(meta_path) else {}

    return ExpertBundle(
        head=head,
        calibrator=calibrator,
        conformal=conformal,
        meta=meta,
        kind=str(kind),
        key=str(key),
    )


# =========================================================
# 5) Cluster map & routing policy
# =========================================================

def save_cluster_map(model_root: str, cluster_map: Dict[str, int], K: int, extra: Optional[Dict[str, Any]] = None) -> str:
    path = os.path.join(model_root, "cluster_map.json")
    obj = {"K": int(K), "cluster_map": {str(k): int(v) for k, v in cluster_map.items()}}
    if extra:
        obj.update(extra)
    _write_json(path, obj)
    return path


def load_cluster_map(model_root: str) -> Dict[str, int]:
    path = os.path.join(model_root, "cluster_map.json")
    if not os.path.exists(path):
        return {}
    obj = _read_json(path) or {}
    cm = obj.get("cluster_map", {}) or {}
    return {str(k): int(v) for k, v in cm.items()}


def save_routing_policy(model_root: str, routing: Dict[str, Any]) -> str:
    path = os.path.join(model_root, "routing.json")
    _write_json(path, routing)
    return path


def load_routing_policy(model_root: str) -> Dict[str, Any]:
    path = os.path.join(model_root, "routing.json")
    if not os.path.exists(path):
        # default policy: type -> cluster -> global
        return {
            "priority": ["type", "cluster", "global"],
            "enable_type": True,
            "enable_cluster": True,
        }
    obj = _read_json(path)
    return obj if isinstance(obj, dict) else {}


# =========================================================
# 6) Save / Load whole bundle
# =========================================================

def save_bundle(
    model_root: str,
    encoder_path_src: str,
    encoder_meta: Dict[str, Any],
    experts: Dict[str, Tuple[BayesianSoftmax, TemperatureCalibrator, ConformalAPS, Dict[str, Any]]],
    cluster_map: Optional[Dict[str, int]] = None,
    routing: Optional[Dict[str, Any]] = None,
) -> None:
    """
    experts: dict key -> (head, calibrator, conformal, meta)
      key examples: "global", "type_12", "cluster_3"
    """
    _ensure_dir(model_root)

    save_encoder_bundle(model_root, encoder_path_src=encoder_path_src, encoder_meta=encoder_meta)

    # save experts
    for key, (head, cal, conf, meta) in experts.items():
        expert_dir = os.path.join(model_root, key)
        save_expert_bundle(expert_dir, head, cal, conf, meta)

    # cluster map / routing
    if cluster_map is not None:
        # infer K if possible
        K = 1
        if len(cluster_map) > 0:
            K = max(int(v) for v in cluster_map.values()) + 1
        save_cluster_map(model_root, cluster_map=cluster_map, K=K)

    if routing is not None:
        save_routing_policy(model_root, routing=routing)


def load_bundle(model_root: str) -> LoadedBundle:
    encoder, enc_meta = load_encoder_bundle(model_root)

    cluster_map = load_cluster_map(model_root)
    routing = load_routing_policy(model_root)

    # scan experts dirs
    experts: Dict[str, ExpertBundle] = {}
    for name in os.listdir(model_root):
        p = os.path.join(model_root, name)
        if not os.path.isdir(p):
            continue
        if name == "encoder":
            continue
        if not os.path.exists(os.path.join(p, "head.npz")):
            continue

        if name == "global":
            kind = "global"
        elif name.startswith("type_"):
            kind = "type"
        elif name.startswith("cluster_"):
            kind = "cluster"
        else:
            # allow arbitrary expert names, treat as global-like
            kind = "custom"

        experts[name] = load_expert_bundle(p, kind=kind, key=name)

    if "global" not in experts:
        raise FileNotFoundError("Bundle missing required expert: global/ (with head.npz)")

    return LoadedBundle(
        encoder=encoder,
        encoder_meta=enc_meta,
        experts=experts,
        cluster_map=cluster_map,
        routing=routing,
    )


# =========================================================
# 7) Routing selection (type -> cluster -> global)
# =========================================================

def pick_expert_key(report_type: int, bundle: LoadedBundle) -> str:
    """
    根据 routing 策略选择 expert key。
    默认优先级：type -> cluster -> global
    """
    rt = int(report_type)

    prio = bundle.routing.get("priority", ["type", "cluster", "global"])
    enable_type = bool(bundle.routing.get("enable_type", True))
    enable_cluster = bool(bundle.routing.get("enable_cluster", True))

    for stage in prio:
        if stage == "type" and enable_type:
            k = f"type_{rt}"
            if k in bundle.experts:
                return k

        if stage == "cluster" and enable_cluster:
            # if cluster expert exists & map contains this type
            cid = bundle.cluster_map.get(str(rt), None)
            if cid is not None:
                k = f"cluster_{int(cid)}"
                if k in bundle.experts:
                    return k

        if stage == "global":
            return "global"

    return "global"  # safe fallback
