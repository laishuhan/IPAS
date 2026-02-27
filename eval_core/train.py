# train.py
# -*- coding: utf-8 -*-
"""
train.py  (ENTRYPOINT, config-driven)
-------------------------------------
融合版训练入口：Project2(SSL Encoder) + Project1(WeakLabel + Bayesian Head + Calib + Conformal + Routing)

所有超参与开关由 config.py 驱动：
- cfg["global"]
- cfg["ssl"]
- cfg["weak_label"]
- cfg["experts"]

命令行只负责：
- --config_path: 指定 JSON 配置文件（可选）
- --train_data_path / --model_root: 方便你运行时覆盖路径（可选）
- --override: 简单 key=value 覆盖（可选，小功能）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from config import get_default_config, load_config, save_config_snapshot

from preprocess import read_jsonl, get_report_type, preprocess_raw_sample
from encoder import (
    SSLAugmentConfig,
    SSLTrainConfig,
    train_ssl_encoder,
    load_encoder,
    encode,
)
from weaklabel import make_soft_labels
from risk_model import train_bayes_head_pipeline
from bundle import save_bundle


# =========================================================
# Utils
# =========================================================

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_by_report_type(dataset: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for i, s in enumerate(dataset):
        rt = int(get_report_type(s, default=-1))
        groups.setdefault(rt, []).append(i)
    return groups

def zscore_standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd

def kmeans(X: np.ndarray, K: int, seed: int = 42, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(int(seed))
    K = int(max(1, min(K, n)))
    idx = rng.choice(n, size=K, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(int(iters)):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.all(new_labels == labels):
            labels = new_labels
            break
        labels = new_labels
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n)]
    return labels, centers

def kmeans_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    diff = X - centers[labels]
    return float(np.sum(diff * diff))

def kmeans_best_of_restarts(X: np.ndarray, K: int, base_seed: int = 42, restarts: int = 5, iters: int = 50) -> Tuple[np.ndarray, np.ndarray, float, int]:
    best_labels = None
    best_centers = None
    best_sse = None
    best_seed = None
    for r in range(int(restarts)):
        seed = int(base_seed + 9973 * r)
        labels, centers = kmeans(X, K=K, seed=seed, iters=iters)
        sse = kmeans_sse(X, labels, centers)
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_labels = labels
            best_centers = centers
            best_seed = seed
    return best_labels, best_centers, float(best_sse), int(best_seed)

def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """
    支持 --override a.b.c=value
    value 支持: int/float/bool/str
    """
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()

        # parse primitive
        vv: Any
        if v.lower() in ("true", "false"):
            vv = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    vv = float(v)
                    if vv.is_integer():
                        # keep floats like 1.0 if user typed; but usually int is fine
                        pass
                else:
                    vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v

        # set nested key
        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = vv
    return out

def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =========================================================
# Main
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config_path", type=str, default="", help="可选：JSON 配置文件路径")
    ap.add_argument("--train_data_path", type=str, default="", help="可选：覆盖 cfg 中的训练数据路径")
    ap.add_argument("--model_root", type=str, default="", help="可选：覆盖 cfg 中的 model_root")

    ap.add_argument("--override", type=str, nargs="*", default=[],
                    help="可选：覆盖配置项，如 ssl.enable=false experts.enable_cluster=true")

    # optional domain-shift knob
    ap.add_argument("--holdout_type", type=int, default=-1, help=">=0 时：该 report_type 完全不参与训练（覆盖 cfg）")
    args = ap.parse_args()

    # 0) config
    cfg = get_default_config()
    if args.config_path:
        cfg = load_config(args.config_path, base_cfg=cfg)

    if args.override:
        ov = parse_overrides(args.override)
        cfg = deep_update(cfg, ov)

    # path overrides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    default_train_data_path = os.path.join(script_dir, "dataset", "eval_train_dataset.jsonl")
    train_data_path = args.train_data_path or default_train_data_path

    
    default_model_root = os.path.join(script_dir, "model")
    model_root = args.model_root or default_model_root
    cfg.setdefault("global", {})
    cfg["global"]["model_root"] = model_root
    # ✅ ALWAYS CLEAN model_root before training
    if os.path.exists(model_root):
        print(f"[Clean] Removing existing model_root: {model_root}")
        shutil.rmtree(model_root)

    ensure_dir(model_root)

    cfg.setdefault("paths", {})
    cfg["paths"]["train_data_path"] = train_data_path

    if int(args.holdout_type) >= 0:
        cfg.setdefault("experiments", {})
        cfg["experiments"]["holdout_type"] = int(args.holdout_type)

    # snapshot config used in this run
    save_config_snapshot(cfg, os.path.join(model_root, "run_config.json"))

    seed = int(cfg["global"]["seed"])
    alpha = float(cfg["global"]["alpha"])
    lambda_reg = float(cfg["global"]["lambda_reg"])

    # 1) load dataset
    print(f"[Load] {train_data_path}")
    dataset = read_jsonl(train_data_path)
    if not isinstance(dataset, list) or len(dataset) < 10:
        raise ValueError(f"dataset too small: n={len(dataset)}")

    # holdout type (if enabled)
    holdout_type = int(cfg.get("experiments", {}).get("holdout_type", -1))
    if holdout_type >= 0:
        dataset = [s for s in dataset if int(get_report_type(s, -1)) != holdout_type]
        print(f"[DomainShift] holdout report_type={holdout_type}, remaining n={len(dataset)}")

    # 2) encoder: train or load existing
    encoder_tmp_path = os.path.join(model_root, "_tmp_encoder.pth")
    encoder_meta_path = os.path.join(model_root, "_tmp_encoder_meta.json")

    ssl_enable = bool(cfg["ssl"]["enable"])
    latent_dim = int(cfg["ssl"]["latent_dim"])
    proj_dim = int(cfg["ssl"]["proj_dim"])

    if ssl_enable:
        print("[SSL] Training encoder ...")
        aug_cfg = SSLAugmentConfig(
            mask_prob=float(cfg["ssl"]["mask_prob"]),
            jitter_std=float(cfg["ssl"]["jitter_std"]),
            jitter_positive_only=True,
        )
        train_cfg = SSLTrainConfig(
            epochs=int(cfg["ssl"]["epochs"]),
            batch_size=int(cfg["ssl"]["batch_size"]),
            lr=float(cfg["ssl"]["lr"]),
            temperature=float(cfg["ssl"]["temperature"]),
            device="auto",
            seed=seed,
            latent_dim=latent_dim,
            output_dim=proj_dim,
            input_dim=32,
        )
        enc_meta = train_ssl_encoder(
            jsonl_path=train_data_path,
            out_encoder_path=encoder_tmp_path,
            out_meta_path=encoder_meta_path,
            augment_cfg=aug_cfg,
            train_cfg=train_cfg,
        )
        encoder = load_encoder(encoder_tmp_path, input_dim=32, latent_dim=latent_dim, output_dim=proj_dim)
        encoder_meta = enc_meta
    else:
        # load from existing bundle
        bundle_enc_path = os.path.join(model_root, "encoder", "encoder.pth")
        bundle_enc_meta = os.path.join(model_root, "encoder", "encoder_meta.json")
        if not os.path.exists(bundle_enc_path):
            raise FileNotFoundError(
                "cfg['ssl']['enable']=false but no existing encoder found at "
                f"{bundle_enc_path}. Either enable ssl training or provide a trained bundle."
            )
        if os.path.exists(bundle_enc_meta):
            with open(bundle_enc_meta, "r", encoding="utf-8") as f:
                encoder_meta = json.load(f)
        else:
            encoder_meta = {"input_dim": 32, "latent_dim": latent_dim, "output_dim": proj_dim}

        encoder = load_encoder(
            bundle_enc_path,
            input_dim=int(encoder_meta.get("input_dim", 32)),
            latent_dim=int(encoder_meta.get("latent_dim", latent_dim)),
            output_dim=int(encoder_meta.get("output_dim", proj_dim)),
        )

        # copy weights to tmp path for save_bundle()
        import torch
        sd = torch.load(bundle_enc_path, map_location="cpu")
        torch.save(sd, encoder_tmp_path)

    # 3) weak labels
    print("[WeakLabel] EM label model ...")
    wl_max_iter = int(cfg["weak_label"]["max_iter"])
    wl_tol = float(cfg["weak_label"]["tol"])
    Y_soft, y_hard, weak_meta = make_soft_labels(dataset, seed=seed, max_iter=wl_max_iter, tol=wl_tol)
    print("[WeakLabel] y_hard distribution:", weak_meta.get("label_distribution_hard"))

    # 4) encode -> Z
    print("[Encode] Computing embeddings ...")
    Z_list: List[np.ndarray] = []
    for s in dataset:
        x = preprocess_raw_sample(s)   # (1,32)
        z = encode(encoder, x)         # (1,d)
        Z_list.append(z[0])
    Z = np.vstack(Z_list).astype(float)
    print(f"[Encode] Z shape = {Z.shape}")

    # 5) train global head
    print("[Train] global head ...")
    head_g, cal_g, conf_g, meta_g = train_bayes_head_pipeline(
        Z=Z,
        Y_soft=Y_soft,
        y_hard=y_hard,
        seed=seed,
        alpha=alpha,
        lambda_reg=lambda_reg,
    )
    meta_g["weak_label_model"] = weak_meta
    meta_g["expert_kind"] = "global"

    experts_to_save: Dict[str, Tuple[Any, Any, Any, Dict[str, Any]]] = {
        "global": (head_g, cal_g, conf_g, meta_g)
    }

    # 6) type experts
    min_type_samples = int(cfg["experts"]["min_type_samples"])
    groups = split_by_report_type(dataset)
    trained_types: List[int] = []

    for rt, idxs in groups.items():
        if int(rt) < 0:
            continue
        if len(idxs) < min_type_samples:
            continue

        Z_rt = Z[idxs]
        Y_rt = Y_soft[idxs]
        y_rt = y_hard[idxs]

        head_t, cal_t, conf_t, meta_t = train_bayes_head_pipeline(
            Z=Z_rt,
            Y_soft=Y_rt,
            y_hard=y_rt,
            seed=seed,
            alpha=alpha,
            lambda_reg=lambda_reg,
        )
        meta_t["weak_label_model"] = weak_meta
        meta_t["expert_kind"] = "type"
        meta_t["report_type"] = int(rt)
        experts_to_save[f"type_{int(rt)}"] = (head_t, cal_t, conf_t, meta_t)
        trained_types.append(int(rt))
        print(f"[Train] type_{rt} (n={len(idxs)})")

    # 7) cluster experts (optional)
    enable_cluster = bool(cfg["experts"]["enable_cluster"])
    cluster_map: Dict[str, int] = {}

    if enable_cluster:
        cluster_k = int(cfg["experts"]["cluster_k"])
        min_cluster_samples = int(cfg["experts"]["min_cluster_samples"])
        kmeans_restarts = int(cfg["experts"]["kmeans_restarts"])
        kmeans_iters = int(cfg["experts"]["kmeans_iters"])

        tail_types = sorted([rt for rt in groups.keys() if int(rt) >= 0 and int(rt) not in set(trained_types)])
        if len(tail_types) == 0:
            print("[Cluster] No tail types to cluster.")
        else:
            # prototype = mean embedding per type
            protos = []
            rts = []
            for rt in tail_types:
                idxs = groups[rt]
                if len(idxs) == 0:
                    continue
                protos.append(Z[idxs].mean(axis=0))
                rts.append(int(rt))
            protos = np.asarray(protos, dtype=float)
            protos_z, mu, sd = zscore_standardize(protos)

            K = int(min(cluster_k, len(rts)))
            labels, centers, best_sse, best_seed = kmeans_best_of_restarts(
                protos_z, K=K, base_seed=seed, restarts=kmeans_restarts, iters=kmeans_iters
            )
            cluster_map = {str(rt): int(labels[i]) for i, rt in enumerate(rts)}
            print(f"[Cluster] Built cluster_map K={K} best_sse={best_sse:.4f} seed={best_seed}")

            for cid in range(K):
                member_types = [rts[i] for i in range(len(rts)) if int(labels[i]) == int(cid)]
                if len(member_types) == 0:
                    continue
                idx_c: List[int] = []
                for rt in member_types:
                    idx_c.extend(groups[int(rt)])
                if len(idx_c) < min_cluster_samples:
                    print(f"[Cluster] Skip cluster_{cid}: n={len(idx_c)} < {min_cluster_samples}")
                    continue

                Z_c = Z[idx_c]
                Y_c = Y_soft[idx_c]
                y_c = y_hard[idx_c]

                head_c, cal_c, conf_c, meta_c = train_bayes_head_pipeline(
                    Z=Z_c,
                    Y_soft=Y_c,
                    y_hard=y_c,
                    seed=seed,
                    alpha=alpha,
                    lambda_reg=lambda_reg,
                )
                meta_c["weak_label_model"] = weak_meta
                meta_c["expert_kind"] = "cluster"
                meta_c["cluster_id"] = int(cid)
                meta_c["member_types"] = member_types
                meta_c["prototype_zscore"] = {"mean": mu.tolist(), "std": sd.tolist()}
                meta_c["kmeans"] = {
                    "K": int(K),
                    "restarts": int(kmeans_restarts),
                    "iters": int(kmeans_iters),
                    "best_sse": float(best_sse),
                    "best_seed": int(best_seed),
                }

                experts_to_save[f"cluster_{int(cid)}"] = (head_c, cal_c, conf_c, meta_c)
                print(f"[Cluster] Trained cluster_{cid} (n={len(idx_c)}, members={len(member_types)})")

    # 8) routing policy
    routing = {
        "priority": ["type", "cluster", "global"],
        "enable_type": True,
        "enable_cluster": bool(enable_cluster),
    }

    # 9) save bundle
    print(f"[Save] bundle -> {model_root}")
    save_bundle(
        model_root=model_root,
        encoder_path_src=encoder_tmp_path,
        encoder_meta={
            "input_dim": 32,
            "latent_dim": int(encoder_meta.get("latent_dim", latent_dim)),
            "output_dim": int(encoder_meta.get("output_dim", proj_dim)),
            "ssl_meta": encoder_meta,
        },
        experts=experts_to_save,
        cluster_map=cluster_map if enable_cluster else None,
        routing=routing,
    )

    print("[Done] Training finished.")
    print("  Experts:", ", ".join(sorted(experts_to_save.keys())))
    if enable_cluster:
        print("  cluster_map size:", len(cluster_map))


if __name__ == "__main__":
    main()
