# === train.py ===
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from models import BayesianSoftmax
from utils import TemperatureCalibrator, ConformalAPS, nll_multiclass, ece_toplabel

from data_adapter import map_status_to_severity

from feature_space import (
    build_report_type_vocab,
    build_global_schema,
    build_type_schema,
    build_cluster_schema,
    featurize_dataset,
    split_by_report_type,
    compute_type_prototypes,
    kmeans,
)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _infer_splits(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = max(1, int(0.6 * n))
    n_cal = max(1, int(0.2 * n))
    return idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]


def weak_label_and_counts(sample: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    indicator_analysis -> (y, abn_cnt, sev_cnt, valid_cnt)
      y: 0/1/2
      abn_cnt: sev>=1 的指标数
      sev_cnt: sev>=2 的指标数
      valid_cnt: sev != None 的指标数（至少 1）
    """
    info0 = sample["info"][0]
    indicators = info0.get("indicator_analysis", []) or []

    has_abn = False
    has_severe = False
    abn_cnt = 0
    sev_cnt = 0
    valid_cnt = 0

    for it in indicators:
        sev = map_status_to_severity(it)  # 0/1/2 or None
        if sev is None:
            continue
        valid_cnt += 1
        sev = int(sev)
        if sev >= 1:
            has_abn = True
            abn_cnt += 1
        if sev >= 2:
            has_severe = True
            sev_cnt += 1

    if has_severe:
        y = 2
    elif has_abn:
        y = 1
    else:
        y = 0

    return y, abn_cnt, sev_cnt, max(valid_cnt, 1)


def build_y_and_counts(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, abn_cnts, sev_cnts, valid_cnts = [], [], [], []
    for s in dataset:
        y, ac, sc, vc = weak_label_and_counts(s)
        ys.append(y)
        abn_cnts.append(ac)
        sev_cnts.append(sc)
        valid_cnts.append(vc)
    return (
        np.asarray(ys, dtype=int),
        np.asarray(abn_cnts, dtype=int),
        np.asarray(sev_cnts, dtype=int),
        np.asarray(valid_cnts, dtype=int),
    )


def _save_bundle(model_dir: str,
                 clf: BayesianSoftmax,
                 temp: TemperatureCalibrator,
                 conf: ConformalAPS,
                 schema: Dict[str, Any],
                 train_meta: Dict[str, Any]) -> None:
    _ensure_dir(model_dir)

    np.savez(
        os.path.join(model_dir, "softmax_model.npz"),
        W_map=clf.W_map,
        Sigma=clf.Sigma if clf.Sigma is not None else np.zeros((1, 1), dtype=float),
        feature_dim=int(clf.W_map.shape[0]),
        n_classes=int(clf.W_map.shape[1]),
        lambda_reg=float(getattr(clf, "lambda_reg", 1.0)),
    )

    with open(os.path.join(model_dir, "calibrator.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "temperature", "T": float(temp.T)}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "conformal.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"method": "APS", "alpha": float(conf.alpha), "qhat": float(conf.qhat) if conf.qhat is not None else None},
            f, ensure_ascii=False, indent=2
        )

    with open(os.path.join(model_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": schema, "train_meta": train_meta}, f, ensure_ascii=False, indent=2)


def _train_one(X: np.ndarray, y: np.ndarray, seed: int, alpha: float, lambda_reg: float) -> Tuple[BayesianSoftmax, TemperatureCalibrator, ConformalAPS, Dict[str, Any]]:
    tr, ca, te = _infer_splits(len(y), seed=seed)

    clf = BayesianSoftmax(n_classes=3, lambda_reg=float(lambda_reg))
    clf.fit(X[tr], y[tr])

    temp = TemperatureCalibrator()
    p_cal_raw = clf.predict_proba(X[ca])
    temp.fit(p_cal_raw, y[ca])
    p_cal = temp.transform(p_cal_raw)

    conf = ConformalAPS(alpha=float(alpha))
    conf.fit(p_cal, y[ca])

    if len(te) > 0:
        p_test = temp.transform(clf.predict_proba(X[te]))
        y_test = y[te]
    else:
        p_test = p_cal
        y_test = y[ca]

    meta = {
        "seed": int(seed),
        "alpha": float(alpha),
        "lambda_reg": float(lambda_reg),
        "n": int(len(y)),
        "n_train": int(len(tr)),
        "n_cal": int(len(ca)),
        "n_test": int(len(te)),
        "T": float(temp.T),
        "qhat": float(conf.qhat) if conf.qhat is not None else None,
        "acc": float(np.mean(np.argmax(p_test, axis=1) == y_test)),
        "nll": float(nll_multiclass(p_test, y_test)),
        "ece_top": float(ece_toplabel(p_test, y_test)),
        "feature_dim": int(X.shape[1]),
    }
    return clf, temp, conf, meta


def zscore_standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def kmeans_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    SSE = sum_i ||x_i - center_{label_i}||^2
    """
    diff = X - centers[labels]
    return float(np.sum(diff * diff))


def kmeans_best_of_restarts(
    X: np.ndarray,
    K: int,
    base_seed: int = 42,
    restarts: int = 5,
    iters: int = 50
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    多次随机重启 kmeans，选 SSE 最小的一次
    返回 labels, centers, best_sse, best_seed
    """
    best_labels = None
    best_centers = None
    best_sse = None
    best_seed = None

    for r in range(int(restarts)):
        seed = int(base_seed + 9973 * r)  # deterministic but different
        labels, centers = kmeans(X, K=K, seed=seed, iters=iters)
        sse = kmeans_sse(X, labels, centers)
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_labels = labels
            best_centers = centers
            best_seed = seed

    return best_labels, best_centers, float(best_sse), int(best_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_report_info_path", type=str, required=True, help="包含 dataset(list) 的 JSON")
    ap.add_argument("--model_root", type=str, required=True, help="输出目录根：global/type_/cluster_")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lambda_reg", type=float, default=1.0)

    ap.add_argument("--min_type_samples", type=int, default=300, help="样本数>=该阈值才训练 type 专属模型")
    ap.add_argument("--cluster_k", type=int, default=8, help="固定 K 聚类（只对长尾 type）")
    ap.add_argument("--min_cluster_samples", type=int, default=500, help="cluster 总样本数>=该阈值才训练 cluster 模型")

    # NEW: kmeans restarts
    ap.add_argument("--kmeans_restarts", type=int, default=5, help="kmeans 多次重启次数")
    ap.add_argument("--kmeans_iters", type=int, default=50, help="每次 kmeans 迭代次数")

    args = ap.parse_args()

    with open(args.processed_report_info_path, "r", encoding="utf-8") as f:
        processed = json.load(f)

    dataset = processed.get("dataset", None)
    if not isinstance(dataset, list) or len(dataset) < 10:
        raise ValueError("必须包含 dataset(list)，且样本数 >= 10")

    y_all, abn_cnt_all, sev_cnt_all, valid_cnt_all = build_y_and_counts(dataset)

    # ===== global =====
    rt_vocab = build_report_type_vocab(dataset)
    global_schema = build_global_schema(dataset, rt_vocab)
    X_global = featurize_dataset(dataset, global_schema)

    global_dir = os.path.join(args.model_root, "global")
    clf_g, temp_g, conf_g, meta_g = _train_one(X_global, y_all, args.seed, args.alpha, args.lambda_reg)
    meta_g["global_max_len"] = int(global_schema["max_len"])
    meta_g["n_report_types"] = int(len(rt_vocab))
    _save_bundle(global_dir, clf_g, temp_g, conf_g, global_schema, meta_g)
    print("[Saved] global ->", global_dir)

    # ===== type-specific =====
    groups = split_by_report_type(dataset)
    trained_types = set()

    for rt, samples_rt in groups.items():
        if rt < 0:
            continue
        if len(samples_rt) < int(args.min_type_samples):
            continue

        idx_rt = [i for i, s in enumerate(dataset) if int(s["info"][0].get("report_type", -1)) == int(rt)]
        y_rt = y_all[idx_rt]

        type_schema = build_type_schema(dataset, rt)
        X_rt = featurize_dataset(samples_rt, type_schema)
        if X_rt.shape[0] != len(y_rt):
            print(f"[Skip] type_{rt}: X({X_rt.shape[0]}) != y({len(y_rt)})")
            continue

        type_dir = os.path.join(args.model_root, f"type_{rt}")
        clf_t, temp_t, conf_t, meta_t = _train_one(X_rt, y_rt, args.seed, args.alpha, args.lambda_reg)
        meta_t["report_type"] = int(rt)
        meta_t["type_max_len"] = int(type_schema["max_len"])
        _save_bundle(type_dir, clf_t, temp_t, conf_t, type_schema, meta_t)
        trained_types.add(int(rt))
        print(f"[Saved] type_{rt} -> {type_dir} (n={len(y_rt)})")

    # ===== tail clustering =====
    tail_types = sorted([rt for rt in groups.keys() if rt >= 0 and int(rt) not in trained_types])
    if len(tail_types) == 0:
        print("[Info] No tail types to cluster.")
        print("\n[Done] Training finished.")
        return

    tail_groups = {rt: groups[rt] for rt in tail_types}

    # Build lookups
    y_lookup: Dict[int, List[int]] = {}
    abn_lookup: Dict[int, List[int]] = {}
    sev_lookup: Dict[int, List[int]] = {}
    valid_lookup: Dict[int, List[int]] = {}

    for rt in tail_types:
        idx_rt = [i for i, s in enumerate(dataset) if int(s["info"][0].get("report_type", -1)) == int(rt)]
        y_lookup[rt] = [int(y_all[i]) for i in idx_rt]
        abn_lookup[rt] = [int(abn_cnt_all[i]) for i in idx_rt]
        sev_lookup[rt] = [int(sev_cnt_all[i]) for i in idx_rt]
        valid_lookup[rt] = [int(valid_cnt_all[i]) for i in idx_rt]

    rts, protos = compute_type_prototypes(
        tail_groups,
        global_max_len=int(global_schema["max_len"]),
        y_lookup=y_lookup,
        abn_count_lookup=abn_lookup,
        sev_count_lookup=sev_lookup,
        valid_count_lookup=valid_lookup,
    )

    # z-score
    protos_z, mu, sd = zscore_standardize(protos, eps=1e-8)

    # kmeans restarts
    K = int(min(args.cluster_k, len(rts)))
    labels, centers, best_sse, best_seed = kmeans_best_of_restarts(
        protos_z,
        K=K,
        base_seed=args.seed,
        restarts=args.kmeans_restarts,
        iters=args.kmeans_iters,
    )

    cluster_map = {str(rt): int(labels[i]) for i, rt in enumerate(rts)}
    _ensure_dir(args.model_root)
    with open(os.path.join(args.model_root, "cluster_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "K": K,
                "cluster_map": cluster_map,
                "prototype_zscore": {"mean": mu.tolist(), "std": sd.tolist()},
                "kmeans": {
                    "restarts": int(args.kmeans_restarts),
                    "iters": int(args.kmeans_iters),
                    "best_sse": float(best_sse),
                    "best_seed": int(best_seed),
                }
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"[Saved] cluster_map.json (K={K}, restarts={args.kmeans_restarts}, best_sse={best_sse:.4f})")

    # ===== train cluster experts =====
    for cid in range(K):
        member_types = [rts[i] for i in range(len(rts)) if int(labels[i]) == int(cid)]
        if len(member_types) == 0:
            continue

        samples_c = []
        y_c = []
        for rt in member_types:
            idx_rt = [i for i, s in enumerate(dataset) if int(s["info"][0].get("report_type", -1)) == int(rt)]
            for i in idx_rt:
                samples_c.append(dataset[i])
                y_c.append(int(y_all[i]))

        if len(samples_c) < int(args.min_cluster_samples):
            print(f"[Skip] cluster_{cid}: n={len(samples_c)} < min_cluster_samples")
            continue

        cluster_schema = build_cluster_schema(dataset, member_types)
        X_c = featurize_dataset(samples_c, cluster_schema)
        y_c = np.asarray(y_c, dtype=int)

        cluster_dir = os.path.join(args.model_root, f"cluster_{cid}")
        clf_c, temp_c, conf_c, meta_c = _train_one(X_c, y_c, args.seed, args.alpha, args.lambda_reg)
        meta_c["member_types"] = member_types
        meta_c["cluster_id"] = int(cid)
        meta_c["cluster_max_len"] = int(cluster_schema["max_len"])
        meta_c["prototype_zscore_used"] = True
        meta_c["kmeans_best_of_restarts"] = {
            "restarts": int(args.kmeans_restarts),
            "iters": int(args.kmeans_iters),
            "best_sse": float(best_sse),
            "best_seed": int(best_seed),
        }
        _save_bundle(cluster_dir, clf_c, temp_c, conf_c, cluster_schema, meta_c)
        print(f"[Saved] cluster_{cid} -> {cluster_dir} (n={len(y_c)}, members={len(member_types)})")

    print("\n[Done] Training finished.")


if __name__ == "__main__":
    main()
