# === train.py ===
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple
import shutil
import sys

import numpy as np

from models import BayesianSoftmax, LabelModelEMMulticlass
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

# =========================================================
# Domain Shift Utilities (NEW)
# =========================================================

def inject_label_noise(y: np.ndarray, noise_rate: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = len(y)
    idx = rng.choice(n, int(n * noise_rate), replace=False)
    for i in idx:
        y_noisy[i] = rng.integers(0, 3)
    return y_noisy


def corrupt_features(X: np.ndarray, rate: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    mask = rng.random(X.shape) < rate
    Xc[mask] = 0.0
    return Xc


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _infer_splits(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = max(1, int(0.6 * n))
    n_cal = max(1, int(0.2 * n))
    return idx[:n_train], idx[n_train:n_train + n_cal], idx[n_train + n_cal:]


# =========================================================
# Weak label stats (counts-only) + multi-class LFs
# =========================================================

def counts_from_indicator_analysis(indicators: List[Any]) -> Tuple[int, int, int, float]:
    """
    indicator_analysis -> (abn_cnt, sev_cnt, valid_cnt, missing_rate)
      sev: 0/1/2 or None
      abn_cnt: sev>=1
      sev_cnt: sev==2
      valid_cnt: sev != None (>=1)
      missing_rate: 1 - valid/total
    """
    abn_cnt = 0
    sev_cnt = 0
    valid_cnt = 0
    total = len(indicators)

    for it in (indicators or []):
        sev = map_status_to_severity(it)
        if sev is None:
            continue
        valid_cnt += 1
        sev = int(sev)
        if sev >= 1:
            abn_cnt += 1
        if sev >= 2:
            sev_cnt += 1

    valid_cnt_safe = max(valid_cnt, 1)
    miss_rate = 1.0 - (valid_cnt / max(total, 1))
    return abn_cnt, sev_cnt, valid_cnt_safe, float(miss_rate)


def build_counts_arrays(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    abn_cnts, sev_cnts, valid_cnts = [], [], []
    for s in dataset:
        info0 = s["info"][0]
        indicators = info0.get("indicator_analysis", []) or []
        abn, sev, valid, _ = counts_from_indicator_analysis(indicators)
        abn_cnts.append(abn)
        sev_cnts.append(sev)
        valid_cnts.append(valid)
    return (
        np.asarray(abn_cnts, dtype=int),
        np.asarray(sev_cnts, dtype=int),
        np.asarray(valid_cnts, dtype=int),
    )


def build_multiclass_lfs() -> Tuple[List[str], List[Any]]:
    """
    Each LF returns vote in {-1,0,1,2}:
      -1 abstain
       0 normal
       1 mild abnormal
       2 severe abnormal
    Votes are based ONLY on indicator_analysis-derived counts (no external truth).
    """
    names: List[str] = []
    lfs: List[Any] = []

    def lf_sev_cnt_ge2(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        return 2 if sev >= 2 else -1

    def lf_sev_cnt_eq1_small(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        # one severe but not many other abnormalities -> more like mild than severe (abstain from 2)
        if sev == 1 and abn <= 2:
            return 1
        return -1

    def lf_abn_ratio_low(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        r = abn / float(valid)
        return 0 if r <= 0.05 else -1

    def lf_abn_ratio_mid(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        r = abn / float(valid)
        # mild zone
        if 0.10 <= r <= 0.40 and sev == 0:
            return 1
        return -1

    def lf_abn_ratio_high_with_sev(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        r = abn / float(valid)
        # high abnormal ratio plus at least one severe -> vote severe
        if r >= 0.60 and sev >= 1:
            return 2
        return -1

    def lf_missing_guard(sample):
        ind = sample["info"][0].get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(ind)
        # too much missing -> abstain (always)
        if miss >= 0.70:
            return -1
        return -1

    names.extend([
        "sev_cnt_ge2",
        "sev_cnt_eq1_small",
        "abn_ratio_low",
        "abn_ratio_mid_mild_only",
        "abn_ratio_high_with_sev",
        "missing_guard",
    ])
    lfs.extend([
        lf_sev_cnt_ge2,
        lf_sev_cnt_eq1_small,
        lf_abn_ratio_low,
        lf_abn_ratio_mid,
        lf_abn_ratio_high_with_sev,
        lf_missing_guard,
    ])
    return names, lfs


def build_L_votes(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    names, lfs = build_multiclass_lfs()
    n = len(dataset)
    m = len(lfs)
    L = np.full((n, m), -1, dtype=int)
    for i, s in enumerate(dataset):
        for j, lf in enumerate(lfs):
            try:
                v = int(lf(s))
            except Exception:
                v = -1
            if v not in (-1, 0, 1, 2):
                v = -1
            L[i, j] = v
    info = {"lf_names": names, "n_lfs": m}
    return L, info


# =========================================================
# Training helpers
# =========================================================

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


def _train_one_soft(
    X: np.ndarray,
    Y_soft: np.ndarray,
    y_hard: np.ndarray,
    seed: int,
    alpha: float,
    lambda_reg: float
) -> Tuple[BayesianSoftmax, TemperatureCalibrator, ConformalAPS, Dict[str, Any]]:
    tr, ca, te = _infer_splits(len(y_hard), seed=seed)

    clf = BayesianSoftmax(n_classes=3, lambda_reg=float(lambda_reg))
    clf.fit_soft(X[tr], Y_soft[tr])

    temp = TemperatureCalibrator()
    p_cal_raw = clf.predict_proba(X[ca])
    temp.fit(p_cal_raw, y_hard[ca])
    p_cal = temp.transform(p_cal_raw)

    conf = ConformalAPS(alpha=float(alpha))
    conf.fit(p_cal, y_hard[ca])

    if len(te) > 0:
        p_test = temp.transform(clf.predict_proba(X[te]))
        y_test = y_hard[te]
    else:
        p_test = p_cal
        y_test = y_hard[ca]

    meta = {
        "seed": int(seed),
        "alpha": float(alpha),
        "lambda_reg": float(lambda_reg),
        "n": int(len(y_hard)),
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
    diff = X - centers[labels]
    return float(np.sum(diff * diff))


def kmeans_best_of_restarts(
    X: np.ndarray,
    K: int,
    base_seed: int = 42,
    restarts: int = 5,
    iters: int = 50
) -> Tuple[np.ndarray, np.ndarray, float, int]:
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


def load_dataset(path: str) -> List[Dict[str, Any]]:
    dataset = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据集文件: {path}")

    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[Warning] 第 {line_no} 行解析 JSON 失败，已跳过。")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "dataset" in data:
                dataset = data["dataset"]
            elif isinstance(data, list):
                dataset = data
            else:
                raise ValueError("JSON 格式不符合要求：应包含 'dataset' 列表")

    return dataset


def main():
    ap = argparse.ArgumentParser()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    default_train_data_path = os.path.join(current_dir, "dataset", "eval_train_dataset.jsonl")
    ap.add_argument("--train_data_path", type=str, default=default_train_data_path,help=f"（默认为同级级中dataset下的eval_train_dataset.jsonl）")

    default_model_root = os.path.join(current_dir, "model")
    ap.add_argument("--model_root", type=str, default=default_model_root,help="输出目录根（默认: 同级中 eval_model）")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lambda_reg", type=float, default=1.0)

    ap.add_argument("--min_type_samples", type=int, default=300, help="样本数>=该阈值才训练 type 专属模型")
    ap.add_argument("--cluster_k", type=int, default=8, help="固定 K 聚类（只对长尾 type）")
    ap.add_argument("--min_cluster_samples", type=int, default=500, help="cluster 总样本数>=该阈值才训练 cluster 模型")

    ap.add_argument("--kmeans_restarts", type=int, default=5, help="kmeans 多次重启次数")
    ap.add_argument("--kmeans_iters", type=int, default=50, help="每次 kmeans 迭代次数")

    ap.add_argument("--label_noise", type=float, default=0.0)
    ap.add_argument("--feature_corrupt", type=float, default=0.0)
    ap.add_argument("--holdout_type", type=int, default=-1,help="如果>=0，则该 report_type 完全不参与训练，用于 domain shift 实验")


    args = ap.parse_args()

    # ==================== 清理旧模型目录 ====================
    target_root = os.path.abspath(args.model_root)

    if os.path.exists(target_root):
        print(f"检测到旧模型目录：{target_root}")
        try:
            shutil.rmtree(target_root)
            print(f"[Success] 已清理旧模型，确保无残留。")
        except Exception as e:
            print(f"[Error] 清理目录失败: {e}")
            sys.exit(1)

    os.makedirs(target_root, exist_ok=True)
    # =======================================================

    print(f"正在加载数据集: {args.train_data_path}")
    dataset = load_dataset(args.train_data_path)

    if not isinstance(dataset, list) or len(dataset) < 10:
        raise ValueError(f"数据集样本量不足 (n={len(dataset)})，无法训练模型。")

    print(f"成功加载 {len(dataset)} 条样本。")

    # ===== Unseen Type Domain Shift =====
    if args.holdout_type >= 0:
        dataset = [
            s for s in dataset
            if int(s["info"][0].get("report_type", -1)) != int(args.holdout_type)
        ]
        print(f"[Domain Shift] holdout report_type={args.holdout_type}")


    # =======================================================
    # NEW: Weak supervision label model -> soft labels
    # =======================================================
    L, lf_info = build_L_votes(dataset)
    lm = LabelModelEMMulticlass(n_classes=3, max_iter=120, tol=1e-6, seed=int(args.seed))
    lm.fit(L)
    Y_soft = lm.predict_proba(L)              # (n,3)
    y_hard = np.argmax(Y_soft, axis=1).astype(int)

    # ===== Label Noise =====
    if args.label_noise > 0:
        y_hard = inject_label_noise(y_hard, noise_rate=args.label_noise)
        print(f"[Domain Shift] Injected label noise rate={args.label_noise}")


    # counts for clustering prototypes
    abn_cnt_all, sev_cnt_all, valid_cnt_all = build_counts_arrays(dataset)

    # record label-model meta once (store in global train_meta and copy to experts)
    weak_meta = {
        "method": "em_multiclass",
        "n_classes": 3,
        "lf_names": lf_info.get("lf_names", []),
        "q_abstain": lm.q_abstain_.tolist() if lm.q_abstain_ is not None else None,
        "pi": lm.pi_.tolist() if lm.pi_ is not None else None,
        "A": lm.A_.tolist() if lm.A_ is not None else None,  # small (m x 3 x 3), ok to store
        "label_distribution_hard": {
            "y0": int(np.sum(y_hard == 0)),
            "y1": int(np.sum(y_hard == 1)),
            "y2": int(np.sum(y_hard == 2)),
        }
    }

    # ===== global =====
    rt_vocab = build_report_type_vocab(dataset)
    global_schema = build_global_schema(dataset, rt_vocab)
    X_global = featurize_dataset(dataset, global_schema)

    # ===== Feature Corruption =====
    if args.feature_corrupt > 0:
        X_global = corrupt_features(X_global, rate=args.feature_corrupt)
        print(f"[Domain Shift] Feature corruption rate={args.feature_corrupt}")

    global_dir = os.path.join(args.model_root, "global")
    clf_g, temp_g, conf_g, meta_g = _train_one_soft(X_global, Y_soft, y_hard, args.seed, args.alpha, args.lambda_reg)
    meta_g["global_max_len"] = int(global_schema["max_len"])
    meta_g["n_report_types"] = int(len(rt_vocab))
    meta_g["weak_label_model"] = weak_meta
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
        y_rt = y_hard[idx_rt]
        Y_rt = Y_soft[idx_rt]

        type_schema = build_type_schema(dataset, rt)
        X_rt = featurize_dataset(samples_rt, type_schema)
        if X_rt.shape[0] != len(y_rt):
            print(f"[Skip] type_{rt}: X({X_rt.shape[0]}) != y({len(y_rt)})")
            continue

        type_dir = os.path.join(args.model_root, f"type_{rt}")
        clf_t, temp_t, conf_t, meta_t = _train_one_soft(X_rt, Y_rt, y_rt, args.seed, args.alpha, args.lambda_reg)
        meta_t["report_type"] = int(rt)
        meta_t["type_max_len"] = int(type_schema["max_len"])
        meta_t["weak_label_model"] = weak_meta
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

    # Build lookups for prototypes
    y_lookup: Dict[int, List[int]] = {}
    abn_lookup: Dict[int, List[int]] = {}
    sev_lookup: Dict[int, List[int]] = {}
    valid_lookup: Dict[int, List[int]] = {}

    for rt in tail_types:
        idx_rt = [i for i, s in enumerate(dataset) if int(s["info"][0].get("report_type", -1)) == int(rt)]
        y_lookup[rt] = [int(y_hard[i]) for i in idx_rt]
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

    protos_z, mu, sd = zscore_standardize(protos, eps=1e-8)

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
        idx_c = []
        for rt in member_types:
            idx_rt = [i for i, s in enumerate(dataset) if int(s["info"][0].get("report_type", -1)) == int(rt)]
            for i in idx_rt:
                samples_c.append(dataset[i])
                idx_c.append(i)

        if len(samples_c) < int(args.min_cluster_samples):
            print(f"[Skip] cluster_{cid}: n={len(samples_c)} < min_cluster_samples")
            continue

        cluster_schema = build_cluster_schema(dataset, member_types)
        X_c = featurize_dataset(samples_c, cluster_schema)
        y_c = y_hard[idx_c]
        Y_c = Y_soft[idx_c]

        cluster_dir = os.path.join(args.model_root, f"cluster_{cid}")
        clf_c, temp_c, conf_c, meta_c = _train_one_soft(X_c, Y_c, y_c, args.seed, args.alpha, args.lambda_reg)
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
        meta_c["weak_label_model"] = weak_meta
        _save_bundle(cluster_dir, clf_c, temp_c, conf_c, cluster_schema, meta_c)
        print(f"[Saved] cluster_{cid} -> {cluster_dir} (n={len(y_c)}, members={len(member_types)})")

    print("\n[Done] Training finished.")


if __name__ == "__main__":
    main()
