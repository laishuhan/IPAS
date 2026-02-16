# === feature_space.py ===
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np


BASIC_FIELDS = ["report_type", "sex", "age", "period_info", "preg_info"]


def _safe_float(x: Any, default: float = 0.0) -> float:
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


def _get_info0(sample: Dict[str, Any]) -> Dict[str, Any]:
    info = sample.get("info", None)
    if not isinstance(info, list) or len(info) == 0 or not isinstance(info[0], dict):
        raise ValueError("sample 缺少 info[0] 或格式不正确")
    return info[0]


def build_report_type_vocab(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    rts = []
    for s in dataset:
        i0 = _get_info0(s)
        rt = i0.get("report_type", None)
        if rt is None:
            continue
        rts.append(int(rt))
    rts = sorted(list(set(rts)))
    return {str(rt): idx for idx, rt in enumerate(rts)}


def split_by_report_type(dataset: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for s in dataset:
        i0 = _get_info0(s)
        rt = int(i0.get("report_type", -1))
        groups.setdefault(rt, []).append(s)
    return groups


def build_global_schema(dataset: List[Dict[str, Any]], rt_vocab: Dict[str, int]) -> Dict[str, Any]:
    max_len = 0
    for s in dataset:
        i0 = _get_info0(s)
        rd = i0.get("report_data", []) or []
        max_len = max(max_len, len(rd))
    return {
        "kind": "global",
        "max_len": int(max_len),
        "rt_vocab": rt_vocab,
        "include_basic_fields": True,
        "include_onehot": True,
        "include_mask": True,
        "basic_fields": BASIC_FIELDS,
    }


def build_type_schema(dataset: List[Dict[str, Any]], report_type: int) -> Dict[str, Any]:
    max_len = 0
    for s in dataset:
        i0 = _get_info0(s)
        if int(i0.get("report_type", -999999)) != int(report_type):
            continue
        rd = i0.get("report_data", []) or []
        max_len = max(max_len, len(rd))
    return {
        "kind": "type",
        "report_type": int(report_type),
        "max_len": int(max_len),
        "include_basic_fields": True,
        "include_onehot": False,
        "include_mask": True,
        "basic_fields": BASIC_FIELDS,
    }


def build_cluster_schema(dataset: List[Dict[str, Any]], member_types: List[int]) -> Dict[str, Any]:
    member_types = sorted(list(set(int(x) for x in member_types)))
    rt_vocab = {str(rt): i for i, rt in enumerate(member_types)}

    max_len = 0
    for s in dataset:
        i0 = _get_info0(s)
        rt = int(i0.get("report_type", -1))
        if rt not in member_types:
            continue
        rd = i0.get("report_data", []) or []
        max_len = max(max_len, len(rd))

    return {
        "kind": "cluster",
        "member_types": member_types,
        "max_len": int(max_len),
        "rt_vocab": rt_vocab,
        "include_basic_fields": True,
        "include_onehot": True,
        "include_mask": True,
        "basic_fields": BASIC_FIELDS,
    }


def _pad_and_mask(report_data: List[Any], max_len: int, missing_value: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.zeros(max_len, dtype=float)
    mask = np.ones(max_len, dtype=float)
    n = min(len(report_data), max_len)
    for i in range(n):
        v = _safe_float(report_data[i], default=missing_value)
        if v == missing_value:
            vals[i] = 0.0
            mask[i] = 1.0
        else:
            vals[i] = float(v)
            mask[i] = 0.0
    return vals, mask


def featurize_sample(sample: Dict[str, Any], schema: Dict[str, Any]) -> np.ndarray:
    i0 = _get_info0(sample)
    max_len = int(schema["max_len"])
    report_data = i0.get("report_data", []) or []
    values, mask = _pad_and_mask(report_data, max_len=max_len)

    feats = [values]
    if schema.get("include_mask", True):
        feats.append(mask)

    if schema.get("include_basic_fields", True):
        basic = []
        for k in schema.get("basic_fields", BASIC_FIELDS):
            basic.append(_safe_float(i0.get(k, -1), default=-1.0))
        feats.append(np.array(basic, dtype=float))

    if schema.get("include_onehot", False):
        rt_vocab = schema.get("rt_vocab", {})
        rt = str(int(i0.get("report_type", -1)))
        onehot = np.zeros(len(rt_vocab), dtype=float)
        if rt in rt_vocab:
            onehot[rt_vocab[rt]] = 1.0
        feats.append(onehot)

    return np.concatenate(feats, axis=0)


def featurize_dataset(dataset: List[Dict[str, Any]], schema: Dict[str, Any]) -> np.ndarray:
    X = [featurize_sample(s, schema) for s in dataset]
    return np.vstack(X)


# =========================
# Clustering (tail types)
# =========================

def compute_type_prototypes(
    groups: Dict[int, List[Dict[str, Any]]],
    global_max_len: int,
    y_lookup: Optional[Dict[int, List[int]]] = None,
    abn_count_lookup: Optional[Dict[int, List[int]]] = None,
    sev_count_lookup: Optional[Dict[int, List[int]]] = None,
    valid_count_lookup: Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    proto = [
      missing_rate(L), mean(L), std(L),
      len_norm,
      abnormal_rate, severe_rate,
      avg_abn_count, avg_sev_count,
      avg_abn_ratio, avg_sev_ratio
    ]
    """
    rts = sorted([rt for rt in groups.keys() if rt >= 0])
    L = int(global_max_len)
    protos = []

    for rt in rts:
        samples = groups[rt]

        vals_list = []
        mask_list = []
        max_len_rt = 0
        for s in samples:
            i0 = _get_info0(s)
            rd = i0.get("report_data", []) or []
            max_len_rt = max(max_len_rt, len(rd))
            v, m = _pad_and_mask(rd, max_len=L)
            vals_list.append(v)
            mask_list.append(m)

        V = np.vstack(vals_list)
        M = np.vstack(mask_list)
        missing_rate = M.mean(axis=0)

        present = (M == 0)
        mean = np.zeros(L, dtype=float)
        std = np.zeros(L, dtype=float)
        for j in range(L):
            idx = present[:, j]
            if np.any(idx):
                xj = V[idx, j]
                mean[j] = float(np.mean(xj))
                std[j] = float(np.std(xj))
            else:
                mean[j] = 0.0
                std[j] = 0.0

        len_norm = float(max_len_rt) / float(max(1, L))

        abnormal_rate = 0.0
        severe_rate = 0.0
        avg_abn_count = 0.0
        avg_sev_count = 0.0
        avg_abn_ratio = 0.0
        avg_sev_ratio = 0.0

        if y_lookup is not None and rt in y_lookup and len(y_lookup[rt]) > 0:
            ys = np.asarray(y_lookup[rt], dtype=int)
            abnormal_rate = float(np.mean(ys >= 1))
            severe_rate = float(np.mean(ys == 2))

        if abn_count_lookup is not None and rt in abn_count_lookup and len(abn_count_lookup[rt]) > 0:
            avg_abn_count = float(np.mean(np.asarray(abn_count_lookup[rt], dtype=float)))

        if sev_count_lookup is not None and rt in sev_count_lookup and len(sev_count_lookup[rt]) > 0:
            avg_sev_count = float(np.mean(np.asarray(sev_count_lookup[rt], dtype=float)))

        # NEW: ratio normalized by valid indicator count
        if (
            valid_count_lookup is not None and rt in valid_count_lookup and len(valid_count_lookup[rt]) > 0
            and abn_count_lookup is not None and rt in abn_count_lookup and len(abn_count_lookup[rt]) > 0
            and sev_count_lookup is not None and rt in sev_count_lookup and len(sev_count_lookup[rt]) > 0
        ):
            v = np.asarray(valid_count_lookup[rt], dtype=float)
            a = np.asarray(abn_count_lookup[rt], dtype=float)
            s = np.asarray(sev_count_lookup[rt], dtype=float)
            v = np.maximum(v, 1.0)
            avg_abn_ratio = float(np.mean(a / v))
            avg_sev_ratio = float(np.mean(s / v))

        proto_tail = np.array(
            [len_norm, abnormal_rate, severe_rate, avg_abn_count, avg_sev_count, avg_abn_ratio, avg_sev_ratio],
            dtype=float
        )
        proto = np.concatenate([missing_rate, mean, std, proto_tail], axis=0)
        protos.append(proto)

    return rts, np.vstack(protos)


def kmeans(protos: np.ndarray, K: int, seed: int = 42, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    n, d = protos.shape
    K = int(min(K, n))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=K, replace=False)
    centers = protos[init_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(iters):
        dist2 = ((protos[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        for k in range(K):
            idx = (labels == k)
            if np.any(idx):
                centers[k] = protos[idx].mean(axis=0)
            else:
                centers[k] = protos[rng.integers(0, n)]

    return labels, centers
