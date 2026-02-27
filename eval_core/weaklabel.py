# weaklabel.py
# -*- coding: utf-8 -*-
"""
weaklabel.py
------------
职责：
1) 从原始样本的 indicator_analysis 统计出异常/重度/缺失等信息
2) 构建多分类弱监督规则（LFs），输出 vote ∈ {-1,0,1,2}
3) EM 多分类 Label Model（Snorkel-style，条件独立假设）学习 LF 可靠性
4) 生成 soft labels (Y_soft) 与 hard labels (y_hard)，并输出 weak_meta 供 bundle 保存

三分类语义约定：
  0: normal
  1: mild abnormal
  2: severe abnormal

LF 投票约定：
 -1: abstain
  0/1/2: vote that class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from preprocess import read_jsonl, map_word_to_score


# =========================================================
# 1) Stats from indicator_analysis
# =========================================================

def counts_from_indicator_analysis(indicators: Any) -> Tuple[int, int, int, float]:
    """
    indicator_analysis -> (abn_cnt, sev_cnt, valid_cnt_safe, missing_rate)

    - abn_cnt: severity>=1 的个数
    - sev_cnt: severity==2 的个数
    - valid_cnt: severity in {0,1,2} 的个数（缺失/无法解析的不计入）
    - missing_rate: 1 - valid/total
    """
    if not isinstance(indicators, list):
        indicators = []

    total = len(indicators)
    abn_cnt = 0
    sev_cnt = 0
    valid_cnt = 0

    for it in indicators:
        sev = map_word_to_score(it)  # 0/1/2 or -1
        if sev < 0:
            continue
        valid_cnt += 1
        if sev >= 1:
            abn_cnt += 1
        if sev >= 2:
            sev_cnt += 1

    valid_cnt_safe = max(valid_cnt, 1)
    missing_rate = 1.0 - (valid_cnt / max(total, 1))
    return int(abn_cnt), int(sev_cnt), int(valid_cnt_safe), float(missing_rate)


def _get_info0(sample: Dict[str, Any]) -> Dict[str, Any]:
    info = sample.get("info", [])
    if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
        return info[0]
    return {}


def build_counts_arrays(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回全数据集的：
      abn_cnts, sev_cnts, valid_cnts, missing_rates
    """
    abn_cnts, sev_cnts, valid_cnts, miss_rates = [], [], [], []
    for s in dataset:
        info0 = _get_info0(s)
        indicators = info0.get("indicator_analysis", []) or []
        abn, sev, valid, miss = counts_from_indicator_analysis(indicators)
        abn_cnts.append(abn)
        sev_cnts.append(sev)
        valid_cnts.append(valid)
        miss_rates.append(miss)
    return (
        np.asarray(abn_cnts, dtype=int),
        np.asarray(sev_cnts, dtype=int),
        np.asarray(valid_cnts, dtype=int),
        np.asarray(miss_rates, dtype=float),
    )


# =========================================================
# 2) Multi-class Labeling Functions (LFs)
# =========================================================

LF = Callable[[Dict[str, Any]], int]


def build_multiclass_lfs() -> Tuple[List[str], List[LF]]:
    """
    每个 LF 仅基于 indicator_analysis 的计数/比例做投票。
    返回 (lf_names, lf_fns)
    """
    names: List[str] = []
    lfs: List[LF] = []

    def _stats(sample: Dict[str, Any]) -> Tuple[int, int, int, float]:
        info0 = _get_info0(sample)
        indicators = info0.get("indicator_analysis", []) or []
        return counts_from_indicator_analysis(indicators)

    # ---- LF 1: severe count >= 2 => severe
    def lf_sev_cnt_ge2(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
        return 2 if sev >= 2 else -1

    # ---- LF 2: exactly one severe but not many abnormals => mild (avoid over-calling severe)
    def lf_sev_cnt_eq1_small(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
        if sev == 1 and abn <= 2:
            return 1
        return -1

    # ---- LF 3: abnormal ratio very low => normal
    def lf_abn_ratio_low(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
        r = abn / float(valid)
        return 0 if r <= 0.05 else -1

    # ---- LF 4: mid abnormal ratio and no severe => mild
    def lf_abn_ratio_mid_mild_only(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
        r = abn / float(valid)
        if 0.10 <= r <= 0.40 and sev == 0:
            return 1
        return -1

    # ---- LF 5: high abnormal ratio + at least one severe => severe
    def lf_abn_ratio_high_with_sev(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
        r = abn / float(valid)
        if r >= 0.60 and sev >= 1:
            return 2
        return -1

    # ---- LF 6: missing guard (too much missing => abstain)
    # 这个 LF 的意义主要是“不给强结论”，所以永远 abstain，但可用于统计/审计
    def lf_missing_guard(sample: Dict[str, Any]) -> int:
        abn, sev, valid, miss = _stats(sample)
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
        lf_abn_ratio_mid_mild_only,
        lf_abn_ratio_high_with_sev,
        lf_missing_guard,
    ])
    return names, lfs


def build_L_votes(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    构建 L: (n, m), vote ∈ {-1,0,1,2}
    """
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

    info = {"lf_names": names, "n_lfs": int(m)}
    return L, info


# =========================================================
# 3) EM Label Model (multi-class)
# =========================================================

@dataclass
class LabelModelEMMulticlass:
    """
    Multi-class EM weak label model (Snorkel-style, conditional independence).

    L_ij in {-1, 0, 1, 2}:
      -1: abstain
       0/1/2: vote for that class

    Learns:
      pi_k = P(Y=k)
      q_j  = P(LF_j abstains)
      A[j, v, k] = P(LF_j votes v | Y=k), where v in {0,1,2}

    Note:
      - This is a lightweight implementation sufficient for project use.
      - It assumes conditional independence across LFs given Y.
    """
    n_classes: int = 3
    max_iter: int = 120
    tol: float = 1e-6
    seed: int = 42
    min_prob: float = 1e-6

    pi_: Optional[np.ndarray] = None        # (K,)
    A_: Optional[np.ndarray] = None         # (m, K, K): A[j, v, k]
    q_abstain_: Optional[np.ndarray] = None # (m,)

    def fit(self, L: np.ndarray) -> "LabelModelEMMulticlass":
        rng = np.random.default_rng(int(self.seed))
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        K = int(self.n_classes)

        # init class prior
        pi = np.ones(K, dtype=float) / K

        # init abstain rate per LF (fixed by data; still store)
        q = np.clip(np.mean(L == -1, axis=0).astype(float), 1e-3, 1 - 1e-3)

        # init A close to diagonal (good LFs vote correct class more often)
        A = np.zeros((m, K, K), dtype=float)  # (j, v, k)
        for j in range(m):
            A[j] = 0.10 / max(1, K - 1)
            np.fill_diagonal(A[j], 0.90)
            A[j] = A[j] / (np.sum(A[j], axis=0, keepdims=True) + 1e-12)

        prev_ll = None

        for _ in range(int(self.max_iter)):
            # -----------------
            # E-step: posterior p(Y|L)
            # -----------------
            logp = np.log(pi + 1e-12)[None, :].repeat(n, axis=0)  # (n,K)

            for j in range(m):
                lj = L[:, j]
                abst = (lj == -1)
                if np.any(abst):
                    logp[abst] += np.log(q[j] + 1e-12)

                non = ~abst
                if np.any(non):
                    v = lj[non]
                    invalid = (v < 0) | (v >= K)
                    if np.any(invalid):
                        # treat invalid as abstain
                        idx_invalid = np.where(non)[0][invalid]
                        logp[idx_invalid] += np.log(q[j] + 1e-12)

                        idx_valid = np.where(non)[0][~invalid]
                        v_valid = v[~invalid]
                        if len(idx_valid) > 0:
                            logp[idx_valid] += np.log(1 - q[j] + 1e-12)
                            # add log P(L=v | Y=k) for each row
                            logp[idx_valid] += np.log(A[j, v_valid, :].T + 1e-12).T
                    else:
                        logp[non] += np.log(1 - q[j] + 1e-12)
                        logp[non] += np.log(A[j, v, :].T + 1e-12).T

            mx = np.max(logp, axis=1, keepdims=True)
            p = np.exp(logp - mx)
            p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)  # (n,K)

            # log-likelihood (up to constant)
            ll = float(np.sum(mx.squeeze() + np.log(np.sum(np.exp(logp - mx), axis=1) + 1e-12)))

            if prev_ll is not None and abs(ll - prev_ll) < float(self.tol):
                break
            prev_ll = ll

            # -----------------
            # M-step
            # -----------------
            # update pi
            pi = np.clip(np.mean(p, axis=0), self.min_prob, 1.0)
            pi = pi / pi.sum()

            # q is determined by data; keep stable but clip
            q = np.clip(np.mean(L == -1, axis=0).astype(float), 1e-3, 1 - 1e-3)

            # update A
            for j in range(m):
                lj = L[:, j]
                non = (lj != -1)
                if not np.any(non):
                    continue

                denom = np.sum(p[non], axis=0) + 1e-12  # (K,)

                Aj = np.zeros((K, K), dtype=float)  # (v,k)
                for v in range(K):
                    idx = non & (lj == v)
                    if np.any(idx):
                        Aj[v, :] = np.sum(p[idx], axis=0) / denom
                    else:
                        Aj[v, :] = self.min_prob

                Aj = Aj / (np.sum(Aj, axis=0, keepdims=True) + 1e-12)
                A[j] = Aj

        self.pi_ = pi
        self.A_ = A
        self.q_abstain_ = q
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.pi_ is None or self.A_ is None or self.q_abstain_ is None:
            raise RuntimeError("LabelModelEMMulticlass not fit")

        L = np.asarray(L, dtype=int)
        n, m = L.shape
        K = int(self.n_classes)

        logp = np.log(self.pi_ + 1e-12)[None, :].repeat(n, axis=0)  # (n,K)

        for j in range(m):
            lj = L[:, j]
            abst = (lj == -1)
            if np.any(abst):
                logp[abst] += np.log(self.q_abstain_[j] + 1e-12)

            non = ~abst
            if np.any(non):
                v = lj[non]
                invalid = (v < 0) | (v >= K)
                if np.any(invalid):
                    idx_invalid = np.where(non)[0][invalid]
                    logp[idx_invalid] += np.log(self.q_abstain_[j] + 1e-12)

                    idx_valid = np.where(non)[0][~invalid]
                    v_valid = v[~invalid]
                    if len(idx_valid) > 0:
                        logp[idx_valid] += np.log(1 - self.q_abstain_[j] + 1e-12)
                        logp[idx_valid] += np.log(self.A_[j, v_valid, :].T + 1e-12).T
                else:
                    logp[non] += np.log(1 - self.q_abstain_[j] + 1e-12)
                    logp[non] += np.log(self.A_[j, v, :].T + 1e-12).T

        mx = np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp - mx)
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)


# =========================================================
# 4) End-to-end: make soft labels
# =========================================================

def make_soft_labels(
    dataset: List[Dict[str, Any]],
    seed: int = 42,
    max_iter: int = 120,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    输入 dataset（list of samples）
    输出：
      Y_soft: (n,3)
      y_hard: (n,)
      weak_meta: 可保存到 bundle 的弱监督元信息
    """
    L, lf_info = build_L_votes(dataset)

    lm = LabelModelEMMulticlass(
        n_classes=3,
        max_iter=int(max_iter),
        tol=float(tol),
        seed=int(seed),
    )
    lm.fit(L)
    Y_soft = lm.predict_proba(L)
    y_hard = np.argmax(Y_soft, axis=1).astype(int)

    weak_meta = {
        "method": "em_multiclass",
        "n_classes": 3,
        "lf_names": lf_info.get("lf_names", []),
        "q_abstain": lm.q_abstain_.tolist() if lm.q_abstain_ is not None else None,
        "pi": lm.pi_.tolist() if lm.pi_ is not None else None,
        "A": lm.A_.tolist() if lm.A_ is not None else None,
        "label_distribution_hard": {
            "y0": int(np.sum(y_hard == 0)),
            "y1": int(np.sum(y_hard == 1)),
            "y2": int(np.sum(y_hard == 2)),
        },
        "n_samples": int(len(dataset)),
        "n_lfs": int(lf_info.get("n_lfs", L.shape[1] if L.ndim == 2 else 0)),
    }

    return Y_soft, y_hard, weak_meta


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    训练入口 train.py 可直接用这个加载 jsonl。
    """
    data = read_jsonl(path)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Cannot load dataset from: {path}")
    return data
