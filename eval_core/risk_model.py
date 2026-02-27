# risk_model.py
# -*- coding: utf-8 -*-
"""
risk_model.py
-------------
职责：
1) BayesianSoftmax：多分类 softmax 回归 + Laplace 近似不确定性（epistemic）
2) TemperatureCalibrator：温度缩放校准（p_raw -> p_final）
3) ConformalAPS：自适应预测集合（APS）
4) abstain_decision_multiclass：风险一致拒判（risk + epistemic）
5) selective_risk：选择性预测评估（coverage, selective_risk）
6) 训练帮助函数：train_bayes_head_pipeline（头部训练+校准+共形）

约定：
- 任务三分类：0 normal, 1 mild, 2 severe
- 输入特征通常是 encoder 输出 embedding Z (n,d)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================================================
# 1) Math helpers
# =========================================================

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_categorical(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def nll_multiclass(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))

def ece_toplabel(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Multi-class ECE using top-label confidence.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    acc = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = len(y)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf >= b0) & (conf < b1) if b1 < 1.0 else (conf >= b0) & (conf <= b1)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask) / n) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)


# =========================================================
# 2) Bayesian Softmax (MAP + Laplace)
# =========================================================

class BayesianSoftmax:
    """
    多分类 softmax 回归 + Laplace 近似
    - 拟合：MAP（L2 正则）
    - 协方差：H^{-1}（Hessian）
    - 不确定性：
        aleatoric = 1 - sum(p^2)
        epistemic = mean_k Var(logit_k)  (delta approx using Sigma)
    """
    def __init__(self, n_classes: int = 3, lambda_reg: float = 1.0):
        self.n_classes = int(n_classes)
        self.lambda_reg = float(lambda_reg)
        self.W_map: Optional[np.ndarray] = None  # (d, K)
        self.Sigma: Optional[np.ndarray] = None  # ((dK),(dK))

    def fit_soft(
        self,
        X: np.ndarray,
        Y_soft: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> "BayesianSoftmax":
        """
        用 soft targets Y_soft (n,K) 训练 MAP 权重，并用 Hessian 近似协方差。
        """
        X = np.asarray(X, dtype=float)
        Y_soft = np.asarray(Y_soft, dtype=float)

        n, d = X.shape
        K = self.n_classes
        if Y_soft.shape != (n, K):
            raise ValueError(f"Y_soft shape {Y_soft.shape} != (n={n},K={K})")

        # normalize
        Y_soft = np.clip(Y_soft, 1e-12, 1.0)
        Y_soft = Y_soft / (np.sum(Y_soft, axis=1, keepdims=True) + 1e-12)

        W = np.zeros((d, K), dtype=float)
        I = np.eye(d * K, dtype=float)

        H = None
        for _ in range(int(max_iter)):
            logits = X @ W                      # (n,K)
            P = softmax(logits, axis=1)         # (n,K)

            # gradient
            G = X.T @ (P - Y_soft) + self.lambda_reg * W   # (d,K)
            g = G.reshape(-1)

            # Hessian
            H = np.zeros((d * K, d * K), dtype=float)
            for i in range(n):
                xi = X[i].reshape(d, 1)
                Si = np.diag(P[i]) - np.outer(P[i], P[i])  # (K,K)
                H += np.kron(Si, (xi @ xi.T))
            H += self.lambda_reg * I

            # Newton step
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, g, rcond=None)[0]

            W_new = (W.reshape(-1) - delta).reshape(d, K)

            if np.linalg.norm(W_new - W) < float(tol):
                W = W_new
                break
            W = W_new

        self.W_map = W
        self.Sigma = np.linalg.pinv(H) if H is not None else None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W_map is None:
            raise RuntimeError("BayesianSoftmax not fit")
        X = np.asarray(X, dtype=float)
        logits = X @ self.W_map
        return softmax(logits, axis=1)

    def predictive_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """
        x: (d,) or (1,d)
        returns:
          p: (K,)
          pred: int
          entropy: float
          aleatoric: float
          epistemic: float (rough)
        """
        if self.W_map is None:
            raise RuntimeError("BayesianSoftmax not fit")

        x = np.asarray(x, dtype=float).reshape(1, -1)  # (1,d)
        p = self.predict_proba(x)[0]                   # (K,)
        pred = int(np.argmax(p))
        ent = entropy_categorical(p)
        ale = float(1.0 - np.sum(p ** 2))

        epi = 0.0
        if self.Sigma is not None:
            d, K = self.W_map.shape
            xvec = x.reshape(-1)
            var_logits = []
            for k in range(K):
                e = np.zeros(d * K, dtype=float)
                e[k * d:(k + 1) * d] = xvec
                var = float(e @ self.Sigma @ e)
                var_logits.append(max(var, 0.0))
            epi = float(np.mean(var_logits))

        return {
            "p": p,
            "pred": pred,
            "entropy": float(ent),
            "aleatoric": float(ale),
            "epistemic": float(epi),
        }


# =========================================================
# 3) Temperature scaling calibrator
# =========================================================

@dataclass
class TemperatureCalibrator:
    """
    多分类温度缩放：p_cal = softmax(logits / T)
    输入 p_raw（概率）时，用 logits=log(p_raw) 近似。
    用数值梯度做简单 GD 优化 T。
    """
    T: float = 1.0
    max_iter: int = 200
    lr: float = 0.05

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        logits = np.log(np.clip(p_raw, 1e-12, 1.0))
        T = float(self.T)

        for _ in range(int(self.max_iter)):
            pT = self.transform_logits(logits, T)
            loss = nll_multiclass(pT, y)

            # numerical gradient
            eps = 1e-4
            pT2 = self.transform_logits(logits, T + eps)
            loss2 = nll_multiclass(pT2, y)
            g = (loss2 - loss) / eps

            T_new = max(0.05, T - self.lr * g)
            if abs(T_new - T) < 1e-6:
                T = T_new
                break
            T = T_new

        self.T = float(T)
        return self

    @staticmethod
    def transform_logits(logits: np.ndarray, T: float) -> np.ndarray:
        z = logits / float(T)
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(np.asarray(p_raw, dtype=float), 1e-12, 1.0))
        return self.transform_logits(logits, self.T)


# =========================================================
# 4) Conformal APS
# =========================================================

class ConformalAPS:
    """
    APS:
      score_i = sum_{k: p_k >= p_{y_i}} p_k  (rank-based cumulative mass)
    qhat = (1-alpha) quantile of scores
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.qhat: Optional[float] = None

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalAPS":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int).reshape(-1)

        scores = []
        for i in range(len(y_cal)):
            pi = p_cal[i]
            yi = int(y_cal[i])
            thr = pi[yi]
            score = float(np.sum(pi[pi >= thr]))
            scores.append(score)

        scores = np.asarray(scores, dtype=float)
        n = len(scores)
        k = min(max(int(np.ceil((n + 1) * (1 - self.alpha))) - 1, 0), n - 1)
        self.qhat = float(np.sort(scores)[k])
        return self

    def predict_set(self, p: np.ndarray) -> List[int]:
        p = np.asarray(p, dtype=float).reshape(-1)
        if self.qhat is None:
            return [int(np.argmax(p))]

        order = np.argsort(-p)
        cum = 0.0
        out: List[int] = []
        for idx in order:
            out.append(int(idx))
            cum += float(p[idx])
            if cum >= float(self.qhat):
                break
        return out if out else [int(np.argmax(p))]


# =========================================================
# 5) Abstention + selective risk
# =========================================================

def abstain_decision_multiclass(
    p: np.ndarray,
    epistemic: float,
    tau_risk: float = 0.3,
    tau_epi: float = 0.5,
) -> bool:
    """
    risk ≈ 1 - max_k p_k
    + epistemic variance threshold
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    risk = 1.0 - float(np.max(p))

    if risk > float(tau_risk):
        return True
    if float(epistemic) > float(tau_epi):
        return True
    return False


def selective_risk(
    p: np.ndarray,
    y: np.ndarray,
    abstain_mask: np.ndarray
) -> Dict[str, float]:
    """
    coverage = 未拒判比例
    selective_risk = 在未拒判样本上的错误率
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    abstain_mask = np.asarray(abstain_mask).astype(bool)

    keep = ~abstain_mask
    if np.sum(keep) == 0:
        return {"coverage": 0.0, "selective_risk": 0.0}

    pred = np.argmax(p[keep], axis=1)
    acc = float(np.mean(pred == y[keep]))
    return {"coverage": float(np.mean(keep)), "selective_risk": float(1.0 - acc)}


# =========================================================
# 6) Training pipeline helper
# =========================================================

def infer_splits(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    60/20/20 split
    """
    idx = np.arange(int(n))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_train = max(1, int(0.6 * n))
    n_cal = max(1, int(0.2 * n))
    tr = idx[:n_train]
    ca = idx[n_train:n_train + n_cal]
    te = idx[n_train + n_cal:]
    return tr, ca, te


def train_bayes_head_pipeline(
    Z: np.ndarray,
    Y_soft: np.ndarray,
    y_hard: np.ndarray,
    seed: int = 42,
    alpha: float = 0.1,
    lambda_reg: float = 1.0,
) -> Tuple[BayesianSoftmax, TemperatureCalibrator, ConformalAPS, Dict[str, Any]]:
    """
    输入：
      Z: (n,d) embedding
      Y_soft: (n,3) soft labels
      y_hard: (n,) hard labels (for calibration/conformal)
    输出：
      head, calibrator, conformal, meta
    """
    Z = np.asarray(Z, dtype=float)
    Y_soft = np.asarray(Y_soft, dtype=float)
    y_hard = np.asarray(y_hard, dtype=int).reshape(-1)

    if Z.shape[0] != Y_soft.shape[0] or Z.shape[0] != y_hard.shape[0]:
        raise ValueError("Z, Y_soft, y_hard must have same n")

    tr, ca, te = infer_splits(len(y_hard), seed=int(seed))

    head = BayesianSoftmax(n_classes=3, lambda_reg=float(lambda_reg))
    head.fit_soft(Z[tr], Y_soft[tr])

    # calibration
    cal = TemperatureCalibrator()
    p_cal_raw = head.predict_proba(Z[ca])
    cal.fit(p_cal_raw, y_hard[ca])
    p_cal = cal.transform(p_cal_raw)

    # conformal
    conf = ConformalAPS(alpha=float(alpha))
    conf.fit(p_cal, y_hard[ca])

    # evaluate
    if len(te) > 0:
        p_test = cal.transform(head.predict_proba(Z[te]))
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
        "T": float(cal.T),
        "qhat": float(conf.qhat) if conf.qhat is not None else None,
        "acc": float(np.mean(np.argmax(p_test, axis=1) == y_test)),
        "nll": float(nll_multiclass(p_test, y_test)),
        "ece_top": float(ece_toplabel(p_test, y_test)),
        "feature_dim": int(Z.shape[1]),
    }
    return head, cal, conf, meta


# =========================================================
# 7) Inference helper (single sample)
# =========================================================

def predict_with_risk_controls(
    z: np.ndarray,
    head: BayesianSoftmax,
    calibrator: TemperatureCalibrator,
    conformal: ConformalAPS,
    tau_risk: float = 0.3,
    tau_epi: float = 0.5,
) -> Dict[str, Any]:
    """
    单样本推理（embedding z -> 风险输出）
    """
    z = np.asarray(z, dtype=float).reshape(1, -1)

    pred_u = head.predictive_with_uncertainty(z[0])
    p_raw = np.asarray(pred_u["p"], dtype=float)
    epistemic = float(pred_u["epistemic"])
    aleatoric = float(pred_u["aleatoric"])
    entropy = float(pred_u["entropy"])

    p_final = calibrator.transform(p_raw.reshape(1, -1))[0]
    pred_set = conformal.predict_set(p_final) if conformal.qhat is not None else [int(np.argmax(p_final))]

    abstain = abstain_decision_multiclass(
        p=p_final,
        epistemic=epistemic,
        tau_risk=float(tau_risk),
        tau_epi=float(tau_epi),
    )

    y_pred = int(np.argmax(p_final))
    p_abn = float(p_final[1] + p_final[2])
    p_sev = float(p_final[2])

    return {
        "p_raw": p_raw,
        "p_final": p_final,
        "pred": y_pred,
        "pred_set": pred_set,
        "p_abnormal": p_abn,
        "p_severe": p_sev,
        "abstain": bool(abstain),
        "entropy": entropy,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
    }
