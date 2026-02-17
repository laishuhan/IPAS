# === models.py ===
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Callable


# =========================================================
# 1. 基础数据架构与质量向量
# =========================================================

@dataclass
class GlobalInfo:
    sex: int = -1
    age: int = -1
    period_info: Any = None
    preg_info: Any = None


@dataclass
class SubReport:
    report_type: int
    status_codes: List[int]


@dataclass
class NeedInfo:
    global_info: GlobalInfo
    reports: List[SubReport] = field(default_factory=list)

    @staticmethod
    def from_legacy(legacy: Sequence[Any]) -> "NeedInfo":
        sex, age, period_info, preg_info = -1, -1, None, None
        reports: List[SubReport] = []

        if legacy:
            gi = legacy[0]
            if isinstance(gi, (list, tuple)):
                if len(gi) >= 1: sex = gi[0]
                if len(gi) >= 2: age = gi[1]
                if len(gi) >= 3: period_info = gi[2]
                if len(gi) >= 4: preg_info = gi[3]

            for sr in legacy[1:]:
                if not isinstance(sr, (list, tuple)) or len(sr) < 2:
                    continue
                r_type = int(sr[0])
                codes = list(map(int, sr[1])) if isinstance(sr[1], (list, tuple)) else []
                reports.append(SubReport(r_type, codes))

        return NeedInfo(
            global_info=GlobalInfo(int(sex), int(age), period_info, preg_info),
            reports=reports,
        )


@dataclass
class QualityVector:
    ocr_missing_rate: float = 0.0
    match_abs_diff_rate: float = 0.0
    image_insufficient: float = 0.0
    plus_exist_flag: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.ocr_missing_rate,
            self.match_abs_diff_rate,
            self.image_insufficient,
            self.plus_exist_flag,
        ], dtype=float)

    def severity(self) -> float:
        return float(
            0.40 * self.ocr_missing_rate +
            0.35 * self.match_abs_diff_rate +
            0.20 * self.image_insufficient +
            0.05 * self.plus_exist_flag
        )


# =========================================================
# 2. 特征提取与数学工具
# =========================================================

_FEATURE_ORDER = [
    "gen_p_high", "gen_entropy", "gen_dimension",
    "abnormal_ratio", "num_reports", "num_codes", "missing_rate_codes",
    "Q_severity", "Q_ocr_missing_rate", "Q_match_abs_diff_rate",
    "Q_image_insufficient", "Q_plus_exist_flag", "bias",
]

def sigmoid(z: Any) -> Any:
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_categorical(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def entropy_bernoulli(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

def extract_indicator_vector(need: NeedInfo) -> np.ndarray:
    vec = []
    for r in need.reports:
        for c in r.status_codes:
            c = int(c)
            vec.append(0.0 if c in (-1, 0, 9) else 1.0)
    return np.array(vec, dtype=float) if vec else np.zeros(1, dtype=float)

def abnormal_ratio(need: NeedInfo) -> float:
    total, abnormal = 0, 0
    for r in need.reports:
        for c in r.status_codes:
            c = int(c)
            if c == -1: continue
            total += 1
            if c not in (0, 9): abnormal += 1
    return float(abnormal / total) if total > 0 else 0.0

def build_context_from_processed(processed: Dict[str, Any]) -> Dict[str, Any]:
    processed = processed or {}
    ctx: Dict[str, Any] = {}
    for k in ("ocr_missing_rate", "match_abs_diff_rate", "image_insufficient", "plus_exist_flag"):
        if k in processed: ctx[k] = processed.get(k)
    if "image_insufficient" not in ctx:
        t = float(processed.get("total_img_number", 0) or 0)
        s = float(processed.get("solvable_img_number", 0) or 0)
        ctx["image_insufficient"] = 1.0 if (t > 0 and s <= 0) else 0.0
    return ctx

def extract_quality(need: NeedInfo, context: Dict[str, Any]) -> QualityVector:
    context = context or {}
    ocr = float(context.get("ocr_missing_rate", 0.0) or 0.0)
    if ocr == 0.0:
        flat_codes = [int(c) for r in need.reports for c in r.status_codes]
        if flat_codes: ocr = flat_codes.count(-1) / len(flat_codes)
    return QualityVector(
        ocr_missing_rate=ocr,
        match_abs_diff_rate=float(context.get("match_abs_diff_rate", 0.0) or 0.0),
        image_insufficient=float(context.get("image_insufficient", 0.0) or 0.0),
        plus_exist_flag=float(context.get("plus_exist_flag", 0.0) or 0.0),
    )


# =========================================================
# 3. 生成式风险模型（保持你原有）
# =========================================================

class GenerativeRiskModel:
    def __init__(self, pi: float = 0.2, alpha_scale: float = 1.2, beta_scale: float = 1.5, gamma_bias: float = -1.0):
        self.pi = pi
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.gamma_bias = gamma_bias

    def likelihood_log_prob(self, x: np.ndarray, Z: int, Q: QualityVector) -> float:
        dim = len(x)
        Qs = Q.severity()
        logp = 0.0
        for i in range(dim):
            p_i = float(sigmoid(self.alpha_scale * Z + self.beta_scale * Qs + self.gamma_bias))
            logp += x[i] * math.log(p_i + 1e-9) + (1 - x[i]) * math.log(1 - p_i + 1e-9)
        return logp

    def posterior(self, need: NeedInfo, Q: QualityVector) -> Dict[str, Any]:
        x = extract_indicator_vector(need)
        logp1 = math.log(self.pi + 1e-9) + self.likelihood_log_prob(x, 1, Q)
        logp0 = math.log(1 - self.pi + 1e-9) + self.likelihood_log_prob(x, 0, Q)
        m = max(logp1, logp0)
        p1, p0 = math.exp(logp1 - m), math.exp(logp0 - m)
        post_p = p1 / (p1 + p0 + 1e-9)
        return {
            "p_high": post_p,
            "entropy": entropy_bernoulli(post_p),
            "dimension": len(x),
        }


# =========================================================
# 4. 弱监督（保持你原有 + 新增多分类 EM Label Model）
# =========================================================

@dataclass
class LabelingFunction:
    name: str
    fn: Callable[[NeedInfo, QualityVector, Dict[str, Any]], int]

def build_default_lfs() -> List[LabelingFunction]:
    def r_high(n, Q, c): return 1 if abnormal_ratio(n) >= 0.45 else (-1 if abnormal_ratio(n) <= 0.05 else 0)
    def img_ins(n, Q, c): return 1 if Q.image_insufficient >= 0.5 else 0
    def q_abst(n, Q, c): return 0
    return [
        LabelingFunction("abnormal_ratio_high", r_high),
        LabelingFunction("image_insufficient_high", img_ins),
        LabelingFunction("quality_abstain", q_abst),
    ]

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
    """
    n_classes: int = 3
    max_iter: int = 100
    tol: float = 1e-6
    seed: int = 42
    min_prob: float = 1e-6

    pi_: Optional[np.ndarray] = None        # (K,)
    A_: Optional[np.ndarray] = None         # (m, K, K): A[j, v, k]
    q_abstain_: Optional[np.ndarray] = None # (m,)

    def fit(self, L: np.ndarray) -> "LabelModelEMMulticlass":
        rng = np.random.default_rng(self.seed)
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        K = int(self.n_classes)

        # init class prior
        pi = np.ones(K, dtype=float) / K

        # abstain rate per LF
        q = np.clip(np.mean(L == -1, axis=0).astype(float), 1e-3, 1 - 1e-3)

        # init A close to diagonal
        A = np.zeros((m, K, K), dtype=float)
        for j in range(m):
            A[j] = 0.10 / max(1, K - 1)
            np.fill_diagonal(A[j], 0.90)
            # ensure normalization over votes v for each class k
            A[j] = A[j] / (np.sum(A[j], axis=0, keepdims=True) + 1e-12)

        prev_ll = None

        for _ in range(int(self.max_iter)):
            # -----------------
            # E-step: posterior
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
                    # safety: clip invalid votes to abstain
                    invalid = (v < 0) | (v >= K)
                    if np.any(invalid):
                        # treat invalid as abstain
                        idx_invalid = np.where(non)[0][invalid]
                        logp[idx_invalid] += np.log(q[j] + 1e-12)
                        idx_valid = np.where(non)[0][~invalid]
                        v_valid = v[~invalid]
                        if len(idx_valid) > 0:
                            logp[idx_valid] += np.log(1 - q[j] + 1e-12)
                            logp[idx_valid] += np.log(A[j, v_valid, :].T + 1e-12).T
                    else:
                        logp[non] += np.log(1 - q[j] + 1e-12)
                        logp[non] += np.log(A[j, v, :].T + 1e-12).T

            mx = np.max(logp, axis=1, keepdims=True)
            p = np.exp(logp - mx)
            p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)

            # log-likelihood
            ll = float(np.sum(mx.squeeze() + np.log(np.sum(np.exp(logp - mx), axis=1) + 1e-12)))
            if prev_ll is not None and abs(ll - prev_ll) < float(self.tol):
                break
            prev_ll = ll

            # -----------------
            # M-step
            # -----------------
            pi = np.clip(np.mean(p, axis=0), self.min_prob, 1.0)
            pi = pi / pi.sum()

            q = np.clip(np.mean(L == -1, axis=0).astype(float), 1e-3, 1 - 1e-3)

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

        self.pi_, self.A_, self.q_abstain_ = pi, A, q
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.pi_ is None or self.A_ is None or self.q_abstain_ is None:
            raise RuntimeError("LabelModelEMMulticlass not fit")
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        K = int(self.n_classes)

        logp = np.log(self.pi_ + 1e-12)[None, :].repeat(n, axis=0)
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
# 5. 贝叶斯多分类 Softmax 回归（新增：三分类用这个）
# =========================================================

class BayesianSoftmax:
    """
    多分类 softmax 回归 + Laplace 近似
    - 拟合：MAP（L2 正则）
    - 协方差：H^{-1}（Hessian）
    不确定性（简单可用版）：
    - epistemic: 使用 logit 的方差（delta 近似）
    - aleatoric: 1 - sum(p^2)（类别内在不确定）
    """
    def __init__(self, n_classes: int = 3, lambda_reg: float = 1.0):
        self.n_classes = int(n_classes)
        self.lambda_reg = float(lambda_reg)
        self.W_map: Optional[np.ndarray] = None     # (d, K)
        self.Sigma: Optional[np.ndarray] = None     # ((dK),(dK))

    def fit_soft(self, X: np.ndarray, Y_soft: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> "BayesianSoftmax":
        """
        Same as fit(), but uses soft targets Y_soft (n,K) where each row is a probability simplex.
        """
        X = np.asarray(X, dtype=float)
        Y_soft = np.asarray(Y_soft, dtype=float)
        n, d = X.shape
        K = self.n_classes
        if Y_soft.shape != (n, K):
            raise ValueError(f"Y_soft shape {Y_soft.shape} != (n={n},K={K})")
        # normalize for safety
        Y_soft = np.clip(Y_soft, 1e-12, 1.0)
        Y_soft = Y_soft / (np.sum(Y_soft, axis=1, keepdims=True) + 1e-12)

        W = np.zeros((d, K), dtype=float)
        I = np.eye(d * K, dtype=float)

        for _ in range(max_iter):
            logits = X @ W
            P = softmax(logits, axis=1)

            G = X.T @ (P - Y_soft) + self.lambda_reg * W
            g = G.reshape(-1)

            H = np.zeros((d * K, d * K), dtype=float)
            for i in range(n):
                xi = X[i].reshape(d, 1)
                Si = np.diag(P[i]) - np.outer(P[i], P[i])
                H += np.kron(Si, (xi @ xi.T))
            H += self.lambda_reg * I

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, g, rcond=None)[0]

            W_new = (W.reshape(-1) - delta).reshape(d, K)

            if np.linalg.norm(W_new - W) < tol:
                W = W_new
                self.W_map = W
                self.Sigma = np.linalg.pinv(H)
                return self

            W = W_new

        self.W_map = W
        self.Sigma = np.linalg.pinv(H)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W_map is None:
            raise RuntimeError("Model not fit")
        X = np.asarray(X, dtype=float)
        logits = X @ self.W_map
        return softmax(logits, axis=1)

    def predictive_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """
        返回：
          p: (K,)
          pred: argmax
          entropy
          aleatoric: 1 - sum(p^2)
          epistemic: 基于 logit 方差的粗略量（可做拒判用）
        """
        if self.W_map is None:
            raise RuntimeError("Model not fit")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        p = self.predict_proba(x)[0]
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
            "entropy": ent,
            "aleatoric": ale,
            "epistemic": epi,
        }
