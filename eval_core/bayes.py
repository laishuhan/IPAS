from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from core import sigmoid

def build_feature_vector(gen_out: Dict[str, Any], q_severity: float) -> np.ndarray:
    """对齐 experiments.py 的特征构建工具"""
    from core import _FEATURE_ORDER
    # 创建一个哑特征字典并填充关键字段
    f = {k: 0.0 for k in _FEATURE_ORDER}
    f["gen_p_high"] = gen_out.get("p_high", 0.0)
    f["gen_entropy"] = gen_out.get("entropy", 0.0)
    f["gen_dimension"] = gen_out.get("dimension", 0.0)
    f["Q_severity"] = q_severity
    f["bias"] = 1.0
    return np.array([f[k] for k in _FEATURE_ORDER], dtype=float)



class BayesianLogistic:
    """
    Bayesian Logistic Regression via Laplace approximation around MAP.

    Objective (soft labels):
      min_w  sum_i w_i * [ -y_i log σ(x_i^T w) - (1-y_i) log(1-σ(x_i^T w)) ] + 0.5*lambda*||w||^2

    where y_i ∈ [0,1], sample_weight w_i >= 0.
    """

    def __init__(self, lambda_reg: float = 1.0):
        self.lambda_reg = float(lambda_reg)
        self.w_map: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        max_iter: int = 25,
        tol: float = 1e-6,
    ) -> "BayesianLogistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        n, d = X.shape
        if len(y) != n:
            raise ValueError("y length must match X rows")

        if sample_weight is None:
            sw = np.ones(n, dtype=float)
        else:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sw) != n:
                raise ValueError("sample_weight length must match X rows")
            sw = np.clip(sw, 0.0, 1e12)

        # init
        w = np.zeros(d, dtype=float)

        # Newton iterations
        H = self.lambda_reg * np.eye(d, dtype=float)

        for _ in range(int(max_iter)):
            z = X @ w
            p = sigmoid(z)  # vectorized via core.sigmoid which supports float -> we implement vectorization below
            if not isinstance(p, np.ndarray):
                # core.sigmoid returns float; make vectorized here
                p = 1.0 / (1.0 + np.exp(-z))

            # gradient: X^T (sw*(p-y)) + lambda*w
            r = sw * (p - y)
            grad = X.T @ r + self.lambda_reg * w

            # Hessian: X^T diag(sw*p*(1-p)) X + lambda I
            s = sw * (p * (1.0 - p))
            # avoid forming huge diag for large n: scale rows
            Xs = X * s[:, None]
            H = X.T @ Xs + self.lambda_reg * np.eye(d, dtype=float)

            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break

            w -= delta

            if float(np.linalg.norm(delta)) < tol:
                break

        self.w_map = w

        # Laplace covariance
        try:
            self.Sigma = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.Sigma = np.linalg.pinv(H)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Batch predict probabilities.
        X: (n,d) or (d,)
        returns: (n,)
        """
        if self.w_map is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        z = X @ self.w_map
        p = 1.0 / (1.0 + np.exp(-z))
        return p.reshape(-1)

    def predictive_variance(self, phi: np.ndarray) -> float:
        """
        Var_w [ φ^T w ] ≈ φ^T Σ φ
        """
        if self.Sigma is None:
            return 0.0
        phi = np.asarray(phi, dtype=float).reshape(-1)
        return float(phi @ self.Sigma @ phi)

    def predictive_with_uncertainty(self, phi: np.ndarray) -> Dict[str, Any]:
        """
        For a single feature vector phi (d,)
        """
        phi = np.asarray(phi, dtype=float).reshape(-1)
        p = float(self.predict_proba(phi)[0])

        # aleatoric (Bernoulli variance)
        aleatoric = p * (1.0 - p)

        # epistemic: linearized variance in logit space
        epistemic = self.predictive_variance(phi)

        total_var = aleatoric + epistemic

        return {
            "p_mean": float(p),
            "aleatoric": float(aleatoric),
            "epistemic": float(epistemic),
            "total_variance": float(total_var),
        }
