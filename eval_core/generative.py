from __future__ import annotations
from typing import Dict, Any
import numpy as np
import math

from core import (
    NeedInfo,
    QualityVector,
    extract_indicator_vector,
    sigmoid,
    entropy_bernoulli,
)


# =========================================================
# 1️⃣  Generative Model Definition
# =========================================================

class GenerativeRiskModel:
    """
    Explicit latent model:

        Z ~ Bernoulli(pi)

        x_i | Z,Q ~ Bernoulli(
            sigmoid(alpha_i * Z + beta_i * Q_severity + gamma_i)
        )

    """

    def __init__(
        self,
        pi: float = 0.2,
        alpha_scale: float = 1.2,
        beta_scale: float = 1.5,
        gamma_bias: float = -1.0,
    ):
        self.pi = float(pi)
        self.alpha_scale = float(alpha_scale)
        self.beta_scale = float(beta_scale)
        self.gamma_bias = float(gamma_bias)

    # -----------------------------------------------------
    # Parameter generator (simple structured parameterization)
    # -----------------------------------------------------

    def _generate_parameters(self, dim: int):
        """
        Structured parameters (paper-grade would learn these).
        Here we create interpretable parameter structure.
        """

        alpha = np.full(dim, self.alpha_scale)     # Z influence
        beta = np.full(dim, self.beta_scale)       # Q influence
        gamma = np.full(dim, self.gamma_bias)      # baseline bias

        return alpha, beta, gamma

    # -----------------------------------------------------
    # Likelihood: P(X | Z, Q)
    # -----------------------------------------------------

    def likelihood_log_prob(
        self,
        x: np.ndarray,
        Z: int,
        Q: QualityVector
    ) -> float:

        dim = len(x)
        alpha, beta, gamma = self._generate_parameters(dim)

        Qs = float(Q.severity())

        logp = 0.0
        for i in range(dim):
            p_i = sigmoid(alpha[i] * Z + beta[i] * Qs + gamma[i])
            xi = x[i]
            logp += xi * math.log(p_i + 1e-9) + (1 - xi) * math.log(1 - p_i + 1e-9)

        return float(logp)

    # -----------------------------------------------------
    # Joint: log P(Z, X | Q)
    # -----------------------------------------------------

    def joint_log_prob(
        self,
        x: np.ndarray,
        Z: int,
        Q: QualityVector
    ) -> float:

        prior = math.log(self.pi + 1e-9) if Z == 1 else math.log(1 - self.pi + 1e-9)
        likelihood = self.likelihood_log_prob(x, Z, Q)

        return float(prior + likelihood)

    # -----------------------------------------------------
    # Posterior: P(Z=1 | X,Q)
    # -----------------------------------------------------

    def posterior(
        self,
        need: NeedInfo,
        Q: QualityVector
    ) -> Dict[str, Any]:

        x = extract_indicator_vector(need)

        logp1 = self.joint_log_prob(x, Z=1, Q=Q)
        logp0 = self.joint_log_prob(x, Z=0, Q=Q)

        m = max(logp1, logp0)
        p1 = math.exp(logp1 - m)
        p0 = math.exp(logp0 - m)

        posterior_prob = p1 / (p1 + p0 + 1e-9)

        entropy = entropy_bernoulli(posterior_prob)

        return {
            "p_high": float(posterior_prob),
            "entropy": float(entropy),
            "log_joint_high": float(logp1),
            "log_joint_low": float(logp0),
            "dimension": int(len(x)),
        }
