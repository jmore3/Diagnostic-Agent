import numpy as np
import pandas as pd
from itertools import product

class BayesianDiagnosticModel:
    def __init__(self, symptom_dims=None, condition_dim=None):
        """
        symptom_dims: tuple of cardinalities for each feature (if None, inferred on fit_joint)
        condition_dim: number of classes (if None, inferred on fit_joint)
        """
        self.symptom_dims = symptom_dims
        self.condition_dim = condition_dim
        self.joint_prob = None
        self.nb_prior = None
        self.nb_cond_probs = None

    def fit_joint(self, data: np.ndarray):
        """
        Learns full joint P(X1,…,Xk,C).
        data: array shape (N, k+1) with last column = condition.
        """
        # infer dims if not set
        if self.symptom_dims is None or self.condition_dim is None:
            maxes = data.max(axis=0).astype(int)
            self.symptom_dims = tuple(maxes[:-1] + 1)
            self.condition_dim = int(maxes[-1] + 1)

        # initialize count array
        counts = np.zeros(self.symptom_dims + (self.condition_dim,), dtype=float)
        for row in data.astype(int):
            *xs, c = row
            idx = tuple(xs) + (c,)
            counts[idx] += 1

        self.joint_prob = counts / counts.sum()

    def inference_by_enumeration(self, query_index: int, evidence: dict):
        """
        P(query_var | evidence) by summing out hidden vars.
        query_index: 0..k-1 for features, k for condition.
        evidence: dict var_index->value.
        """
        assert self.joint_prob is not None, "Call fit_joint first"
        domain = list(self.symptom_dims) + [self.condition_dim]
        Q = domain[query_index]
        unnorm = np.zeros(Q, dtype=float)
        for idx in np.ndindex(*domain):
            if any(idx[i] != v for i, v in evidence.items()):
                continue
            unnorm[idx[query_index]] += self.joint_prob[idx]
        total = unnorm.sum()
        return unnorm / total if total > 0 else np.zeros_like(unnorm)

    def marginal(self, var_index: int):
        """P(var_index) with no evidence."""
        return self.inference_by_enumeration(var_index, {})

    def fit_naive_bayes(self, data: np.ndarray, alpha: float = 1.0):
        """
        Learns P(C) and P(X_i = v | C) for categorical features.
        data: (N, k+1) array; last column is condition.
        """
        X = data[:, :len(self.symptom_dims)].astype(int)
        y = data[:, -1].astype(int)

        # prior
        cond_counts = np.bincount(y, minlength=self.condition_dim) + alpha
        self.nb_prior = cond_counts / cond_counts.sum()

        # conditional
        self.nb_cond_probs = []
        for i, dim in enumerate(self.symptom_dims):
            counts = np.zeros((dim, self.condition_dim), dtype=float)
            for c in range(self.condition_dim):
                hist = np.bincount(X[y == c, i], minlength=dim)
                counts[:, c] = alpha + hist
            totals = cond_counts + alpha * dim
            self.nb_cond_probs.append(counts / totals[np.newaxis, :])

    def naive_bayes_inference(self, evidence: dict):
        """
        P(C | evidence) under NB assumption.
        evidence: dict feature_index -> observed value.
        """
        assert self.nb_prior is not None, "Call fit_naive_bayes first"
        logp = np.log(self.nb_prior.copy())
        for i, v in evidence.items():
            logp += np.log(self.nb_cond_probs[i][v])
        exp = np.exp(logp - np.max(logp))
        return exp / exp.sum()

    @staticmethod
    def load_csv(filepath: str) -> np.ndarray:
        df = pd.read_csv(filepath)
        cols = df.columns.tolist()
        return df[cols].values.astype(int)

    @staticmethod
    def compute_all_marginals(model):
        k = len(model.symptom_dims)
        names = [f"Symptom{i+1}" for i in range(k)] + ["Condition"]
        return {n: model.marginal(i) for i, n in enumerate(names)}