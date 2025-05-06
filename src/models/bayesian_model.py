import numpy as np
import pandas as pd
from itertools import product
from copy import deepcopy


class BayesianDiagnosticModel:
    def __init__(self, symptom_dims=(2, 2, 2, 2), condition_dim=4):
        self.symptom_dims = symptom_dims
        self.condition_dim = condition_dim
        self.joint_prob = None
        self.nb_cond_probs = None
        self.nb_prior = None

    def fit_joint(self, data: np.ndarray):
        """
        data: array of shape (N, 5): cols [s1, s2, s3, s4, cond]
        Learns full joint P(S1,…,S4,C).
        """
        counts = np.zeros(self.symptom_dims + (self.condition_dim,), dtype=float)
        for s1, s2, s3, s4, c in data:
            counts[s1, s2, s3, s4, c] += 1
        self.joint_prob = counts / counts.sum()

    def inference_by_enumeration(self, query_index: int, evidence: dict):
        """
        Returns P(query_var | evidence).

        query_index: 0–3 for Symptom1–4, 4 for Condition
        evidence: dict var_index->value, e.g. {0:1, 2:0}
        """
        assert self.joint_prob is not None, "Call fit_joint first"

        # determine domain sizes
        domain_sizes = list(self.symptom_dims) + [self.condition_dim]
        # prepare unnormalized posterior
        Q = domain_sizes[query_index]
        unnorm = np.zeros(Q, dtype=float)

        # iterate over all joint cells
        # using np.ndindex is a bit cleaner than nested loops
        for idx in np.ndindex(*domain_sizes):
            if any(idx[var] != val for var, val in evidence.items()):
                continue
            unnorm[idx[query_index]] += self.joint_prob[idx]

        total = unnorm.sum()
        if total == 0:
            # no support - return uniform or zeros
            return np.zeros_like(unnorm)
        return unnorm / total

    def marginal(self, var_index: int):
        """
        Compute P(var) by enumeration with no evidence.
        var_index: 0–3 for symptoms, 4 for condition.
        """
        return self.inference_by_enumeration(var_index, evidence={})

    # ---- Naive Bayes alternative ----
    def fit_naive_bayes(self, data: np.ndarray, alpha: float = 1.0):
        """
        Learns P(condition) and P(symptom_i | condition) with Laplace smoothing.
        data: same format as fit_joint.
        alpha: smoothing strength.
        """
        # separate inputs
        symptoms = data[:, :4].astype(int)
        conditions = data[:, 4].astype(int)

        # prior P(condition)
        cond_counts = np.bincount(conditions, minlength=self.condition_dim) + alpha
        self.nb_prior = cond_counts / cond_counts.sum()

        # conditional P(symptom_i = 1 | condition = c) for each symptom
        self.nb_cond_probs = []
        for i in range(4):
            counts = np.zeros(self.condition_dim, dtype=float)

            for c in range(self.condition_dim):
                counts[c] = alpha + np.sum(symptoms[conditions == c, i])

            # total for each c = (#cases with cond=c) + 2*alpha (binary feature)
            totals = cond_counts + 2 * alpha
            self.nb_cond_probs.append(counts / totals)

    def naive_bayes_inference(self, evidence: dict):
        """
        P(condition | evidence) under NB assumption.
        evidence: dict of symptom_index -> 0/1.
        """
        assert self.nb_prior is not None, "Call fit_naive_bayes first"
        log_joint = np.log(self.nb_prior.copy())
        # for each piece of evidence multiply in P(symptom | cond)
        for i, val in evidence.items():
            p_i = self.nb_cond_probs[i]
            log_joint += np.log(p_i if val == 1 else (1 - p_i))

        # normalize
        joint = np.exp(log_joint - log_joint.max())  # for numeric stability
        return joint / joint.sum()

    # Utility methods
    @staticmethod
    def load_csv(filepath: str) -> np.ndarray:
        """
        Reads a CSV with columns [Symptom1..Symptom4,Condition] into a numpy array.
        """
        df = pd.read_csv(filepath)
        return df[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Condition']].values.astype(int)

    @staticmethod
    def compute_all_marginals(model):
        """
        Given an instance of this model with joint_prob fitted,
        returns a dict of marginals for each variable.
        """
        names = ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Condition']
        return {name: model.marginal(i) for i, name in enumerate(names)}