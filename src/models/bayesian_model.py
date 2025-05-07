"""
BayesianDiagnosticModel: A class for performing Bayesian diagnostic inference.

This class supports:
  - Full joint probability model estimation (with Laplace smoothing)
  - Exact inference by enumeration
  - Naïve Bayes parameter estimation and inference
  - Marginal distribution computation
"""
import numpy as np
import pandas as pd

class BayesianDiagnosticModel:
    """
    A model for diagnosing conditions given binary symptoms.
    Implements both full joint and Naïve Bayes approaches.
    """

    @staticmethod
    def load_csv(filepath):
        """
        Load normalized symptoms and condition codes from CSV.
        Returns an (N,5) numpy array with columns [S1,S2,S3,S4,Condition].
        """
        df = pd.read_csv(filepath)
        return df[["Symptom1","Symptom2","Symptom3","Symptom4","Condition"]].values

    def __init__(self):
        # Full joint distribution
        self.joint_prob = None
        # Naive Bayes parameters
        self.nb_prior = None        # shape (C,)
        self.nb_likelihoods = None  # shape (4,2,C)

    def fit_joint(self, data, alpha=0):
        """
        Estimate the full joint P(S1,S2,S3,S4,Condition) with optional Laplace smoothing.
        data: numpy array of shape (N,5)
        alpha: Laplace smoothing constant
        """
        # infer number of conditions
        C = int(data[:, 4].max()) + 1
        # initialize counts (with smoothing)
        counts = np.ones((2, 2, 2, 2, C)) * alpha
        # tally frequencies
        for s1, s2, s3, s4, c in data:
            counts[int(s1), int(s2), int(s3), int(s4), int(c)] += 1
        # normalize
        self.joint_prob = counts / counts.sum()

    def inference_by_enumeration(self, query_index, evidence):
        """
        Perform exact inference by enumeration using the full joint.
        query_index: int, variable to query (0-4)
        evidence: dict mapping var_index -> observed value
        Returns posterior array of length C
        """
        # build query vector
        qv = [-1] * 5
        for idx, val in evidence.items():
            qv[idx] = val
        qv[query_index] = -2
        # delegate to diagnostic_agent's function
        from src.diagnostic_agent import inference_by_enumeration
        return inference_by_enumeration(self.joint_prob, qv)

    def fit_naive_bayes(self, data, alpha=1.0):
        """
        Estimate Naive Bayes parameters: P(Condition) and P(Symptom|Condition).
        alpha: Laplace smoothing for prior and likelihoods
        """
        # infer number of conditions
        C = int(data[:, 4].max()) + 1
        # prior counts with smoothing
        class_counts = np.zeros(C) + alpha * C
        for _, _, _, _, c in data:
            class_counts[int(c)] += 1
        self.nb_prior = class_counts / class_counts.sum()
        # likelihood counts
        likes = np.zeros((4, 2, C)) + alpha * 2
        for s1, s2, s3, s4, c in data:
            for idx, val in enumerate([s1, s2, s3, s4]):
                likes[idx, int(val), int(c)] += 1
        # normalize P(S_i | c)
        for c in range(C):
            for idx in range(4):
                likes[idx, :, c] /= likes[idx, :, c].sum()
        self.nb_likelihoods = likes

    def naive_bayes_inference(self, evidence):
        """
        Compute Naive Bayes posterior P(Condition | evidence).
        evidence: dict mapping symptom_index -> observed 0/1
        """
        C = self.nb_prior.shape[0]
        posterior = np.array(self.nb_prior, copy=True)
        for c in range(C):
            for idx, val in evidence.items():
                posterior[c] *= self.nb_likelihoods[idx, int(val), c]
        return posterior / posterior.sum()

    def compute_all_marginals(self):
        """
        Compute marginal distributions for all variables using enumeration.
        Returns dict var_name -> array
        """
        names = ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Condition']
        from src.diagnostic_agent import inference_by_enumeration
        marginals = {}
        for i, name in enumerate(names):
            qv = [-1] * 5
            qv[i] = -2
            marginals[name] = inference_by_enumeration(self.joint_prob, qv)
        return marginals
