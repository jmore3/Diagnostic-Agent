# src/diagnostic_agent.py
import pandas as pd
import numpy as np
from itertools import product


def load_dataset(filepath):
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)


def build_joint_probability(data: np.ndarray,
                            num_conditions: int = None,
                            alpha: float = 0.0) -> np.ndarray:
    """
    Learns the full joint distribution P(X1,X2,...,Xk,C) via enumeration with optional Laplace smoothing.

    Parameters
    ----------
    data : np.ndarray, shape (N, k+1)
        Rows of integer codes for k features followed by the condition code.
    num_conditions : int, optional
        Number of distinct condition codes; if None, inferred as max(data[:,-1]) + 1.
    alpha : float, default=0.0
        Laplace smoothing constant to add to every cell.

    Returns
    -------
    joint_prob : np.ndarray
        Normalized joint probability table of shape (d0, d1, ..., d_{k-1}, num_conditions).
    """
    data = np.asarray(data, dtype=int)
    N, D = data.shape
    k = D - 1  # number of feature columns

    # determine condition dimension
    if num_conditions is None:
        num_conditions = int(data[:, -1].max()) + 1

    # determine feature dimensions
    if k == 4:
        # always treat the first four features as binary
        feature_dims = [2, 2, 2, 2]
    else:
        # infer each dimension from the data
        feature_dims = [int(data[:, i].max()) + 1 for i in range(k)]

    dims = feature_dims + [num_conditions]
    counts = np.full(tuple(dims), fill_value=alpha, dtype=float)

    # tally occurrences
    for row in data:
        *features, cond = row.tolist()
        c = int(cond)
        if 0 <= c < num_conditions:
            idx = tuple(int(f) for f in features) + (c,)
            counts[idx] += 1

    total = counts.sum()
    if total <= 0:
        raise ValueError("No counts in joint distribution; check data or alpha.")
    return counts / total


def inference_by_enumeration(joint_prob: np.ndarray,
                             query_vector: list) -> np.ndarray:
    """
    Exact inference by enumeration on the joint probability table.

    Parameters
    ----------
    joint_prob : np.ndarray
        The joint probability table of arbitrary dims.
    query_vector : list of int
        Length must equal joint_prob.ndim.
        Use -2 for the query variable, -1 for hidden vars, and 0..(dim_i-1) for evidence.

    Returns
    -------
    posterior : np.ndarray
        Normalized posterior distribution for the query variable.
    """
    jp = np.asarray(joint_prob, dtype=float)
    dims = jp.shape
    if len(query_vector) != len(dims):
        raise ValueError("query_vector length must match joint_prob.ndim")

    qidx = query_vector.index(-2)
    qdim = dims[qidx]
    posterior = np.zeros(qdim, dtype=float)

    for qv in range(qdim):
        mask = list(query_vector)
        mask[qidx] = qv
        subtotal = 0.0
        for idx in product(*(range(d) for d in dims)):
            if all((mask[i] == -1 or mask[i] == idx[i]) for i in range(len(dims))):
                subtotal += jp[idx]
        posterior[qv] = subtotal

    # restore placeholder (though original query_vector likely not reused)
    query_vector[qidx] = -2
    norm = posterior.sum()
    return (posterior / norm) if norm > 0 else posterior
