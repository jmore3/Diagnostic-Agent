import numpy as np
from src.diagnostic_agent import inference_by_enumeration

def compute_all_marginals(joint_prob):
    """
    Compute marginal distributions for each variable in the joint.
    Returns a dict:
      {
        'Symptom1': array([P=0, P=1]),
        ...,
        'Condition': array([P=0, P=1, P=2, P=3])
      }
    """
    names = ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Condition']
    marginals = {}

    for i, name in enumerate(names):
        # build query vector: -2 for query var, -1 for hidden
        query = [-1] * 5
        query[i] = -2
        dist = inference_by_enumeration(joint_prob, query)
        # sanity check: should sum to 1
        assert np.isclose(np.sum(dist), 1.0), f"Marginal of {name} sums to {dist.sum()}"
        marginals[name] = dist

    return marginals

