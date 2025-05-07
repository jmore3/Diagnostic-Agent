# test_diagnostic_agent.py
import numpy as np
import pytest
from src.diagnostic_agent import build_joint_probability, inference_by_enumeration

# A tiny hand-crafted dataset
# Two records: same symptoms, two different conditions (0 and 1)
S1 = [1, 1]
S2 = [0, 0]
S3 = [0, 0]
S4 = [1, 1]
Cond = [0, 1]
data = np.array(list(zip(S1, S2, S3, S4, Cond)))

def test_build_joint_no_smoothing():
    joint = build_joint_probability(data, num_conditions=2, alpha=0)
    # Check the two observed cells have counts 1 each, so P=1/2 each
    assert pytest.approx(joint[1,0,0,1,0]) == 0.5
    assert pytest.approx(joint[1,0,0,1,1]) == 0.5

def test_build_joint_with_smoothing():
    # With alpha=1, each of the 2x2x2x2x2=32 cells gets +1
    joint = build_joint_probability(data, num_conditions=2, alpha=1)
    total_cells = 32
    total_count = data.shape[0] + total_cells  # 2 records + smoothing
    # The two observed cells have count = 2 now
    assert pytest.approx(joint[1,0,0,1,0]) == 2/total_count
    assert pytest.approx(joint[1,0,0,1,1]) == 2/total_count

def test_inference_simple():
    joint = build_joint_probability(data, num_conditions=2, alpha=0)
    # P(Cond | S1=1,S4=1) should be uniform [0.5,0.5]
    posterior = inference_by_enumeration(joint, [1,-1,-1,1,-2])
    assert posterior.shape == (2,)
    assert pytest.approx(posterior[0]) == 0.5
    assert pytest.approx(posterior[1]) == 0.5

def test_marginal_sums_to_one():
    # Build a random small joint
    random_data = np.random.randint(0,2,size=(10,5))
    joint = build_joint_probability(random_data, num_conditions=2, alpha=1)
    # Check every marginal sums to 1
    for idx in range(5):
        qv = [-1]*5
        qv[idx] = -2
        marg = inference_by_enumeration(joint, qv)
        assert pytest.approx(marg.sum()) == 1.0
