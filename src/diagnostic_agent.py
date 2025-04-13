# 1. Load Data
import pandas as pd
import numpy as np

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

# 2. Build Joint Distribution
def build_joint_probability(data):
    joint_counts = np.zeros((2, 2, 2, 2, 4))
    for row in data:
        s1, s2, s3, s4, condition = row
        joint_counts[s1, s2, s3, s4, condition] += 1
    joint_prob = joint_counts / np.sum(joint_counts)
    return joint_prob

# 3. Inference Engine
def inference_by_enumeration(joint_prob, query_vector):
    query_idx = query_vector.index(-2)
    query_range = range(2) if query_idx < 4 else range(4)
    result = np.zeros(len(query_range))

    for val in query_range:
        query_vector[query_idx] = val
        total = 0
        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    for s4 in range(2):
                        for cond in range(4):
                            config = [s1, s2, s3, s4, cond]
                            match = all(query_vector[i] in [-1, config[i]] for i in range(5))
                            if match:
                                total += joint_prob[s1, s2, s3, s4, cond]
        result[val] = total
    return result / np.sum(result)