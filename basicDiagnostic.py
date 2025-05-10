#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#  Locate CSV next to this script
here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, 'Health_Data_Set.csv')

#  Load & rename columns
df = pd.read_csv(csv_path, header=0)
df = df.rename(columns={
    'Symptom 1': 'S0',
    'Symptom 2': 'S1',
    'Symptom 3': 'S2',
    'Symptom 4': 'S3'
}).astype(int)

#  Build joint distributions
joint_counts = np.zeros((2,2,2,2,4), dtype=float)
for s0, s1, s2, s3, cond in df[['S0','S1','S2','S3','Condition']].itertuples(index=False):
    joint_counts[s0, s1, s2, s3, cond] += 1
joint = joint_counts / joint_counts.sum()

#  Inference by enumeration
def enumeration_ask(evidence):
    """
    evidence: list of length 5
      -2 at the query variable index
      -1 for hidden variables
      0/1 for known symptoms
      0–3 for known condition
    """
    ev = list(evidence)
    qvar = ev.index(-2)
    domain = joint.shape[qvar]
    Q = np.zeros(domain, dtype=float)

    for qval in range(domain):
        ev[qvar] = qval
        total = 0.0
        # sum over all assignments consistent with evidence
        for a0 in range(2):
            for a1 in range(2):
                for a2 in range(2):
                    for a3 in range(2):
                        for a4 in range(4):
                            idx = [a0, a1, a2, a3, a4]
                            if all(ev[k] < 0 or ev[k] == idx[k] for k in range(5)):
                                total += joint[tuple(idx)]
        Q[qval] = total

    return Q / Q.sum()

# Compute and print marginals
vars_ = ['S0','S1','S2','S3','Condition']
margs = {}
for vidx, name in enumerate(vars_):
    ev = [-1] * 5
    ev[vidx] = -2
    dist = enumeration_ask(ev)
    margs[name] = dist
    print(f"{name} marginal: {np.round(dist,4)}")

# Plot marginals for visualization
for vidx, name in enumerate(vars_):
    dist = margs[name]
    labels = ['Absent','Present'] if vidx < 4 else ['0','1','2','3']
    plt.figure()
    plt.bar(range(len(dist)), dist)
    plt.xticks(range(len(dist)), labels)
    plt.xlabel(name)
    plt.ylabel("Probability")
    plt.title(f"Marginal of {name}")
    plt.tight_layout()
    plt.show()

#   Enumeration‐based posterior (cached) + ROC graph 
#   P(disease) for each of the 16 symptom combinations
combos = df[['S0','S1','S2','S3']].drop_duplicates().reset_index(drop=True)
post_cache = {}
for s0, s1, s2, s3 in combos.itertuples(index=False):
    ev = [s0, s1, s2, s3, -2]
    post = enumeration_ask(ev)
    post_cache[(s0,s1,s2,s3)] = post[1:].sum()  # sum P(cond=1,2,3)

#   Map back to full DataFrame
df['DiseaseProb'] = df.apply(lambda r: post_cache[(r.S0,r.S1,r.S2,r.S3)], axis=1)
df['Label']       = (df.Condition > 0).astype(int)

#   Print first 10
print("\nEnumeration-based posteriors:")
print(df[['S0','S1','S2','S3','Condition','DiseaseProb']].head(10).to_string(index=False))

#   ROC curve
fpr, tpr, _ = roc_curve(df.Label, df.DiseaseProb)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], '--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC: Disease vs. No Disease")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#  Personal combination examples
print("\nExample inferences for selected symptom combinations:")
examples = [
    [1, 0, -1, -1, -2],  # S0=1, S1=0, S2/S3 unknown
    [0, -1, 1, -1, -2],  # S0=0, S2=1, S1/S3 unknown
    [-1, 1, 1, 0,  -2],  # S1=1, S2=1, S3=0, S0 unknown
]
for ev in examples:
    post = enumeration_ask(ev)
    print(f"  P(Condition | {ev[:-1]}) = {np.round(post,4)}")
