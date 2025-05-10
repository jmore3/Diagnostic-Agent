#!/usr/bin/env python3

# SECTION 0: SSL fix for ucimlrepo
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_curve,
    auc,
)
from src.models.bayesian_model import BayesianDiagnosticModel

# SECTION 1: Fetch & normalize heart disease data

heart = fetch_ucirepo(id=45)
var_df = heart.variables
feature_cols = var_df.loc[var_df['role']=='Feature', 'name'].tolist()

df_hd = pd.DataFrame(heart.data.features, columns=feature_cols)
df_hd['num'] = heart.data.targets
df_hd['Condition'] = (df_hd['num'] > 0).astype(int)

symptom_cols = ['sex', 'fbs', 'exang', 'restecg']
normalized_hd = df_hd[symptom_cols + ['Condition']].copy()
normalized_hd.columns = ['Symptom1','Symptom2','Symptom3','Symptom4','Condition']

# Determine project root: go up two levels from src/data/
BASE_DIR = Path(__file__).resolve().parents[2]  # now points to Diagnostic-Agent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

OUT_CSV = DATA_DIR / "normalized_heart_data.csv"
normalized_hd.to_csv(OUT_CSV, index=False)
print(f"→ Saved normalized heart data ({len(normalized_hd)} rows) to {OUT_CSV}")

# SECTION 2: Load & evaluate with BayesianDiagnosticModel

df = pd.read_csv(OUT_CSV)
data = df[['Symptom1','Symptom2','Symptom3','Symptom4','Condition']].values.astype(int)
y = data[:, 4]
labels = [0, 1]

train_idx, test_idx = train_test_split(
    np.arange(len(data)), test_size=61, random_state=42, stratify=y
)
train, test = data[train_idx], data[test_idx]

model = BayesianDiagnosticModel(symptom_dims=(2,2,2,3), condition_dim=2)
model.fit_joint(train)
model.fit_naive_bayes(train, alpha=1.0)

y_true = test[:, 4]
probs_enum, probs_nb = [], []
y_pred_enum, y_pred_nb = [], []

for row in test:
    evidence = {i: int(row[i]) for i in range(4)}

    pe = model.inference_by_enumeration(query_index=4, evidence=evidence)
    if pe.sum() == 0:
        pe = np.ones_like(pe)
    pe /= pe.sum()

    pn = model.naive_bayes_inference(evidence)
    if pn.sum() == 0:
        pn = np.ones_like(pn)
    pn /= pn.sum()

    probs_enum.append(pe)
    probs_nb.append(pn)
    y_pred_enum.append(int(np.argmax(pe)))
    y_pred_nb.append(int(np.argmax(pn)))

probs_enum = np.vstack(probs_enum)
probs_nb   = np.vstack(probs_nb)

# SECTION 3: Compute & print metrics

acc_e = accuracy_score(y_true, y_pred_enum)
acc_nb = accuracy_score(y_true, y_pred_nb)
ll_e  = log_loss(y_true, probs_enum, labels=labels)
ll_nb = log_loss(y_true, probs_nb,   labels=labels)

cm_e = confusion_matrix(y_true, y_pred_enum, labels=labels)
cm_nb = confusion_matrix(y_true, y_pred_nb,   labels=labels)

print("\n=== Quantitative Evaluation (Heart Disease) ===")
print(f"Enumeration  — Accuracy: {acc_e:.3f}, Log-Loss: {ll_e:.3f}")
print(f"Naïve-Bayes  — Accuracy: {acc_nb:.3f}, Log-Loss: {ll_nb:.3f}\n")
print("Confusion Matrix (Enum):\n", cm_e)
print("Confusion Matrix (NB):\n",   cm_nb)

# SECTION 4: ROC Curve — Heart Disease (binary)

scores_enum_pos = probs_enum[:, 1]
scores_nb_pos   = probs_nb[:, 1]

fpr_e, tpr_e, _   = roc_curve(y_true, scores_enum_pos)
fpr_nb, tpr_nb, _ = roc_curve(y_true, scores_nb_pos)
auc_e = auc(fpr_e, tpr_e)
auc_nb= auc(fpr_nb, tpr_nb)

plt.figure(figsize=(8,6))
plt.plot(fpr_e, tpr_e,  label=f'Enum (AUC={auc_e:.2f})')
plt.plot(fpr_nb, tpr_nb,'--',label=f'NB   (AUC={auc_nb:.2f})')
plt.plot([0,1],[0,1],'k:', label='Chance')
plt.title('ROC Curve — Heart Disease')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# SECTION 5: Bar Chart — Heart Disease
labels_plot = ['Enumeration','Naïve-Bayes']
scores_acc     = [acc_e, acc_nb]
scores_logloss = [ll_e, ll_nb]
x = np.arange(len(labels_plot)); w = 0.35

fig, ax = plt.subplots(figsize=(6,4))
r1 = ax.bar(x-w/2, scores_acc,     w, label='Accuracy')
r2 = ax.bar(x+w/2, scores_logloss, w, label='Log-Loss')
ax.set_xticks(x); ax.set_xticklabels(labels_plot)
ax.set_title('Heart Disease: Accuracy vs Log-Loss')
ax.legend()
for r in (r1 + r2):
    h = r.get_height()
    ax.annotate(f'{h:.2f}', xy=(r.get_x()+r.get_width()/2, h),
                xytext=(0,3), textcoords='offset points',
                ha='center')
plt.tight_layout()
plt.show()