import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from src.models.bayesian_model import BayesianDiagnosticModel


# 1) Load pre-normalized data
df = pd.read_csv('data/normalized_health_data.csv')
# Columns: Symptom1,Symptom2,Symptom3,Symptom4,Condition

# 2) Prepare numpy array and count classes
data = df[['Symptom1','Symptom2','Symptom3','Symptom4','Condition']].values.astype(int)
X, y = data[:, :4], data[:, 4]
n_conditions = y.max() + 1
labels = list(range(n_conditions))

# 3) Stratified 450/250 split
idx = np.arange(len(data))
train_idx, test_idx = train_test_split(
    idx, test_size=250, random_state=42, stratify=y
)
train, test = data[train_idx], data[test_idx]

# 4) Fit both models
model = BayesianDiagnosticModel(symptom_dims=(2,2,2,2), condition_dim=n_conditions)
model.fit_joint(train)
model.fit_naive_bayes(train, alpha=1.0)

# 5) Predict on test
y_true = test[:, 4]
probs_enum, probs_nb = [], []
y_pred_enum, y_pred_nb = [], []

for row in test:
    evidence = {i: int(row[i]) for i in range(4)}
    post_e = model.inference_by_enumeration(query_index=4, evidence=evidence)
    post_nb = model.naive_bayes_inference(evidence)
    probs_enum.append(post_e)
    probs_nb.append(post_nb)
    y_pred_enum.append(int(np.argmax(post_e)))
    y_pred_nb.append(int(np.argmax(post_nb)))

# 6) Compute metrics
acc_e  = accuracy_score(y_true, y_pred_enum)
acc_nb = accuracy_score(y_true, y_pred_nb)
ll_e   = log_loss(y_true, probs_enum, labels=labels)
ll_nb  = log_loss(y_true, probs_nb,   labels=labels)

cm_e = confusion_matrix(y_true, y_pred_enum, labels=labels)
cm_nb = confusion_matrix(y_true, y_pred_nb,   labels=labels)

print("=== Quantitative Evaluation ===")
print(f"Enumeration  — Accuracy: {acc_e:.4f}, Log-Loss: {ll_e:.4f}")
print(f"Naïve-Bayes  — Accuracy: {acc_nb:.4f}, Log-Loss: {ll_nb:.4f}\n")
print("Confusion Matrix (Enumeration):\n", cm_e)
print("Confusion Matrix (Naïve-Bayes):\n",   cm_nb)

# 7) Representative ROC Curves (top 5 + micro/macro averages)
# Binarize true labels
y_true_bin = label_binarize(y_true, classes=labels)

# Stack scores
scores_enum = np.vstack(probs_enum)
scores_nb   = np.vstack(probs_nb)

# 7.1 Top-5 most frequent in TRAIN
train_labels, train_counts = np.unique(train[:,4], return_counts=True)
top5 = train_labels[np.argsort(train_counts)[-5:]][::-1]

# 7.2 Micro-average ROC
fpr_e_micro, tpr_e_micro, _ = roc_curve(y_true_bin.ravel(), scores_enum.ravel())
fpr_nb_micro, tpr_nb_micro, _ = roc_curve(y_true_bin.ravel(), scores_nb.ravel())
auc_e_micro = auc(fpr_e_micro, tpr_e_micro)
auc_nb_micro = auc(fpr_nb_micro, tpr_nb_micro)

# 7.3 Macro-average ROC
all_fpr = np.linspace(0, 1, 100)
mean_tpr_e = np.zeros_like(all_fpr)
mean_tpr_nb = np.zeros_like(all_fpr)
for i in labels:
    y_bin = y_true_bin[:, i]
    if y_bin.sum() < 1 or (y_bin==0).sum() < 1:
        continue
    fpr_e, tpr_e, _  = roc_curve(y_bin, scores_enum[:, i])
    fpr_nb, tpr_nb, _ = roc_curve(y_bin, scores_nb[:, i])
    mean_tpr_e  += np.interp(all_fpr, fpr_e, tpr_e)
    mean_tpr_nb += np.interp(all_fpr, fpr_nb, tpr_nb)
mean_tpr_e  /= len(labels)
mean_tpr_nb /= len(labels)
auc_e_macro = auc(all_fpr, mean_tpr_e)
auc_nb_macro = auc(all_fpr, mean_tpr_nb)

# 7.4 Plot
plt.figure(figsize=(10, 8))

# Micro
plt.plot(fpr_e_micro, tpr_e_micro, 'k--', label=f'Enum micro (AUC={auc_e_micro:.2f})')
plt.plot(fpr_nb_micro, tpr_nb_micro, 'k:',  label=f'NB   micro (AUC={auc_nb_micro:.2f})')

# Macro
plt.plot(all_fpr, mean_tpr_e, 'k-',  label=f'Enum macro (AUC={auc_e_macro:.2f})')
plt.plot(all_fpr, mean_tpr_nb, 'k-.', label=f'NB   macro (AUC={auc_nb_macro:.2f})')

# Top-5
colors = cycle(['b','g','r','c','m'])
for color, i in zip(colors, top5):
    # Enum
    fpr_e, tpr_e, _ = roc_curve(y_true_bin[:, i], scores_enum[:, i])
    auc_e = auc(fpr_e, tpr_e)
    plt.plot(fpr_e, tpr_e, color=color, lw=1.5,
             label=f'Enum C{i} (AUC={auc_e:.2f})')
    # NB
    fpr_nb, tpr_nb, _ = roc_curve(y_true_bin[:, i], scores_nb[:, i])
    auc_nb = auc(fpr_nb, tpr_nb)
    plt.plot(fpr_nb, tpr_nb, color=color, linestyle='--', lw=1.5,
             label=f'NB   C{i} (AUC={auc_nb:.2f})')

plt.plot([0,1],[0,1],'k:', label='Chance')
plt.title('Representative ROC: Top-5 Classes + Micro/Macro')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# 8) Bar Chart: Accuracy vs. Log-Loss
labels_plot = ['Enumeration', 'Naïve-Bayes']
scores_acc = [acc_e, acc_nb]
scores_ll  = [ll_e, ll_nb]
x = np.arange(len(labels_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, scores_acc, width, label='Accuracy')
rects2 = ax.bar(x + width/2, scores_ll,  width, label='Log-Loss')

ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()

for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}',
                xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()