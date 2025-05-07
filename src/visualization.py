import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import CalibrationDisplay

from src.diagnostic_agent import build_joint_probability, inference_by_enumeration
from src.models.bayesian_model import BayesianDiagnosticModel

def plot_symptom_marginals(joint):
    """Bar charts of P(S_i=0) vs P(S_i=1) for each symptom."""
    symptom_names = ["Symptom1","Symptom2","Symptom3","Symptom4"]
    margs = {}
    for idx, name in enumerate(symptom_names):
        qv = [-1]*5
        qv[idx] = -2
        margs[name] = inference_by_enumeration(joint, qv)

    fig, axes = plt.subplots(1, 4, figsize=(12,3))
    for ax, (name, dist) in zip(axes, margs.items()):
        ax.bar([0,1], dist, tick_label=["absent","present"])
        ax.set_title(f"{name}\nP(present)={dist[1]:.2f}")
    plt.suptitle("Symptom Marginals")
    plt.tight_layout(rect=[0,0,1,0.9])
    plt.show()

def plot_heatmap_top1(joint):
    """Heatmap of the most likely condition code for each (S1,S4) combo."""
    grid = np.zeros((2,2))
    for s1 in [0,1]:
        for s4 in [0,1]:
            qv = [s1, -1, -1, s4, -2]
            post = inference_by_enumeration(joint, qv)
            grid[s1, s4] = np.argmax(post)

    df_grid = pd.DataFrame(grid, index=["S1=0","S1=1"], columns=["S4=0","S4=1"])
    plt.figure(figsize=(4,3))
    sns.heatmap(df_grid, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Top-1 Condition by S1/S4")
    plt.show()

def plot_calibration_curve(X_train, y_train, X_test, y_test, cls):
    """Calibration curve for a single class index `cls`."""
    # train NB
    model = BayesianDiagnosticModel()
    arr = np.hstack((X_train, y_train.reshape(-1,1)))
    model.fit_naive_bayes(arr, alpha=1.0)

    # predict probs
    probas = np.array([model.naive_bayes_inference({
        0: x[0], 1: x[1], 2: x[2], 3: x[3]
    }) for x in X_test])

    fig, ax = plt.subplots(figsize=(5,5))
    CalibrationDisplay.from_predictions(
        y_test == cls,
        probas[:, cls],
        n_bins=10,
        ax=ax,
        name=f"Class {cls}"
    )
    ax.set_title(f"Calibration Curve (Class {cls})")
    plt.tight_layout()
    plt.show()

def plot_pr_curves(y_bin, probas, top_classes):
    """Precision–Recall curves for the top N classes."""
    plt.figure(figsize=(6,4))
    for cls in top_classes:
        p, r, _ = precision_recall_curve(y_bin[:,cls], probas[:,cls])
        plt.plot(r, p, lw=2, label=f"Class {cls} (AUC={auc(r,p):.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (Top Classes)")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.show()

def plot_confusion(y_test, probas):
    """Argmax confusion matrix."""
    y_pred = probas.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (argmax)")
    plt.tight_layout()
    plt.show()

def main():
    # load & split
    df = pd.read_csv("data/normalized_health_data.csv")
    X = df[["Symptom1","Symptom2","Symptom3","Symptom4"]].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # joint model (no smoothing)
    data = np.hstack((X_train, y_train.reshape(-1,1)))
    joint = build_joint_probability(data, alpha=0)

    # 1. symptom marginals
    plot_symptom_marginals(joint)

    # 2. heatmap
    plot_heatmap_top1(joint)

    # 3. train NB, get probabilities
    nb = BayesianDiagnosticModel()
    nb.fit_naive_bayes(data, alpha=1.0)
    probas = np.array([
        nb.naive_bayes_inference({0:x[0],1:x[1],2:x[2],3:x[3]})
        for x in X_test
    ])
    # binarize for multiclass metrics
    C = int(y.max())+1
    y_bin = label_binarize(y_test, classes=np.arange(C))

    # pick top classes by AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(C):
        if y_bin[:,i].sum()==0: continue
        fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], probas[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    top_classes = sorted(roc_auc, key=roc_auc.get, reverse=True)[:5]

    # 4. calibration on best class
    plot_calibration_curve(X_train, y_train, X_test, y_test, top_classes[0])

    # 5. precision–recall
    plot_pr_curves(y_bin, probas, top_classes)

    # 6. confusion matrix
    plot_confusion(y_test, probas)

if __name__ == "__main__":
    main()
