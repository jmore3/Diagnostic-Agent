# src/evaluate.py

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.diagnostic_agent import inference_by_enumeration
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(joint, nb_model, X_test, y_test, disease_names, output_dir="output"):
    """
    - joint: the joint-probability array
    - nb_model: a fitted BayesianDiagnosticModel (naïve-bayes)
    - X_test: array of shape (n_samples, 4) with symptom values
    - y_test: array of true condition codes
    - disease_names: list mapping codes→string names
    """
    # 1) Full‐joint MAP predictions
    y_pred_enum = []
    for x in X_test:
        # build query-vector: [s1,s2,s3,s4, ?]
        qv = [int(x[0]), int(x[1]), int(x[2]), int(x[3]), -2]
        post = inference_by_enumeration(joint, qv)
        y_pred_enum.append(int(np.argmax(post)))
    y_pred_enum = np.array(y_pred_enum)

    # 2) Naïve‐Bayes MAP predictions
    y_pred_nb = []
    for x in X_test:
        post_nb = nb_model.naive_bayes_inference({
            0: int(x[0]), 1: int(x[1]), 2: int(x[2]), 3: int(x[3])
        })
        y_pred_nb.append(int(np.argmax(post_nb)))
    y_pred_nb = np.array(y_pred_nb)

    # 3) Accuracy scores
    acc_enum = accuracy_score(y_test, y_pred_enum)
    acc_nb   = accuracy_score(y_test, y_pred_nb)
    print(f"Full‐joint accuracy:   {acc_enum:.4f}")
    print(f"Naïve‐Bayes accuracy: {acc_nb:.4f}\n")

    # 4) Classification report for NB
    print("Classification Report (Naïve‐Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=disease_names))

    # 5) Confusion matrix for NB
    cm = confusion_matrix(y_test, y_pred_nb)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=disease_names, yticklabels=disease_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Naïve‐Bayes)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_nb.png", dpi=300)
    plt.show()
