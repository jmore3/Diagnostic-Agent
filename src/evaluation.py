import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.diagnostic_agent import inference_by_enumeration


def evaluate_models(joint, nb_model, X_test, y_test, labels, target_names, output_dir):
    """
    Evaluate full-joint and Naïve-Bayes models on the test set.

    Parameters:
    - joint: ndarray, joint probability table from build_joint_probability
    - nb_model: BayesianDiagnosticModel instance
    - X_test: array-like, test features
    - y_test: array-like, test labels
    - labels: list of int, the class indices present in y_test
    - target_names: list of str, names corresponding to each label
    - output_dir: str, directory where output plots will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Full-joint predictions via enumeration
    y_pred_joint = []
    for x in X_test:
        # build a query vector: symptoms followed by -2 (to predict condition)
        qv = [int(v) for v in x] + [-2]
        post = inference_by_enumeration(joint, qv)
        y_pred_joint.append(int(np.argmax(post)))
    y_pred_joint = np.array(y_pred_joint)

    # 2. Naïve-Bayes predictions
    y_pred_nb = []
    for x in X_test:
        evidence = {i: int(x[i]) for i in range(len(x))}
        post = nb_model.naive_bayes_inference(evidence)
        y_pred_nb.append(int(np.argmax(post)))
    y_pred_nb = np.array(y_pred_nb)

    # 3. Print accuracy
    acc_joint = accuracy_score(y_test, y_pred_joint)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Full‐joint accuracy:   {acc_joint:.4f}")
    print(f"Naïve‐Bayes accuracy: {acc_nb:.4f}\n")

    # 4. Classification report for Naïve-Bayes
    print("Classification Report (Naïve‐Bayes):")
    print(classification_report(
        y_test,
        y_pred_nb,
        labels=labels,
        target_names=target_names
    ))

    # 5. Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred_nb, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Naïve‐Bayes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
