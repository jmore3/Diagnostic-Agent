import os
import logging
import pandas as pd
import numpy as np
from itertools import product
from src.diagnostic_agent import build_joint_probability, inference_by_enumeration
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Formatter for colored console output
def setup_logger():
    class ColoredFormatter(logging.Formatter):
        RESET = "\x1b[0m"
        WHITE = "\x1b[37m"
        RED   = "\x1b[31m"

        def format(self, record):
            msg = super().format(record)
            color = self.RED if record.levelno >= logging.WARNING else self.WHITE
            return f"{color}{msg}{self.RESET}"

    logger = logging.getLogger("DiagnosticAgent")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt = "%(asctime)s %(name)-15s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    ch.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting Bayesian diagnostic agent with saved outputs and models")

    # Create output directory
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and clean data
    data_path = "data/Disease_symptom_and_patient_profile_dataset.csv"
    df = pd.read_csv(data_path)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype('Int64')
    before = len(df)
    df.dropna(subset=["Age"], inplace=True)
    df["Age"] = df["Age"].astype(int)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows due to invalid Age")

    # 2. Encode features
    df["Symptom1"] = df["Fever"].map({"No":0, "Yes":1})
    df["Symptom2"] = df["Cough"].map({"No":0, "Yes":1})
    df["Symptom3"] = df["Fatigue"].map({"No":0, "Yes":1})
    df["Symptom4"] = df["Difficulty Breathing"].map({"No":0, "Yes":1})
    df["AgeBin"]    = pd.cut(df["Age"], bins=[0,20,30,40,50,60,999], labels=[0,1,2,3,4,5]).astype(int)
    df["GenderCode"] = df["Gender"].map({"Male":0, "Female":1})
    df["BPCode"]     = pd.Categorical(df["Blood Pressure"], categories=["Low","Normal","High"]).codes
    df["CholCode"]   = pd.Categorical(df["Cholesterol Level"], categories=["Low","Normal","High"]).codes

    # 3. Encode target
    df["Condition"] = pd.Categorical(df["Disease"]).codes
    disease_cat      = pd.Categorical(df["Disease"])
    disease_names    = list(disease_cat.categories)
    num_conditions   = len(disease_names)
    logger.info(f"Detected {num_conditions} distinct conditions")

    # 4. Build joint & save
    features   = ["Symptom1","Symptom2","Symptom3","Symptom4",
                  "AgeBin","GenderCode","BPCode","CholCode"]
    data_array = df[features + ["Condition"]].values.astype(int)
    joint = build_joint_probability(data_array, num_conditions=num_conditions, alpha=0.0)
    logger.info("Built joint distribution without smoothing")

    # Save joint probability array
    joint_path = os.path.join(OUTPUT_DIR, "joint_probability.npy")
    np.save(joint_path, joint)
    logger.info(f"Saved joint probability array to {joint_path}")

    # Query by enumeration & save chart
    query_vector = [1, -1, -1, 1, -1, -1, -1, -1, -2]
    posterior = inference_by_enumeration(joint, query_vector)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(posterior)), posterior)
    ax.set_xticks(range(len(posterior)))
    ax.set_xticklabels(disease_names, rotation=90, fontsize=8)
    ax.set_ylabel("P(condition | evidence)")
    ax.set_title("Enumeration-based Posterior")
    plt.tight_layout()
    enum_png = os.path.join(OUTPUT_DIR, "enumeration_posterior.png")
    fig.savefig(enum_png, dpi=300)
    logger.info(f"Saved enumeration posterior chart to {enum_png}")

    # Also save posterior array
    post_path = os.path.join(OUTPUT_DIR, "enumeration_posterior.npy")
    np.save(post_path, posterior)
    logger.info(f"Saved enumeration posterior array to {post_path}")

    # 5. Train/test split & NB
    X = df[features].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    logger.info(f"Train/test split: {len(X_train)} / {len(X_test)}")

    clf = CategoricalNB()
    clf.fit(X_train, y_train)
    logger.info("Fitted CategoricalNB on training set")

    # Save the NB model
    nb_path = os.path.join(OUTPUT_DIR, "categorical_nb_model.joblib")
    joblib.dump(clf, nb_path)
    logger.info(f"Saved Naïve-Bayes model to {nb_path}")

    y_proba = clf.predict_proba(X_test)

    # 6. Plot & save ROC curves
    fig, ax = plt.subplots(figsize=(8,6))
    for i, name in enumerate(disease_names):
        y_true = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves by Condition")
    ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    roc_png = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.savefig(roc_png, dpi=300)
    logger.info(f"Saved ROC curves to {roc_png}")

    logger.info("All done — outputs and models saved in 'output/' directory.")
