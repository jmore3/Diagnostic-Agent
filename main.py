import logging
import pandas as pd
import numpy as np
from itertools import product

from src.diagnostic_agent import build_joint_probability, inference_by_enumeration

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ColoredFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    WHITE = "\x1b[37m"
    RED   = "\x1b[31m"

    def format(self, record):
        msg = super().format(record)
        color = self.RED if record.levelno >= logging.WARNING else self.WHITE
        return f"{color}{msg}{self.RESET}"

def setup_logger():
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
    logger.info("Starting Bayesian diagnostic agent with ROC evaluation")

    # --- 1. Load raw data ---
    data_path = "data/Disease_symptom_and_patient_profile_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {df.shape[0]} records with {df.shape[1]} columns")

    # --- 1a. Clean Age ---
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    before = len(df)
    df.dropna(subset=["Age"], inplace=True)
    df["Age"] = df["Age"].astype(int)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows due to invalid Age")

    # --- 2. Encode symptoms ---
    df["Symptom1"] = df["Fever"].map({"No":0, "Yes":1})
    df["Symptom2"] = df["Cough"].map({"No":0, "Yes":1})
    df["Symptom3"] = df["Fatigue"].map({"No":0, "Yes":1})
    df["Symptom4"] = df["Difficulty Breathing"].map({"No":0, "Yes":1})

    # --- 3. Bin Age ---
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0,20,30,40,50,60,999],
        labels=[0,1,2,3,4,5]
    ).astype(int)

    # --- 4. Encode Gender, BP, Cholesterol ---
    df["GenderCode"] = df["Gender"].map({"Male":0,"Female":1})
    df["BPCode"]   = pd.Categorical(
        df["Blood Pressure"],
        categories=["Low","Normal","High"]
    ).codes
    df["CholCode"] = pd.Categorical(
        df["Cholesterol Level"],
        categories=["Low","Normal","High"]
    ).codes

    # --- 5. Encode target condition & map names ---
    df["Condition"] = pd.Categorical(df["Disease"]).codes
    disease_cat     = pd.Categorical(df["Disease"])
    disease_names   = list(disease_cat.categories)
    num_conditions  = len(disease_names)
    logger.info(f"Detected {num_conditions} distinct conditions")

    # --- 6. Build joint distrib & query by enumeration ---
    features   = ["Symptom1","Symptom2","Symptom3","Symptom4",
                  "AgeBin","GenderCode","BPCode","CholCode"]
    data_array = df[features + ["Condition"]].values.astype(int)

    logger.debug("Building joint probability (no smoothing)")
    joint = build_joint_probability(data_array, num_conditions=num_conditions, alpha=0.0)
    logger.info("Joint distribution built")

    # sample query: Fever=Yes, DiffBreath=Yes, others unknown → Condition
    query_vector = [1, -1, -1, 1, -1, -1, -1, -1, -2]
    logger.info(f"Performing inference for query {query_vector}")
    posterior = inference_by_enumeration(joint, query_vector)

    # top-10 results
    ranked = sorted(
        enumerate(posterior),
        key=lambda x: x[1],
        reverse=True
    )
    logger.info("Top 10 predicted conditions:")
    for code, prob in ranked[:10]:
        logger.info(f"  {disease_names[code]} (code {code}): {prob:.4f}")

    # --- 7. Hold-out split & Naïve-Bayes ROC evaluation ---
    X = df[features].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    logger.info(f"Train/test split: {len(X_train)} / {len(X_test)}")

    # fit a CategoricalNB on the discrete features
    clf = CategoricalNB()
    clf.fit(X_train, y_train)
    logger.info("Naïve-Bayes model trained on training set")

    # predict class probabilities on test set
    y_proba = clf.predict_proba(X_test)

    # plot ROC curve for each condition
    plt.figure(figsize=(8,6))
    for i, name in enumerate(disease_names):
        # binary ground truth for class i
        y_true = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Condition")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

    logger.info("All done — exiting.")
