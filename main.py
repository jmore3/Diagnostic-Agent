import os
import logging
import numpy as np
import pandas as pd

from src.diagnostic_agent import load_dataset, build_joint_probability, inference_by_enumeration
from src.analyze_marginals import compute_all_marginals
from src.models.bayesian_model import BayesianDiagnosticModel
from src.evaluation import evaluate_models

# Colored console output
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

def main():
    logger = setup_logger()
    logger.info("Starting Bayesian diagnostic agent")

    # ─── 0. Load raw file purely to get disease names ───────────────────────────
    raw = pd.read_csv("data/Disease_symptom_and_patient_profile_dataset.csv")
    disease_names = list(raw["Disease"].astype("category").cat.categories)
    logger.debug(f"Found {len(disease_names)} disease labels")

    # ─── 1. Load normalized data for modeling ──────────────────────────────────
    data_path = "data/normalized_health_data.csv"
    logger.debug(f"Loading normalized dataset from {data_path}")
    df = load_dataset(data_path)
    logger.info(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

    # ─── 2. Train/test split ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X = df[["Symptom1","Symptom2","Symptom3","Symptom4"]].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test")

    # ─── 3. Build joint over training set ──────────────────────────────────────
    train_data = np.hstack((X_train, y_train.reshape(-1,1)))
    logger.debug("Building joint probability table (α=0)")
    joint_prob = build_joint_probability(
        train_data,
        num_conditions=int(y.max()) + 1,
        alpha=0
    )
    logger.info("Joint probability table built")

    # ─── 4. Example single‐query inference ─────────────────────────────────────
    qv = [1, -1, -1, 1, -2]
    logger.info(f"Performing inference by enumeration for query {qv}")
    post = inference_by_enumeration(joint_prob, qv)
    for i, p in enumerate(post):
        logger.info(f"  Condition {i}: {p:.4f}")

    # ─── 5. Legacy marginals ────────────────────────────────────────────────────
    logger.info("Computing all marginals (legacy interface)")
    legacy_margs = compute_all_marginals(joint_prob)
    for name, dist in legacy_margs.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # ─── 6. Fit our BayesianDiagnosticModel ────────────────────────────────────
    logger.debug("Instantiating BayesianDiagnosticModel")
    model = BayesianDiagnosticModel()
    model.fit_joint(train_data)
    model.fit_naive_bayes(train_data, alpha=1.0)
    logger.info("Fitted both joint‐ and Naïve‐Bayes models")

    # ─── 7. Model marginals ─────────────────────────────────────────────────────
    logger.info("Computing all marginals (model interface)")
    model_margs = BayesianDiagnosticModel.compute_all_marginals(model)
    for name, dist in model_margs.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # ─── 8. Compare enumeration vs NB on sample evidence ───────────────────────
    evidence = {0: 1, 2: 0}
    logger.info(f"Comparing inference methods for evidence {evidence}")
    post_enum = model.inference_by_enumeration(query_index=4, evidence=evidence)
    post_nb   = model.naive_bayes_inference(evidence)
    logger.info("Enumeration‐based posterior:")
    for i, p in enumerate(post_enum): logger.info(f"  Condition {i}: {p:.4f}")
    logger.info("Naïve‐Bayes posterior:")
    for i, p in enumerate(post_nb):   logger.info(f"  Condition {i}: {p:.4f}")

    # 9. Evaluate performance on test set
    os.makedirs("output", exist_ok=True)
    logger.info("Evaluating model accuracy and confusion (test set)")

    # build the full disease list from the raw CSV
    all_diseases = raw["Disease"].astype("category").cat.categories.tolist()

    # find which condition codes actually appear in the test set
    present_labels = sorted(set(y_test))

    # filter only those disease names
    disease_names = [all_diseases[i] for i in present_labels]

    # pass both labels and their names to evaluation
    evaluate_models(
        joint=joint_prob,
        nb_model=model,
        X_test=X_test,
        y_test=y_test,
        labels=present_labels,
        target_names=disease_names,
        output_dir="output"
    )


    logger.info("All done — exiting.")

if __name__ == "__main__":
    main()
