import os
import logging
import numpy as np
import pandas as pd

from src.diagnostic_agent import load_dataset, build_joint_probability, inference_by_enumeration
from src.analyze_marginals import compute_all_marginals
from src.models.bayesian_model import BayesianDiagnosticModel
from src.evaluate import evaluate_models

# Colored console output
class ColoredFormatter(logging.Formatter):
    RESET  = "\x1b[0m"
    WHITE  = "\x1b[37m"
    RED    = "\x1b[31m"

    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{self.RED}{msg}{self.RESET}"
        else:
            return f"{self.WHITE}{msg}{self.RESET}"


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

    # 1. Load the data
    data_path = "data/normalized_health_data.csv"
    logger.debug(f"Loading dataset from {data_path}")
    df = load_dataset(data_path)
    logger.info(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

    # 2. Split into train/test sets
    from sklearn.model_selection import train_test_split
    X = df[["Symptom1","Symptom2","Symptom3","Symptom4"]].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test")

    # 3. Build joint distribution on training data
    train_data = np.hstack((X_train, y_train.reshape(-1,1)))
    logger.debug("Building joint probability table")
    joint_prob = build_joint_probability(train_data, num_conditions=int(y.max())+1, alpha=0)
    logger.info("Joint probability table built")

    # 4. Single conditional query example
    query_vector = [1, -1, -1, 1, -2]
    logger.info(f"Performing inference by enumeration for query: {query_vector}")
    post = inference_by_enumeration(joint_prob, query_vector)
    for i, p in enumerate(post):
        logger.info(f"  Condition {i}: {p:.4f}")

    # 5. Compute marginals (legacy)
    logger.info("Computing all marginals (legacy function)")
    marginals_legacy = compute_all_marginals(joint_prob)
    for name, dist in marginals_legacy.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # 6. Fit BayesianDiagnosticModel
    logger.debug("Instantiating BayesianDiagnosticModel")
    model = BayesianDiagnosticModel()
    model.fit_joint(train_data, num_conditions=int(y.max())+1)
    model.fit_naive_bayes(train_data, alpha=1.0)
    logger.info("Fitted both joint and Naïve-Bayes models")

    # 7. Compute marginals (model interface)
    logger.info("Computing all marginals (model methods)")
    marginals_model = BayesianDiagnosticModel.compute_all_marginals(model)
    for name, dist in marginals_model.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # 8. Compare enumeration vs NB for a sample evidence
    evidence = {0:1, 2:0}
    logger.info(f"Comparing inference methods for evidence: {evidence}")
    post_enum = model.inference_by_enumeration(query_index=4, evidence=evidence)
    post_nb   = model.naive_bayes_inference(evidence)
    logger.info("Enumeration-based posterior:")
    for i, p in enumerate(post_enum): logger.info(f"  Condition {i}: {p:.4f}")
    logger.info("Naïve-Bayes posterior:")
    for i, p in enumerate(post_nb):   logger.info(f"  Condition {i}: {p:.4f}")

    # 9. Evaluate performance on test set
    os.makedirs("output", exist_ok=True)
    logger.info("Evaluating model accuracy and confusion (test set)")
    disease_names = list(df['Disease'].astype('category').cat.categories)
    evaluate_models(
        joint=joint_prob,
        nb_model=model,
        X_test=X_test,
        y_test=y_test,
        disease_names=disease_names,
        output_dir="output"
    )

    logger.info("All done — exiting.")


if __name__ == "__main__":
    main()
