import logging
from src.diagnostic_agent import load_dataset, build_joint_probability, inference_by_enumeration
from src.analyze_marginals import compute_all_marginals
from src.models.bayesian_model import BayesianDiagnosticModel

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
    colored_formatter = ColoredFormatter(fmt, datefmt=datefmt)

    ch.setFormatter(colored_formatter)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting Bayesian diagnostic agent")

    # 1. Load the data
    data_path = "data/simplified_health_data.csv"
    logger.debug(f"Loading dataset from {data_path}")
    df = load_dataset(data_path)
    logger.info(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

    # 2. Build joint distribution
    logger.debug("Building joint probability table")
    joint_prob = build_joint_probability(df.values)
    logger.info("Joint probability table built")

    # 3. Single conditional query
    query_vector = [1, -1, -1, 1, -2]
    logger.info(f"Performing inference by enumeration for query vector: {query_vector}")
    result = inference_by_enumeration(joint_prob, query_vector)
    logger.info("Query results:")
    for i, p in enumerate(result):
        logger.info(f"  Condition {i}: {p:.4f}")

    # 4. Compute & log all marginals via the old interface
    logger.info("Computing all marginals (legacy function)")
    marginals_legacy = compute_all_marginals(joint_prob)
    logger.info("Marginal distributions (legacy):")
    for name, dist in marginals_legacy.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # 5. Compute & log with our BayesianDiagnosticModel class
    logger.debug("Instantiating BayesianDiagnosticModel")
    data_array = BayesianDiagnosticModel.load_csv(data_path)
    model = BayesianDiagnosticModel()
    model.fit_joint(data_array)
    model.fit_naive_bayes(data_array, alpha=1.0)
    logger.info("Fitted both joint and Naïve-Bayes models")

    logger.info("Computing all marginals (model methods)")
    marginals_model = BayesianDiagnosticModel.compute_all_marginals(model)
    for name, dist in marginals_model.items():
        vals = ", ".join(f"{p:.4f}" for p in dist)
        logger.info(f"  {name}: [{vals}]")

    # 6. Compare full-joint vs. Naïve-Bayes for a sample evidence
    evidence = {0: 1, 2: 0}
    logger.info(f"Comparing inference methods for evidence: {evidence}")
    post_enum = model.inference_by_enumeration(query_index=4, evidence=evidence)
    post_nb   = model.naive_bayes_inference(evidence)

    logger.info("Enumeration-based posterior:")
    for i, p in enumerate(post_enum):
        logger.info(f"  Condition {i}: {p:.4f}")

    logger.info("Naïve-Bayes posterior:")
    for i, p in enumerate(post_nb):
        logger.info(f"  Condition {i}: {p:.4f}")

    logger.info("All done — exiting.")