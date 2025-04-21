from src.diagnostic_agent import load_dataset, build_joint_probability, inference_by_enumeration
from src.analyze_marginals import compute_all_marginals

if __name__ == "__main__":
    # Load the data and build the joint distribution
    df = load_dataset("data/simplified_health_data.csv")
    joint_prob = build_joint_probability(df.values)

    # Conditional query
    print("P(Condition | Symptom1=1, Symptom4=1):")
    result = inference_by_enumeration(joint_prob, [1, -1, -1, 1, -2])
    for i, p in enumerate(result):
        print(f"  Condition {i}: {p:.4f}")

    # Compute & print all marginals
    print("\n=== Marginal Distributions ===")
    marginals = compute_all_marginals(joint_prob)
    for name, dist in marginals.items():
        values = ", ".join(f"{p:.4f}" for p in dist)
        print(f"  {name}: [{values}]")