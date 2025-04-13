from src.diagnostic_agent import load_dataset, build_joint_probability, inference_by_enumeration

if __name__ == "__main__":
    # Load data
    data = load_dataset("data/simplified_health_data.csv")
    joint_prob = build_joint_probability(data.values)

    # Run a test inference: P(Condition | Symptom1=1, Symptom4=1)
    query_vector = [1, -1, -1, 1, -2]
    result = inference_by_enumeration(joint_prob, query_vector)

    print("P(Condition | Symptom1=1, Symptom4=1):")
    for i, prob in enumerate(result):
        print(f"Condition {i}: {prob:.4f}")
