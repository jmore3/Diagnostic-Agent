import os
import pandas as pd

# 1) Paths
INPUT_CSV  = "/Users/joelmartinez/Documents/bloomberg/bpuzzled/Diagnostic-Agent/data/Disease_symptom_and_patient_profile_dataset.csv"
OUTPUT_DIR = "/Users/joelmartinez/Documents/bloomberg/bpuzzled/Diagnostic-Agent/data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "normalized_health_data.csv")

# 2) Load
df = pd.read_csv(INPUT_CSV)

# 2.1) Drop any stray header rows where 'Disease' column literally contains the string "Disease"
df = df[df['Disease'] != 'Disease']

# 3) Rename the breathing column
df = df.rename(columns={"Difficulty Breathing": "Difficulty_Breathing"})

# 4) Map Yes/No → 1/0
symptom_cols = ["Fever", "Cough", "Fatigue", "Difficulty_Breathing"]
df[symptom_cols] = df[symptom_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}))

# 5) Check for unmapped values
bad = df[symptom_cols].isnull().any(axis=1)
if bad.any():
    print("Rows with unexpected symptom values:")
    print(df.loc[bad, symptom_cols + ["Disease"]])
    raise ValueError("Found non-Yes/No entries in symptom columns.")

# 6) Encode Disease → Condition (integer codes)
df["Condition"] = df["Disease"].astype("category").cat.codes
mapping = dict(enumerate(df["Disease"].astype("category").cat.categories))
print("Condition mapping:", mapping)

# 7) Build the normalized frame
normalized = df[symptom_cols + ["Condition"]].copy()
normalized.columns = ["Symptom1", "Symptom2", "Symptom3", "Symptom4", "Condition"]

# 8) Force integer dtype (should be clean now)
normalized = normalized.astype(int)

# 9) Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 10) Save or continue in memory
normalized.to_csv(OUTPUT_CSV, index=False)
print(f"Written normalized CSV to {OUTPUT_CSV} ({len(normalized)} rows)")

# If you need the NumPy array for immediate use:
data_array = normalized.values
print("Sample rows:\n", data_array[:5])