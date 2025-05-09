#!/usr/bin/env python3
import ssl
# ─────────────────────────────────────────────────────
# Monkey-patch the SSL context to skip certificate checks
ssl._create_default_https_context = ssl._create_unverified_context
# ─────────────────────────────────────────────────────
from ucimlrepo import fetch_ucirepo, list_available_datasets

print("-------------------------------------")
print("The following datasets are available:")
print("-------------------------------------")
list_available_datasets()

# Now fetching Heart Disease (ID=45)
heart_disease = fetch_ucirepo(id=45)

# access data
X = heart_disease.data.features
y = heart_disease.data.targets

# train example (uncomment to use)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression().fit(X, y)

# access metadata
print("Dataset ID:", heart_disease.metadata.uci_id)
print("Instances:", heart_disease.metadata.num_instances)
print("Summary:", heart_disease.metadata.additional_info.summary)

# access variable info
print("\nVariables:")
print(heart_disease.variables)