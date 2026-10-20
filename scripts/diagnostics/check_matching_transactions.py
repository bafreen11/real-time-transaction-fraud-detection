import pandas as pd

from fraud_detection.config import FEATURES_DATA_PATH

df = pd.read_csv(FEATURES_DATA_PATH)

matches = df[
    (df["step"] == 1)
    & (df["amount"] == 181)
    & (df["oldbalanceOrg"] == 181)
    & (df["newbalanceOrig"] == 0)
    & (df["oldbalanceDest"] == 0)
    & (df["newbalanceDest"] == 181)
]

print("Matching transactions:", len(matches))

if len(matches) > 0:
    print(
        matches[["type", "isFraud", "isFlaggedFraud"]]
        .value_counts(dropna=False)
        .to_string()
    )
else:
    print("No exact matching transaction found.")
