import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    SHARED_TEST_FRACTION,
    SHARED_TEST_PATH,
    SHARED_TRAIN_LEGITIMATE_SAMPLE,
    SHARED_TRAIN_PATH,
    ensure_output_dirs,
)

ensure_output_dirs()

print("Loading original PaySim dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("Original rows:", len(df))
print("Original fraud transactions:", int(df["isFraud"].sum()))

# Reserve the test set before sampling training data.
train_df, test_df = train_test_split(
    df,
    test_size=SHARED_TEST_FRACTION,
    random_state=RANDOM_STATE,
    stratify=df["isFraud"],
)

# Keep the full held-out test partition.
# Sample the training partition for manageable initial training.
fraud_train = train_df[train_df["isFraud"] == 1]
legitimate_train = train_df[train_df["isFraud"] == 0]

legitimate_count = min(
    SHARED_TRAIN_LEGITIMATE_SAMPLE,
    len(legitimate_train),
)

legitimate_sample = legitimate_train.sample(
    n=legitimate_count,
    random_state=RANDOM_STATE,
)

train_sample = pd.concat(
    [fraud_train, legitimate_sample],
    ignore_index=True,
)

# Save stable transaction IDs to help verify separation.
train_sample.to_csv(SHARED_TRAIN_PATH, index=False)
test_df.to_csv(SHARED_TEST_PATH, index=False)

print("\nShared split saved.")
print("Training rows:", len(train_sample))
print("Training fraud rows:", int(train_sample["isFraud"].sum()))
print("Test rows:", len(test_df))
print("Test fraud rows:", int(test_df["isFraud"].sum()))
print("Training file:", SHARED_TRAIN_PATH)
print("Test file:", SHARED_TEST_PATH)
