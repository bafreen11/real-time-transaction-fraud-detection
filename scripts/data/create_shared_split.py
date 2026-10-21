import pandas as pd

from fraud_detection.config import (
    RAW_DATA_PATH,
    SHARED_TEST_PATH,
    SHARED_TRAIN_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import shared_split

ensure_output_dirs()

print("Loading original PaySim dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("Original rows:", len(df))
print("Original fraud transactions:", int(df["isFraud"].sum()))

# Reserve the test set before sampling training data.
# The training side keeps all fraud plus a legitimate sample.
train_sample, test_df = shared_split(df)

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
