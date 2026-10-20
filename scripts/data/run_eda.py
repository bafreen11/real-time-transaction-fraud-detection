import pandas as pd

from fraud_detection import eda
from fraud_detection.config import RAW_DATA_PATH

# Step 1: Load the PaySim dataset
df = pd.read_csv(RAW_DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Run all 10 EDA functions

eda.summarize_fraud_data(df)

eda.analyze_fraud_distribution(df)

eda.analyze_fraud_by_type(df)

eda.analyze_transaction_amounts(df)

eda.analyze_fraud_over_time(df)

eda.analyze_balance_patterns(df)

eda.analyze_zero_balances(df)

eda.analyze_existing_fraud_flag(df)

eda.analyze_numeric_correlations(df)

eda.analyze_account_reuse(df)

print("\nAll EDA functions executed successfully!")
