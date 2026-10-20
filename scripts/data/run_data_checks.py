from fraud_detection.config import RAW_DATA_PATH
from fraud_detection.data.loading import (
    load_data,
    inspect_data,
    validate_required_columns,
    check_data_quality
)

# Step 1: Load the dataset
df = load_data(RAW_DATA_PATH)

# Step 2: Inspect the dataset
inspect_data(df)

# Step 3: Validate the dataset
validate_required_columns(df)

# Step 4: Perform data quality checks
check_data_quality(df)

print("\nInitial data processing completed.")
