
#!/usr/bin/env python
# coding: utf-8

# Main pipeline: PaySim Fraud Detection

from ps_load_data import load_data
from ps1_data_loading import (
    inspect_data, 
    validate_required_columns,
    check_data_quality
)

# Step 1: Load the dataset
df = load_data(r"E:\paysim.csv")

# Step 2: Inspect the dataset
inspect_data(df)

# Step 3: Validate required columns
validate_required_columns(df)

# Step 4: Check data quality
check_data_quality(df)

print("\nInitial data processing completed.")
