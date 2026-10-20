from fraud_detection.config import FEATURES_DATA_PATH, RAW_DATA_PATH
from fraud_detection.data.loading import load_data
from fraud_detection.features.engineering import (
    ENGINEERED_FEATURES,
    create_features,
)

# Step 1: Load the original dataset
df = load_data(RAW_DATA_PATH)

# Step 2: Create engineered features
df_features = create_features(df)

# Step 3: Display dataset information
print("Original dataset shape:", df.shape)
print("Engineered dataset shape:", df_features.shape)

print("\nNew features:")
print(ENGINEERED_FEATURES)

# Step 4: Preview the engineered features
print("\nFeature sample:")
print(df_features[ENGINEERED_FEATURES].head())

# Step 5: Save the engineered dataset
df_features.to_csv(FEATURES_DATA_PATH, index=False)

print("\nEngineered dataset saved successfully!")
print("Saved location:", FEATURES_DATA_PATH)
