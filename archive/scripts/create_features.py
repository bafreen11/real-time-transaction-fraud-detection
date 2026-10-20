#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from feature_engineering import create_features

df_features = create_features(df)

print("Original dataset shape:", df.shape)
print("Engineered dataset shape:", df_features.shape)

print("\nNew features:")
print(df_features.columns[-6:].tolist())

print("\nFeature sample:")
print(df_features[
    [
        "origin_balance_error",
        "destination_balance_error",
        "amount_to_origin_balance",
        "amount_to_destination_balance",
        "step_day",
        "step_hour"
    ]
].head())

