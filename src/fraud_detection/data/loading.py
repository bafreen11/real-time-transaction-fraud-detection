import pandas as pd


def load_data(file_path):
    """Load the fraud detection dataset."""
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return df


def inspect_data(df):
    """Inspect the basic structure of the dataset."""
    print("\n--- Dataset Inspection ---")
    print("Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst Five Records:")
    print(df.head())


def check_data_quality(df):
    """Check basic data quality issues."""
    print("\n--- Data Quality Checks ---")

    print("\nMissing Values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nTotal Missing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    print("Duplicate Column Names:", duplicate_columns)


def validate_required_columns(df):
    """Verify columns required for fraud analysis."""
    required_columns = [
        "type",
        "amount",
        "isFraud"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    print("\nRequired columns validation passed.")
