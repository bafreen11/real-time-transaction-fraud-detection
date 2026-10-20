# ============================================================
# PROJECT: Real-Time Fraud Detection
# PURPOSE: Exploratory Data Analysis (EDA)
# DATASET: PaySim
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rich tables inside Jupyter; plain print when run as a script.
try:
    from IPython.display import display
except ImportError:
    display = print


# ============================================================
# 1. FRAUD DISTRIBUTION
# ============================================================

def analyze_fraud_distribution(df):
    """Analyze legitimate and fraudulent transaction counts."""

    counts = df["isFraud"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    summary = pd.DataFrame({
        "transaction_count": counts,
        "percentage": (counts / len(df) * 100).round(4)
    })

    summary.index = ["Legitimate", "Fraudulent"]
    summary.index.name = "transaction_class"

    print("\nFraud Distribution")
    display(summary)

    summary["transaction_count"].plot(
        kind="bar",
        title="Legitimate vs Fraudulent Transactions"
    )
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return summary


# ============================================================
# 2. FRAUD BY TRANSACTION TYPE
# ============================================================

def analyze_fraud_by_type(df):
    """Analyze transaction volume, fraud counts and fraud rates."""

    summary = df.groupby("type", observed=True)["isFraud"].agg(
        total_transactions="count",
        fraud_cases="sum"
    )

    summary["legitimate_cases"] = (
        summary["total_transactions"] - summary["fraud_cases"]
    )

    summary["fraud_rate_pct"] = (
        summary["fraud_cases"]
        / summary["total_transactions"] * 100
    ).round(4)

    summary = summary.sort_values(
        "fraud_rate_pct", ascending=False
    )

    print("\nFraud Analysis by Transaction Type")
    display(summary)

    summary["fraud_rate_pct"].plot(
        kind="bar",
        title="Fraud Rate by Transaction Type"
    )
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return summary


# ============================================================
# 3. TRANSACTION AMOUNT ANALYSIS
# ============================================================

def analyze_transaction_amounts(df):
    """Compare transaction amount statistics by fraud class."""

    summary = df.groupby("isFraud")["amount"].agg(
        ["count", "mean", "median", "min", "max", "std"]
    )

    summary.index = ["Legitimate", "Fraudulent"]

    print("\nTransaction Amount Analysis")
    display(summary.round(2))

    # Sample each class for visualization to reduce memory use.
    class_counts = df["isFraud"].value_counts()
    sample_size = min(10000, int(class_counts.min()))

    sample = (
        df.groupby("isFraud", group_keys=False)
        .sample(n=sample_size, random_state=42)
    )

    for label, group in sample.groupby("isFraud"):
        plt.hist(
            np.log1p(group["amount"]),
            bins=50,
            alpha=0.6,
            label="Fraudulent" if label == 1 else "Legitimate"
        )

    plt.title("Transaction Amount Distribution (log1p scale)")
    plt.xlabel("log(1 + amount)")
    plt.ylabel("Sampled Transaction Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return summary


# ============================================================
# 4. FRAUD AND TRANSACTION VOLUME OVER TIME
# ============================================================

def analyze_fraud_over_time(df):
    """Examine transaction counts and fraud rates by step."""

    summary = df.groupby("step")["isFraud"].agg(
        total_transactions="count",
        fraud_cases="sum"
    )

    summary["fraud_rate_pct"] = (
        summary["fraud_cases"]
        / summary["total_transactions"] * 100
    ).round(4)

    print("\nTransaction and Fraud Patterns Over Time")

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        summary.index,
        summary["total_transactions"],
        label="Transaction Count"
    )
    ax1.set_xlabel("Step (simulated time unit)")
    ax1.set_ylabel("Transaction Count")

    ax2 = ax1.twinx()
    ax2.plot(
        summary.index,
        summary["fraud_rate_pct"],
        linestyle="--",
        label="Fraud Rate"
    )
    ax2.set_ylabel("Fraud Rate (%)")

    plt.title("Transaction Volume and Fraud Rate Over Time")
    fig.tight_layout()
    plt.show()

    display(
        summary.sort_values(
            "fraud_rate_pct", ascending=False
        ).head(10)
    )

    return summary


# ============================================================
# 5. SENDER AND RECIPIENT BALANCE PATTERNS
# ============================================================

def analyze_balance_patterns(df):
    """Compare balance movements for legitimate and fraud transactions."""

    columns = [
        "isFraud",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    work = df[columns].copy()

    work["sender_balance_change"] = (
        work["oldbalanceOrg"] - work["newbalanceOrig"]
    )

    work["recipient_balance_change"] = (
        work["newbalanceDest"] - work["oldbalanceDest"]
    )

    summary = work.groupby("isFraud")[
        [
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "sender_balance_change",
            "recipient_balance_change"
        ]
    ].median().round(2)

    summary.index = ["Legitimate", "Fraudulent"]

    print("\nMedian Balance Patterns by Fraud Class")
    display(summary)

    summary.T.plot(
        kind="bar",
        figsize=(12, 5),
        title="Balance Patterns by Fraud Class"
    )
    plt.ylabel("Median Amount")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return summary


# ============================================================
# 6. ZERO-BALANCE ANALYSIS
# ============================================================

def analyze_zero_balances(df):
    """Measure zero balances in sender and recipient accounts."""

    balance_cols = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    rows = []

    for col in balance_cols:
        for label, group in df.groupby("isFraud"):
            zero_count = int((group[col] == 0).sum())

            rows.append({
                "balance_column": col,
                "class": (
                    "Fraudulent" if label == 1 else "Legitimate"
                ),
                "zero_count": zero_count,
                "zero_percentage": round(
                    zero_count / len(group) * 100, 3
                )
            })

    summary = pd.DataFrame(rows)

    print("\nZero-Balance Analysis")
    display(summary)

    return summary


# ============================================================
# 7. EXISTING FRAUD FLAG ANALYSIS
# ============================================================

def analyze_existing_fraud_flag(df):
    """Compare the existing flag with the actual fraud label."""

    summary = pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        rownames=["Existing Flag"],
        colnames=["Actual Fraud"],
        margins=True
    )

    print("\nExisting Fraud Flag vs Actual Fraud")
    display(summary)

    flagged = df["isFlaggedFraud"] == 1

    print("Transactions flagged:", int(flagged.sum()))
    print(
        "Flagged transactions that are actual fraud:",
        int(df.loc[flagged, "isFraud"].sum())
    )

    return summary


# ============================================================
# 8. NUMERIC FEATURE CORRELATION
# ============================================================

def analyze_numeric_correlations(df):
    """Display correlations between numerical features."""

    columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ]

    corr = df[columns].corr()

    print("\nNumeric Feature Correlation Matrix")
    display(corr.round(3))

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        corr,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=60, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)

    ax.set_title("Numeric Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")

    fig.tight_layout()
    plt.show()

    return corr


# ============================================================
# 9. ACCOUNT REUSE ANALYSIS
# ============================================================

def analyze_account_reuse(df):
    """Examine repeated sender and recipient identifiers."""

    sender_counts = df["nameOrig"].value_counts()
    recipient_counts = df["nameDest"].value_counts()

    summary = {
        "unique_sender_accounts": df["nameOrig"].nunique(),
        "unique_recipient_accounts": df["nameDest"].nunique(),
        "senders_appearing_multiple_times": int(
            (sender_counts > 1).sum()
        ),
        "recipients_appearing_multiple_times": int(
            (recipient_counts > 1).sum()
        ),
        "unique_senders_in_fraud_transactions": df.loc[
            df["isFraud"] == 1, "nameOrig"
        ].nunique(),
        "unique_recipients_in_fraud_transactions": df.loc[
            df["isFraud"] == 1, "nameDest"
        ].nunique()
    }

    print("\nAccount Reuse Analysis")

    for key, value in summary.items():
        print(f"{key}: {value:,}")

    return summary


# ============================================================
# 10. OVERALL FRAUD SUMMARY
# ============================================================

def summarize_fraud_data(df):
    """Print high-level statistics for the fraud dataset."""

    total_transactions = len(df)
    total_fraud = int(df["isFraud"].sum())
    total_legitimate = total_transactions - total_fraud

    fraud_percentage = (
        total_fraud / total_transactions * 100
        if total_transactions else 0
    )

    summary = {
        "total_transactions": total_transactions,
        "legitimate_transactions": total_legitimate,
        "fraudulent_transactions": total_fraud,
        "fraud_percentage": round(fraud_percentage, 4),
        "legitimate_percentage": round(
            100 - fraud_percentage, 4
        )
    }

    print("\nOverall Fraud Detection Summary")
    print("-" * 40)

    for key, value in summary.items():
        print(f"{key}: {value:,}" if isinstance(value, int)
              else f"{key}: {value}%")

    return summary

