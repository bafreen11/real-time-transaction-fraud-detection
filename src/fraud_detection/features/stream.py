"""Feature set of the streaming-track model.

STREAM_FEATURE_COLUMNS = the 12 baseline features + transaction type
flags + recipient behavioral features from Feast. build_stream_features
is the single implementation, used by training and by the scoring
service.
"""

import numpy as np
import pandas as pd

from fraud_detection.features.behavioral import (
    BEHAVIORAL_FEATURES,
    recipient_features,
)
from fraud_detection.features.engineering import (
    FEATURE_COLUMNS,
    RAW_TRANSACTION_FIELDS,
    create_features,
)

# PaySim fraud only occurs in TRANSFER and CASH_OUT transactions.
TYPE_FEATURES = ["type_TRANSFER", "type_CASH_OUT"]

STREAM_FEATURE_COLUMNS = FEATURE_COLUMNS + TYPE_FEATURES + BEHAVIORAL_FEATURES

# Fields a streamed transaction must carry.
STREAM_TRANSACTION_FIELDS = RAW_TRANSACTION_FIELDS + [
    "type",
    "nameOrig",
    "nameDest",
]


def build_stream_features(transactions, state):
    """Model input for transactions given their recipient state.

    transactions: DataFrame with STREAM_TRANSACTION_FIELDS.
    state: recipient state aligned row-for-row (see behavioral.py).
    """
    transactions = transactions.reset_index(drop=True)
    state = state.reset_index(drop=True)

    base = create_features(transactions[RAW_TRANSACTION_FIELDS])

    for type_feature in TYPE_FEATURES:
        base[type_feature] = (
            transactions["type"] == type_feature.removeprefix("type_")
        ).astype(np.int64)

    behavioral = recipient_features(transactions, state)

    return pd.concat([base, behavioral], axis=1)[STREAM_FEATURE_COLUMNS]
