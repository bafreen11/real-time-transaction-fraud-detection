"""Feature set of the streaming-track model.

STREAM_FEATURE_COLUMNS = the 14 baseline features (including the
transaction type flags) + recipient behavioral features from Feast. build_stream_features
is the single implementation, used by training and by the scoring
service.
"""

import pandas as pd

from fraud_detection.features.behavioral import (
    BEHAVIORAL_FEATURES,
    recipient_features,
)
from fraud_detection.features.engineering import (
    FEATURE_COLUMNS,
    TRANSACTION_FIELDS,
    create_features,
)

STREAM_FEATURE_COLUMNS = FEATURE_COLUMNS + BEHAVIORAL_FEATURES

# Fields a streamed transaction must carry.
STREAM_TRANSACTION_FIELDS = TRANSACTION_FIELDS + [
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

    base = create_features(transactions[TRANSACTION_FIELDS])

    behavioral = recipient_features(transactions, state)

    return pd.concat([base, behavioral], axis=1)[STREAM_FEATURE_COLUMNS]
