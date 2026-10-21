# `feast apply` imports every module in this folder. The objects are
# defined once in the package so training and serving share them.
from fraud_detection.feature_store import (  # noqa: F401
    recipient,
    recipient_stats_source,
    recipient_stats_view,
)
