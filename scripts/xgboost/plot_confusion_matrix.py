import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from fraud_detection.config import (
    CONFUSION_MATRIX_PATH,
    FEATURES_DATA_PATH,
    XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import baseline_split

print("Loading dataset and model...")
df = pd.read_csv(FEATURES_DATA_PATH)
saved = joblib.load(XGBOOST_MODEL_PATH)

model = saved["model"]
features = saved["feature_columns"]

# Reconstruct the same seeded 10% sample and test split
X_train, X_test, y_train, y_test = baseline_split(df, features)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

display.plot(values_format="d")
plt.title("PaySim Fraud Detection - Confusion Matrix")
plt.tight_layout()

ensure_output_dirs()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
print("Saved confusion matrix to:", CONFUSION_MATRIX_PATH)

plt.show()
