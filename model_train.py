import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("MT_clean_data.csv")

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))


# =====================
# PREDICTIONS
# =====================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.7).astype(int)

# =====================
# METRICS
# =====================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================
# CLASSIFICATION REPORT
# =====================
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================
# ROC CURVE
# =====================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# =====================
# OVERFITTING CHECK
# =====================
print("\nTrain Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))


# SAVING MODEL FOR TESTING DATA-------------------------------------------------------------
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "train_columns.pkl")

print("Model + columns saved")
