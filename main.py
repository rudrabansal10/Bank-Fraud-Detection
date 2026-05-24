import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from cleaning_pipeline import clean_dataset
from feature_engineering import feature_engineering
from dirty_pipeline import dirty_dataset
from eda import run_eda


# -------------------------------
# 1. LOAD MODEL + TRAIN META
# -------------------------------
model = joblib.load("model.pkl")
train_cols = joblib.load("train_columns.pkl")

# -------------------------------
# 2. LOAD TEST DATA
# -------------------------------
df_test = pd.read_csv("fraudTest.csv",index_col=0)

# -------------------------------
# 3. DIRTYING DATA ( TO SIMULATE REAL WORLD DATA ) ( OUT TEST DATA IS TOO PURE )
# -------------------------------
df_test = dirty_dataset(df_test)
df_test_dirty = df_test.copy()

# -------------------------------
# 3. CLEANING
# -------------------------------
df_test = clean_dataset(df_test)
df_test_clean = df_test.copy()

#---------------------------------
# 4. EDA( EXPLORATORY DATA ANALYSIS )
#--------------------------------
run_eda(df_test_dirty,"BEFORE CLEANING")
run_eda(df_test_clean,"AFTER CLEANING")

# -------------------------------
# 5. FEATURE ENGINEERING
# -------------------------------
df_test = feature_engineering(df_test)

# -------------------------------
# 6. ENCODING (ONE-HOT)
# -------------------------------

print(df_test.select_dtypes(include='object').columns)

cat_cols = ["gender", "category"]
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
dummy_cols = df_test.select_dtypes(include='bool').columns
df_test[dummy_cols] = df_test[dummy_cols].astype(int)

# -------------------------------
# 7. ALIGN WITH TRAIN DATA
# -------------------------------
df_test = df_test.reindex(columns=train_cols, fill_value=0)

# -------------------------------
# 8. PREDICTIONS
# -------------------------------
y_prob = model.predict_proba(df_test)[:, 1]

threshold = 0.75
y_pred = (y_prob > threshold).astype(int)

# -------------------------------
# 9. RISK SCORING SYSTEM
# -------------------------------
df_test['risk_score'] = y_prob
df_test['risk_score_100'] = (y_prob * 100).round(2)

def risk_level(score):
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"

def decision(score):
    if score > 0.8:
        return "BLOCK"
    elif score > 0.5:
        return "REVIEW"
    else:
        return "ALLOW"

df_test['risk_level'] = df_test['risk_score'].apply(risk_level)
df_test['decision'] = df_test['risk_score'].apply(decision)

# -------------------------------
# 10. FEATURES
# -------------------------------
df_test['confidence'] = abs(df_test['risk_score'] - 0.5) * 2
df_test['fraud_rank'] = df_test['risk_score'].rank(ascending=False)

# high-value flag
if 'amt' in df_test.columns:
    df_test['high_value_flag'] = (
        df_test['amt'] > df_test['amt'].quantile(0.95) ).astype(int)

# -------------------------------
# 11. OUTPUT
# -------------------------------
final_cols = [
    'risk_score',
    'risk_score_100',
    'risk_level',
    'decision',
    'confidence',
    'fraud_rank',
    'high_value_flag'
]

print("\nTop Risk Transactions:")
print(df_test.sort_values(by='risk_score', ascending=False)[final_cols].head(100))

# save results
df_test_results = df_test[[ 'amt',
                            'risk_score',
                            'risk_score_100',
                            'risk_level',
                            'decision',
                            'confidence',
                            'fraud_rank',
                            'high_value_flag']]
df_test_results.to_csv("test_predictions.csv", index=False)

print("\nPipeline executed successfully!")