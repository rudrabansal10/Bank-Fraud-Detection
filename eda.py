
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# df_dirty = pd.read_csv("raw_train_data.csv")
# df_clean = pd.read_csv("clean_data.csv")

def run_eda(df, title="DATA"):
    print(f"\n===== {title} =====")
    print("Shape:", df.shape)

    # -------------------------------
    # 1. Missing Values
    # -------------------------------
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        plt.figure(figsize=(8,4))
        missing.plot(kind='bar')
        plt.title(f"{title} - Missing (%)")
        plt.show()

    # -------------------------------
    # 4. Time Pattern 
    # -------------------------------
    if 'trans_date_trans_time' in df.columns:
        temp = df.copy()
        temp['hour'] = pd.to_datetime(
            temp['trans_date_trans_time'], errors='coerce'
        ).dt.hour

        plt.figure(figsize=(8,4))
        sns.countplot(x='hour', hue='is_fraud', data=temp)
        plt.title(f"{title} - Fraud by Hour")
        plt.show()

    # -------------------------------
    # 5. Correlation
    # -------------------------------
    corr = df.corr(numeric_only=True)
    fraud_corr = corr['is_fraud'].sort_values(ascending=False)

    print("CORRELATION WITH FRAUD ")
    print(fraud_corr)

    fraud_corr.drop('is_fraud').plot(kind='bar')
    ax = fraud_corr.plot(kind='bar')

    plt.title("Top Features Correlated with Fraud")
    plt.ylabel("Correlation")

    for i, v in enumerate(fraud_corr):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    plt.show()

    num_cols = df.select_dtypes(include=np.number).columns

    plt.figure(figsize=(6,4))
    sns.heatmap(df[num_cols].corr(), cmap='coolwarm')
    plt.title(f"{title} - Correlation")
    plt.show()

    # -------------------------------
    # 6. AMOUNT vs FRAUD (VERY IMPORTANT)
    # -------------------------------
    if 'amt' in df.columns and 'is_fraud' in df.columns:

        # 1. Boxplot
        sns.boxplot(x='is_fraud', y='amt', data=df)
        plt.title(f"{title} - Amount vs Fraud")
        plt.show()

        # 2. Distribution
        sns.histplot(data=df, x='amt', hue='is_fraud', bins=20, kde=True)
        plt.xscale('log')
        plt.title(f"{title} - Amount Distribution by Fraud")
        plt.show()

        # 3. Summary stats
        print("\nAMOUNT STATS BY FRAUD:")
        print(df.groupby('is_fraud')['amt'].describe())


# Run EDA
# run_eda(df_dirty, "BEFORE CLEANING")
# run_eda(df_clean, "AFTER CLEANING")