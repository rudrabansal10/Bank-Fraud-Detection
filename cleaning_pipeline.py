import pandas as pd
import numpy as np

def clean_dataset(df):
    df_clean = df.copy()

    # -------------------------------
    # 1. REMOVE DUPLICATES
    # -------------------------------
    df_clean = df_clean.drop_duplicates()

    # -------------------------------
    # 2. HANDLE 'unknown' VALUES
    # -------------------------------
    df_clean.replace('unknown', np.nan, inplace=True)

    # -------------------------------
    # 3. FIX DATA TYPES (important)
    # -------------------------------
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except:
                pass

    # -------------------------------
    # 4. FIX INVALID NUMERIC VALUES
    # -------------------------------
    num_cols = df_clean.select_dtypes(include=np.number).columns

    if 'is_fraud' in num_cols:
        num_cols = num_cols.drop('is_fraud')

    for col in num_cols:
        
        if 'amt' in df_clean.columns:
            df_clean.loc[df_clean['amt'] <= 0, 'amt'] = np.nan

    # -------------------------------
    # 5. HANDLE MISSING VALUES
    # -------------------------------
    # Numeric → median
    for col in num_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Categorical → mode
    cat_cols = df_clean.select_dtypes(include='object').columns

    for col in cat_cols:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # -------------------------------
    # 6. FIX CATEGORY INCONSISTENCIES
    # -------------------------------
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()

    # -------------------------------
    # 7. REMOVE TEXT NOISE
    # -------------------------------
    noise_tokens = ['!!!', '@@@', '###']

    for col in cat_cols:
        for token in noise_tokens:
            df_clean[col] = df_clean[col].str.replace(token, '', regex=False)

    print("After:", df_clean.shape)
    
    print("\nMissing After:\n", df_clean.isnull().sum())

    print("Cleaning completed!")

    return df_clean


# # ----------- DURING TRAINING PHASE -----------
# df_dirty = pd.read_csv("raw_train_data.csv")

# df_clean = clean_dataset(df_dirty)

# df_clean.to_csv("clean_data.csv", index=False)

# print("Before:", df_dirty.shape)
# print("After:", df_clean.shape)

# print("\nMissing Before:\n", df_dirty.isnull().sum())
# print("\nMissing After:\n", df_clean.isnull().sum())

# print("Cleaning completed!")
