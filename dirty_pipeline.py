import numpy as np
import pandas as pd

def dirty_dataset(df):
    df_dirty = df.copy()

    num_cols = df_dirty.select_dtypes(include=np.number).columns.tolist()
    if "is_fraud" in num_cols:
        num_cols.remove("is_fraud")
    cat_cols = df_dirty.select_dtypes(include='object').columns.tolist()
    
    # -------------------------------
    # 1. MISSING VALUES (safe)
    # -------------------------------
    for col in df_dirty.columns:
        if col=="is_fraud":
            continue
        frac = np.random.uniform(0.02, 0.1)
        idx = df_dirty.sample(frac=frac).index
        df_dirty.loc[idx, col] = np.nan

    # -------------------------------
    # 3. INVALID VALUES (numeric)
    # -------------------------------
    for col in num_cols:
        idx = df_dirty.sample(frac=0.005).index
        df_dirty.loc[idx, col] = np.random.choice([-1, -100, 0], size=len(idx))

    # -------------------------------
    # 4. CATEGORY ISSUEs
    # -------------------------------
    for col in cat_cols:
        idx1 = df_dirty.sample(frac=0.03).index
        idx2 = df_dirty.sample(frac=0.02).index

        df_dirty.loc[idx1, col] = df_dirty.loc[idx1, col].astype(str).str.lower()
        df_dirty.loc[idx2, col] = df_dirty.loc[idx2, col].astype(str).str.upper()

    # -------------------------------
    # 5. TEXT NOISE
    # -------------------------------
    noise_tokens = ['!!!', '@@@', '###']

    for col in cat_cols:
        idx = df_dirty.sample(frac=0.01).index
        noise = np.random.choice(noise_tokens, size=len(idx))
        df_dirty.loc[idx, col] = df_dirty.loc[idx, col].astype(str) + noise

    # -------------------------------
    # 6. DUPLICATES
    # -------------------------------
    df_dirty = pd.concat(
        [df_dirty, df_dirty.sample(frac=0.03)],
        ignore_index=True
    )

    # -------------------------------
    # 7. DATA TYPE CORRUPTION
    # -------------------------------
    if len(num_cols) > 0:
        col = num_cols[0]

        idx = df_dirty.sample(frac=0.02).index
        df_dirty.loc[idx, col] = df_dirty.loc[idx, col].astype(str)

        idx2 = df_dirty.sample(frac=0.01).index
        df_dirty.loc[idx2, col] = 'unknown'


   
    print("Safe dirty dataset created!")
    print("Before:", df_dirty.shape)
    print("\nMissing Before:\n", df_dirty.isnull().sum())
    
    return df_dirty


# # ----------- USAGE -----------
# df = pd.read_csv("fraudTrain.csv")

# df_dirty = dirty_dataset_safe(df)

# df_dirty.to_csv("raw_train_data.csv", index=False)

# print("Safe dirty dataset created!")