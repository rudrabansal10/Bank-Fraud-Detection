import pandas as pd
import numpy as np


def feature_engineering(df):
    # df = pd.read_csv("clean_data.csv")

    # =====================
    # TIME FEATURES
    # =====================
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day'] >= 5).astype(int)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.day

    df.drop('datetime',axis=1,inplace=True,errors="ignore")

    # night flag
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)

    # DISTANCE FEATURE
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)

    # AMOUNT FEATURES
    # category
    df['amt_category'] = pd.qcut(df['amt'], q=5, labels=False)

    # high amount flag
    df['high_amt'] = (df['amt'] > df['amt'].quantile(0.9)).astype(int)

    # =====================
    # USER BEHAVIOR FEATURES
    # =====================
    if 'cc_num' in df.columns:
        # total transactions
        df['txn_count'] = df.groupby('cc_num')['amt'].transform('count')
    
        # average spending
        df['avg_amt_user'] = df.groupby('cc_num')['amt'].transform('mean')
    
        # deviation from normal behavior
        df['amt_vs_avg'] = df['amt'] / (df['avg_amt_user'] + 1)


    # AGE FEATURE
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365

    # LOCATION BEHAVIOR
    df['same_location'] = (df['distance'] < df['distance'].median()).astype(int)

    # TIME FREQUENCY
    df['hour_freq'] = df.groupby('hour')['amt'].transform('count')

    # =====================
    # DONE
    # =====================
    print("New columns added:")
    print([col for col in df.columns if col not in ['unix_time']])
    print(df.head())

    # DROP IRRELEVANT COLUMNS
    drop_cols = [
        'cc_num', 'trans_num', 'job',
        'first', 'last', 'trans_date_trans_time',
        'street', 'city', 'state', 
        'dob', 'unix_time', 'merchant'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')

    print("FEATURE ENGINNERING DONE!!!")
    print("FEATURE SELECTION DONE!!!")
    
    

    # FEATURE SELECTION ----------------------------------------------------------------------------------------

    # 1. DROP IRRELEVANT COLUMNS
    drop_cols = [
        'cc_num', 'trans_num', 'job',
        'first', 'last', 'trans_date_trans_time',
        'street', 'city', 'state', 
        'dob', 'unix_time'
        
    ]

    df = df.drop(columns=drop_cols, errors='ignore')


    # 2. DROP HIGH-CARDINALITY TEXT
    if 'merchant' in df.columns:
        df = df.drop('merchant', axis=1)

    # 3. REMOVE CONSTANT / LOW-VARIANCE COLUMNS
    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(col, axis=1, inplace=True)

    # 4. REMOVE HIGHLY CORRELATED FEATURES
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # drop columns with correlation > 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df = df.drop(columns=to_drop, errors='ignore')

    print("Remaining columns:")
    print(df.columns)
    print("Shape:", df.shape)
    print(df.head())

    return df

    # df.to_csv("FE_clean_data.csv", index=False)

    # LABEL ENCODING -------------------------------------------------------------------------------

    # df = pd.read_csv("FE_clean_data.csv")

    # 1. IDENTIFY CATEGORICAL COLUMNS
    cat_cols = ["gender", "category"]

    print("Categorical columns:", cat_cols)

    # 2. APPLY ONE-HOT ENCODING
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    dummy_cols = df.select_dtypes(include='bool').columns
    df[dummy_cols] = df[dummy_cols].astype(int)

    # DONE
    print("After encoding shape:", df.shape)
    print(df.head())

    # df.to_csv("MT_clean_data.csv", index=False)