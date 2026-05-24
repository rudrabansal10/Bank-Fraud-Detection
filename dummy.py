import pandas as pd

df = pd.read_csv("FE_clean_data.csv")
print(df.info())

print(df.nunique())