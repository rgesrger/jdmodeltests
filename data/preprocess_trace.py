import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AzureTrace.csv")
df = df.iloc[:int(len(df) * 0.1)]  # slice returns a DataFrame
print("Token statistics:")
print(df['ContextTokens'].describe())
bins = [0, 256, 1000, 4000, 8000]
labels = ["small", "medium", "large", "xl"]

df["token_bucket"] = pd.cut(df["ContextTokens"], bins=bins, labels=labels, include_lowest=True)
print(df["token_bucket"].value_counts(normalize=True))

plt.figure(figsize=(10, 6))
df["token_bucket"].value_counts().sort_index().plot(kind='bar')
plt.title("Token Bucket Distribution")
plt.xlabel("Bucket")
plt.ylabel("Count")
plt.show()

df = df[["TIMESTAMP", "token_bucket"]]
print(df.head())
# newdf.to_parquet("AzureTrace.parquet", engine="pyarrow", index=False)
