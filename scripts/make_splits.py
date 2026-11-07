import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("reports/audit.csv")
df = df[~df.corrupt]
df = df.drop_duplicates("sha256", keep="first")  # drops image34.jpg
df["label"] = (df.cls == "ai").astype(int)       # ai = 1, real = 0

train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label"]].to_csv(f"data/{name}.csv", index=False)
    print(name, len(part), part.label.value_counts().to_dict())
print("total", len(df))
