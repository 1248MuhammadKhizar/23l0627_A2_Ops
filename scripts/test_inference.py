import sys
sys.path.insert(0, "src")
import pandas as pd
import inference
df = pd.read_csv("data/test.csv")
for label in (0, 1):
    path = df[df.label == label].path.iloc[0]
    r = inference.predict(open(path, "rb").read())
    print("true:", "ai" if label else "real", "| file:", path.split("\\")[-1], "->", r["label"], f"({r['confidence']:.2%})")
