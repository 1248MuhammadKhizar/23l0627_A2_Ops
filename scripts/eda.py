import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("reports/audit.csv")
ok = df[~df.corrupt]
os.makedirs("reports/figures", exist_ok=True)

n = 8
fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
for r, cls in enumerate(["real", "ai"]):
    sample = ok[ok.cls == cls].sample(n, random_state=42)
    for c, path in enumerate(sample.path):
        ax = axes[r, c]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
        if c == 0:
            ax.set_title(cls, loc="left")
plt.tight_layout()
plt.savefig("reports/figures/samples.png", dpi=150)
plt.close()

for col, label in [("width", "Width (px)"), ("height", "Height (px)"), ("size_bytes", "File size (bytes)")]:
    plt.figure(figsize=(6, 4))
    for cls in ["real", "ai"]:
        plt.hist(ok[ok.cls == cls][col], bins=40, alpha=0.6, label=cls)
    plt.xlabel(label)
    plt.ylabel("Images")
    plt.title(label + " by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/hist_{col}.png", dpi=150)
    plt.close()
print("saved")
