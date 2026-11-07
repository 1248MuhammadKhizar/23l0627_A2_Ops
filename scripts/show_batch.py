import sys
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_ds

ds = make_ds("data/train.csv", training=True, batch=16)
imgs, labels = next(iter(ds))
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for ax, im, lb in zip(axes.flat, imgs, labels):
    ax.imshow(im.numpy().astype("uint8"))
    ax.set_title("ai" if lb.numpy() == 1 else "real")
    ax.axis("off")
plt.tight_layout()
plt.savefig("reports/figures/augmented_batch.png", dpi=150)
print("batch shape:", imgs.shape, "dtype:", imgs.dtype)
