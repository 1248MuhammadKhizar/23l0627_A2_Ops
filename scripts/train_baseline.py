import os, sys, random
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_ds
from baseline import build_baseline

SEED = 42
EPOCHS = 5
random.seed(SEED); np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

os.makedirs("models", exist_ok=True)
train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

model = build_baseline()
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/baseline_best.keras", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
]
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

h = hist.history
best = int(np.argmin(h["val_loss"]))
print("BEST EPOCH:", best + 1, "val_loss:", round(h["val_loss"][best], 4), "val_acc:", round(h["val_accuracy"][best], 4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, k, t in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
    a.plot(h[k], label="train"); a.plot(h["val_" + k], label="val")
    a.axvline(best, color="red", ls="--", label=f"best epoch {best + 1}")
    a.set_title("Baseline " + t); a.set_xlabel("epoch"); a.legend()
plt.tight_layout()
plt.savefig("reports/figures/baseline_curves.png", dpi=150)
