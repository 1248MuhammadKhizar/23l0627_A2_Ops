import os, sys, random
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_ds, IMG_SIZE

SEED = 42
random.seed(SEED); np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

def build_transfer(img_size=IMG_SIZE):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = layers.Input((img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="mobilenetv2_transfer"), base

def compile_model(m, lr):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

def trainable_report(model, base, title):
    names = [l.name for l in base.layers if l.trainable]
    head = [l.name for l in model.layers if l is not base and l.trainable and l.count_params() > 0]
    return "\n".join([f"== {title} ==",
                      f"Backbone layers trainable: {len(names)} of {len(base.layers)}",
                      f"Head layers trainable: {head}",
                      f"Backbone trainable names: {names if names else 'none (frozen)'}", ""])

def cbs(path, threshold=None):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True,
                                           initial_value_threshold=threshold),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
    ]

os.makedirs("models", exist_ok=True)
train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")
model, base = build_transfer()
report = []

compile_model(model, 1e-3)
report.append(trainable_report(model, base, "Stage 1 (backbone frozen, lr 1e-3)"))
print(report[-1])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=cbs("models/transfer_best.keras"))
best1 = min(h1.history["val_loss"])

base.trainable = True
for l in base.layers:
    l.trainable = (l.name.startswith("block_16") or l.name.startswith("Conv_1")) \
                  and not isinstance(l, layers.BatchNormalization)
compile_model(model, 1e-5)
report.append(trainable_report(model, base, "Stage 2 (last block unfrozen, lr 1e-5)"))
print(report[-1])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=cbs("models/transfer_best.keras", best1))

open("reports/trainable_layers.txt", "w").write("\n".join(report))

H = {k: h1.history[k] + h2.history[k] for k in ["loss", "val_loss", "accuracy", "val_accuracy"]}
best = int(np.argmin(H["val_loss"]))
b1 = len(h1.history["loss"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, k, t in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
    a.plot(H[k], label="train"); a.plot(H["val_" + k], label="val")
    a.axvline(b1 - 0.5, color="grey", ls=":", label="unfreeze last block")
    a.axvline(best, color="red", ls="--", label=f"best epoch {best + 1}")
    a.set_title("Transfer " + t); a.set_xlabel("epoch (stage 1 then 2)"); a.legend()
plt.tight_layout()
plt.savefig("reports/figures/transfer_curves.png", dpi=150)
print("BEST EPOCH (overall):", best + 1, "val_loss:", round(H["val_loss"][best], 4),
      "val_acc:", round(H["val_accuracy"][best], 4))
