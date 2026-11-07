import os, sys, subprocess
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dotenv import load_dotenv
load_dotenv()
import numpy as np, pandas as pd, tensorflow as tf, mlflow, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from data import make_ds, IMG_SIZE

LR = float(os.environ.get("LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
AUG = os.environ.get("AUG", "1") == "1"
UNF = int(os.environ.get("UNF", "0"))
BB = os.environ.get("BB", "mobilenetv2")
RUN_NAME = os.environ.get("RUN_NAME", "run")
SEED = 42
tf.keras.utils.set_random_seed(SEED)
os.makedirs("artifacts", exist_ok=True); os.makedirs("models", exist_ok=True)

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
git_commit = sh("git rev-parse --short HEAD")
dvc_hash = yaml.safe_load(open("data/raw.dvc"))["outs"][0]["md5"]

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

shape = (IMG_SIZE, IMG_SIZE, 3)
if BB == "efficientnetb0":
    base = tf.keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet")
else:
    base = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet")
base.trainable = True
for l in base.layers:
    l.trainable = False
if UNF:
    k = 0
    for l in reversed(base.layers):
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True
            k += 1
            if k == UNF:
                break

inp = layers.Input(shape)
x = inp
if BB == "mobilenetv2":
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)   # EfficientNet has its own rescaling inside
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

train_ds = make_ds("data/train.csv", training=True, aug=AUG)
val_ds = make_ds("data/val.csv")
test_ds = make_ds("data/test.csv")

class MLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tags({"git_commit": git_commit, "dvc_data_md5": dvc_hash, "seed": str(SEED),
                     "logging_mode": "manual", "series": "comparison"})
    mlflow.log_params({"lr": LR, "epochs": EPOCHS, "img_size": IMG_SIZE, "backbone": BB,
                       "unfrozen_layers": UNF, "augmentation": int(AUG), "dropout": 0.3})
    h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                              restore_best_weights=True),
                             MLflowCallback()]).history
    best = int(np.argmin(h["val_loss"]))
    pv = model.predict(val_ds, verbose=0).ravel()
    pt = model.predict(test_ds, verbose=0).ravel()
    yv = pd.read_csv("data/val.csv").label.values
    y = pd.read_csv("data/test.csv").label.values
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[int(np.argmax([f1_score(yv, pv >= t, zero_division=0) for t in grid]))])
    yp = (pt >= thr).astype(int)
    mlflow.log_metrics({"best_val_loss": h["val_loss"][best], "best_val_auc": h["val_auc"][best],
                        "best_epoch": best + 1, "threshold": thr,
                        "val_f1": f1_score(yv, pv >= thr),
                        "test_accuracy": accuracy_score(y, yp),
                        "test_precision": precision_score(y, yp, zero_division=0),
                        "test_recall": recall_score(y, yp), "test_f1": f1_score(y, yp),
                        "test_auc": roc_auc_score(y, pt)})

    # ---- 3.4 artifacts
    cm = confusion_matrix(y, yp)
    plt.figure(figsize=(3.5, 3.5)); plt.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        plt.text(b, a, v, ha="center", va="center")
    plt.xticks([0, 1], ["real", "ai"]); plt.yticks([0, 1], ["real", "ai"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(RUN_NAME); plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(y, pt)
    plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {RUN_NAME}"); plt.tight_layout()
    plt.savefig("artifacts/roc_curve.png", dpi=120); plt.close()
    open("artifacts/packages.txt", "w").write(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    model.save(f"models/{RUN_NAME}.keras")
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/roc_curve.png")
    mlflow.log_artifact("artifacts/packages.txt")
    for f in ("train", "val", "test"):
        mlflow.log_artifact(f"data/{f}.csv", artifact_path="splits")
    mlflow.log_artifact(f"models/{RUN_NAME}.keras", artifact_path="model")
    print("done", RUN_NAME, "val_loss", round(h["val_loss"][best], 4), "val_auc", round(h["val_auc"][best], 4))
