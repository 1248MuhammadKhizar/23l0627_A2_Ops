import os, sys, subprocess
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dotenv import load_dotenv
load_dotenv()
import numpy as np, pandas as pd, tensorflow as tf, mlflow, yaml
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data import make_ds, IMG_SIZE

MODE = sys.argv[1] if len(sys.argv) > 1 else "manual"
LR = float(os.environ.get("LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
RUN_NAME = os.environ.get("RUN_NAME", f"transfer_{MODE}")
SEED = 42
tf.keras.utils.set_random_seed(SEED)

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
git_commit = sh("git rev-parse --short HEAD")
dvc_hash = yaml.safe_load(open("data/raw.dvc"))["outs"][0]["md5"]

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("visionguard")

base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False
inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

class MLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)

if MODE == "auto":
    mlflow.tensorflow.autolog(log_models=False)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tags({"git_commit": git_commit, "dvc_data_md5": dvc_hash, "seed": str(SEED), "logging_mode": MODE})
    if MODE == "manual":
        mlflow.log_params({"lr": LR, "epochs": EPOCHS, "img_size": IMG_SIZE, "backbone": "MobileNetV2",
                           "frozen": "all", "dropout": 0.3, "augmentation": "flip,rot,zoom,contrast"})
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    if MODE == "manual":
        cbs.append(MLflowCallback())
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=2)

    tdf = pd.read_csv("data/test.csv")
    ptest = model.predict(make_ds("data/test.csv"), verbose=0).ravel()
    y, yp = tdf.label.values, (ptest >= 0.5).astype(int)
    mlflow.log_metrics({"test_accuracy": accuracy_score(y, yp), "test_precision": precision_score(y, yp, zero_division=0),
                        "test_recall": recall_score(y, yp), "test_f1": f1_score(y, yp), "test_auc": roc_auc_score(y, ptest)})
    print("done", MODE, "test_auc", round(roc_auc_score(y, ptest), 4))
