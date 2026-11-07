import os, sys
sys.path.insert(0, "src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf, keras_tuner as kt, pandas as pd
from tensorflow.keras import layers, models
from data import make_ds, IMG_SIZE
tf.keras.utils.set_random_seed(42)

def build(hp):
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                             include_top=False, weights="imagenet")
    n = hp.Choice("unfrozen_layers", [0, 10, 30])
    base.trainable = True
    for l in base.layers:
        l.trainable = False
    if n:
        k = 0
        for l in reversed(base.layers):
            if not isinstance(l, layers.BatchNormalization):
                l.trainable = True
                k += 1
                if k == n:
                    break
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-5, 1e-3, sampling="log")),
              loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")
t = kt.RandomSearch(build, objective="val_loss", max_trials=8, seed=42,
                    directory="tuner_dir", project_name="visionguard", overwrite=True)
t.search(train_ds, validation_data=val_ds, epochs=3, verbose=0)
rows = [dict(trial=tr.trial_id, **tr.hyperparameters.values, val_loss=round(tr.score, 4))
        for tr in t.oracle.get_best_trials(8)]
tab = pd.DataFrame(rows)
tab.to_csv("reports/tuner_visionguard.csv", index=False)
print(tab.to_string(index=False))
print("BEST:", t.get_best_hyperparameters(1)[0].values)
print("\nMean val_loss by unfrozen_layers:\n", tab.groupby("unfrozen_layers").val_loss.mean().round(4))
print("Correlation of lr with val_loss:", round(tab.lr.corr(tab.val_loss), 3))
print("Correlation of dropout with val_loss:", round(tab.dropout.corr(tab.val_loss), 3))
