import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf, keras_tuner as kt, pandas as pd
tf.keras.utils.set_random_seed(42)
(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x, y, xv, yv = x[:10000] / 255.0, y[:10000], xv[:2000] / 255.0, yv[:2000]

def build(hp):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 128, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

t = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, seed=42,
                    directory="toy_tuner", project_name="fmnist", overwrite=True)
t.search(x, y, validation_data=(xv, yv), epochs=3, verbose=0)
rows = [dict(trial=tr.trial_id, **tr.hyperparameters.values, val_acc=round(tr.score, 4))
        for tr in t.oracle.get_best_trials(6)]
tab = pd.DataFrame(rows)
tab.to_csv("reports/tuner_toy.csv", index=False)
print(tab.to_string(index=False))
print("BEST:", t.get_best_hyperparameters(1)[0].values)
