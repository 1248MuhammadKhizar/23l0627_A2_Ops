import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dotenv import load_dotenv
load_dotenv()
import mlflow, tensorflow as tf
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy_fashion_mnist_dagshub")
tf.keras.utils.set_random_seed(42)
(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x, y, xv, yv = x[:10000] / 255.0, y[:10000], xv[:2000] / 255.0, yv[:2000]
for lr in (0.01, 0.0005):
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.set_tag("purpose", "toy lab 3.2")
        mlflow.log_params({"lr": lr, "units": 64, "epochs": 5})
        m = tf.keras.Sequential([tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="softmax")])
        m.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(x, y, validation_data=(xv, yv), epochs=5, verbose=0).history
        for ep in range(5):
            mlflow.log_metrics({"loss": h["loss"][ep], "val_accuracy": h["val_accuracy"][ep]}, step=ep)
        print("logged lr", lr, "val_acc", round(h["val_accuracy"][-1], 4))
