import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import mlflow, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("sqlite:///mlflow_toy.db")
mlflow.set_experiment("toy_fashion_mnist")
tf.keras.utils.set_random_seed(42)

(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x, y, xv, yv = x[:10000] / 255.0, y[:10000], xv[:2000] / 255.0, yv[:2000]

for lr in (0.01, 0.0005):
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.set_tag("purpose", "toy lab 3.1")
        mlflow.log_params({"lr": lr, "units": 64, "epochs": 5, "optimizer": "adam"})
        m = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")])
        m.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(x, y, validation_data=(xv, yv), epochs=5, verbose=0).history
        for ep in range(5):
            mlflow.log_metrics({"loss": h["loss"][ep], "val_loss": h["val_loss"][ep],
                                "accuracy": h["accuracy"][ep],
                                "val_accuracy": h["val_accuracy"][ep]}, step=ep)
        plt.figure(figsize=(5, 3.5))
        plt.plot(h["loss"], label="train"); plt.plot(h["val_loss"], label="val")
        plt.title(f"Loss, lr={lr}"); plt.legend(); plt.tight_layout()
        plt.savefig("toy_loss.png", dpi=120); plt.close()
        mlflow.log_artifact("toy_loss.png")
        m.save("toy_model.keras")
        mlflow.log_artifact("toy_model.keras", artifact_path="model")
        print("lr", lr, "final val_acc", round(h["val_accuracy"][-1], 4))
