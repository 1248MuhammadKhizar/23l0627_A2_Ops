import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline(img_size=224):
    inp = layers.Input((img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inp)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

if __name__ == "__main__":
    m = build_baseline()
    m.summary()
    print()
    print(f"{'Layer':32s}{'Output shape':26s}{'Params':>10s}")
    for l in m.layers:
        print(f"{l.name:32s}{str(l.output.shape):26s}{l.count_params():>10,d}")
    print("Total params:", f"{m.count_params():,}")
