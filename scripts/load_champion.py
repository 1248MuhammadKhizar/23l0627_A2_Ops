import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dotenv import load_dotenv
load_dotenv()
import mlflow, numpy as np, pandas as pd, tensorflow as tf
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
NAME = "visionguard_classifier"
mv = c.get_model_version_by_alias(NAME, "champion")
run = c.get_run(mv.run_id)
size = int(run.data.params["img_size"])
print("alias champion -> version", mv.version, "run", run.data.tags.get("mlflow.runName"), "img_size", size)

folder = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model")
path = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".keras")][0]
model = tf.keras.models.load_model(path)

img_path = sys.argv[1] if len(sys.argv) > 1 else pd.read_csv("data/test.csv").path.iloc[0]
img = tf.io.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
img = tf.image.resize(img, (size, size))[None, ...]
p = float(model.predict(img, verbose=0)[0][0])
print("image:", os.path.basename(img_path))
print("p(ai) = %.3f -> %s" % (p, "AI-generated" if p >= 0.5 else "real"))
