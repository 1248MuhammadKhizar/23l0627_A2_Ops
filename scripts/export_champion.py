import os, json, shutil
from dotenv import load_dotenv
load_dotenv()
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
NAME = "visionguard_classifier"
mv = c.get_model_version_by_alias(NAME, "champion")
run = c.get_run(mv.run_id)
folder = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model")
src = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".keras")][0]
os.makedirs("models", exist_ok=True)
shutil.copy(src, "models/champion.keras")
meta = {"name": NAME, "version": mv.version, "run_name": run.data.tags.get("mlflow.runName"),
        "img_size": int(run.data.params["img_size"]), "threshold": run.data.metrics["threshold"]}
json.dump(meta, open("models/champion.json", "w"), indent=2)
print(meta)
