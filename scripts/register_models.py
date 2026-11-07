import os
from dotenv import load_dotenv
load_dotenv()
import mlflow, pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
t = pd.read_csv("reports/comparison.csv").sort_values(["val_auc", "val_loss"], ascending=[False, True])
best, second = t.iloc[0], t.iloc[1]
print("champion candidate:", best.run, "val_auc", best.val_auc, "| runner-up:", second.run, "val_auc", second.val_auc)

NAME = "visionguard_classifier"
try:
    c.create_registered_model(NAME)
except Exception as e:
    print("model exists:", str(e)[:80])

v = {}
for tag, row in (("v_best", best), ("v_second", second)):
    mv = c.create_model_version(NAME, source=f"runs:/{row.run_id}/model", run_id=row.run_id,
                                description=f"{row.run}, val_auc {row.val_auc}")
    v[tag] = mv.version
    print("registered", row.run, "as version", mv.version)

c.set_registered_model_alias(NAME, "champion", v["v_best"])
print("alias champion ->", v["v_best"])
for mv in c.search_model_versions(f"name='{NAME}'"):
    print("version", mv.version, "run", mv.run_id[:8], "aliases", mv.aliases)
