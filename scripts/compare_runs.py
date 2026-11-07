import os
from dotenv import load_dotenv
load_dotenv()
import mlflow, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
df = mlflow.search_runs(experiment_names=["visionguard"], filter_string="tags.series = 'comparison'")
df = df.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
cols = {"tags.mlflow.runName": "run", "params.lr": "lr", "params.augmentation": "aug",
        "params.unfrozen_layers": "unfrozen", "params.img_size": "size", "params.backbone": "backbone",
        "metrics.best_val_loss": "val_loss", "metrics.best_val_auc": "val_auc",
        "metrics.val_f1": "val_f1", "run_id": "run_id"}
t = df[list(cols)].rename(columns=cols).sort_values("val_loss").round(4)
t.to_csv("reports/comparison.csv", index=False)
print(t.drop(columns="run_id").to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].barh(t.run, t.val_loss); ax[0].set_title("Validation loss (lower is better)")
ax[1].barh(t.run, t.val_auc); ax[1].set_title("Validation AUC (higher is better)")
plt.tight_layout(); plt.savefig("reports/figures/comparison.png", dpi=150)
