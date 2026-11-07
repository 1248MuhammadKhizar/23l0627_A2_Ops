import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"baseline": ("models/baseline_best.keras", 112),
          "transfer": ("models/transfer_best.keras", 128)}

def load_ds(csv, size):
    df = pd.read_csv(csv)
    def _load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        img.set_shape((size, size, 3))
        return img
    ds = tf.data.Dataset.from_tensor_slices(df.path.values).map(_load).batch(32)
    return df, ds

rows, roc, preds, thr_out = [], {}, {}, {}
fig_cm, axcm = plt.subplots(1, 2, figsize=(9, 4))
for i, (name, (path, size)) in enumerate(MODELS.items()):
    model = tf.keras.models.load_model(path)
    vdf, vds = load_ds("data/val.csv", size)
    tdf, tds = load_ds("data/test.csv", size)
    pv = model.predict(vds, verbose=0).ravel()
    pt = model.predict(tds, verbose=0).ravel()
    # threshold chosen on VALIDATION only: maximise F1
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(vdf.label, pv >= t, zero_division=0) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    thr_out[name] = thr
    yp = (pt >= thr).astype(int)
    y = tdf.label.values
    rows.append(dict(model=name, threshold=round(thr, 2),
                     accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                     recall=recall_score(y, yp), f1=f1_score(y, yp),
                     roc_auc=roc_auc_score(y, pt)))
    cm = confusion_matrix(y, yp)
    axcm[i].imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        axcm[i].text(b, a, v, ha="center", va="center", fontsize=14)
    axcm[i].set_xticks([0, 1]); axcm[i].set_yticks([0, 1])
    axcm[i].set_xticklabels(["real", "ai"]); axcm[i].set_yticklabels(["real", "ai"])
    axcm[i].set_xlabel("predicted"); axcm[i].set_ylabel("true")
    axcm[i].set_title(f"{name} (thr {thr:.2f})")
    roc[name] = roc_curve(y, pt)
    preds[name] = (tdf, pt, yp)
fig_cm.tight_layout(); fig_cm.savefig("reports/figures/confusion_matrices.png", dpi=150)

plt.figure(figsize=(5, 5))
for name, (fpr, tpr, _) in roc.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC {rows[list(MODELS).index(name)]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test)"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/roc_curves.png", dpi=150)

res = pd.DataFrame(rows).round(3)
res.to_csv("reports/test_metrics.csv", index=False)
json.dump(thr_out, open("reports/thresholds.json", "w"))
print(res.to_string(index=False))

# 2.6 error analysis on the transfer model: 10 most confident mistakes
tdf, pt, yp = preds["transfer"]
tdf = tdf.assign(prob_ai=pt, pred=yp)
wrong = tdf[tdf.pred != tdf.label].copy()
wrong["conf"] = np.where(wrong.pred == 1, wrong.prob_ai, 1 - wrong.prob_ai)
wrong = wrong.sort_values("conf", ascending=False).head(10)
audit = pd.read_csv("reports/audit.csv")[["path", "width", "height", "format", "size_bytes"]]
wrong = wrong.merge(audit, on="path", how="left")
wrong[["path", "label", "pred", "prob_ai", "width", "height", "format", "size_bytes"]].to_csv(
    "reports/misclassified.csv", index=False)
print("\nMisclassified (n=%d total wrong on test: %d)" % (len(wrong), int((tdf.pred != tdf.label).sum())))
print(wrong[["path", "label", "pred", "prob_ai", "width", "height", "format", "size_bytes"]]
      .assign(path=wrong.path.str.split("\\").str[-1]).round(2).to_string())
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, (_, r) in zip(axes.flat, wrong.iterrows()):
    ax.imshow(Image.open(r.path).convert("RGB")); ax.axis("off")
    ax.set_title(f"true {'ai' if r.label else 'real'} / pred {'ai' if r.pred else 'real'}\np(ai)={r.prob_ai:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig("reports/figures/misclassified.png", dpi=150)
