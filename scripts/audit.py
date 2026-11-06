import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("data/raw")
CLASSES = {"AiArtData": "ai", "RealArt": "real"}

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

rows = []
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    cls = next((CLASSES[part] for part in p.parts if part in CLASSES), None)
    if cls is None:
        continue
    row = dict(path=str(p), cls=cls, size_bytes=p.stat().st_size, width=None,
               height=None, mode=None, format=None, sha256=sha256(p),
               corrupt=False, error="")
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as e:
        row["corrupt"] = True
        row["error"] = repr(e)[:200]
    rows.append(row)

df = pd.DataFrame(rows)
df["is_duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("reports/audit.csv", index=False)

ok = df[~df.corrupt]
pd.set_option("display.width", 200)
print("\n== Images per class ==\n", df.groupby("cls").size())
print("\n== Format / colour mode per class ==\n", ok.groupby(["cls", "format", "mode"]).size())
print("\n== min / median / max ==\n", ok.groupby("cls")[["width", "height", "size_bytes"]].agg(["min", "median", "max"]).T)
print("\nCorrupt/unreadable files:", int(df.corrupt.sum()))
print("Exact duplicate files (extra copies):", int(df.is_duplicate.sum()))
cross = df.groupby("sha256")["cls"].nunique()
print("Hashes that appear in BOTH classes:", int((cross > 1).sum()))
