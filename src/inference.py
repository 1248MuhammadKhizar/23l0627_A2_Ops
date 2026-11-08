import os, io, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.keras")
META_PATH = os.environ.get("MODEL_META", "models/champion.json")
meta = json.load(open(META_PATH))
SIZE = int(meta["img_size"])
THRESHOLD = float(os.environ.get("THRESHOLD", meta["threshold"]))
MODEL_NAME = f"{meta['name']} v{meta['version']} ({meta['run_name']})"

# loaded ONCE at start-up, not per request
_model = tf.keras.models.load_model(MODEL_PATH)

class BadImage(Exception):
    pass

def preprocess(data: bytes):
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        img = tf.io.decode_image(data, channels=3, expand_animations=False)   # same as training
    except Exception as e:
        raise BadImage(str(e))
    img = tf.image.resize(img, (SIZE, SIZE))                                  # same resize as training
    return img[None, ...]        # the model applies its own rescaling internally

def predict(data: bytes):
    p = float(_model.predict(preprocess(data), verbose=0)[0][0])
    ai = p >= THRESHOLD
    return {"label": "AI-generated" if ai else "Real", "confidence": p if ai else 1 - p,
            "p_ai": p, "model": MODEL_NAME, "threshold": THRESHOLD}
