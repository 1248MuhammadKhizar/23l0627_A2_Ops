import os, sys
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.utils import secure_filename
import inference

ALLOWED = {"jpg", "jpeg", "png"}
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "5"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(200))
    label = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    p_ai = db.Column(db.Float)
    model_name = db.Column(db.String(120))
    threshold = db.Column(db.Float)

with app.app_context():
    db.create_all()

def error(msg, code):
    return render_template("error.html", message=msg, code=code), code

@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f or not f.filename:
        return error("Please choose an image file first.", 400)
    name = secure_filename(f.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        return error("Only .jpg, .jpeg and .png files are accepted.", 400)
    data = f.read()                      # kept in memory, never written to disk
    try:
        r = inference.predict(data)
    except inference.BadImage:
        return error("That file looks like an image but could not be decoded. It may be corrupt.", 400)
    db.session.add(Prediction(filename=name, label=r["label"], confidence=r["confidence"],
                              p_ai=r["p_ai"], model_name=r["model"], threshold=r["threshold"]))
    db.session.commit()
    return render_template("result.html", r=r, filename=name)

@app.route("/history")
def history():
    rows = Prediction.query.order_by(Prediction.id.desc()).limit(200).all()
    return render_template("history.html", rows=rows)

@app.route("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        dbok = True
    except Exception:
        dbok = False
    return jsonify(status="ok" if dbok else "degraded", model=inference.MODEL_NAME, database=dbok), (200 if dbok else 503)

@app.errorhandler(413)
def too_big(e):
    return error(f"File is too large. The limit is {MAX_MB} MB.", 413)

@app.errorhandler(404)
def not_found(e):
    return error("Page not found.", 404)

@app.errorhandler(500)
def server_error(e):
    return error("Something went wrong on our side. Please try again.", 500)

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1")
