import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "shaganaas-workspace")
WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "fruit-counting-pipeline-1792730881345")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
URL = f"https://serverless.roboflow.com/infer/workflows/{WORKSPACE}/{WORKFLOW}"


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/detect")
def detect():
    if not API_KEY:
        return jsonify(error="ROBOFLOW_API_KEY is not set"), 500
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400
    b64 = base64.b64encode(file.read()).decode()
    body = {"inputs": {"image": {"type": "base64", "value": b64}}}
    try:
        r = requests.post(
            URL, json=body, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return jsonify(error=str(e)), 502
    return jsonify(r.json())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
