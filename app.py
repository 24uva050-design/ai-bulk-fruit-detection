import ast

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

SIZE = 640
CONF = 0.25


def load(path, allowed):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])
    mask = np.array([names[i].lower() in allowed for i in range(len(names))], dtype=np.float32)
    return sess, sess.get_inputs()[0].name, names, mask


# Individual fruits come from the COCO model, the rest from our own model.
MODELS = [
    load("yolov8n.onnx", {"apple", "orange", "banana"}),
    load("best.onnx", {"pineapple", "watermelon", "grape", "grapes"}),
]


def run(model, x, r):
    sess, inp, names, mask = model
    out = sess.run(None, {inp: x})[0][0].T
    scores = out[:, 4:] * mask
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf > CONF
    boxes, cls, conf = out[keep, :4] / r, cls[keep], conf[keep]
    rects = [[int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)] for cx, cy, bw, bh in boxes]
    idx = cv2.dnn.NMSBoxes(rects, conf.tolist(), CONF, 0.5) if rects else []
    return [
        {
            "x": float(boxes[i][0]), "y": float(boxes[i][1]),
            "width": float(boxes[i][2]), "height": float(boxes[i][3]),
            "class": names[int(cls[i])].title(), "confidence": float(conf[i]),
        }
        for i in np.array(idx).flatten()
    ]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return "ok"


@app.post("/api/detect")
def detect():
    try:
        f = request.files.get("image")
        if not f:
            return jsonify(error="Please choose an image first."), 400
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="Could not read this image. Try a JPG or PNG."), 400

        h, w = img.shape[:2]
        r = SIZE / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255

        preds = []
        for m in MODELS:
            preds += run(m, x, r)
        return jsonify(predictions=preds)
    except Exception:
        return jsonify(error="Something went wrong while detecting. Please try another image."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
