import ast

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

SIZE = 640
CONF = 0.35
sess = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
INPUT = sess.get_inputs()[0].name
NAMES = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/detect")
def detect():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded"), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Invalid image"), 400

    h, w = img.shape[:2]
    r = SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255

    out = sess.run(None, {INPUT: x})[0][0].T  # (8400, 4 + classes)
    scores = out[:, 4:]
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf > CONF
    boxes, cls, conf = out[keep, :4] / r, cls[keep], conf[keep]

    rects = [[int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)] for cx, cy, bw, bh in boxes]
    idx = cv2.dnn.NMSBoxes(rects, conf.tolist(), CONF, 0.5) if rects else []
    preds = [
        {
            "x": float(boxes[i][0]), "y": float(boxes[i][1]),
            "width": float(boxes[i][2]), "height": float(boxes[i][3]),
            "class": NAMES[int(cls[i])], "confidence": float(conf[i]),
        }
        for i in np.array(idx).flatten()
    ]
    return jsonify(predictions=preds)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
