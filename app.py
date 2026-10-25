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


def infer(model, img):
    """Run one model on a BGR image. Returns (x1, y1, x2, y2, conf, name) tuples."""
    sess, inp, names, mask = model
    h, w = img.shape[:2]
    r = SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255
    out = sess.run(None, {inp: x})[0][0].T
    scores = out[:, 4:] * mask
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf > CONF
    boxes, cls, conf = out[keep, :4] / r, cls[keep], conf[keep]
    return [
        (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(c), names[int(k)].title())
        for (cx, cy, bw, bh), c, k in zip(boxes, conf, cls)
    ]

   def tiles(img, n=3, overlap=0.2):
    h, w = img.shape[:2]
    th, tw = int(h / n * (1 + overlap)), int(w / n * (1 + overlap))
    for i in range(n):
        for j in range(n):
            y0 = min(int(i * h / n), h - th)
            x0 = min(int(j * w / n), w - tw)
            yield x0, y0, img[y0:y0 + th, x0:x0 + tw]


def nms(dets, thr=0.5):
    out = []
    for name in {d[5] for d in dets}:
        ds = [d for d in dets if d[5] == name]
        rects = [[int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])] for d in ds]
        idx = cv2.dnn.NMSBoxes(rects, [d[4] for d in ds], CONF, thr)
        out += [ds[int(i)] for i in np.array(idx).flatten()]
    return out


def drop_groups(dets):
    """Remove one big box that covers 3 or more smaller boxes of the same fruit."""
    keep = []
    for a in dets:
        inside = sum(
            1 for b in dets
            if b is not a and b[5] == a[5]
            and a[0] <= (b[0] + b[2]) / 2 <= a[2] and a[1] <= (b[1] + b[3]) / 2 <= a[3]
        )
        if inside < 3:
            keep.append(a)
    return keep


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

        dets = []
        for m in MODELS:
            dets += infer(m, img)
        for x0, y0, t in tiles(img):  # small fruits: also scan tiles (COCO model only)
            for a, b, c, d, cf, nm in infer(MODELS[0], t):
                dets.append((a + x0, b + y0, c + x0, d + y0, cf, nm))
                 dets = drop_groups(nms(dets))
           H, W = img.shape[:2]
           if len(dets) >= 4:
               dets = [
                   d for d in dets
                   if d[5] in ("Pineapple", "Watermelon", "Grape", "Grapes")
                   or (d[2] - d[0]) * (d[3] - d[1]) < 0.10 * W * H
               ]
        preds = [
            {
                "x": float((d[0] + d[2]) / 2), "y": float((d[1] + d[3]) / 2),
                "width": float(d[2] - d[0]), "height": float(d[3] - d[1]),
                "class": d[5], "confidence": d[4],
            }
            for d in dets
        ]
        return jsonify(predictions=preds)
    except Exception:
        return jsonify(error="Something went wrong while detecting. Please try another image."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
