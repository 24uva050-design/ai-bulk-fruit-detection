import ast

import cv2
import numpy as np
import onnxruntime as ort
from flask import Blueprint, jsonify, render_template, request

bp = Blueprint("quality", __name__)
SIZE, CONF = 640, 0.30
_m = None


def _load():
    global _m
    if _m is None:
        s = ort.InferenceSession("quality.onnx", providers=["CPUExecutionProvider"])
        names = ast.literal_eval(s.get_modelmeta().custom_metadata_map["names"])
        _m = (s, s.get_inputs()[0].name, names)
    return _m


def _infer(img):
    s, inp, names = _load()
    h, w = img.shape[:2]
    r = SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255
    out = s.run(None, {inp: x})[0][0].T
    cls, conf = out[:, 4:].argmax(1), out[:, 4:].max(1)
    k = conf > CONF
    boxes, cls, conf = out[k, :4] / r, cls[k], conf[k]
    return [(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(c), names[int(i)])
            for (cx, cy, bw, bh), c, i in zip(boxes, conf, cls)]


def _tiles(img, n=2, ov=0.2):
    h, w = img.shape[:2]
    th, tw = int(h / n * (1 + ov)), int(w / n * (1 + ov))
    for i in range(n):
        for j in range(n):
            y0, x0 = min(int(i * h / n), h - th), min(int(j * w / n), w - tw)
            yield x0, y0, img[y0:y0 + th, x0:x0 + tw]


@bp.get("/quality")
def page():
    return render_template("quality.html")


@bp.post("/api/quality")
def quality():
    try:
        f = request.files.get("image")
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
        if img is None:
            return jsonify(error="Please choose a JPG or PNG image."), 400
        try:
            dets = _infer(img)
        except Exception as e:
            print("quality model error:", repr(e))
            return jsonify(error="Quality model is not available yet (quality.onnx missing)."), 503
        for x0, y0, t in _tiles(img):
            dets += [(a + x0, b + y0, c + x0, d + y0, cf, n) for a, b, c, d, cf, n in _infer(t)]
        keep = []
        for n in {d[5] for d in dets}:
            ds = [d for d in dets if d[5] == n]
            rects = [[int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])] for d in ds]
            idx = cv2.dnn.NMSBoxes(rects, [d[4] for d in ds], CONF, 0.5)
            keep += [ds[int(i)] for i in np.array(idx).flatten()]
        preds = []
        for d in keep:
            name = d[5].replace("-", " ").replace("_", " ").title().replace("Benana", "Banana")
            preds.append({"x": (d[0] + d[2]) / 2, "y": (d[1] + d[3]) / 2, "width": d[2] - d[0],
                          "height": d[3] - d[1], "class": name,
                          "state": "rotten" if "rotten" in name.lower() else "fresh"})
        return jsonify(predictions=preds)
    except Exception as e:
        print("quality failed:", repr(e))
        return jsonify(error="Something went wrong. Please try another image."), 500
