import ast
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

app = Flask(__name__)

SIZE = 640
CONF = 0.25
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
LOCAL_FALLBACK = os.getenv("LOCAL_FALLBACK", "0") == "1"
client = genai.Client(api_key=GEMINI_KEY) if (genai and GEMINI_KEY) else None

MODEL_CANDIDATES = []
for _m in [os.getenv("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]:
    if _m and _m not in MODEL_CANDIDATES:
        MODEL_CANDIDATES.append(_m)
WORKING_MODEL = None

FRUITS = ["pineapple", "watermelon", "orange", "banana", "apple"]  # pineapple first (contains "apple")
ALIASES = {"mandarin": "Orange", "tangerine": "Orange", "clementine": "Orange"}

PROMPT = (
    "Detect every individual fruit of these types only: apple, orange, banana, pineapple, "
    "watermelon (whole or cut). Ignore all other fruits and objects. Include small, distant "
    "and partly hidden fruits. Use exactly one box per single fruit (one per banana, one per "
    "apple), never one box for a pile, bunch or crate. Return a JSON list of "
    "{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"apple|orange|banana|pineapple|watermelon\"} "
    "with box values normalized to 0-1000. If there are none, return []."
)


def clean_label(label):
    l = label.lower()
    if "custard" in l or "sugar apple" in l:
        return None
    for k, v in ALIASES.items():
        if k in l:
            return v
    for f in FRUITS:
        if f in l:
            return f.title()
    return None


# ---------- local models (optional fallback) ----------
def load(path, allowed):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])
    mask = np.array([names[i].lower() in allowed for i in range(len(names))], dtype=np.float32)
    return sess, sess.get_inputs()[0].name, names, mask


try:
    MODELS = [
        load("yolov8n.onnx", {"apple", "orange", "banana"}),
        load("best.onnx", {"pineapple", "watermelon"}),
    ]
except Exception as e:
    print("Local models not loaded:", repr(e))
    MODELS = []


def infer(model, img):
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


def tiles(img, n=2, overlap=0.2):
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


def local_detect(img):
    if not MODELS:
        return []
    dets = []
    for m in MODELS:
        dets += infer(m, img)
    for x0, y0, t in tiles(img):
        for a, b, c, d, cf, nm in infer(MODELS[0], t):
            dets.append((a + x0, b + y0, c + x0, d + y0, cf, nm))
    return drop_groups(nms(dets))


# ---------- Gemini (main) ----------
def generate(contents, config=None):
    """Call Gemini. Tries other model names if one is not found, retries on rate limit."""
    global WORKING_MODEL
    last = None
    for model in ([WORKING_MODEL] if WORKING_MODEL else MODEL_CANDIDATES):
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=model, contents=contents, config=config)
                WORKING_MODEL = model
                return r
            except Exception as e:
                last = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(4 * (attempt + 1))
                    continue
                break
    raise last


def ask(img):
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img)
    resp = generate(
        [types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), PROMPT],
        types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    return json.loads(resp.text)


def to_dets(items, x0, y0, w, h):
    dets = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = clean_label(str(it.get("label", "")))
        if not name:
            continue
        y1, x1, y2, x2 = [float(v) / 1000 for v in it["box_2d"]]
        dets.append((x0 + x1 * w, y0 + y1 * h, x0 + x2 * w, y0 + y2 * h, 1.0, name))
    return dets


def gemini_detect(img):
    jobs = [(0, 0, img)]
    if os.getenv("GEMINI_TILES", "1") == "1":
        jobs += list(tiles(img))

    def safe(job):
        x0, y0, t = job
        try:
            return to_dets(ask(t), x0, y0, t.shape[1], t.shape[0])
        except Exception as e:
            print("Gemini call failed:", repr(e))
            if x0 == 0 and y0 == 0 and t is img:
                raise
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(safe, jobs))
    merged = [d for r in results if r for d in r]
    return drop_groups(nms(merged, 0.4))


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return (
        f"ok | key: {'set' if GEMINI_KEY else 'MISSING'} | library: {'ok' if genai else 'MISSING'}"
        f" | models: {MODEL_CANDIDATES} | local models loaded: {len(MODELS)}"
    )


@app.get("/gemini-test")
def gemini_test():
    if genai is None:
        return "FAIL: google-genai library is not installed. Add google-genai to requirements.txt"
    if not GEMINI_KEY:
        return "FAIL: GEMINI_API_KEY is not set in Render Environment"
    try:
        r = generate("Reply with the single word OK")
        return f"WORKS: model={WORKING_MODEL} reply={(r.text or '').strip()[:50]}"
    except Exception as e:
        return "FAIL: " + repr(e)[:600]


@app.post("/api/detect")
def detect():
    try:
        f = request.files.get("image")
        if not f:
            return jsonify(error="Please choose an image first."), 400
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="Could not read this image. Try a JPG or PNG."), 400

        dets, reason = None, None
        if genai is None:
            reason = "google-genai library is not installed"
        elif not GEMINI_KEY:
            reason = "GEMINI_API_KEY is not set"
        else:
            try:
                dets = gemini_detect(img)
            except Exception as e:
                reason = str(e)[:200]
                print("Gemini failed:", repr(e))

        if dets is None:
            if LOCAL_FALLBACK:
                dets = local_detect(img)
            else:
                return jsonify(error=f"Gemini could not process this image: {reason}. Open /gemini-test for details."), 502

        preds = [
            {
                "x": float((d[0] + d[2]) / 2), "y": float((d[1] + d[3]) / 2),
                "width": float(d[2] - d[0]), "height": float(d[3] - d[1]),
                "class": d[5], "confidence": d[4],
            }
            for d in dets
        ]
        return jsonify(predictions=preds)
    except Exception as e:
        print("Detect failed:", repr(e))
        return jsonify(error="Something went wrong while detecting. Please try another image."), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
