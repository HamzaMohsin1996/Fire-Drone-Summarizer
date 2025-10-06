# server.py
import os
import cv2
import gc
import time
import base64
import asyncio
import traceback
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from sort import Sort  # keep sort.py in same folder

# ---------------- CONFIG ----------------
MOBILE_SAM_CHECKPOINT = "models/mobile_sam.pt"
YOLO_MODEL_PATH = "yolov8m.pt"             # use yolov8n.pt on CPU if slow
QWEN_MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"

HOST = "0.0.0.0"
PORT = 8001                                # <— WebSocket on 8001 to match your frontend

CONF_THRESHOLD      = 0.15
IOU_THRESHOLD       = 0.45
FRAME_SKIP          = 3
MOTION_MIN_MEAN     = 5
PERSON_MOTION_MIN   = 3
VALID_CLASSES       = {"person", "car", "bus", "truck", "train"}

SUMMARY_INTERVAL    = 30.0                 # seconds between summaries
SUMMARY_MAX_IMAGES  = 1                    # keep 1 on CPU to avoid OOM
SUMMARY_IMG_SIZE    = (640, 360)
MASK_DOWNSCALE      = 1.0
TRACK_EXPIRY        = 5.0                  # dedup per track for N seconds
MISSION_SAVE_INTERVAL = 30.0
THUMB_MAX_W         = 360                  # downscale thumbnails for frontend

os.makedirs("data", exist_ok=True)

# ---------------- LOAD MobileSAM (optional) ----------------
print(f"[BOOT] Checking MobileSAM checkpoint: {MOBILE_SAM_CHECKPOINT}")
print("DEBUG: MobileSAM path exists:", Path(MOBILE_SAM_CHECKPOINT).exists())
sam_predictor = None
try:
    import mobile_sam
    print("DEBUG: mobile_sam module @", mobile_sam.__file__)
    from mobile_sam import sam_model_registry, SamPredictor
    if Path(MOBILE_SAM_CHECKPOINT).exists():
        sam = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_CHECKPOINT)
        sam.to("cpu")
        sam_predictor = SamPredictor(sam)
        print("[BOOT] ✅ MobileSAM loaded")
    else:
        print("ℹ️ MobileSAM checkpoint not found; running without SAM.")
except Exception as e:
    print(f"ℹ️ MobileSAM not available: {e}. Running without SAM.")

# ---------------- LOAD YOLO + SORT ----------------
print(f"[BOOT] Loading YOLO: {YOLO_MODEL_PATH}")
det_model = YOLO(YOLO_MODEL_PATH)
print("[BOOT] ✅ YOLO loaded")

tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

# ---------------- LOAD Qwen2-VL ----------------
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

print("[BOOT] Loading Qwen2-VL model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.bfloat16),
    device_map="auto",
    trust_remote_code=True
)
print(f"[BOOT] ✅ Qwen2-VL loaded (device: {device})")

# ---------------- APP STATE ----------------
app = FastAPI(title="Fire-Drone Mission Server")

track_memory         = {}     # track_id -> last_seen_ts
latest_summary_data  = None
last_summary_time    = 0.0
mission_timeline     = []     # list of {ts, coord, summary, detections}

# Zone memory (mission memory buffer)
MISSION_MEMORY = defaultdict(lambda: {
    "detections": {},   # aggregated counts (label -> count)
    "timeline": [],     # [{ts, summary, detections, deltas}]
    "last_summary": "",
})

# background subtractor for motion
bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# ---------------- HELPERS ----------------
def decode_frame(frame_b64: str):
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("⚠️ decode_frame: cv2.imdecode returned None")
        return frame
    except Exception as e:
        print(f"❌ decode_frame error: {e}")
        return None

def _resize_keep_w(img_bgr, max_w=THUMB_MAX_W):
    h, w = img_bgr.shape[:2]
    if w <= max_w:
        return img_bgr
    scale = max_w / float(w)
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def crop_to_base64(frame, box):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], [0,0,0,0], [w, h, w, h])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        print("⚠️ crop_to_base64: empty ROI", box)
        return None
    roi = _resize_keep_w(roi, THUMB_MAX_W)   # downscale for smaller payloads
    ok, buffer = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        print("⚠️ cv2.imencode failed for ROI")
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def mask_to_base64(mask, box, frame_shape):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], [0,0,0,0], [w, h, w, h])
    if y2 <= y1 or x2 <= x1:
        return None
    submask = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    # encode as RGBA with alpha = mask
    rgba = np.zeros((submask.shape[0], submask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = submask
    ok, buffer = cv2.imencode(".png", rgba)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def motion_score(fgmask, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fgmask.shape[1]-1, x2), min(fgmask.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = fgmask[y1:y2, x1:x2]
    return float(patch.mean())

def run_sam_on_box(frame_bgr, box_xyxy):
    if sam_predictor is None:
        return None
    try:
        sam_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        masks, _, _ = sam_predictor.predict(box=np.array(box_xyxy)[None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8)
        return mask
    except Exception as e:
        print(f"SAM error: {e}")
        return None

def compute_scene_delta(prev_detections, new_detections):
    # prev_detections/new_detections: list of det dicts
    def tally(dets):
        counts = {}
        for d in dets:
            lbl = d["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    prev = tally(prev_detections)
    curr = tally(new_detections)

    labels = set(prev) | set(curr)
    changes = []
    for lbl in sorted(labels):
        a, b = prev.get(lbl, 0), curr.get(lbl, 0)
        if b > a:
            changes.append(f"🟢 +{b-a} {lbl}(s)")
        elif a > b:
            changes.append(f"🔴 -{a-b} {lbl}(s)")

    return "No major changes." if not changes else "; ".join(changes)

def qwen_summarize_images(pil_images):
    """
    Provide 1..N small PIL images to Qwen and get a dispatcher-focused summary.
    """
    try:
        images = [img.resize(SUMMARY_IMG_SIZE) for img in pil_images[:SUMMARY_MAX_IMAGES]]

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {
                        "type": "text",
                        "text": (
                            "You are an emergency mission assistant. "
                            "Summarize this drone footage concisely for a dispatcher. "
                            "Include counts of people and vehicles (approx.), any visible fire/smoke, "
                            "rescue activity, blocked/at-risk areas, and urgent hazards or changes."
                        )
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(device)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=160)

        summary = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        # Remove system preface if present
        return summary.replace("system\nYou are a helpful assistant.\n", "").strip()
    except Exception as e:
        print(f"❌ Qwen summarize error: {e}")
        return f"[summary_error] {e}"

# ---------- Zone memory helpers ----------
def get_zone_id(coord):
    """Bucket GPS into ~50–100m tiles (coarse)."""
    if not coord or len(coord) != 2:
        return "unknown"
    lon, lat = coord
    return f"zone_{round(lat, 3)}_{round(lon, 3)}"

def update_zone_memory(coord, detections, summary_text):
    zone = get_zone_id(coord)
    now = int(time.time())
    zone_data = MISSION_MEMORY[zone]

    # aggregate label counts for snapshot
    snapshot = {}
    for d in detections:
        snapshot[d["label"]] = snapshot.get(d["label"], 0) + 1

    # compute deltas vs previous snapshot
    prev = zone_data["detections"]
    deltas = []
    for label, cnt in snapshot.items():
        prev_cnt = prev.get(label, 0)
        if cnt > prev_cnt:
            deltas.append(f"🟢 +{cnt - prev_cnt} {label}(s)")
    for label, prev_cnt in prev.items():
        if snapshot.get(label, 0) < prev_cnt:
            deltas.append(f"🔴 -{prev_cnt - snapshot.get(label, 0)} {label}(s)")

    if deltas:
        print(f"[ZONE Δ] {zone}: {', '.join(deltas)}")

    zone_data["detections"] = snapshot
    zone_data["last_summary"] = summary_text
    zone_data["timeline"].append({
        "ts": now,
        "summary": summary_text,
        "detections": snapshot,
        "deltas": deltas
    })

    # persist zone memory
    try:
        with open("data/mission_memory.json", "w") as f:
            json.dump(MISSION_MEMORY, f, indent=2)
    except Exception as e:
        print("⚠️ mission_memory.json write error:", e)

    return zone

# ---------------- WEBSOCKET ----------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connected")
    global latest_summary_data, last_summary_time

    frame_counter = 0
    last_persist = time.time()

    try:
        while True:
            data = await ws.receive_json()
            frame_b64 = data.get("frame")
            coord     = data.get("coord")

            if not frame_b64:
                continue

            # Decode frame
            frame = decode_frame(frame_b64)
            if frame is None:
                continue

            frame_counter += 1
            if frame_counter % FRAME_SKIP != 0:
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bg_sub.apply(gray)
            fg = cv2.medianBlur(fg, 3)

            # YOLO inference
            results = det_model(frame, imgsz=960, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

            dets_for_sort, raw_boxes = [], []
            for det in results[0].boxes:
                cls_id = int(det.cls)
                label  = det_model.names[cls_id].lower()
                if label not in VALID_CLASSES:
                    continue

                conf = float(det.conf)
                x1, y1, x2, y2 = map(int, det.xyxy[0])

                # drop very large static vehicle if low conf
                area = (x2 - x1) * (y2 - y1)
                rel_area = area / (h * w + 1e-6)
                if label in {"car", "bus", "truck", "train"} and rel_area > 0.20 and conf < 0.80:
                    print(f"🪟 dropped large static {label} (conf {conf:.2f}, rel_area {rel_area:.2f})")
                    continue

                mscore = motion_score(fg, (x1, y1, x2, y2))
                if (label == "person" and mscore < PERSON_MOTION_MIN) or (label != "person" and mscore < MOTION_MIN_MEAN):
                    continue

                dets_for_sort.append([x1, y1, x2, y2, conf])
                raw_boxes.append((x1, y1, x2, y2, conf, label, mscore))

            # SORT tracking
            detections = []
            if len(dets_for_sort) > 0:
                tracks = tracker.update(np.array(dets_for_sort))
                now = time.time()

                for t in tracks:
                    # SORT returns [x1, y1, x2, y2, id]
                    x1, y1, x2, y2, track_id = map(int, t)

                    # nearest raw box
                    match = min(raw_boxes, key=lambda b: abs(b[0]-x1)+abs(b[1]-y1))
                    _, _, _, _, conf, label, mscore = match

                    # de-dup
                    last_seen = track_memory.get(track_id, 0)
                    if now - last_seen < TRACK_EXPIRY:
                        continue
                    track_memory[track_id] = now

                    # Mask (SAM or rectangle fallback)
                    mask = run_sam_on_box(frame, (x1, y1, x2, y2))
                    if mask is None:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        mask[y1:y2, x1:x2] = 1

                    thumb    = crop_to_base64(frame, (x1, y1, x2, y2))
                    mask_png = mask_to_base64(mask, (x1, y1, x2, y2), frame.shape)

                    # Debug: why thumbnails not showing?
                    if thumb is None:
                        print(f"⚠️ Thumbnail is None for {label} id={track_id} bbox={x1,y1,x2,y2}")
                    else:
                        print(f"🖼️ thumb len={len(thumb)} label={label} id={track_id}")

                    detections.append({
                        "id": str(track_id),
                        "label": label,
                        "score": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "coord": coord,
                        "thumbnail": thumb,
                        "mask_png": mask_png,
                        "motion": mscore,
                        "ts": int(now * 1000),
                    })

                    print(f"🧩 Detection: {label} ({conf:.2f}) motion={mscore:.1f} id={track_id}")

            # ---- periodic snapshot summary ----
            now_t = time.time()
            if (now_t - last_summary_time) > SUMMARY_INTERVAL:
                prev_det = mission_timeline[-1]["detections"] if mission_timeline else []
                delta_text = compute_scene_delta(prev_det, detections)

                # one PIL snapshot for Qwen
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                summary_text = qwen_summarize_images([pil])
                summary_text = f"{delta_text}\n{summary_text}"

                latest_summary_data = {"ts": int(now_t * 1000), "text": summary_text}
                last_summary_time = now_t

                mission_timeline.append({
                    "ts": int(now_t * 1000),
                    "coord": coord,
                    "summary": summary_text,
                    "detections": detections
                })

                # zone memory
                zone_id = update_zone_memory(coord, detections, summary_text)
                print(f"📍 Zone updated: {zone_id}")

                # persist timeline (periodically)
                if (now_t - last_persist) > MISSION_SAVE_INTERVAL:
                    try:
                        with open("data/mission_timeline.json", "w") as f:
                            json.dump(mission_timeline, f, indent=2)
                        last_persist = now_t
                        print(f"💾 mission_timeline.json saved ({len(mission_timeline)} entries)")
                    except Exception as e:
                        print("⚠️ mission_timeline.json write error:", e)

                # memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Send to client
            payload = {"events": detections, "scene_summary": latest_summary_data}
            await ws.send_json(payload)

            await asyncio.sleep(0.03)

    except Exception as e:
        print("❌ WebSocket error:", e)
        traceback.print_exc()
    finally:
        print("🔌 WebSocket disconnected")

# ---------------- HTTP ENDPOINTS ----------------
@app.get("/timeline")
def get_timeline():
    return JSONResponse({"count": len(mission_timeline), "timeline": mission_timeline})

@app.get("/mission/recap")
def get_mission_recap():
    recap = {
        zone: {
            "detections": data["detections"],
            "last_summary": data["last_summary"],
            "entries": len(data["timeline"])
        }
        for zone, data in MISSION_MEMORY.items()
    }
    return JSONResponse(recap)

@app.get("/mission/zone/{zone_id}")
def get_zone_timeline(zone_id: str):
    data = MISSION_MEMORY.get(zone_id)
    if not data:
        return JSONResponse({"error": "zone not found"}, status_code=404)
    return JSONResponse(data)

@app.get("/health")
def health():
    return JSONResponse({"ok": True, "qwen_device": device})

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
