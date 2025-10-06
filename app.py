from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
import os, json, traceback

app = FastAPI(title="Fire Drone Summarizer", version="1.0")

# Load model safely
print("[BOOT] Loading Qwen2-VL model...")
try:
    from summarizer import build_timeline  # ✅ only load after app starts
    from video_processor import split_video
    print("[BOOT] Model loaded ✅")
except Exception as e:
    print("[BOOT ❌] Failed to load model:")
    traceback.print_exc()

# ✅ Debug handler — shows *every* exception in terminal & API
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("\n🔥 ERROR TRACEBACK 🔥\n", tb)
    return PlainTextResponse(tb, status_code=500)

@app.post("/process_video/")
def process_video(video_path: str = Query(..., description="Path to the input video file")):
    print(f"\n[PROCESS] Request received for video: {video_path}\n")

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return {"error": f"Video not found: {video_path}"}

    os.makedirs("data/chunks", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)

    print("[STEP 1] Splitting video into chunks...")
    from video_processor import split_video
    split_video(video_path, out_dir="data/chunks")
    print("[STEP 1 ✅] Video split done.")

    print("[STEP 2] Building timeline (summarizing)...")
    from summarizer import build_timeline
    timeline = build_timeline("data/chunks", "data/frames")
    print("[STEP 2 ✅] Summaries done.")

    with open("data/timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)
    print("[STEP 3 ✅] Timeline saved to data/timeline.json")

    return {"message": "Processing complete ✅", "timeline": timeline}

@app.get("/summaries/")
def get_summaries():
    if not os.path.exists("data/timeline.json"):
        return {"timeline": []}
    with open("data/timeline.json") as f:
        return json.load(f)
