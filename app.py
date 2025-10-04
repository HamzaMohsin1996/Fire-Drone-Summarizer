from fastapi import FastAPI, Query
from video_processor import split_video
from summarizer import build_timeline
import os, json

app = FastAPI()

@app.post("/process_video/")
def process_video(video_path: str = Query(..., description="Path to the input video file")):
    """Split, summarize, and build timeline for given video."""
    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    # Ensure output directories exist
    os.makedirs("data/chunks", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)

    # Split the video into chunks
    split_video(video_path, out_dir="data/chunks")

    # Summarize each chunk
    timeline = build_timeline("data/chunks", "data/frames")

    # Save the timeline
    with open("data/timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)

    return {"timeline": timeline}


@app.get("/summaries/")
def get_summaries():
    """Return latest mission log (timeline)."""
    if not os.path.exists("data/timeline.json"):
        return {"timeline": []}
    with open("data/timeline.json") as f:
        return json.load(f)
