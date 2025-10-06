import torch, glob, os, gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from video_processor import extract_frames_from_video  # ✅ import frame extractor
import psutil  # optional for monitoring memory usage


# =====================================================
# ✅ Load Qwen2-VL model once at startup
# =====================================================
print("[BOOT] Loading Qwen2-VL model...")

model_id = "Qwen/Qwen2-VL-2B-Instruct"  # ✅ smaller, lighter model
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # ✅ bfloat16 can be slower on CPU
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print(f"[BOOT] Model loaded ✅ (using device: {device})")


# =====================================================
# ✅ Summarize a single video chunk
# =====================================================
def summarize_chunk(frame_dir: str) -> str:
    """Summarize one chunk by sampling and resizing its frames."""
    frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
    if not frames:
        return "No frames found in this segment."

    # ✅ Take up to 2 frames only (for low memory usage)
    images = [Image.open(f).convert("RGB") for f in frames[:2]]

    # ✅ Resize to smaller resolution (reduce memory footprint)
    images = [img.resize((480, 270)) for img in images]

    # 🧠 Use chat-style input expected by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {
                    "type": "text",
                    "text": (
                        "You are an AI fire-response assistant analyzing drone footage. "
                        "Summarize this segment — describe visible fire, smoke, "
                        "rescue activity, vehicle movement, and structural damage."
                    ),
                },
            ],
        }
    ]

    # Prepare model input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)

    print(f"[DEBUG] RAM before generate: {psutil.virtual_memory().percent}%")

    with torch.inference_mode():
        try:
            output = model.generate(**inputs, max_new_tokens=120)
            summary = processor.batch_decode(output, skip_special_tokens=True)[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "Skipped due to GPU out-of-memory."
        except Exception as e:
            return f"Error generating summary: {e}"

    # ✅ Clean up memory after inference
    del inputs, images, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary.strip()


# =====================================================
# ✅ Build timeline from all chunks
# =====================================================
def build_timeline(chunks_dir: str, frames_dir: str):
    """Process all chunks sequentially and return structured timeline."""
    timeline = []
    chunk_paths = sorted(glob.glob(f"{chunks_dir}/*.mp4"))

    for idx, chunk_path in enumerate(chunk_paths):
        print(f"[SUMMARY] Processing chunk {idx} → {chunk_path}")
        frame_dir = f"{frames_dir}/chunk_{idx}"
        os.makedirs(frame_dir, exist_ok=True)

        # ✅ Extract frames (0.5 fps = 1 frame every 2 seconds)
        try:
            extract_frames_from_video(chunk_path, frame_dir, fps=0.5)
            print(f"[FRAMES ✅] Extracted frames from {chunk_path} → {frame_dir}")
        except Exception as e:
            print(f"[FRAMES ❌] Failed extracting frames: {e}")
            continue

        # ✅ Summarize frames
        try:
            summary = summarize_chunk(frame_dir)
            print(f"[SUMMARY ✅] {summary[:120]}...")
        except Exception as e:
            print(f"[SUMMARY ❌] Failed on chunk {idx}: {e}")
            summary = f"Error processing chunk {idx}: {e}"

        # ✅ Memory cleanup after each chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        timeline.append({
            "chunk": idx,
            "video_file": os.path.basename(chunk_path),
            "summary": summary
        })

    return timeline
