import os, glob, json, torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from video_processor import extract_keyframes

# ✅ Define model to use
MODEL_ID = "liuhaotian/llava-v1.6-mistral-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[BOOT] Loading LLaVA model...")
model_name = get_model_name_from_path(MODEL_ID)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_ID, None, model_name
)
print("[BOOT] Model loaded ✅")

def summarize_chunk(frame_dir: str) -> str:
    """Summarize one video chunk by sampling frames."""
    frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
    if not frames:
        return "No frames found in this segment."

    # Take a few frames for context
    images = [Image.open(f).convert("RGB") for f in frames[:5]]

    prompt = (
        "You are an AI fire response assistant observing drone footage. "
        "Summarize what happens in this segment — note fire intensity, smoke behavior, "
        "building condition, firefighter movement, and visible hazards."
    )

    # Encode prompt with an image token
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(DEVICE)
    image_tensor = image_processor.preprocess(images[0], return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=180,
            temperature=0.7,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def build_timeline(chunks_dir="data/chunks", frames_dir="data/frames"):
    """Generate summaries for all video chunks and build a mission timeline."""
    os.makedirs(frames_dir, exist_ok=True)
    timeline = []

    for chunk_file in sorted(os.listdir(chunks_dir)):
        chunk_name = os.path.splitext(chunk_file)[0]
        frame_dir = os.path.join(frames_dir, chunk_name)
        os.makedirs(frame_dir, exist_ok=True)

        # Extract frames from chunk
        extract_keyframes(os.path.join(chunks_dir, chunk_file), frame_dir, fps=1)

        # Summarize frames
        summary = summarize_chunk(frame_dir)
        print(f"[{chunk_name}] {summary}")
        timeline.append({"chunk": chunk_name, "summary": summary})

    # Save timeline to JSON
    with open("data/timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)

    return timeline
