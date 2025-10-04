import cv2, os

def split_video(video_path, out_dir="data/chunks", chunk_len=60):
    """Split video into chunks (default: 60s)."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        raise ValueError("Cannot read video FPS.")

    os.makedirs(out_dir, exist_ok=True)
    chunk_idx, frame_idx = 0, 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (fps * chunk_len) == 0:
            if writer: writer.release()
            out_path = f"{out_dir}/chunk_{chunk_idx}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (frame.shape[1], frame.shape[0]))
            chunk_idx += 1

        writer.write(frame)
        frame_idx += 1

    if writer: writer.release()
    cap.release()
    print(f"[VIDEO] Split into {chunk_idx} chunks.")


def extract_keyframes(video_path, out_dir, fps=1):
    """Extract frames (default: 1 per sec) from video."""
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if video_fps == 0:
        raise ValueError("Cannot read FPS.")
    frame_interval = max(1, video_fps // fps)

    os.makedirs(out_dir, exist_ok=True)
    count, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1
    cap.release()
    print(f"[FRAMES] Extracted {saved} frames → {out_dir}")
