# 🔥 Fire Drone Summarizer

This project automatically analyzes **drone footage of fire incidents** and generates **AI-powered summaries** for each part of the video using a **vision-language model (LLaVA)**.

It is built to help **fire command centers** or **first responders** get a timeline of what happened — including **fire spread, building conditions, firefighter actions, civilians, and hazards**.

---

## 🧠 Features

✅ Splits videos into smaller, easy-to-analyze segments  
✅ Extracts keyframes from each video chunk  
✅ Generates scene summaries using **LLaVA (Large Language and Vision Assistant)**  
✅ Produces a structured mission log (`timeline.json`)  
✅ Offers a **FastAPI** server for easy integration into dashboards or web apps  

---

## 🗂️ Project Structure

fire_drone_summarizer/
│
├── app.py # FastAPI app (main backend entry)
├── summarizer.py # LLaVA summarization logic
├── video_processor.py # Video chunking and frame extraction
├── requirements.txt # Dependencies list
├── .gitignore # Git ignored files/folders
│
└── data/
├── input/ # Input drone videos (ignored in git)
├── chunks/ # Temporary split video segments
├── frames/ # Extracted keyframes
└── timeline.json # Final summary output

mkdir -p data/input data/chunks data/frames


python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

fastapi
uvicorn
transformers
torch
torchvision
pillow
opencv-python
llava==1.2.2.post1
pip install -r requirements.txt
cd data/input
curl -L -o cityoverview.mp4 "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
cd ../..
uvicorn app:app --reload
curl -X POST "http://127.0.0.1:8000/process_video/?video_path=data/input/cityoverview.mp4"
