import requests

url = "http://127.0.0.1:8000/process_video/"
params = {"video_path": "data/input/cityoverview.mp4"}

response = requests.post(url, params=params)

print("STATUS:", response.status_code)
print("RESPONSE:")
print(response.text)
