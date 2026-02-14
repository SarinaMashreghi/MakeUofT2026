from ultralytics import YOLO
from pathlib import Path
import torch

# Where you want to store models
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Load (this auto-downloads if not present)
# model = YOLO("yolov5n-face.pt")
model = torch.hub.load("deepcam-cn/yolov5-face", "yolov5n-face", pretrained=True)


# Explicitly save a copy locally (optional but clean)
local_path = model_dir / "yolov5n-face.pt"
model.save(local_path)

print("Model saved to:", local_path)
