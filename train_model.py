# train_model.py
# Trains YOLOv8 hand detection model
# Run: python3 train_model.py

from ultralytics import YOLO
import torch
import os

# ── Check GPU ─────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device.upper()}")
if device == 'cpu':
    print("No GPU detected — training on CPU. Takes longer but works fine.")

# ── Check dataset ─────────────────────────────────────────────────
DATA_YAML = '/home/krrish/safety_project/datasets/EgoHands Public.v1-specific.yolo26/data.yaml'
if not os.path.exists(DATA_YAML):
    print(f"\nERROR: Dataset not found at {DATA_YAML}")
    print("Please:")
    print("  1. Go to https://universe.roboflow.com/roboflow-100/hand-gestures-jkgyo")
    print("  2. Download Dataset → YOLOv8 format → Download zip")
    print("  3. Extract into datasets/hand-dataset/")
    exit()

print(f"Dataset found at: {DATA_YAML}")

# ── Load base model ───────────────────────────────────────────────
# yolov8n = nano  (fastest — good for Jetson Nano)
# yolov8s = small (more accurate, slightly slower)
model = YOLO('yolov8n.pt')   # auto-downloads pretrained weights first time

# ── Train ─────────────────────────────────────────────────────────
print("\nStarting training — this will take 20-60 minutes...")
print("Watch the mAP50 column — it should rise toward 0.85+\n")

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=512,        # 🔥 safer than 640
    batch=4,          # good for 4GB GPU
    device=device,
    workers=2,        # 🔥 prevents WSL overload
    cache=False,      # 🔥 avoids RAM spikes
    name='hand_safety_v1',
    project='models',
    patience=15,
    save=True,
    plots=True,
    verbose=True
)

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best weights : models/hand_safety_v1/weights/best.pt")
print(f"Results graph: models/hand_safety_v1/results.png")
print(f"\nNext step: run python3 test_model.py")