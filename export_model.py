# export_model.py
# Exports trained model to ONNX format for Jetson
# Run AFTER training is complete and tested
# Run: python3 export_model.py

from ultralytics import YOLO
import os

MODEL_PATH = 'models/hand_safety_v1/weights/best.pt'

if not os.path.exists(MODEL_PATH):
    print("ERROR: Train the model first using train_model.py")
    exit()

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print("Exporting to ONNX format...")
model.export(format='onnx', imgsz=640, simplify=True)

onnx_path = MODEL_PATH.replace('.pt', '.onnx')
print(f"\nExport complete!")
print(f"  PT file  : {MODEL_PATH}    — copy this to Jetson")
print(f"  ONNX file: {onnx_path}  — for TensorRT on Jetson")
print(f"\nCopy both files to a USB drive for Jetson next week.")