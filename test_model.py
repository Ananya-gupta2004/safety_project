# test_model.py
# Tests trained model accuracy + live webcam preview
# Run: python3 test_model.py

from ultralytics import YOLO
import cv2
import os

# ── Check model exists ────────────────────────────────────────────
MODEL_PATH = 'runs/detect/models/hand_safety_v14/weights/best.pt'
DATA_YAML  = 'datasets/EgoHands Public.v1-specific.yolo26/data.yaml'

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run train_model.py first.")
    exit()

model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}\n")

# ── Validate on test set ──────────────────────────────────────────
print("Running validation on test dataset...")
metrics = model.val(data=DATA_YAML, verbose=False)

map50     = metrics.box.map50
precision = metrics.box.mp
recall    = metrics.box.mr

print("\n── Model Performance ──────────────────────────")
print(f"  mAP50     : {map50:.3f}   (target > 0.85)")
print(f"  Precision : {precision:.3f}   (target > 0.85)")
print(f"  Recall    : {recall:.3f}   (target > 0.85)")
print("───────────────────────────────────────────────")

if map50 >= 0.85:
    print("  RESULT: GOOD — model is ready to use")
elif map50 >= 0.70:
    print("  RESULT: ACCEPTABLE — try 100 epochs for better accuracy")
else:
    print("  RESULT: LOW — switch to yolov8s.pt and 100 epochs in train_model.py")

# ── Live webcam test ──────────────────────────────────────────────
print("\nStarting live webcam test...")
print("Hold your hand in front of the camera")
print("Press Q to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    print("If on WSL, make sure your webcam is connected and accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    count = len(results[0].boxes)
    color = (0, 200, 0) if count == 0 else (0, 0, 255)

    cv2.putText(annotated, f"Hands detected: {count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(annotated, "Press Q to quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Hand Detection Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test ended.")