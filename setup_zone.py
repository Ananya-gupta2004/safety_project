# setup_zone.py
# Draw your danger zone on the camera feed and save it
# Run ONCE during machine setup
# Left-click to add corners, S to save, C to clear, Q to quit

import cv2
import json
import numpy as np

zone_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))
        print(f"  Point added: ({x}, {y})  — total: {len(zone_points)} points")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    print("If on WSL, webcam access may need setup — see note below.")
    exit()

cv2.namedWindow("Draw Danger Zone")
cv2.setMouseCallback("Draw Danger Zone", mouse_callback)

print("── Zone Setup ──────────────────────────────────")
print("  Click on the camera feed to mark zone corners")
print("  Minimum 3 points needed (4 recommended for a box)")
print("  S = Save    C = Clear    Q = Quit without saving")
print("────────────────────────────────────────────────\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw each clicked point
    for pt in zone_points:
        cv2.circle(frame, pt, 6, (0, 0, 255), -1)

    # Draw polygon when 3+ points exist
    if len(zone_points) >= 3:
        pts_array = np.array(zone_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # Semi-transparent green fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    # On-screen instructions
    cv2.putText(frame, f"Points: {len(zone_points)}  |  S=Save  C=Clear  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if len(zone_points) >= 3:
        cv2.putText(frame, "Zone ready — press S to save",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Draw Danger Zone", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(zone_points) < 3:
            print("  Need at least 3 points. Keep clicking on the feed.")
        else:
            with open("zone_config.json", "w") as f:
                json.dump({
              "zone": zone_points,
              "width": 640,
              "height": 480
                           }, f, indent=2)
            print(f"\n  Zone saved to zone_config.json")
            print(f"  Points saved: {zone_points}")
            print(f"\n  Next step: run python3 main_safety_system.py")
            break

    elif key == ord('c'):
        zone_points = []
        print("  Zone cleared — click again to restart")

    elif key == ord('q'):
        print("  Quit without saving.")
        break
    elif key == ord('u'):
     if len(zone_points) > 0:
        removed = zone_points.pop()
        print(f"Removed point {removed}")

cap.release()
cv2.destroyAllWindows()
