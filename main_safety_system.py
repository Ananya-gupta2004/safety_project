# main_safety_system.py
# Full safety system: YOLO detection + state machine + GPIO
# On PC/WSL: runs in simulation mode (GPIO printed, not real)
# On Jetson: change SIMULATION_MODE = False
# Run: python3 main_safety_system.py

from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import cv2
import json
import time
import sqlite3
import numpy as np
import os
from datetime import datetime
def load_config():
    with open("safety_config.json", "r") as f:
        return json.load(f)

config = load_config()

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH      = 'models/hand_safety_v1/weights/best.pt'
DB_PATH = "logs/safety_log.db"
CAMERA_ID = config["camera"]["camera_id"]

CONFIDENCE_THR = config["model"]["confidence_threshold"]

WARNING_DISTANCE = config["tracking"]["warning_distance"]

PREDICT_TIME = config["tracking"]["prediction_time"]

CLEAR_TIMER_SEC = config["safety"]["clear_timer_sec"]
# ── Mode ──────────────────────────────────────────────────────────
SIMULATION_MODE = True   # True = PC/WSL,  False = Jetson (change when on Jetson)

# ── GPIO ──────────────────────────────────────────────────────────
if not SIMULATION_MODE:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO_PIN = config["gpio"]["pin"]
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.HIGH)

def gpio_safe():
    if SIMULATION_MODE:
        print("  [SIM GPIO] → HIGH (SAFE)")
    else:
        GPIO.output(GPIO_PIN, GPIO.HIGH)

def gpio_stop():
    if SIMULATION_MODE:
        print("  [SIM GPIO] → LOW  (STOP ✋)")
    else:
        GPIO.output(GPIO_PIN, GPIO.LOW)
# ── States ────────────────────────────────────────────────────────
STATE_SAFE        = "SAFE"
STATE_HAZARD      = "HAZARD"
STATE_LOCK        = "LOCK"
STATE_CLEAR_TIMER = "CLEAR_TIMER"
STATE_REARMED     = "REARMED"

# ── Database ──────────────────────────────────────────────────────
def init_db():
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp  TEXT,
        event      TEXT,
        state      TEXT,
        confidence REAL,
        gpio_level TEXT,
        fps        REAL
    )''')
    conn.commit()
    conn.close()


def log_event(event, state, confidence=None, gpio_level="HIGH", fps=0.0):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events VALUES (NULL,?,?,?,?,?,?)",
        (datetime.now().isoformat(), event, state, confidence, gpio_level, fps)
    )
    conn.commit()
    conn.close()

# ── Zone ──────────────────────────────────────────────────────────
def load_zone():
    if not os.path.exists('zone_config.json'):
        print("ERROR: zone_config.json not found.")
        print("Run setup_zone.py first to draw your danger zone.")
        exit()
    with open('zone_config.json') as f:
        data = json.load(f)
    return np.array(data["zone"], np.int32)

def box_in_zone(box_xyxy, zone_poly):
    x1, y1, x2, y2 = map(int, box_xyxy)
    test_pts = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        ((x1+x2)//2, (y1+y2)//2)
    ]
    poly = zone_poly.reshape((-1, 1, 2))
    return any(cv2.pointPolygonTest(poly, pt, False) >= 0 for pt in test_pts)
def point_distance_to_zone(point, zone_poly):
    poly = zone_poly.reshape((-1,1,2))
    return cv2.pointPolygonTest(poly, point, True)

# ── Main ──────────────────────────────────────────────────────────
def run():
    init_db()
    zone = load_zone()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run train_model.py first.")
        exit()

    

    model = YOLO(MODEL_PATH)

    tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=15,
    minimum_matching_threshold=0.8
)


    

    track_history = defaultdict(list)

    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    config["camera"]["width"]
)

    cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    config["camera"]["height"]
)

    if not cap.isOpened():
        gpio_stop()
        log_event("CAMERA_FAIL", STATE_HAZARD, gpio_level="LOW")
        print("[CRITICAL] Camera failed — GPIO LOW (fail-safe).")
        return

    state          = STATE_SAFE
    clear_start    = None
    fps_timer      = time.time()
    frame_count    = 0
    current_fps    = 0.0

    gpio_safe()
    log_event("SYSTEM_START", state)

    print("\n" + "="*50)
    print("  SAFETY SYSTEM RUNNING")
    print("="*50)
    print("  Platform  : Raspberry Pi 5 + AI Booster")
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Confidence: {CONFIDENCE_THR}")
    print(f"  Timer     : {CLEAR_TIMER_SEC}s")
    print(f"  Press Q in the camera window to quit")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                gpio_stop()
                log_event("FRAME_TIMEOUT", STATE_HAZARD, gpio_level="LOW")
                print("[CRITICAL] Camera frame failed — GPIO LOW (fail-safe).")
                break
            hand_in_zone = False
            warning_zone = False
            best_conf = 0.0

            # FPS counter
            frame_count += 1
            if frame_count >= 20:
                current_fps = 20 / (time.time() - fps_timer)
                fps_timer   = time.time()
                frame_count = 0

            # YOLO detection
            results = model(
               frame,
               conf=CONFIDENCE_THR,
               verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            detections = tracker.update_with_detections(
            detections
)

            hand_in_zone = False
            warning_zone = False
            best_conf = 0.0

            for idx in range(len(detections)):

             x1, y1, x2, y2 = detections.xyxy[idx]

             conf = float(detections.confidence[idx])

             if (
             detections.tracker_id is not None
             and len(detections.tracker_id) > idx
):
              track_id = int(detections.tracker_id[idx])
             else:
              track_id = idx

             cx = int((x1 + x2) / 2)
             cy = int((y1 + y2) / 2)

             current_time = time.time()

             vx = 0
             vy = 0
             dt=0

             if len(track_history[track_id]) > 0:

              px, py, pt = track_history[track_id][-1]

              dt = current_time - pt

             if dt > 0:
               vx = (cx - px) / dt
               vy = (cy - py) / dt
             velocity = (vx**2 + vy**2)**0.5

             track_history[track_id].append(
             (cx, cy, current_time)
    )

             if len(track_history[track_id]) > 30:
              track_history[track_id].pop(0)

             distance = point_distance_to_zone(
             (cx, cy),
             zone
    )

             future_x = cx + vx * PREDICT_TIME
             future_y = cy + vy * PREDICT_TIME

             future_distance = point_distance_to_zone(
             (int(future_x), int(future_y)),
             zone
    )

             if distance >= 0:
              hand_in_zone = True
              best_conf = max(best_conf, conf)

             elif (
             distance > -WARNING_DISTANCE
             or future_distance >= 0
    ):
              warning_zone = True

             cv2.rectangle(
             frame,
             (int(x1), int(y1)),
             (int(x2), int(y2)),
             (0,255,255),
             2
    )
             cv2.putText(
             frame,
             f"V:{velocity:.1f}",
             (int(x1), int(y2)+20),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.5,
             (255,255,255),
             1
    )

             cv2.putText(
             frame,
             f"ID:{track_id}",
             (int(x1), int(y1)-10),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.5,
             (0,255,255),
             2
    )

            # ── State machine ──────────────────────────────────────
            if state == STATE_SAFE:
               if hand_in_zone:
                    state = STATE_HAZARD
                    gpio_stop()
                    log_event("HAZARD_DETECTED", state, best_conf, "LOW", current_fps)
                    print(f"[HAZARD]      Hand detected! conf={best_conf:.2f}  fps={current_fps:.1f}")

            elif state == STATE_HAZARD:
                state = STATE_LOCK
                log_event("ENTER_LOCK", state, gpio_level="LOW", fps=current_fps)
                print("[LOCK]        Zone locked. Waiting for hand to leave...")

            elif state == STATE_LOCK:
                if not hand_in_zone:
                    state       = STATE_CLEAR_TIMER
                    clear_start = time.time()
                    log_event("CLEAR_TIMER_START", state, gpio_level="LOW")
                    print(f"[CLEAR_TIMER] Zone clear. Counting {CLEAR_TIMER_SEC}s...")

            elif state == STATE_CLEAR_TIMER:
                if hand_in_zone:
                    state       = STATE_LOCK
                    clear_start = None
                    log_event("TIMER_RESET", state, best_conf, "LOW")
                    print("[LOCK]        Hand returned — timer reset.")
                elif time.time() - clear_start >= CLEAR_TIMER_SEC:
                    state = STATE_REARMED
                    gpio_safe()
                    log_event("REARMED", state, gpio_level="HIGH")
                    print("[REARMED]     Zone verified clear. GPIO HIGH. Awaiting START.")

            elif state == STATE_REARMED:
                if hand_in_zone:
                    state = STATE_HAZARD
                    gpio_stop()
                    log_event("HAZARD_DETECTED", state, best_conf, "LOW")
                    print(f"[HAZARD]      New hazard detected. GPIO LOW.")
                else:
                    state = STATE_SAFE
                    log_event("SAFE", state, gpio_level="HIGH")
                    print("[SAFE]        Monitoring normally.")

            # ── Display overlay ────────────────────────────────────
            is_safe    = state in [STATE_SAFE, STATE_REARMED]
            zone_color = (0, 200, 0) if is_safe else (0, 0, 255)
            s_color    = (0, 200, 0) if is_safe else (0, 0, 255)

            cv2.polylines(frame, [zone.reshape((-1, 1, 2))], True, zone_color, 2)
            cv2.putText(frame, f"STATE: {state}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, s_color, 2)
            cv2.putText(frame, f"GPIO : {'HIGH — SAFE' if is_safe else 'LOW  — STOP'}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)
            cv2.putText(frame, f"FPS  : {current_fps:.1f}",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            if state == STATE_CLEAR_TIMER and clear_start:
                remaining = max(0, CLEAR_TIMER_SEC - (time.time() - clear_start))
                cv2.putText(frame, f"Clearing: {remaining:.1f}s",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if warning_zone and not hand_in_zone:
             cv2.putText(
                frame,
                "WARNING: HAND APPROACHING",
                (10,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
    
    )
            cv2.imshow("Safety System Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break

    except KeyboardInterrupt:
        print("\nShutdown requested by user.")

    finally:
        gpio_stop()
        log_event("SHUTDOWN", "SHUTDOWN", gpio_level="LOW")
        cap.release()
        cv2.destroyAllWindows()
        if not SIMULATION_MODE:
            GPIO.cleanup()
        print("System shut down cleanly.")

if __name__ == "__main__":
    run()
