import time
import serial
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from flask import Flask, Response, jsonify, render_template
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

ESP = None
try:
    ESP = serial.Serial("COM3", 115200, timeout=0.1, write_timeout=0.05)
    time.sleep(2)
    print("Connected to ESP32 on COM3")
except Exception as e:
    print(f"WARN: Could not connect to ESP32 on COM3: {e}")
    print("Running in camera/web-only mode.")

# Motion-control state to reduce jitter and command spam
_LAST_CMD = "S"
_LAST_CMD_TS = 0.0
_LATEST_JPEG = None
_FRAME_LOCK = threading.Lock()
_FRAME_COND = threading.Condition(_FRAME_LOCK)
_LATEST_FRAME_ID = 0
_FRAME_COUNT = 0
_PROCESSING_STARTED = False

def on_happy_detected(person_id: int, emotion: str, confidence: float) -> None:
    """TODO: Add robot/audio behavior for happy emotion."""
    pass


def on_sad_detected(person_id: int, emotion: str, confidence: float) -> None:
    """TODO: Add robot/audio behavior for non-happy emotion (treated as sad)."""
    pass


def classify_person_emotion(
    person_bgr,
    image_processor: AutoImageProcessor,
    emotion_model: AutoModelForImageClassification,
    id2label: Dict[int, str],
) -> Tuple[str, float]:
    person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    inputs = image_processor(images=person_rgb, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    label = str(id2label.get(idx, idx)).lower()
    return label, conf


def send_motion_command(cmd: str, cooldown_s: float) -> None:
    """Send motion command with cooldown and duplicate suppression."""
    global _LAST_CMD, _LAST_CMD_TS, ESP
    now = time.time()
    if cmd == _LAST_CMD and (now - _LAST_CMD_TS) < cooldown_s:
        return
    
    print("sending motion", cmd)
    if ESP is not None:
        try:
            ESP.write(cmd.encode())
        except Exception as e:
            # Prevent serial issues from stalling the vision/web loop.
            print(f"WARN: ESP write failed, disabling serial output: {e}")
            try:
                ESP.close()
            except Exception:
                pass
            ESP = None
    _LAST_CMD = cmd
    _LAST_CMD_TS = now


def get_ultrasonic_distance_cm() -> float | None:
    """TODO: Parse ultrasonic distance from ESP serial and return cm."""
    return None


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)


@app.get("/")
def index() -> str:
    return render_template("index.html")


def _mjpeg_generator():
    last_sent_id = -1
    while True:
        with _FRAME_COND:
            # Wait until a new frame is published.
            while _LATEST_JPEG is None or _LATEST_FRAME_ID == last_sent_id:
                _FRAME_COND.wait(timeout=0.5)
            frame = _LATEST_JPEG
            last_sent_id = _LATEST_FRAME_ID
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.get("/video_feed")
def video_feed():
    resp = Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/status")
def status():
    with _FRAME_LOCK:
        has_frame = _LATEST_JPEG is not None
    return jsonify(
        {
            "streaming": has_frame,
            "processing_started": _PROCESSING_STARTED,
            "frame_count": _FRAME_COUNT,
        }
    )


def publish_frame(frame_bgr) -> None:
    global _LATEST_JPEG, _FRAME_COUNT, _LATEST_FRAME_ID
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if not ok:
        return
    with _FRAME_COND:
        _LATEST_JPEG = encoded.tobytes()
        _LATEST_FRAME_ID += 1
        _FRAME_COND.notify_all()
    _FRAME_COUNT += 1


def run_web_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def open_camera_with_fallback(preferred_index: int):
    # Windows webcams often vary by backend; validate by reading a frame.
    trial_indices = [preferred_index, 0, 1, 2]
    seen = set()
    ordered_indices = []
    for idx in trial_indices:
        if idx not in seen:
            ordered_indices.append(idx)
            seen.add(idx)

    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for idx in ordered_indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Give backend a short warm-up and verify we can actually read frames.
            ok = False
            for _ in range(20):
                read_ok, test_frame = cap.read()
                if read_ok and test_frame is not None and test_frame.size > 0:
                    ok = True
                    break
                time.sleep(0.03)

            if ok:
                print(f"Camera opened at index={idx}, backend={backend} (frame read OK)")
                return cap, idx

            print(f"WARN: Camera index={idx}, backend={backend} opened but returned no frames.")
            cap.release()
    raise RuntimeError("Could not open any camera index (tried 0,1,2).")


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = max(1, area_a + area_b - inter)
    return inter / union


def smooth_bbox(
    prev_bbox: Tuple[int, int, int, int] | None,
    curr_bbox: Tuple[int, int, int, int],
    alpha: float = 0.7,
) -> Tuple[int, int, int, int]:
    """Exponential smoothing to reduce per-frame bbox jitter."""
    if prev_bbox is None:
        return curr_bbox
    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = curr_bbox
    x1 = int(alpha * px1 + (1.0 - alpha) * cx1)
    y1 = int(alpha * py1 + (1.0 - alpha) * cy1)
    x2 = int(alpha * px2 + (1.0 - alpha) * cx2)
    y2 = int(alpha * py2 + (1.0 - alpha) * cy2)
    return x1, y1, x2, y2


def go_to_person(person_id: int, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> None:
    """Orient robot toward person center and move forward when centered."""
    del person_id  # reserved for future multi-target tracking

    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape
    del frame_h

    cx = (x1 + x2) * 0.5
    mx = frame_w * 0.5
    e = (cx - mx) / max(1.0, mx)  # normalized center error in [-1, 1]
    abs_e = abs(e)

    # Hysteresis-style thresholds reduce oscillation.
    deadband = 0.10
    enter_turn = 0.18
    strong_turn = 0.45

    if abs_e <= deadband:
        send_motion_command("F", cooldown_s=0.5)
        return

    if abs_e < enter_turn:
        send_motion_command("F", cooldown_s=0.5)
        return

    if e < 0:
        if abs_e >= strong_turn:
            send_motion_command("L", cooldown_s=0.5)
        else:
            send_motion_command("L", cooldown_s=1)
    else:
        if abs_e >= strong_turn:
            send_motion_command("R", cooldown_s=0.5)
        else:
            send_motion_command("R", cooldown_s=1)


def reached_person(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    area_ratio_threshold: float = 0.20,
    ultrasonic_distance_cm: float | None = None,
    ultrasonic_threshold_cm: float = 45.0,
) -> bool:
    """Return True when vision OR ultrasonic says robot is close enough."""
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_h * frame_w)
    vision_close = (box_area / frame_area) >= area_ratio_threshold

    ultrasonic_close = False
    if ultrasonic_distance_cm is not None and ultrasonic_distance_cm > 0:
        ultrasonic_close = ultrasonic_distance_cm <= ultrasonic_threshold_cm

    return vision_close or ultrasonic_close


def detect_emotion(
    person_id: int,
    person_roi,
    image_processor: AutoImageProcessor,
    emotion_model: AutoModelForImageClassification,
    id2label: Dict[int, str],
) -> Tuple[str, float, Tuple[int, int, int]]:
    """Classify emotion for a person ROI and trigger happy/sad callbacks."""
    emotion_label, emotion_conf = classify_person_emotion(
        person_roi, image_processor, emotion_model, id2label
    )

    if emotion_label == "happy":
        on_happy_detected(person_id, emotion_label, emotion_conf)
        return emotion_label, emotion_conf, (0, 220, 0)

    on_sad_detected(person_id, emotion_label, emotion_conf)
    return emotion_label, emotion_conf, (0, 80, 255)


def main() -> None:
    global _PROCESSING_STARTED
    # ---- Config ----
    webcam_index = 1
    yolo_model_path = "./models/yolov8n.pt"
    emotion_model_dir = "./models/hugging_face_vit"
    yolo_conf = 0.50
    imgsz = 416
    web_host = "0.0.0.0"
    web_port = 5000
    stream_width = 640
    stream_height = 360
    # --------------

    server_thread = threading.Thread(
        target=run_web_server, args=(web_host, web_port), daemon=True
    )
    server_thread.start()
    print(f"Web UI: http://localhost:{web_port}")
    print("Loading models...")

    person_detector = YOLO(yolo_model_path)

    # Load locally exported HF emotion model + processor
    image_processor = AutoImageProcessor.from_pretrained(emotion_model_dir, use_fast=False)
    emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_dir)
    emotion_model.eval()
    id2label = emotion_model.config.id2label
    print("Models loaded.")

    names = person_detector.names
    if isinstance(names, dict):
        person_ids = [k for k, v in names.items() if str(v).lower() == "person"]
    else:
        person_ids = [i for i, v in enumerate(names) if str(v).lower() == "person"]
    if not person_ids:
        raise RuntimeError("YOLO model does not contain a 'person' class.")

    cap, actual_index = open_camera_with_fallback(webcam_index)
    print(f"Starting main loop with camera index {actual_index}")

    frame_idx = 0
    t0 = time.time()
    active_target_id = None
    next_target_id = 1
    active_target_bbox = None
    target_miss_count = 0
    max_target_miss_frames = 12
    reacquire_block_until = 0.0

    _PROCESSING_STARTED = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if frame_idx % 30 == 0:
                    print("WARN: Camera read failed; retrying...")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            print("frame received")
            h, w = frame.shape[:2]
            detections: List[Tuple[int, int, int, int, float, str]] = []

            try:
                yolo_results = person_detector.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=yolo_conf,
                    verbose=False,
                )

                boxes = yolo_results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                        x1i = max(0, min(int(x1), w - 1))
                        y1i = max(0, min(int(y1), h - 1))
                        x2i = max(0, min(int(x2), w - 1))
                        y2i = max(0, min(int(y2), h - 1))
                        if x2i > x1i and y2i > y1i:
                            cls_idx = int(cls_id)
                            if isinstance(names, dict):
                                class_name = str(names.get(cls_idx, "unknown")).lower()
                            else:
                                class_name = str(names[cls_idx]).lower() if 0 <= cls_idx < len(names) else "unknown"
                            detections.append((x1i, y1i, x2i, y2i, float(conf), class_name))

                person_candidates = [d for d in detections if d[5] == "person"]
                ultrasonic_distance_cm = get_ultrasonic_distance_cm()

                target_detection = None
                target_bbox_for_frame = None
                now = time.time()

                # Fast path: if there is exactly one person, it is always the target.
                if len(person_candidates) == 1:
                    only = person_candidates[0]
                    target_detection = only
                    target_bbox_for_frame = (only[0], only[1], only[2], only[3])
                    if active_target_id is None and now >= reacquire_block_until:
                        active_target_id = next_target_id
                        next_target_id += 1
                    active_target_bbox = target_bbox_for_frame
                    target_miss_count = 0

                # Acquire a target once (pick largest person) when unlocked.
                elif active_target_id is None and now >= reacquire_block_until and person_candidates:
                    target_detection = max(
                        person_candidates,
                        key=lambda d: (d[2] - d[0]) * (d[3] - d[1]),
                    )
                    target_bbox_for_frame = (
                        target_detection[0],
                        target_detection[1],
                        target_detection[2],
                        target_detection[3],
                    )
                    active_target_id = next_target_id
                    next_target_id += 1
                    active_target_bbox = (target_detection[0], target_detection[1], target_detection[2], target_detection[3])
                    target_miss_count = 0
                elif active_target_id is not None and active_target_bbox is not None and person_candidates:
                    # Keep following the same target by IoU matching.
                    best = max(
                        person_candidates,
                        key=lambda d: bbox_iou(active_target_bbox, (d[0], d[1], d[2], d[3])),
                    )
                    best_iou = bbox_iou(active_target_bbox, (best[0], best[1], best[2], best[3]))
                    if best_iou >= 0.15:
                        smoothed = smooth_bbox(active_target_bbox, (best[0], best[1], best[2], best[3]), alpha=0.7)
                        target_detection = (smoothed[0], smoothed[1], smoothed[2], smoothed[3], best[4], best[5])
                        target_bbox_for_frame = smoothed
                        active_target_bbox = smoothed
                        target_miss_count = 0
                    else:
                        target_miss_count += 1
                elif active_target_id is not None:
                    target_miss_count += 1

                if active_target_id is not None and target_miss_count > max_target_miss_frames:
                    active_target_id = None
                    active_target_bbox = None
                    target_miss_count = 0
                    send_motion_command("W", cooldown_s=0.5)

                if active_target_id is None:
                    send_motion_command("W", cooldown_s=0.5)

                for _, (x1, y1, x2, y2, p_conf, class_name) in enumerate(detections):
                    if class_name != "person":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 140, 0), 2)
                        cv2.putText(
                            frame,
                            f"{class_name} {p_conf:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (220, 140, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        continue

                    if target_bbox_for_frame is None:
                        is_target = False
                    else:
                        is_target = bbox_iou(
                            (x1, y1, x2, y2),
                            target_bbox_for_frame,
                        ) >= 0.25

                    if not is_target:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
                        cv2.putText(
                            frame,
                            f"{class_name} {p_conf:.2f} | non-target",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            2,
                            cv2.LINE_AA,
                        )
                        continue

                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size == 0:
                        continue

                    if not reached_person(
                        (x1, y1, x2, y2),
                        (h, w),
                        ultrasonic_distance_cm=ultrasonic_distance_cm,
                    ):
                        go_to_person(active_target_id, (x1, y1, x2, y2), (h, w))
                        box_color = (255, 180, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            frame,
                            f"target {p_conf:.2f} | approaching",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            box_color,
                            2,
                            cv2.LINE_AA,
                        )
                        continue

                    send_motion_command("S", cooldown_s=0.2)
                    emotion_label, emotion_conf, box_color = detect_emotion(
                        active_target_id, person_roi, image_processor, emotion_model, id2label
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{class_name}#{active_target_id} {p_conf:.2f} | {emotion_label} {emotion_conf:.2f}"
                    label = f"target {p_conf:.2f} | {emotion_label} {emotion_conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )
                    # Target reached and processed; reset to waiting.
                    active_target_id = None
                    active_target_bbox = None
                    target_miss_count = 0
                    reacquire_block_until = time.time() + 1.0
                    send_motion_command("W", cooldown_s=0.2)
                    break

            except Exception as e:
                print(f"ERROR in frame processing: {e}")
                cv2.putText(
                    frame,
                    f"Processing error: {e}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            dt = time.time() - t0
            fps = (frame_idx + 1) / dt if dt > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Persons: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            stream_frame = cv2.resize(
                frame,
                (stream_width, stream_height),
                interpolation=cv2.INTER_AREA,
            )
            publish_frame(stream_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Loop alive. frame_idx={frame_idx}, detections={len(detections)}")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
