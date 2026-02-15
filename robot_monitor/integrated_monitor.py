import time
import serial
from typing import Dict, List, Tuple

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

ESP = serial.Serial("COM9", 115200)

time.sleep(2)
print("Connected to ESP32!")

# Motion-control state to reduce jitter and command spam
_LAST_CMD = "S"
_LAST_CMD_TS = 0.0

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


def turn_right():
    cmd = "R"
    print("turn right")
    ESP.write(cmd.encode())
    
def turn_left():
    cmd = "L"
    print("turn left")
    ESP.write(cmd.encode())

def move_forward():
    cmd = "F"
    print("move forward")
    ESP.write(cmd.encode())
    
def move_backward():
    cmd = "B"
    print("move backward")
    ESP.write(cmd.encode())
    
def stop():
    cmd = "S"
    print("stop")
    ESP.write(cmd.encode())


def send_motion_command(cmd: str, cooldown_s: float) -> None:
    """Send motion command with cooldown and duplicate suppression."""
    global _LAST_CMD, _LAST_CMD_TS
    now = time.time()
    if cmd == _LAST_CMD and (now - _LAST_CMD_TS) < cooldown_s:
        return
    
    print("sending motion", cmd)
    ESP.write(cmd.encode())
    _LAST_CMD = cmd
    _LAST_CMD_TS = now


def get_ultrasonic_distance_cm() -> float | None:
    """TODO: Parse ultrasonic distance from ESP serial and return cm."""
    return None


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
        send_motion_command("F", cooldown_s=1)
        return

    if abs_e < enter_turn:
        send_motion_command("F", cooldown_s=1)
        return

    if e < 0:
        if abs_e >= strong_turn:
            send_motion_command("L", cooldown_s=0.5)
        else:
            send_motion_command("L", cooldown_s=0.5)
    else:
        if abs_e >= strong_turn:
            send_motion_command("R", cooldown_s=0.5)
        else:
            send_motion_command("R", cooldown_s=0.5)


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
    # ---- Config ----
    webcam_index = 1
    yolo_model_path = "./models/yolov8n.pt"
    emotion_model_dir = "./models/hugging_face_vit"
    yolo_conf = 0.50
    imgsz = 416
    window_name = "Robot Monitor"
    # --------------

    person_detector = YOLO(yolo_model_path)

    # Load locally exported HF emotion model + processor
    image_processor = AutoImageProcessor.from_pretrained(emotion_model_dir, use_fast=False)
    emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_dir)
    emotion_model.eval()
    id2label = emotion_model.config.id2label

    names = person_detector.names
    if isinstance(names, dict):
        person_ids = [k for k, v in names.items() if str(v).lower() == "person"]
    else:
        person_ids = [i for i, v in enumerate(names) if str(v).lower() == "person"]
    if not person_ids:
        raise RuntimeError("YOLO model does not contain a 'person' class.")

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {webcam_index}.")

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            yolo_results = person_detector.predict(
                source=frame,
                imgsz=imgsz,
                conf=yolo_conf,
                verbose=False,
            )

            detections: List[Tuple[int, int, int, int, float, str]] = []
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

            person_candidates = [
                (idx, d) for idx, d in enumerate(detections) if d[5] == "person"
            ]
            target_person_idx = None
            if person_candidates:
                target_person_idx, _ = max(
                    person_candidates,
                    key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                )
            ultrasonic_distance_cm = get_ultrasonic_distance_cm()

            for person_idx, (x1, y1, x2, y2, p_conf, class_name) in enumerate(detections):
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

                if target_person_idx is not None and person_idx != target_person_idx:
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
                    go_to_person(person_idx, (x1, y1, x2, y2), (h, w))
                    box_color = (255, 180, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {p_conf:.2f} | approaching",
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
                    person_idx, person_roi, image_processor, emotion_model, id2label
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{class_name} {p_conf:.2f} | {emotion_label} {emotion_conf:.2f}"
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

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
