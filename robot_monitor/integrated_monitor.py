import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

try:
    from conversations import machine_conversation_turn
    from conversations import fixed_phrase_tts
    from conversations import emotion_demo_tts
    from conversations import phrase_tts, get_emotion_turn_script
except BaseException as e:  # pragma: no cover - runtime dependency guard
    machine_conversation_turn = None
    fixed_phrase_tts = None
    emotion_demo_tts = None
    phrase_tts = None
    get_emotion_turn_script = None
    print(f"WARN: conversations module unavailable: {e}")

from io_control import ESPMotionController, WebStreamer

PICKUP_INVITE_PHRASE = "hey im cupid bot, pick me up if you want company"
INVITE_REPEAT_INTERVAL_S = 10.0
INVITE_MAX_PLAYS = 2
USER_REPLY_GRACE_S = 12.0
POST_REPLY_CONVERSATION_HOLD_S = 15.0
AUDIO_TURN_DELAY_S = 12.0
RUNTIME_ERROR_LOG = "runtime_errors.log"


def persist_fatal_error(base_dir: Path, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = base_dir / RUNTIME_ERROR_LOG
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception as e:
        print(f"WARN: Could not write fatal error log: {e}")


def on_happy_detected(
    person_id: int,
    emotion: str,
    confidence: float,
    conversation_history: List[str],
    web: WebStreamer,
) -> float:
    del person_id, emotion, confidence, conversation_history
    if emotion_demo_tts is None:
        return 0.0

    try:
        assistant_text, audio_url = emotion_demo_tts("happy")
        web.publish_audio(text=assistant_text, audio_url=audio_url, kind="conversation_happy_demo")
        print(f"[conversation/happy/demo] {assistant_text}")
        print(f"[conversation/happy/demo] audio: {audio_url}")
        return 14.0
    except Exception as e:
        print(f"WARN: Happy conversation failed: {e}")
        return 0.0


def on_sad_detected(
    person_id: int,
    emotion: str,
    confidence: float,
    conversation_history: List[str],
    web: WebStreamer,
) -> float:
    del person_id, emotion, confidence, conversation_history
    if emotion_demo_tts is None:
        return 0.0

    try:
        assistant_text, audio_url = emotion_demo_tts("sad")
        web.publish_audio(text=assistant_text, audio_url=audio_url, kind="conversation_sad_demo")
        print(f"[conversation/sad/demo] {assistant_text}")
        print(f"[conversation/sad/demo] audio: {audio_url}")
        return 14.0
    except Exception as e:
        print(f"WARN: Sad conversation failed: {e}")
        return 0.0


def speak_pickup_invite(web: WebStreamer) -> float:
    if fixed_phrase_tts is None:
        return 0.0
    try:
        spoken_text, audio_url = fixed_phrase_tts(
            phrase=PICKUP_INVITE_PHRASE,
            cache_key="pickup_invite_cupid_bot",
        )
        web.publish_audio(text=spoken_text, audio_url=audio_url, kind="pickup_invite")
        print(f"[pickup-invite] audio: {audio_url}")
        return 0.0
    except Exception as e:
        print(f"WARN: Fixed pickup invite TTS failed: {e}")
        return 0.0


def speak_timed_line(web: WebStreamer, text: str, kind: str) -> bool:
    if phrase_tts is None:
        return False
    try:
        spoken_text, audio_url = phrase_tts(text=text, prefix=kind)
        web.publish_audio(text=spoken_text, audio_url=audio_url, kind=kind)
        print(f"[timed-line/{kind}] {spoken_text}")
        print(f"[timed-line/{kind}] audio: {audio_url}")
        return True
    except Exception as e:
        print(f"WARN: timed line TTS failed ({kind}): {e}")
        return False


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


def open_camera_with_fallback(preferred_index: int):
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
    if prev_bbox is None:
        return curr_bbox
    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = curr_bbox
    x1 = int(alpha * px1 + (1.0 - alpha) * cx1)
    y1 = int(alpha * py1 + (1.0 - alpha) * cy1)
    x2 = int(alpha * px2 + (1.0 - alpha) * cx2)
    y2 = int(alpha * py2 + (1.0 - alpha) * cy2)
    return x1, y1, x2, y2


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def lonely_score(
    candidate_bbox: Tuple[int, int, int, int],
    all_bboxes: List[Tuple[int, int, int, int]],
) -> float:
    if len(all_bboxes) <= 1:
        return float("inf")

    cx, cy = bbox_center(candidate_bbox)
    total = 0.0
    count = 0
    for other in all_bboxes:
        if other == candidate_bbox:
            continue
        ox, oy = bbox_center(other)
        dx = cx - ox
        dy = cy - oy
        total += (dx * dx + dy * dy) ** 0.5
        count += 1
    return total / max(1, count)


def select_loneliest_person(
    person_candidates: List[Tuple[int, int, int, int, float, str]]
) -> Tuple[int, int, int, int, float, str]:
    all_bboxes = [(p[0], p[1], p[2], p[3]) for p in person_candidates]
    return max(
        person_candidates,
        key=lambda p: (
            lonely_score((p[0], p[1], p[2], p[3]), all_bboxes),
            (p[2] - p[0]) * (p[3] - p[1]),
            p[4],
        ),
    )


def go_to_person(
    motion: ESPMotionController,
    person_id: int,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> str:
    del person_id
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape
    del frame_h

    cx = (x1 + x2) * 0.5
    mx = frame_w * 0.5
    e = (cx - mx) / max(1.0, mx)
    abs_e = abs(e)

    deadband = 0.10
    enter_turn = 0.18
    strong_turn = 0.45

    if abs_e <= deadband or abs_e < enter_turn:
        motion.send_motion_command("F", cooldown_s=0.5)
        return "forward"

    if e < 0:
        motion.send_motion_command("L", cooldown_s=0.5 if abs_e >= strong_turn else 1.0)
        return "left"
    else:
        motion.send_motion_command("R", cooldown_s=0.5 if abs_e >= strong_turn else 1.0)
        return "right"


def reached_person(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    area_ratio_threshold: float = 0.20,
    ultrasonic_distance_cm: float | None = None,
    ultrasonic_threshold_cm: float = 45.0,
) -> bool:
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_h * frame_w)
    vision_close = (box_area / frame_area) >= area_ratio_threshold

    ultrasonic_close = False
    if ultrasonic_distance_cm is not None and ultrasonic_distance_cm > 0:
        ultrasonic_close = ultrasonic_distance_cm <= ultrasonic_threshold_cm

    reached_target = vision_close or ultrasonic_close
    
    if(reached_target):
        print("reached target")
        
    return reached_target


def detect_emotion(
    person_id: int,
    person_roi,
    image_processor: AutoImageProcessor,
    emotion_model: AutoModelForImageClassification,
    id2label: Dict[int, str],
    conversation_history: List[str],
    web: WebStreamer,
    trigger_conversation: bool = True,
) -> Tuple[str, float, Tuple[int, int, int], float]:
    emotion_label, emotion_conf = classify_person_emotion(
        person_roi, image_processor, emotion_model, id2label
    )

    if emotion_label == "happy":
        hold_s = 0.0
        if trigger_conversation:
            hold_s = on_happy_detected(
                person_id, emotion_label, emotion_conf, conversation_history, web
            )
        return emotion_label, emotion_conf, (0, 220, 0), hold_s

    hold_s = 0.0
    if trigger_conversation:
        hold_s = on_sad_detected(
            person_id, emotion_label, emotion_conf, conversation_history, web
        )
    return emotion_label, emotion_conf, (0, 80, 255), hold_s


def main() -> None:
    webcam_index = 1
    yolo_model_path = "./models/yolov8n.pt"
    emotion_model_dir = "./models/hugging_face_vit"
    yolo_conf = 0.50
    imgsz = 416
    web_host = "0.0.0.0"
    web_port = 5005
    stream_width = 600
    stream_height = 360

    motion = ESPMotionController(port="COM3")
    web = WebStreamer(base_dir=Path(__file__).resolve().parent)
    web.start_in_thread(host=web_host, port=web_port)
    web.set_runtime_state(
        robot_state="idle",
        robot_direction="none",
        esp_connected=motion.is_connected(),
        esp_error=motion.get_last_error(),
    )
    
    print(f"Web UI: http://localhost:{web_port}")
    print("Loading models...")

    person_detector = YOLO(yolo_model_path)
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
    target_reached_hold_until = 0.0
    timed_conversation_active = False
    timed_conversation_lines: List[str] = []
    timed_conversation_idx = 0
    timed_conversation_next_at = 0.0
    conversation_history: List[str] = []

    web.set_processing_started(True)
    web.set_fatal_error("")

    try:
        while True:
            robot_state = "idle"
            robot_direction = "none"
            ok, frame = cap.read()
            if not ok or frame is None:
                if frame_idx % 30 == 0:
                    print("WARN: Camera read failed; retrying...")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            detections: List[Tuple[int, int, int, int, float, str]] = []
            now = time.time()

            if timed_conversation_active:
                active_target_id = None
                active_target_bbox = None
                target_miss_count = 0
                motion.send_motion_command("S", cooldown_s=0.3)
                robot_state = "conversation_ongoing"

                if now >= timed_conversation_next_at:
                    if timed_conversation_idx < len(timed_conversation_lines):
                        line = timed_conversation_lines[timed_conversation_idx]
                        speak_timed_line(web, line, "timed_conversation")
                        timed_conversation_idx += 1
                        timed_conversation_next_at = time.time() + AUDIO_TURN_DELAY_S
                    else:
                        timed_conversation_active = False
                        timed_conversation_lines = []
                        timed_conversation_idx = 0
                        timed_conversation_next_at = 0.0
                        reacquire_block_until = time.time() + 0.3
                        robot_state = "idle"

                cv2.putText(
                    frame,
                    "Timed conversation running..." if robot_state == "conversation_ongoing" else "Conversation done.",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 200),
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
                web.set_runtime_state(
                    robot_state=robot_state,
                    robot_direction=robot_direction,
                    esp_connected=motion.is_connected(),
                    esp_error=motion.get_last_error(),
                )
                stream_frame = cv2.resize(frame, (stream_width, stream_height), interpolation=cv2.INTER_AREA)
                web.publish_frame(stream_frame)
                frame_idx += 1
                continue

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
                ultrasonic_distance_cm = motion.get_ultrasonic_distance_cm()

                target_detection = None
                target_bbox_for_frame = None
                now = time.time()

                if active_target_id is None and now >= reacquire_block_until and person_candidates:
                    target_detection = select_loneliest_person(person_candidates)
                    target_bbox_for_frame = (
                        target_detection[0],
                        target_detection[1],
                        target_detection[2],
                        target_detection[3],
                    )
                    active_target_id = next_target_id
                    next_target_id += 1
                    active_target_bbox = target_bbox_for_frame
                    target_miss_count = 0
                elif active_target_id is not None and active_target_bbox is not None and person_candidates:
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
                    motion.send_motion_command("W", cooldown_s=0.5)

                if active_target_id is None:
                    motion.send_motion_command("W", cooldown_s=0.5)

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
                        is_target = bbox_iou((x1, y1, x2, y2), target_bbox_for_frame) >= 0.25

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
                        robot_direction = go_to_person(
                            motion, active_target_id, (x1, y1, x2, y2), (h, w)
                        )
                        robot_state = "moving_towards_target"
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

                    motion.send_motion_command("S", cooldown_s=0.2)
                    robot_state = "target_reached"
                    robot_direction = "none"
                    target_reached_hold_until = time.time() + 2.0
                    speak_pickup_invite(web)
                    emotion_label, emotion_conf, box_color, conversation_hold_s = detect_emotion(
                        active_target_id,
                        person_roi,
                        image_processor,
                        emotion_model,
                        id2label,
                        conversation_history,
                        web,
                        trigger_conversation=False,
                    )
                    mood = "happy" if emotion_label == "happy" else "sad"
                    timed_conversation_lines = (
                        get_emotion_turn_script(mood) if get_emotion_turn_script is not None else []
                    )
                    timed_conversation_active = len(timed_conversation_lines) > 0
                    timed_conversation_idx = 0
                    timed_conversation_next_at = time.time() + AUDIO_TURN_DELAY_S
                    web.set_last_emotion(f"{emotion_label} ({emotion_conf:.2f})")
                    print(f"[emotion] label={emotion_label} conf={emotion_conf:.2f}")
                    web.set_conversation_mood(mood)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
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

                    active_target_id = None
                    active_target_bbox = None
                    target_miss_count = 0
                    reacquire_block_until = time.time() + 1.0
                    motion.send_motion_command("S", cooldown_s=0.2)
                    break

            except Exception as e:
                fatal_message = f"FATAL: frame processing error: {e}"
                print(fatal_message)
                persist_fatal_error(Path(__file__).resolve().parent, fatal_message)
                web.set_fatal_error(fatal_message)
                motion.send_motion_command("S", cooldown_s=0.1)
                break

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

            display_state = robot_state
            if (
                not timed_conversation_active
                and robot_state != "moving_towards_target"
                and time.time() < target_reached_hold_until
            ):
                display_state = "target_reached"

            web.set_runtime_state(
                robot_state=display_state,
                robot_direction=robot_direction,
                esp_connected=motion.is_connected(),
                esp_error=motion.get_last_error(),
            )
            stream_frame = cv2.resize(frame, (stream_width, stream_height), interpolation=cv2.INTER_AREA)
            web.publish_frame(stream_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Loop alive. frame_idx={frame_idx}, detections={len(detections)}")

    except KeyboardInterrupt:
        print("Shutdown requested from terminal (Ctrl+C).")
    except Exception as e:
        fatal_message = f"FATAL: main loop crashed: {e}"
        print(fatal_message)
        persist_fatal_error(Path(__file__).resolve().parent, fatal_message)
        web.set_fatal_error(fatal_message)
    finally:
        web.set_processing_started(False)
        web.stop()
        cap.release()
        motion.close()


if __name__ == "__main__":
    main()
