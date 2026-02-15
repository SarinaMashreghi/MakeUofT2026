import sys
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from tensorflow import keras
except ImportError:
    try:
        import keras
    except ImportError:
        keras = None

# Box format: (x1, y1, x2, y2, conf)
Box = Tuple[float, float, float, float, float]


def pick_target_largest(boxes: List[Box]) -> Optional[Box]:
    """Pick the largest-area box (good heuristic for the closest person)."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def preprocess_roi_for_emotion(roi_bgr: np.ndarray, model_input_shape: Tuple[Optional[int], ...]) -> np.ndarray:
    """Resize + normalize ROI to match emotion model input."""
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    if len(model_input_shape) != 4:
        raise ValueError(f"Unexpected emotion model input shape: {model_input_shape}")

    _, in_h, in_w, in_c = model_input_shape
    if in_h is None or in_w is None:
        in_h, in_w = 48, 48
    in_h, in_w = int(in_h), int(in_w)

    if in_c == 1:
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (in_w, in_h))
        roi = roi.astype(np.float32) / 255.0
        roi = np.expand_dims(roi, axis=-1)  # H, W, 1
    else:
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (in_w, in_h))
        roi = roi.astype(np.float32) / 255.0

    return np.expand_dims(roi, axis=0)  # 1, H, W, C


def main():
    # --------- Config ----------
    model_path = "yolov8n.pt"   # or your trained .pt file
    emotion_model_path = "face_recognition_v1.keras"
    webcam_index = 0            # change to 1/2 if you have multiple cameras
    conf_thres = 0.35
    imgsz = 416                 # smaller = faster
    only_person = True          # restrict to "person" class
    print_every_n_frames = 5    # reduce spam
    window_name = "YOLO Detection"
    target_color = (0, 255, 255)  # BGR
    box_color = (0, 255, 0)       # BGR
    display_width = 480
    display_height = 600
    emotion_labels = ["angry", "surprise", "neutral", "happy", "fear", "disgust", "sad"]
    # ---------------------------

    model = YOLO(model_path)
    if keras is None:
        print("ERROR: Could not import tensorflow/keras. Install one of them to run emotion inference.")
        sys.exit(1)

    emotion_model = keras.models.load_model(emotion_model_path, compile=False)
    emotion_input_shape = emotion_model.input_shape

    # Identify the COCO class id for "person" (usually 0)
    # Ultralytics stores names as dict {id: name} or list
    names = model.names
    if isinstance(names, dict):
        person_ids = [k for k, v in names.items() if v == "person"]
    else:
        person_ids = [i for i, v in enumerate(names) if v == "person"]

    if only_person and not person_ids:
        print("ERROR: Could not find 'person' class in model.names. "
              "Set only_person=False or check your model.")
        sys.exit(1)

    classes = person_ids if only_person else None

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam index {webcam_index}")
        sys.exit(1)

    # Optional: reduce latency (may not work on all backends)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    print("Running. Press Ctrl+C to stop.")
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARN: Failed to read frame.")
                time.sleep(0.05)
                continue

            # Run YOLO (BGR frame is fine)
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf_thres,
                classes=classes,
                verbose=False
            )

            r0 = results[0]
            boxes_out: List[Box] = []

            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                cls_ids = r0.boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, cls_ids):
                    boxes_out.append((float(x1), float(y1), float(x2), float(y2), float(c)))

                    class_name = "unknown"
                    cls_idx = int(cls_id)
                    if isinstance(names, dict):
                        class_name = str(names.get(cls_idx, "unknown"))
                    elif 0 <= cls_idx < len(names):
                        class_name = str(names[cls_idx])

                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    x1i = max(0, min(x1i, frame.shape[1] - 1))
                    x2i = max(0, min(x2i, frame.shape[1] - 1))
                    y1i = max(0, min(y1i, frame.shape[0] - 1))
                    y2i = max(0, min(y2i, frame.shape[0] - 1))

                    emotion_text = "emotion N/A"
                    if x2i > x1i and y2i > y1i:
                        person_roi = frame[y1i:y2i, x1i:x2i]
                        if person_roi.size > 0:
                            roi_input = preprocess_roi_for_emotion(person_roi, emotion_input_shape)
                            probs = emotion_model.predict(roi_input, verbose=0)[0]
                            emo_idx = int(np.argmax(probs))
                            if 0 <= emo_idx < len(emotion_labels):
                                emotion_text = f"{emotion_labels[emo_idx]} {probs[emo_idx]:.2f}"

                    label = f"{class_name} {c:.2f} | {emotion_text}"
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), box_color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1i, max(20, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

            # Pick a target box for rover steering (largest = closest)
            target = pick_target_largest(boxes_out)

            # Highlight target box, if any
            if target:
                x1, y1, x2, y2, c = target
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), target_color, 3)
                cv2.putText(
                    frame,
                    "TARGET",
                    (x1i, max(20, y1i - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    target_color,
                    2,
                    cv2.LINE_AA,
                )

            # Print occasionally so terminal stays readable
            if frame_idx % print_every_n_frames == 0:
                # FPS estimate
                dt = time.time() - t0
                fps = (frame_idx + 1) / dt if dt > 0 else 0.0

                print(f"\nFrame {frame_idx} | FPS~{fps:.1f} | detections={len(boxes_out)}")

                for b in boxes_out:
                    x1, y1, x2, y2, c = b
                    print(f"  person conf={c:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

                if target:
                    x1, y1, x2, y2, c = target
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    area = (x2 - x1) * (y2 - y1)
                    print(f"  TARGET conf={c:.2f} center=({cx:.0f},{cy:.0f}) area={area:.0f}")
                else:
                    print("  TARGET: None")

            dt_now = time.time() - t0
            fps_now = (frame_idx + 1) / dt_now if dt_now > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps_now:.1f} | Detections: {len(boxes_out)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow(window_name, display_frame)
            # Press 'q' to quit the visualization window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'q' pressed. Stopping...")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
