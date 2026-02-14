import time

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    model_dir = "./models/hugging_face_vit"
    webcam_index = 0
    window_name = "Hugging Face Emotion Test"

    image_processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()

    id2label = model.config.id2label

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {webcam_index}.")

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_bgr = frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

            inputs = image_processor(images=roi_rgb, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                idx = int(torch.argmax(probs).item())
                conf = float(probs[idx].item())

            label = str(id2label.get(idx, idx))
            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        dt = time.time() - t0
        fps = (frame_idx + 1) / dt if dt > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Faces: {len(faces)}",
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
