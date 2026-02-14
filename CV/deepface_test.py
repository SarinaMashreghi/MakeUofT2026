from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

window_name = "DeepFace Emotion Recognition"

while True:
    success, frame = cap.read()
    if not success or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    try:
        # DeepFace can take a numpy image directly.
        # enforce_detection=False avoids exceptions when no face is present.
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
        )

        if isinstance(analysis, dict):
            analysis = [analysis]

        for face_data in analysis:
            region = face_data.get("region", {})
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("w", 0))
            h = int(region.get("h", 0))

            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            emotion = face_data.get("dominant_emotion", "unknown")
            emotion_scores = face_data.get("emotion", {})
            emotion_conf = float(emotion_scores.get(emotion, 0.0))
            label = f"{emotion} {emotion_conf:.1f}%"

            text_x = x if w > 0 else 10
            text_y = max(20, y - 10) if h > 0 else 30
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    except Exception as e:
        cv2.putText(
            frame,
            f"DeepFace error: {e}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# dfs = DeepFace.find(
#   img_path = "img1.jpg", db_path = "C:/my_db", model_name = models[1]
# )

# embeddings = DeepFace.represent(
#   img_path = "img.jpg", model_name = models[2]
# )
