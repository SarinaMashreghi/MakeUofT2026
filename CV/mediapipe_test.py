import cv2
import mediapipe as mp

# Initialize
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 6, 10, 14, 18]

# Finger names
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        if ret and frame is not None:
            print("Likely camera index:", i, "shape:", frame.shape)
    cap.release()
    
# Start webcam
cap = cv2.VideoCapture(0)
print("video capture started")
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    print(h, w)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    # results = hands.process(img_rgb)

    # if results.multi_hand_landmarks and results.multi_handedness:
    #     for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    #         label = hand_handedness.classification[0].label  # 'Left' or 'Right'
    #         fingers = []

    #         # Thumb logic depends on left/right
    #         if label == 'Right':
    #             if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_BASES[0]].x:
    #                 fingers.append(1)
    #             else:
    #                 fingers.append(0)
    #         else:
    #             if hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_BASES[0]].x:
    #                 fingers.append(1)
    #             else:
    #                 fingers.append(0)

    #         # Other 4 fingers
    #         for tip, base in zip(FINGER_TIPS[1:], FINGER_BASES[1:]):
    #             if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
    #                 fingers.append(1)
    #             else:
    #                 fingers.append(0)

    #         # Count open fingers
    #         finger_count = sum(fingers)
    #         cv2.putText(img, f"{label} Hand: {finger_count} fingers",
    #                     (10, 30 if label == "Right" else 60),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    #         # Visual feedback for each finger
    #         for i, is_up in enumerate(fingers):
    #             status = "Up" if is_up else "Down"
    #             cv2.putText(img, f"{label}-{FINGER_NAMES[i]}: {status}",
    #                         (10, 90 + i * 20 if label == "Left" else 220 + i * 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_up else (0, 0, 255), 1)

    #         mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # else:
    #     cv2.putText(img, "No Hands Detected: STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # cv2.imshow("Fixed Hand Tracker", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cap.release()
# cv2.destroyAllWindows()