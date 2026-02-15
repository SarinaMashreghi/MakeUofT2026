import cv2
import mediapipe as mp

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 6, 10, 14, 18]

# Finger names
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


    
# Start webcam
cap = cv2.VideoCapture(0)
print("video capture started")
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    print(h, w)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
