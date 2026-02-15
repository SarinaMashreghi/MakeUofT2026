# Robot Monitor Pipeline

Run:

```bash
python robot_monitor/integrated_monitor.py
```

Current flow:
- Webcam frame capture on laptop
- YOLO person detection
- Face detection inside each person bounding box
- Emotion classification using local Hugging Face export (`./hugging_face_vit`)
- Event hooks:
  - `on_happy_detected(...)`
  - `on_sad_detected(...)` (`sad` means any non-`happy` emotion)
- Web UI:
  - Template: `robot_monitor/web/templates/index.html`
  - Static CSS: `robot_monitor/web/static/styles.css`
  - Stream endpoint: `/video_feed`
  - Health endpoint: `/status`

Integration points to add later:
- ESP motion commands (inside the event hooks)
- Microphone input/output logic (inside the event hooks or a side worker)
