import argparse
import time

import cv2
import requests
from ultralytics import YOLO


def post(server_url: str, session_id: str, person_detected: bool, people_count: int) -> None:
    r = requests.post(
        server_url,
        json={
            "sessionId": session_id,
            "personDetected": person_detected,
            "peopleCount": people_count,
        },
        timeout=2,
    )
    print(r.json())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO and post person presence events.")
    parser.add_argument(
        "--model",
        help="Path to local YOLO model file (.pt/.onnx/.engine/etc)",
    )
    parser.add_argument(
        "--hf-repo",
        help="Hugging Face repo id (for example: username/my-yolo-model)",
    )
    parser.add_argument(
        "--hf-file",
        default="best.pt",
        help="Model filename inside Hugging Face repo",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Hugging Face branch/tag/commit",
    )
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument(
        "--server",
        default="http://localhost:8080/api/vision_event",
        help="Vision event endpoint",
    )
    parser.add_argument("--session-id", default="demo-session", help="Session id used by the app")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between server updates",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detections",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model:
        return args.model

    if args.hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: huggingface_hub. Install with 'pip install huggingface_hub'."
            ) from exc

        return hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_file,
            revision=args.hf_revision,
        )

    raise RuntimeError("Provide either --model (local file) or --hf-repo (Hugging Face model).")


def main() -> None:
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    model_path = resolve_model_path(args)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    last_post = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)
        people_count = 0

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            if class_id == 0:
                people_count += 1

        now = time.time()
        if now - last_post >= args.interval:
            post(
                server_url=args.server,
                session_id=args.session_id,
                person_detected=people_count > 0,
                people_count=people_count,
            )
            last_post = now

    cap.release()


if __name__ == "__main__":
    main()
