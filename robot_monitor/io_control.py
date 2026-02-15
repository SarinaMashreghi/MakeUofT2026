import time
import threading
import os
from pathlib import Path

import cv2
import serial
from flask import Flask, Response, jsonify, render_template, make_response, send_file
from werkzeug.serving import make_server

try:
    from conversations import conversation_process, conversation_reset
    conversation_import_error = ""
except BaseException as e:  # pragma: no cover - optional dependency for robot voice UI
    conversation_process = None
    conversation_reset = None
    conversation_import_error = str(e)
    print(f"WARN: conversation routes unavailable: {e}")


class ESPMotionController:
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 0.1,
        write_timeout: float = 0.05,
    ) -> None:
        self._esp = None
        self._esp_error = ""
        self._last_cmd = "S"
        self._last_cmd_ts = 0.0

        try:
            self._esp = serial.Serial(
                port, baudrate, timeout=timeout, write_timeout=write_timeout
            )
            time.sleep(2)
            self._esp_error = ""
            print(f"Connected to ESP32 on {port}")
        except Exception as e:
            self._esp_error = str(e)
            print(f"WARN: Could not connect to ESP32 on {port}: {e}")
            print("Running in camera/web-only mode.")

    def send_motion_command(self, cmd: str, cooldown_s: float) -> None:
        now = time.time()
        if cmd == self._last_cmd and (now - self._last_cmd_ts) < cooldown_s:
            return

        # print("sending motion", cmd)
        if self._esp is not None:
            try:
                self._esp.write(cmd.encode())
            except Exception as e:
                self._esp_error = f"Serial write failed: {e}"
                print(f"WARN: ESP write failed, disabling serial output: {e}")
                try:
                    self._esp.close()
                except Exception:
                    pass
                self._esp = None

        self._last_cmd = cmd
        self._last_cmd_ts = now

    def get_ultrasonic_distance_cm(self) -> float | None:
        # TODO: Parse ultrasonic distance from ESP serial and return cm.
        return None

    def close(self) -> None:
        if self._esp is not None:
            try:
                self._esp.close()
            except Exception:
                pass
            self._esp = None

    def is_connected(self) -> bool:
        return self._esp is not None

    def get_last_error(self) -> str:
        return self._esp_error


class WebStreamer:
    def __init__(self, base_dir: Path) -> None:
        web_dir = base_dir / "web"
        self._app = Flask(
            __name__,
            template_folder=str(web_dir / "templates"),
            static_folder=str(web_dir / "static"),
        )
        self._app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
        self._latest_jpeg = None
        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)
        self._server = None
        self._server_thread = None
        self._latest_frame_id = 0
        self._frame_count = 0
        self._latest_audio = {
            "id": 0,
            "text": "",
            "audio_url": "",
            "created_at": 0.0,
            "kind": "none",
        }
        self._conversation_api_available = conversation_process is not None
        self._conversation_api_error = conversation_import_error
        self._last_transcript = ""
        self._last_emotion = ""
        self._conversation_started = False
        self._conversation_processing = False
        self._processing_started = False
        self._robot_state = "idle"
        self._robot_direction = "none"
        self._conversation_mood = "happy"
        self._esp_connected = False
        self._esp_error = "ESP32 not connected."
        self._fatal_error = ""
        self._fatal_error_ts = 0.0
        self._register_routes()

    def _register_routes(self) -> None:
        self._app.add_url_rule("/", "index", self._index)
        self._app.add_url_rule("/robot", "robot", self._robot)
        self._app.add_url_rule("/emoji.png", "emoji_png", self._emoji_png)
        self._app.add_url_rule("/video_feed", "video_feed", self._video_feed)
        self._app.add_url_rule("/status", "status", self._status)
        self._app.add_url_rule("/api/audio/latest", "audio_latest", self._audio_latest)

        if conversation_process is not None:
            self._app.add_url_rule("/api/process", "api_process", self._api_process, methods=["POST"])
        if conversation_reset is not None:
            self._app.add_url_rule("/api/reset", "api_reset", self._api_reset, methods=["POST"])

    def _index(self):
        return render_template("index.html")

    def _robot(self):
        return render_template("robot.html")

    def _emoji_png(self):
        img_path = Path(self._app.root_path) / "emoji.png"
        if img_path.exists():
            return send_file(str(img_path), mimetype="image/png")
        return ("", 404)

    def _mjpeg_generator(self):
        last_sent_id = -1
        while True:
            with self._frame_cond:
                while self._latest_jpeg is None or self._latest_frame_id == last_sent_id:
                    self._frame_cond.wait(timeout=0.5)
                frame = self._latest_jpeg
                last_sent_id = self._latest_frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    def _video_feed(self):
        resp = Response(
            self._mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    def _status(self):
        with self._frame_lock:
            has_frame = self._latest_jpeg is not None
            frame_count = self._frame_count
            processing_started = self._processing_started
            robot_state = self._robot_state
            robot_direction = self._robot_direction
            conversation_mood = self._conversation_mood
            conversation_processing = self._conversation_processing
            conversation_api_available = self._conversation_api_available
            conversation_api_error = self._conversation_api_error
            last_transcript = self._last_transcript
            last_emotion = self._last_emotion
            esp_connected = self._esp_connected
            esp_error = self._esp_error
            fatal_error = self._fatal_error
            fatal_error_ts = self._fatal_error_ts
        return jsonify(
            {
                "streaming": has_frame,
                "processing_started": processing_started,
                "frame_count": frame_count,
                "robot_state": robot_state,
                "robot_direction": robot_direction,
                "conversation_mood": conversation_mood,
                "conversation_processing": conversation_processing,
                "conversation_api_available": conversation_api_available,
                "conversation_api_error": conversation_api_error,
                "last_transcript": last_transcript,
                "last_emotion": last_emotion,
                "esp_connected": esp_connected,
                "esp_error": esp_error,
                "fatal_error": fatal_error,
                "fatal_error_ts": fatal_error_ts,
            }
        )

    def _audio_latest(self):
        with self._frame_lock:
            latest_audio = dict(self._latest_audio)
        return jsonify(latest_audio)

    def _api_process(self):
        print("[api/process] request received")
        # Mark processing immediately so motion loop can stay paused end-to-end.
        self.set_conversation_processing(True)
        resp = make_response(conversation_process())
        try:
            payload = resp.get_json(silent=True) or {}
            if resp.status_code < 400 and not payload.get("error"):
                self.set_last_transcript(str(payload.get("user_text", "")))
                self.mark_conversation_started()
                self.set_fatal_error("")
            else:
                # If processing failed, clear the signal so robot can resume.
                self.clear_conversation_started()
                err_msg = str(payload.get("error", "Unknown conversation API error"))
                print(f"[api/process] error: {err_msg}")
                self.set_fatal_error(f"/api/process failed: {err_msg}")
        except Exception:
            self.clear_conversation_started()
        finally:
            self.set_conversation_processing(False)
        return resp

    def _api_reset(self):
        self.clear_conversation_started()
        return conversation_reset()

    def publish_frame(self, frame_bgr, jpeg_quality: int = 65) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            return
        with self._frame_cond:
            self._latest_jpeg = encoded.tobytes()
            self._latest_frame_id += 1
            self._frame_count += 1
            self._frame_cond.notify_all()

    def set_processing_started(self, started: bool) -> None:
        with self._frame_lock:
            self._processing_started = started

    def set_runtime_state(
        self,
        robot_state: str,
        robot_direction: str,
        esp_connected: bool,
        esp_error: str = "",
    ) -> None:
        with self._frame_lock:
            self._robot_state = robot_state
            self._robot_direction = robot_direction
            self._esp_connected = esp_connected
            self._esp_error = esp_error

    def publish_audio(self, text: str, audio_url: str, kind: str = "speech") -> None:
        with self._frame_lock:
            self._latest_audio["id"] += 1
            self._latest_audio["text"] = text
            self._latest_audio["audio_url"] = audio_url
            self._latest_audio["created_at"] = time.time()
            self._latest_audio["kind"] = kind

    def set_conversation_mood(self, mood: str) -> None:
        with self._frame_lock:
            self._conversation_mood = mood if mood in {"happy", "sad"} else "happy"

    def mark_conversation_started(self) -> None:
        with self._frame_lock:
            self._conversation_started = True

    def clear_conversation_started(self) -> None:
        with self._frame_lock:
            self._conversation_started = False

    def consume_conversation_started(self) -> bool:
        with self._frame_lock:
            started = self._conversation_started
            self._conversation_started = False
        return started

    def set_conversation_processing(self, processing: bool) -> None:
        with self._frame_lock:
            self._conversation_processing = bool(processing)

    def is_conversation_processing(self) -> bool:
        with self._frame_lock:
            return self._conversation_processing

    def set_last_transcript(self, transcript: str) -> None:
        with self._frame_lock:
            self._last_transcript = transcript

    def set_last_emotion(self, emotion: str) -> None:
        with self._frame_lock:
            self._last_emotion = emotion

    def set_fatal_error(self, message: str) -> None:
        with self._frame_lock:
            self._fatal_error = message
            if message:
                self._fatal_error_ts = time.time()
                self._processing_started = False
            else:
                self._fatal_error_ts = 0.0

    def start_in_thread(self, host: str = "0.0.0.0", port: int = 5000) -> threading.Thread:
        if self._server_thread is not None and self._server_thread.is_alive():
            return self._server_thread

        self._server = make_server(host, port, self._app, threaded=True)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        return self._server_thread

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
