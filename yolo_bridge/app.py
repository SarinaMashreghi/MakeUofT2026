import os
import uuid
import subprocess
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

# --- Load .env next to this file (bulletproof) ---
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY in .env")
if not ELEVENLABS_API_KEY:
    raise SystemExit("Missing ELEVENLABS_API_KEY in .env")
if not VOICE_ID:
    raise SystemExit("Missing ELEVENLABS_VOICE_ID in .env")

# You can change model if you want
GEMINI_MODEL = "gemini-3-flash-preview"

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "static" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


def maybe_convert_webm_to_wav(input_path: Path) -> Path:
    """
    Browsers often record as audio/webm.
    Gemini may accept audio/webm, but to be robust we convert to WAV if ffmpeg is available.
    If ffmpeg isn't installed, we just return the original file.
    """
    if input_path.suffix.lower() != ".webm":
        return input_path

    wav_path = input_path.with_suffix(".wav")

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        # No ffmpeg; return original webm
        return input_path

    # Convert webm -> wav (mono, 16k) good for speech
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    return input_path


def gemini_generate_from_audio(audio_bytes: bytes, mime_type: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "You are my assistant."
        "Then respond with a helpful short answer. "
        "Return ONLY your final response (no labels)."
        "After 3 respond, ask the user if they want "
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )

    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text


def elevenlabs_tts_to_mp3(text: str, out_mp3_path: Path):
    eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Streaming generator -> write chunks
    audio_stream = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        output_format="mp3_22050_32",
        model_id="eleven_turbo_v2_5",
    )

    with open(out_mp3_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)


@app.post("/api/process")
def api_process():
    """
    Expects multipart/form-data with a file field named "audio".
    Returns: { text: "...", audio_url: "/static/out/xxx.mp3" }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded (field name must be 'audio')."}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    req_id = uuid.uuid4().hex
    # Save incoming file
    in_suffix = Path(f.filename).suffix.lower() or ".webm"
    in_path = OUT_DIR / f"in_{req_id}{in_suffix}"
    f.save(in_path)

    # Convert if needed
    converted_path = maybe_convert_webm_to_wav(in_path)

    # Read bytes and decide mime
    audio_bytes = converted_path.read_bytes()
    mime_type = "audio/wav" if converted_path.suffix.lower() == ".wav" else "audio/webm"

    try:
        text = gemini_generate_from_audio(audio_bytes, mime_type=mime_type)
    except Exception as e:
        return jsonify({"error": f"Gemini error: {e}"}), 500

    out_mp3 = OUT_DIR / f"tts_{req_id}.mp3"

    try:
        elevenlabs_tts_to_mp3(text, out_mp3)
    except Exception as e:
        return jsonify({"error": f"ElevenLabs error: {e}"}), 500

    audio_url = f"/static/out/{out_mp3.name}"
    return jsonify({"text": text, "audio_url": audio_url})


if __name__ == "__main__":
    # Use 0.0.0.0 if you want other devices on your network to access it
    app.run(host="127.0.0.1", port=5000, debug=True)
