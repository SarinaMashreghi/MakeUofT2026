import os
import uuid
import subprocess
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv


from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

# ---------------- Config ----------------
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
OUT_DIR = BASE_DIR / "static" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
HAPPY_PROMPT = """
You are a friendly social robot approaching someone who appears happy and open to chatting.
Your goal is to create a light, positive, and brief social interaction.

Behavior rules:
- Speak in a warm, upbeat, and natural tone.
- Use only 1–2 short sentences per turn.
- Ask ONE simple, fun question to keep the conversation going.
- You may offer a quick optional activity such as:
  • a fun fact
  • a short joke
  • a tiny question game
  • casual friendly company
- Be playful but not awkward, childish, or overly slang-heavy.
- Never mention emotion detection, AI models, cameras, or analysis.
- Never say “you look happy” or reference how the person was detected.
- Keep the interaction socially appropriate for a public campus or lounge.

Opening style:
Start with a warm, natural greeting that feels comfortable in a public space.
Example tone (do NOT copy exactly):
“Hey! I’m doing quick friendly check-ins around here. Want a fun fact or a quick question game?”
"""
SAD_PROMPT = """
You are a gentle and supportive social robot approaching someone who may be feeling down or stressed.
Your goal is to offer calm, respectful presence without pressure.

Behavior rules:
- Use a calm, kind, and soft tone.
- Speak only 1–2 short sentences per turn.
- Do NOT assume emotions and do NOT diagnose feelings.
- Never say “you look sad” or reference detection, AI, cameras, or analysis.
- Begin with a neutral, caring check-in that allows an easy opt-out.
- Ask at most ONE gentle question.
- Offer simple low-effort options such as:
  • quiet company for a moment
  • a short breathing pause
  • a light distraction like a gentle fun fact
- If the user declines, says stop, or shows disinterest:
  respond politely, respect their space, and end the interaction briefly.

Tone guidance:
Be human, warm, and non-intrusive.
Avoid being overly cheerful, clinical, or robotic.

Opening style:
Start with a soft, respectful check-in.
Example tone (do NOT copy exactly):
“Hey — just a quiet check-in. I can keep you company for a minute, or give you space. What would you prefer?”
"""


if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY in .env")
if not ELEVENLABS_API_KEY:
    raise SystemExit("Missing ELEVENLABS_API_KEY in .env")
if not VOICE_ID:
    raise SystemExit("Missing ELEVENLABS_VOICE_ID in .env")

GEMINI_MODEL = "gemini-3-flash-preview"
MAX_TURNS_BEFORE_CHECK = 3
STOP_WORDS = {"stop", "exit", "quit", "bye", "goodbye", "cancel", "end"}


# ---------------- App ----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")  # change for production


@app.get("/")
def index():
    return render_template("index.html")


def maybe_convert_webm_to_wav(input_path: Path) -> Path:
    """Convert browser audio/webm -> wav (mono 16k) if ffmpeg exists."""
    if input_path.suffix.lower() != ".webm":
        return input_path

    wav_path = input_path.with_suffix(".wav")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        return input_path

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(wav_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return wav_path if wav_path.exists() and wav_path.stat().st_size > 0 else input_path


def gemini_transcribe(audio_bytes: bytes, mime_type: str) -> str:
    """Audio -> text (just transcription)."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "Transcribe the user's speech accurately. "
        "Return ONLY the transcript text, no extra words, no punctuation fixes, no response."
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
        raise RuntimeError("Gemini returned empty transcript.")
    return text



def build_prompt_with_history(user_text: str,mood: str = "happy") -> str:
    history = session.get("history", [])
    turns = len(history) // 2  # user+assistant pairs
    if mood == "happy":
        base = HAPPY_PROMPT
    else:
        base = SAD_PROMPT
    # base = (
    #     "You are a helpful voice assistant in a natural conversation.\n"
    #     "Keep responses SHORT (1-3 sentences) and easy to speak.\n"
    # )

    if turns >= MAX_TURNS_BEFORE_CHECK:
        base += (
            "This is the 3rd+ turn. "
            "After answering, ask: 'Do you want to continue or stop?' "
            "If the user wants to stop, say a short goodbye.\n"
        )

    history_text = "\n".join(history) if history else "(no history yet)"

    return f"""{base}

Conversation so far:
{history_text}

User: {user_text}

Assistant:""".strip()

def gemini_reply_with_context(user_text: str,mood: str = "happy") -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_prompt_with_history(user_text,mood)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
    )

    reply = (resp.text or "").strip()
    if not reply:
        reply = "Sorry, I didn't catch that. Can you say it again?"
    return reply


def elevenlabs_tts_to_mp3(text: str, out_mp3_path: Path):
    eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

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


def normalize_mood(mood: str | None) -> str:
    mood = (mood or "happy").strip().lower()
    return mood if mood in {"happy", "sad"} else "happy"


def infer_mode_from_first_reply(text: str) -> str:
    """
    Very simple heuristic:
    If reply contains gentle/opt-out/support language -> sad mode, else happy mode.
    """
    t = (text or "").lower()

    sad_signals = [
        "give you space", "no pressure", "quiet", "check-in", "take a breath",
        "i can leave", "up to you", "want me to go", "are you okay",
        "stress", "overwhelmed", "here if you want"
    ]
    happy_signals = [
        "fun fact", "joke", "game", "quick question", "awesome", "nice",
        "want to play", "haha"
    ]

    if any(s in t for s in sad_signals):
        return "sad"
    if any(s in t for s in happy_signals):
        return "happy"

    # default if uncertain
    return "happy"

@app.post("/api/process")
def api_process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded (field name must be 'audio')."}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    req_id = uuid.uuid4().hex
    in_suffix = Path(f.filename).suffix.lower() or ".webm"
    in_path = OUT_DIR / f"in_{req_id}{in_suffix}"
    f.save(in_path)

    converted = maybe_convert_webm_to_wav(in_path)
    audio_bytes = converted.read_bytes()
    mime_type = "audio/wav" if converted.suffix.lower() == ".wav" else "audio/webm"

    # YOLO mood only matters on FIRST approach
    incoming_mood = normalize_mood(request.form.get("mood"))  # default happy
    if "mode" not in session:
        session["mode"] = incoming_mood
    mode = session["mode"]

    # A) Transcribe
    try:
        user_text = gemini_transcribe(audio_bytes, mime_type=mime_type)
    except Exception as e:
        return jsonify({"error": f"Gemini transcription error: {e}"}), 500

    # Stop intent check
    user_lower = user_text.lower()
    if any(w in user_lower for w in STOP_WORDS):
        session.pop("history", None)
        session.pop("mode", None)
        goodbye = "Okay, ending now. Talk to you later!"
        out_mp3 = OUT_DIR / f"tts_{req_id}.mp3"
        try:
            elevenlabs_tts_to_mp3(goodbye, out_mp3)
        except Exception as e:
            return jsonify({"error": f"ElevenLabs error: {e}"}), 500

        return jsonify({
            "user_text": user_text,
            "text": goodbye,
            "audio_url": f"/static/out/{out_mp3.name}",
            "end_session": True
        }), 200

    # B) Add user to history BEFORE reply
    history = session.get("history", [])
    history.append(f"User: {user_text}")
    session["history"] = history

    # Determine if this is the first assistant message (before we add Assistant:)
    is_first_assistant = not any(h.startswith("Assistant:") for h in history)

    # Generate assistant reply using the locked mode
    try:
        assistant_text = gemini_reply_with_context(user_text, mode)
    except Exception as e:
        return jsonify({"error": f"Gemini reply error: {e}"}), 500

    # If first assistant reply, infer mode from it and lock it (overrides YOLO if needed)
    if is_first_assistant:
        session["mode"] = infer_mode_from_first_reply(assistant_text)
        mode = session["mode"]

    # Save assistant reply
    history = session.get("history", [])
    history.append(f"Assistant: {assistant_text}")
    session["history"] = history

    # TTS
    out_mp3 = OUT_DIR / f"tts_{req_id}.mp3"
    try:
        elevenlabs_tts_to_mp3(assistant_text, out_mp3)
    except Exception as e:
        return jsonify({"error": f"ElevenLabs error: {e}"}), 500

    return jsonify({
        "user_text": user_text,
        "text": assistant_text,
        "audio_url": f"/static/out/{out_mp3.name}",
        "turns": len(session.get("history", [])) // 2,
        "end_session": False,
        "mode": mode
    })




@app.post("/api/reset")
def api_reset():
    session.pop("history", None)
    session.pop("mode", None)
    return jsonify({"ok": True})

    


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
