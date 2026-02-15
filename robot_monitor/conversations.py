import os
import uuid
import subprocess
import tempfile
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv


from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

# ---------------- Config ----------------
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
# Shared static output served by WebStreamer (/static/...)
OUT_DIR = BASE_DIR / "web" / "static" / "out"
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


GEMINI_MODEL = "gemini-3-flash-preview"
MAX_TURNS_BEFORE_CHECK = 3
STOP_WORDS = {"stop", "exit", "quit", "bye", "goodbye", "cancel", "end"}
RUNTIME_ERROR_LOG = "runtime_errors.log"
TRANSCRIBE_TIMEOUT_S = float(os.getenv("TRANSCRIBE_TIMEOUT_S", "60"))
REPLY_TIMEOUT_S = float(os.getenv("REPLY_TIMEOUT_S", "35"))
TTS_TIMEOUT_S = float(os.getenv("TTS_TIMEOUT_S", "40"))
CANDY_CLOSING_LINE = "here is a candy for you, happy valentines"

HAPPY_DEMO_SCRIPTS = [
    (
        "Would you pick chocolate hearts or red roses for Valentine’s Day? "
        "I love that choice. "
        "Would you choose a rom-com movie night or cozy love songs this evening? "
        f"{CANDY_CLOSING_LINE}"
    ),
    (
        "Here is a Valentine fun fact: around 250 million roses are grown every year for Valentine's Day. "
        "Would you choose a picnic date or a coffee date this weekend? "
        f"{CANDY_CLOSING_LINE}"
    ),
]

SAD_DEMO_SCRIPTS = [
    (
        "Here is a kind compliment: your presence makes this space feel warmer. "
        "Take a short calming breath with me. "
        "Here is a soft Valentine fun fact: the first heart-shaped chocolate box was introduced in the 1800s. "
        f"{CANDY_CLOSING_LINE}"
    ),
    (
        "Would you like a quiet check-in or a tiny smile challenge? "
        "Good choice. "
        "Should I send you a warm Valentine wish or a short love quote? "
        f"{CANDY_CLOSING_LINE}"
    ),
]


# # ---------------- App ----------------
# app = Flask(__name__)
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")  # change for production


def maybe_convert_webm_to_wav(input_path: Path) -> Path:
    """Convert browser audio/webm -> wav (mono 16k) if ffmpeg exists."""
    if input_path.suffix.lower() != ".webm":
        return input_path

    wav_path = input_path.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3,
        )
    except FileNotFoundError:
        return input_path
    except subprocess.TimeoutExpired:
        return input_path

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(wav_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=12,
        )
    except subprocess.TimeoutExpired:
        return input_path
    return wav_path if wav_path.exists() and wav_path.stat().st_size > 0 else input_path


def run_with_timeout(fn, timeout_s: float, label: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError as e:
            raise RuntimeError(f"{label} timed out after {timeout_s:.0f}s") from e


def persist_runtime_error(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = BASE_DIR / RUNTIME_ERROR_LOG
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception as e:
        print(f"[conversation] WARN: failed to write runtime error log: {e}")


def ensure_api_keys() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")
    if not VOICE_ID:
        raise RuntimeError("Missing ELEVENLABS_VOICE_ID in .env")


def gemini_transcribe(audio_bytes: bytes, mime_type: str) -> str:
    """Audio -> text (just transcription)."""
    ensure_api_keys()
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "Transcribe the user's speech accurately. "
        "Return ONLY the transcript text, no extra words, no punctuation fixes, no response."
    )

    print(
        f"[gemini][transcribe][request] model={GEMINI_MODEL} mime={mime_type} bytes={len(audio_bytes)}"
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


def build_prompt_with_history(
    user_text: str,
    mood: str = "happy",
    history: list[str] | None = None,
) -> str:
    if history is None:
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

def gemini_reply_with_context(
    user_text: str,
    mood: str = "happy",
    history: list[str] | None = None,
) -> str:
    ensure_api_keys()
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_prompt_with_history(user_text, mood, history=history)
    preview = prompt.replace("\n", " ")[:160]
    print(
        f"[gemini][reply][request] model={GEMINI_MODEL} mood={mood} prompt_chars={len(prompt)} preview={preview!r}"
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
    )

    reply = (resp.text or "").strip()
    if not reply:
        reply = "Sorry, I didn't catch that. Can you say it again?"
    return reply


def elevenlabs_tts_to_mp3(text: str, out_mp3_path: Path):
    ensure_api_keys()
    eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    preview = text.replace("\n", " ")[:160]
    print(
        f"[elevenlabs][tts][request] model=eleven_turbo_v2_5 voice_id={VOICE_ID} text_chars={len(text)} out={out_mp3_path.name} preview={preview!r}"
    )

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


def machine_conversation_turn(
    mood: str,
    history: list[str] | None = None,
    user_text: str | None = None,
) -> tuple[str, str, list[str]]:
    mood = normalize_mood(mood)
    turn_history = list(history or [])
    prompt_text = (user_text or "Start with a brief greeting and one short follow-up question.").strip()

    turn_history.append(f"User: {prompt_text}")
    assistant_text = gemini_reply_with_context(prompt_text, mood, history=turn_history)
    turn_history.append(f"Assistant: {assistant_text}")

    req_id = uuid.uuid4().hex
    out_mp3 = OUT_DIR / f"tts_{req_id}.mp3"
    elevenlabs_tts_to_mp3(assistant_text, out_mp3)

    return assistant_text, f"/static/out/{out_mp3.name}", turn_history


def fixed_phrase_tts(phrase: str, cache_key: str = "pickup_invite") -> tuple[str, str]:
    """
    Generate (or reuse) a fixed TTS phrase and return:
    (spoken_text, static_audio_url)
    """
    clean_key = "".join(c for c in cache_key.lower() if c.isalnum() or c in {"_", "-"}).strip()
    if not clean_key:
        clean_key = "fixed_phrase"
    out_mp3 = OUT_DIR / f"{clean_key}.mp3"

    if not out_mp3.exists() or out_mp3.stat().st_size == 0:
        elevenlabs_tts_to_mp3(phrase, out_mp3)

    return phrase, f"/static/out/{out_mp3.name}"


def emotion_demo_tts(mood: str) -> tuple[str, str]:
    mood = normalize_mood(mood)
    if mood == "happy":
        intro = "Hey, I'm Cupid Bot. You look happy today. "
        text = intro + random.choice(HAPPY_DEMO_SCRIPTS)
    else:
        intro = "Hey, I'm Cupid Bot. You look sad today. "
        text = intro + random.choice(SAD_DEMO_SCRIPTS)

    out_mp3 = OUT_DIR / f"demo_{mood}_{uuid.uuid4().hex}.mp3"
    elevenlabs_tts_to_mp3(text, out_mp3)
    return text, f"/static/out/{out_mp3.name}"


def phrase_tts(text: str, prefix: str = "line") -> tuple[str, str]:
    out_mp3 = OUT_DIR / f"{prefix}_{uuid.uuid4().hex}.mp3"
    elevenlabs_tts_to_mp3(text, out_mp3)
    return text, f"/static/out/{out_mp3.name}"


def get_emotion_turn_script(mood: str) -> list[str]:
    mood = normalize_mood(mood)
    if mood == "happy":
        return [
            "Hey, I'm Cupid Bot. You look happy today.",
            "Would you pick chocolate hearts or red roses for Valentine's Day?",
            "Would you choose a rom-com movie night or cozy love songs this evening?",
            CANDY_CLOSING_LINE,
        ]
    return [
        "Hey, I'm Cupid Bot. You look sad today.",
        "Would you like a kind compliment or a short calming breathing pause?",
        "Should I give you a soft Valentine fun fact or a gentle positive reminder?",
        CANDY_CLOSING_LINE,
    ]


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

def conversation_process():
    t_req = time.time()
    if "audio" not in request.files:
        persist_runtime_error("conversation_process: missing 'audio' in request")
        return jsonify({"error": "No audio file uploaded (field name must be 'audio')."}), 400

    f = request.files["audio"]
    if not f.filename:
        persist_runtime_error("conversation_process: empty uploaded filename")
        return jsonify({"error": "Empty filename."}), 400

    in_suffix = Path(f.filename).suffix.lower() or ".webm"
    with tempfile.TemporaryDirectory(prefix="cupidbot_audio_") as tmp_dir:
        in_path = Path(tmp_dir) / f"in{in_suffix}"
        f.save(in_path)
        converted = maybe_convert_webm_to_wav(in_path)
        audio_bytes = converted.read_bytes()
        mime_type = "audio/wav" if converted.suffix.lower() == ".wav" else "audio/webm"
    print(f"[conversation] input_bytes={len(audio_bytes)} mime={mime_type}")

    # YOLO mood only matters on FIRST approach
    incoming_mood = normalize_mood(request.form.get("mood"))  # default happy
    if "mode" not in session:
        session["mode"] = incoming_mood
    mode = session["mode"]

    # A) Transcribe (Gemini). If it times out after 10s, continue to next step.
    transcription_timed_out = False
    try:
        t0 = time.time()
        print("[conversation] starting transcription (backend=gemini)")
        user_text = run_with_timeout(
            lambda: gemini_transcribe(audio_bytes, mime_type=mime_type),
            timeout_s=10.0,
            label="Transcription",
        )
        print(f"[conversation] transcribe_s={time.time() - t0:.2f}")
        print(f"[conversation] transcript={user_text}")
    except Exception as e:
        if "timed out" in str(e).lower():
            transcription_timed_out = True
            user_text = "(no user response detected)"
            print("[conversation] transcription timed out at 10s; continuing without user transcript")
            persist_runtime_error("Gemini transcription timed out at 10s; continuing to next step")
        else:
            persist_runtime_error(f"Gemini transcription error: {e}")
            return jsonify({"error": f"Gemini transcription error: {e}"}), 500

    # Stop intent check
    user_lower = user_text.lower()
    if any(w in user_lower for w in STOP_WORDS):
        session.pop("history", None)
        session.pop("mode", None)
        goodbye = "Okay, ending now. Talk to you later!"
        out_mp3 = OUT_DIR / f"tts_{uuid.uuid4().hex}.mp3"
        try:
            elevenlabs_tts_to_mp3(goodbye, out_mp3)
        except Exception as e:
            persist_runtime_error(f"ElevenLabs error (goodbye): {e}")
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
        t1 = time.time()
        print("[conversation] starting prompt reply generation")
        # Pass an explicit history snapshot because this call runs in a worker thread.
        # Accessing Flask `session` from that thread raises request-context errors.
        history_snapshot = list(session.get("history", []))
        assistant_text = run_with_timeout(
            lambda: gemini_reply_with_context(user_text, mode, history=history_snapshot),
            timeout_s=REPLY_TIMEOUT_S,
            label="Prompt reply generation",
        )
        print(f"[conversation] reply_s={time.time() - t1:.2f}")
    except Exception as e:
        persist_runtime_error(f"Gemini reply error: {e}")
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
    out_mp3 = OUT_DIR / f"tts_{uuid.uuid4().hex}.mp3"
    try:
        t2 = time.time()
        print("[conversation] starting TTS generation")
        run_with_timeout(
            lambda: elevenlabs_tts_to_mp3(assistant_text, out_mp3),
            timeout_s=TTS_TIMEOUT_S,
            label="TTS generation",
        )
        print(f"[conversation] tts_s={time.time() - t2:.2f}")
    except Exception as e:
        persist_runtime_error(f"ElevenLabs error: {e}")
        return jsonify({"error": f"ElevenLabs error: {e}"}), 500
    print(f"[conversation] total_s={time.time() - t_req:.2f}")

    return jsonify({
        "user_text": user_text,
        "text": assistant_text,
        "audio_url": f"/static/out/{out_mp3.name}",
        "turns": len(session.get("history", [])) // 2,
        "end_session": False,
        "mode": mode
    })


def conversation_reset():
    session.pop("history", None)
    session.pop("mode", None)
    return jsonify({"ok": True})

