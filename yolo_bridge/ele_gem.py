import os
import uuid
import subprocess
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from google import genai
from google.genai import types

from elevenlabs.client import ElevenLabs

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY in .env")
if not ELEVENLABS_API_KEY:
    raise SystemExit("Missing ELEVENLABS_API_KEY in .env")
if not VOICE_ID:
    raise SystemExit("Missing ELEVENLABS_VOICE_ID in .env (use your cloned voice id)")

# --------- Settings ----------
RECORD_SECONDS = 5
SAMPLE_RATE = 16000  # good for speech
GEMINI_MODEL = "gemini-3-flash-preview"  # example from docs
# ----------------------------

def record_wav(path: str, seconds: int = RECORD_SECONDS, sr: int = SAMPLE_RATE):
    print(f"Recording for {seconds}s... (speak now)")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, sr, subtype="PCM_16")
    print(f"Saved mic audio to: {path}")

def gemini_reply_from_audio(wav_path: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    prompt = (
        "You are my assistant."
        "Then respond with a helpful short answer. "
        "Return ONLY your final response (no labels)."
        "Act as a valentine robot, give some punchline if the context is appropriate"
        "You are a robot assistant that's come and ask people if they need help"
        "Keep the answer short in 1 or 2 sentence max"
    )

    # Gemini accepts inline audio bytes using Part.from_bytes
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text

def elevenlabs_tts_to_mp3(text: str, out_path: str):
    eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # convert() returns a generator/stream -> write chunks
    audio_stream = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        output_format="mp3_22050_32",
        model_id="eleven_turbo_v2_5",
    )

    with open(out_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)

def play_mp3(path: str):
    # Uses ffplay if available (common on Linux). Otherwise just skip.
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=False,
        )
    except FileNotFoundError:
        print("ffplay not found (install ffmpeg) — skipping playback.")

def main():
    wav_path = f"mic_{uuid.uuid4().hex}.wav"
    mp3_path = f"tts_{uuid.uuid4().hex}.mp3"

    record_wav(wav_path)

    print("Sending audio to Gemini...")
    reply = gemini_reply_from_audio(wav_path)
    print("\nGemini reply:\n", reply)

    print("\nSending text to ElevenLabs TTS...")
    try:
        elevenlabs_tts_to_mp3(reply, mp3_path)
    except Exception as e:
        print("\nElevenLabs error:", e)
        print(
            "\nIf you see a 402 payment_required, your voice_id is likely a library voice.\n"
            "Use your OWN cloned voice_id for API on the free plan."
        )
        raise

    print(f"\nSaved: {mp3_path}")
    play_mp3(mp3_path)

if __name__ == "__main__":
    main()
