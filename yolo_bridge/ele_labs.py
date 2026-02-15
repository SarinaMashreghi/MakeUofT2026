from elevenlabs import ElevenLabs

client = ElevenLabs(api_key ="sk_8ec5cf867b7614c823bd40c73c69b7885ce923c05483c89d")

audio = client.text_to_speech.convert(
    voice_id="hpp4J3VqNfWAUOO0d1Us",
    text = "Hello Tom, your system is ready"
)

with open("speech.mp3", "wb") as f:
    for chunk in audio:
        if chunk:          # avoid empty chunks
            f.write(chunk)

print("Saved speech.mp3")