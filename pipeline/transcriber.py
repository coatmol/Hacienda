import os
import requests


def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Groq's Whisper-large-v3 model.

    Audio should improve captions when it works, but it must never cause a
    video task to fall back. Missing keys, missing files, and provider errors
    all return an empty transcript.
    """
    if not file_path or not os.path.exists(file_path):
        return ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set; continuing without transcription.")
        return ""

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {api_key}"
        # Do NOT set "Content-Type" manually here.
        # The requests library automatically sets the correct multipart boundary when you pass 'files'.
    }

    try:
        with open(file_path, "rb") as audio_file:

            files = {"file": (os.path.basename(file_path), audio_file)}

            data = {
                "model": "whisper-large-v3",
                "temperature": "0",
                "response_format": "text",
            }

            print(f"Sending {os.path.basename(file_path)} to Groq for transcription...")

            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            response.raise_for_status()

            text = response.text.strip()
            if len(text.split()) < 3:
                return ""
            return text
    except Exception as exc:
        print(f"Transcription failed; continuing without audio text: {exc}")
        return ""
