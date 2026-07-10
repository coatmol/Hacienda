import os
import requests


def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Groq's Whisper-large-v3 model.
    """
    # Grab the API key from the environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {api_key}"
        # Do NOT set "Content-Type" manually here.
        # The requests library automatically sets the correct multipart boundary when you pass 'files'.
    }

    # We must open the file in binary mode ("rb") for requests to upload it
    with open(file_path, "rb") as audio_file:

        # -F "file=@./audio.m4a" translates to the 'files' parameter
        files = {"file": (os.path.basename(file_path), audio_file)}

        # The rest of the -F flags translate to the standard 'data' dictionary
        data = {
            "model": "whisper-large-v3",
            "temperature": "0",
            "response_format": "text",
        }

        print(f"Sending {os.path.basename(file_path)} to Groq for transcription...")

        response = requests.post(url, headers=headers, files=files, data=data)

        # Catch any errors (like invalid API keys or unsupported file formats)
        response.raise_for_status()

        return response.text