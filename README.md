# 🏠 Hacienda

**AI-powered video captioning agent** — built for **Track 2** of the [LabLabAI × AMD Developer Hackathon: ACT II](https://lablab.ai).

Hacienda watches a video so you don't have to. It downloads clips, samples keyframes, transcribes audio, asks a vision-language model what it sees, and writes four stylistically distinct captions — all inside a single Docker container.

---

## ✨ Features

- **Adaptive frame sampling** — picks 8–24 keyframes scaled to clip duration, chunked for long videos
- **Audio transcription** — extracts speech via [Groq Whisper](https://groq.com/) (large-v3) when an audio track is present
- **Vision-language analysis** — extracts visual facts using minimax-m3, then drafts captions using a custom fine-tuned Gemma-4-e4b model on Fireworks AI
- **Interactive Web UI** — upload videos or test URLs interactively via the built-in FastAPI demo interface
- **Multi-style captioning** — generates `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech` captions per clip
- **Self-evaluation & repair** — scores each caption for accuracy and tone, then rewrites weak ones automatically
- **Deterministic fallbacks** — every task always produces valid output, even when the model is unavailable

---

## 🏗️ Architecture

```
tasks.json
    │
    ▼
┌──────────┐   ┌────────────┐   ┌──────────────┐
│  Reader   │──▶│  Extractor  │──▶│  Transcriber  │
│ download  │   │  ffmpeg     │   │  Groq Whisper │
└──────────┘   └────────────┘   └──────────────┘
                     │                   │
                     ▼                   ▼
              ┌──────────────────────────────┐
              │         Captioner            │
              │  visual evidence → 4 styles  │
              └──────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────────────┐
              │         Evaluator            │
              │  score → repair weak caps    │
              └──────────────────────────────┘
                          │
                          ▼
                    results.json
```

### Pipeline modules

| Module | File | Role |
|--------|------|------|
| **Reader** | `pipeline/reader.py` | Resolves I/O paths, downloads video clips, reads/writes JSON |
| **Extractor** | `pipeline/extractor.py` | Uses `ffprobe`/`ffmpeg` to get duration, sample frames (768 px wide), and extract 16 kHz mono WAV audio |
| **Transcriber** | `pipeline/transcriber.py` | Sends audio to Groq's Whisper-large-v3 endpoint for speech-to-text |
| **Captioner** | `pipeline/captioner.py` | Collects per-chunk visual evidence via Gemma, drafts four caption styles, then repairs off-tone or unfaithful results |
| **Evaluator** | `pipeline/evaluator.py` | Scores each caption on `accuracy` and `style_match` (0–1); flags weak styles for re-generation |
| **Gemma Client** | `gemma_client.py` | Client for Fireworks AI (minimax-m3 for vision, Gemma-4-e4b for text) with robust JSON extraction |

---

## 🚀 Quick start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (or Docker Desktop on Windows/macOS)

### 1. Clone the repo

```bash
git clone https://github.com/coatmol/Hacienda.git
cd Hacienda
```

### 2. Configure environment

Create a `.env` file in the project root (already in `.gitignore`):

```env
HACIENDA_GEMMA_BASE_URL=https://api.fireworks.ai/inference/v1
HACIENDA_GEMMA_TOKEN=your-fireworks-api-key
HACIENDA_GEMMA_MODEL=accounts/gamal004/deployments/ie6yw9o2
GROQ_API_KEY=your-groq-api-key
```

| Variable | Purpose |
|----------|---------|
| `HACIENDA_GEMMA_BASE_URL` | Base URL of the Fireworks AI inference endpoint |
| `HACIENDA_GEMMA_TOKEN` | Bearer token for Fireworks AI (API Key) |
| `HACIENDA_GEMMA_MODEL` | The custom fine-tuned Gemma-4-e4b model on Fireworks AI |
| `GROQ_API_KEY` | API key for Groq's Whisper audio transcription |

### 3. Build & run with Docker Compose (recommended)

```bash
docker compose up --build
```

This automatically injects your `.env` and live-mounts the project directory.

### 4. Or build & run manually

```bash
docker build -t hacienda .
docker run --rm -it --env-file .env \
  -v ${PWD}/inputs:/input \
  -v ${PWD}/output:/output \
  hacienda
```

---

## 📦 Runtime contract

The hackathon judging harness mounts volumes at fixed paths:

| Direction | Container path | Local fallback |
|-----------|---------------|----------------|
| **Input** | `/input/tasks.json` | `inputs/tasks.json` |
| **Output** | `/output/results.json` | `output/results.json` |

### Input format (`tasks.json`)

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

### Output format (`results.json`)

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "A person walks through a sunlit corridor while examining the surrounding architecture.",
      "sarcastic": "Apparently walking through a hallway is now an extreme sport worth filming.",
      "humorous_tech": "User navigates the hallway like a packet traversing a well-routed network.",
      "humorous_non_tech": "Someone decided this hallway deserved its own movie premiere."
    }
  }
]
```

Each caption is a single English sentence, 10–28 words, faithful to the visual evidence.

---

## 🐳 Building for submission

To securely bake credentials into the final image for judging without exposing them in your GitHub repository:

1. Ensure your `.env` file contains your actual API keys.
2. We have configured the `.dockerignore` to **include** `.env` during the build process, while `.gitignore` prevents it from being pushed to GitHub.
3. Build and push the image:

```bash
docker build -t your-username/hacienda:latest .
docker push your-username/hacienda:latest
```

When the container runs, the built-in `python-dotenv` loader will automatically read the `.env` file baked inside the image!

---

## 🛠️ Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Container | Docker (slim base) |
| Media processing | FFmpeg / FFprobe |
| Audio transcription | Groq Whisper (large-v3) |
| Vision model | minimax-m3 via Fireworks AI |
| Text model | Fine-tuned Gemma-4-e4b via Fireworks AI |
| HTTP client | Requests |

---

## 📁 Project structure

```
Hacienda/
├── main.py                 # Entry point — orchestrates the full pipeline
├── gemma_client.py         # OpenAI-compatible Gemma chat client
├── pipeline/
│   ├── reader.py           # I/O: download clips, read/write JSON
│   ├── extractor.py        # Frame sampling & audio extraction (ffmpeg)
│   ├── transcriber.py      # Speech-to-text via Groq Whisper
│   ├── captioner.py        # Evidence collection, caption generation & repair
│   └── evaluator.py        # Self-evaluation scoring
├── inputs/
│   └── tasks.json          # Sample tasks for local testing
├── Dockerfile              # Production container definition
├── docker-compose.yml      # Dev convenience (auto .env + volume mount)
├── requirements.txt        # Python dependencies
└── .env                    # Local secrets (git-ignored)
```

---

## 👥 Team

Built by **Coatmol** and **Gabdelrahman** for the LabLabAI × AMD Developer Hackathon: ACT II.

---

## 📄 License

This project was created for a hackathon and is provided as-is.
