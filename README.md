# 🏠 Hacienda

**AI-powered video captioning agent** — built for **Track 2** of the [LabLabAI × AMD Developer Hackathon: ACT II](https://lablab.ai).

Hacienda watches a video so you don't have to. It downloads clips, samples keyframes, builds a factual scene understanding with a vision model, and turns it into four stylistically distinct captions — all inside a single Docker container, within the harness's 10-minute budget.

---

## ✨ Features

- **Two-stage caption pipeline** — a vision model produces a structured, factual scene analysis from 16 keyframes; a text model then writes all four styles from that analysis in one pass. Style writers never see raw pixels, so they cannot assert visual details the analysis did not establish
- **Alternative grounded mode** — a three-stage variant (`HACIENDA_PIPELINE=generic`): structured JSON brief → verification pass that removes or generalizes unsupported claims (quoted text, brands, locations) → four styles written sequentially with prior captions fed forward for variety
- **Multi-style captioning** — `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech` captions per clip, each 2–4 sentences, grounded in the same scene analysis
- **High-fidelity frame sampling** — 16 uniformly spaced keyframes per clip at up to 1920 px, high JPEG quality, chunk-aware for long videos
- **Resilient by construction** — download retries with backoff, API retry on 429/5xx, per-task pipeline retries, template fallbacks as a last resort, and `results.json` snapshotted after every completed task so a hard kill still leaves valid output

---

## 🏗️ Architecture

```
tasks.json
    │  (tasks processed in parallel)
    ▼
┌──────────┐   ┌────────────┐
│  Reader   │──▶│  Extractor  │
│ download  │   │  ffmpeg     │
│ + retries │   │  16 frames  │
└──────────┘   └────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │            Captioner               │
        │  scene analysis (vision model)     │
        │     → 4 styled captions (text)     │
        └────────────────────────────────────┘
                          │
                          ▼
                    results.json
              (snapshot per task)
```

The clips in the benchmark task set carry no audio streams, so the audio
transcription stage (ffmpeg WAV extraction + Whisper) is not wired into
the runtime path; `pipeline/transcriber.py` remains available for clips that
do have speech.

### Pipeline modules

| Module | File | Role |
|--------|------|------|
| **Reader** | `pipeline/reader.py` | Resolves I/O paths, downloads clips with retries, reads/writes JSON |
| **Extractor** | `pipeline/extractor.py` | Uses `ffprobe`/`ffmpeg` to get duration and sample keyframes |
| **Transcriber** | `pipeline/transcriber.py` | Speech-to-text (not in the runtime path — the benchmark clips have no audio) |
| **Captioner** | `pipeline/captioner.py` | Scene analysis → four styled captions (submission pipeline) |
| **Generic Captioner** | `pipeline/generic_captioner.py` | Brief → verify/generalize → sequential styles (alternative pipeline) |
| **Gemma Client** | `gemma_client.py` | OpenAI-compatible client for Fireworks AI with retry/backoff, per-call model/timeout overrides, and JSON extraction |

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
HACIENDA_GEMMA_MODEL=accounts/fireworks/models/gpt-oss-120b
HACIENDA_VISION_MODEL=accounts/fireworks/models/minimax-m3
HACIENDA_PIPELINE=simple
HACIENDA_WORKERS=2
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `HACIENDA_GEMMA_BASE_URL` | — | Base URL of the Fireworks AI inference endpoint |
| `HACIENDA_GEMMA_TOKEN` | — | Bearer token for Fireworks AI (API key) |
| `HACIENDA_GEMMA_MODEL` | `gemma` | Text model that writes the styled captions |
| `HACIENDA_VISION_MODEL` | same as `HACIENDA_GEMMA_MODEL` | Vision model for the scene analysis |
| `HACIENDA_CAPTION_MODEL` | same as `HACIENDA_GEMMA_MODEL` | Caption model for the generic pipeline's sequential style writes |
| `HACIENDA_PIPELINE` | `generic` | `simple` = two-stage pipeline (submission default); `generic` = brief → verify → sequential styles |
| `HACIENDA_WORKERS` | `3` | Parallel task workers |
| `HACIENDA_TIME_BUDGET` | `570` | Wall-clock budget in seconds |
| `HACIENDA_MAX_FRAMES` / `HACIENDA_MIN_FRAMES` | `16` / `8` | Keyframe count per clip |
| `HACIENDA_FRAME_WIDTH` | `896` | Frame downscale cap in pixels (`0` = native) |
| `HACIENDA_GEMMA_TIMEOUT` / `HACIENDA_VISION_TIMEOUT` | `90` / `180` | Per-request timeouts (seconds) for text / vision calls |

### 3. Build & run with Docker Compose (recommended)

```bash
docker compose up --build
```

This automatically injects your `.env` and live-mounts the project directory.

### 4. Or build & run manually (simulates the judging harness)

```bash
docker build -t hacienda .
docker run --rm \
  -v ${PWD}/examples:/input \
  -v ${PWD}/output:/output \
  hacienda
```

Note: no `--env-file` — this verifies the baked `.env` works exactly as it will under the harness. The first log line after the entrypoint should read `Pipeline: simple | vision=..., text=...`; if it says `FATAL CONFIG`, the image has no usable credentials.

---

## 📦 Runtime contract

The hackathon judging harness mounts volumes at fixed paths and enforces a **~10-minute wall-clock limit**:

| Direction | Container path | Local fallback |
|-----------|---------------|----------------|
| **Input** | `/input/tasks.json` | `inputs/tasks.json` |
| **Output** | `/output/results.json` | `output/results.json` |

`results.json` is rewritten after every completed task (in input order), so even a hard kill leaves valid captions for everything finished so far.

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
      "formal": "A person walks through a sunlit corridor while examining the surrounding architecture, pausing at each doorway to look inside before continuing toward the far exit.",
      "sarcastic": "Someone strolls down a sunlit corridor inspecting doorways with the gravity of a building inspector who has definitely never approved anything on the first visit.",
      "humorous_tech": "A person traverses the sunlit corridor door by door like a crawler indexing every room, politely ignoring the ones that return nothing interesting.",
      "humorous_non_tech": "A visitor wanders the bright corridor peeking through each doorway like a hotel guest convinced a better room exists somewhere on this floor."
    }
  }
]
```

Each caption is natural English, 2–4 sentences, faithful to the visual evidence, with no hedging and no references to frames or models.

---

## 🐳 Building for submission

To securely bake credentials into the final image for judging without exposing them in your GitHub repository:

1. Ensure your `.env` file contains your actual API keys.
2. `.dockerignore` **includes** `.env` during the build, while `.gitignore` prevents it from being pushed to GitHub.
3. Build, verify, and push the image:

```bash
docker build -t your-username/hacienda:latest .
docker run --rm -v ${PWD}/examples:/input -v ${PWD}/output:/output your-username/hacienda:latest
docker push your-username/hacienda:latest
```

The `GemmaClient` loads the baked `.env` at runtime and ignores empty-string environment presets, so the image works whether or not the harness injects any environment variables.

---

## 🛠️ Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Container | Docker (slim base) |
| Media processing | FFmpeg / FFprobe |
| Vision + caption models | Fireworks AI (AMD-powered inference; configurable via `HACIENDA_*_MODEL`) |
| HTTP client | Requests (retry with exponential backoff) |

---

## 📁 Project structure

```
Hacienda/
├── main.py                    # Entry point — parallel orchestration, retries, snapshots
├── gemma_client.py            # OpenAI-compatible Fireworks client (retry, env handling)
├── pipeline/
│   ├── reader.py              # I/O: download clips (with retries), read/write JSON
│   ├── extractor.py           # Frame sampling & audio extraction (ffmpeg)
│   ├── transcriber.py         # Speech-to-text (not in the runtime path)
│   ├── captioner.py           # Scene analysis → four styled captions
│   └── generic_captioner.py   # Brief → verify/generalize → sequential styles
├── examples/
│   └── tasks.json             # Sample tasks for local testing
├── demo/
│   └── web_app.py             # Interactive FastAPI demo UI
├── Dockerfile                 # Production container definition
├── docker-compose.yml         # Dev convenience (auto .env + volume mount)
├── requirements.txt           # Python dependencies
└── .env                       # Local secrets (git-ignored, baked into the image)
```

---

## 👥 Team

Built by **Coatmol** and **Gabdelrahman** for the LabLabAI × AMD Developer Hackathon: ACT II.

---

## 📄 License

This project was created for a hackathon and is provided as-is.
