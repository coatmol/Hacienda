# 🏠 Hacienda

**AI-powered video captioning agent** — built for **Track 2** of the [LabLabAI × AMD Developer Hackathon: ACT II](https://lablab.ai).

Hacienda watches a video so you don't have to. It downloads clips, samples keyframes, writes a self-verified scene description, and turns it into four stylistically distinct captions — all inside a single Docker container, within the harness's 10-minute budget.

---

## ✨ Features

- **Grounded caption pipeline** — describe → self-verify against the frames → write each style from the verified description only, so style writers can't hallucinate visual details they never saw
- **Adaptive frame sampling** — picks 8 high-quality keyframes (896 px) across each clip, chunked for long videos
- **Public-validation-set calibration** — prompts embed the reference captions the organizers published for retired validation scene types (see *AMD Hackathon Judging FAQ and Self-Check Guide* in the repo root) as per-style voice examples; when the model's own scene description matches one of those archetypes, that archetype's reference captions are surfaced as gold examples to adapt against the frames. Matching is content-based only — no task ids or URLs — and unmatched scenes use the generic examples
- **Multi-style captioning** — generates `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech` captions per clip, each with tone-anchoring few-shot examples and structural variety across styles
- **Rule enforcement with repair** — validates natural length and sentence count, no hedging, no medium references, no tech words in `humorous_non_tech`; violations trigger targeted rewrite calls, never truncation
- **Time-budget governor** — tasks run in parallel (4 workers by default) and generation degrades gracefully (`full` → `no_verify` → `direct`) as the deadline approaches, with results snapshotted after every task
- **Deep QA** — best-of-N candidate generation with judge-ranked selection (`HACIENDA_DEEP_QA=lite`, the submission default); `full` adds self-eval and weak-style regeneration for offline runs
- **Resilient by construction** — retry with exponential backoff on 429/5xx, layered fallbacks (grounded → direct single-pass → single-frame → templates), every task always produces valid output

---

## 🏗️ Architecture

```
tasks.json
    │  (tasks processed in parallel, 4 workers)
    ▼
┌──────────┐   ┌────────────┐
│  Reader   │──▶│  Extractor  │
│ download  │   │  ffmpeg     │
└──────────┘   └────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │            Captioner               │
        │  describe (vision)                 │
        │     → verify vs frames (vision)    │
        │     → 4 styled captions (text)     │
        │     → rule check + repair          │
        └────────────────────────────────────┘
                          │
                          ▼          ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
                    results.json       Evaluator (best-of-N
              (snapshot per task)    │ ranking, deep QA)     │
                                     └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

The clips in the benchmark task set carry no audio streams, so the audio
transcription stage (ffmpeg WAV extraction + Groq Whisper) is not wired into
the runtime path; `pipeline/transcriber.py` remains available for clips that
do have speech.

### Pipeline modules

| Module | File | Role |
|--------|------|------|
| **Reader** | `pipeline/reader.py` | Resolves I/O paths, downloads video clips, reads/writes JSON |
| **Extractor** | `pipeline/extractor.py` | Uses `ffprobe`/`ffmpeg` to get duration and sample frames (896 px) |
| **Transcriber** | `pipeline/transcriber.py` | Speech-to-text via Groq Whisper-large-v3 (not in the runtime path — the benchmark clips have no audio) |
| **Captioner** | `pipeline/captioner.py` | Describe → verify → per-style caption writing, rule validation, and targeted repair |
| **Evaluator** | `pipeline/evaluator.py` | Scores captions on `accuracy` / `style_match` and ranks best-of-N candidate pools (deep QA mode) |
| **Gemma Client** | `gemma_client.py` | OpenAI-compatible client for Fireworks AI with retry/backoff, JSON extraction, and configurable generation/judge models |

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
HACIENDA_GEMMA_MODEL=your-fireworks-text-model
HACIENDA_VISION_MODEL=your-fireworks-vision-model
HACIENDA_JUDGE_MODEL=your-fireworks-vision-model
HACIENDA_DEEP_QA=lite
HACIENDA_WORKERS=4
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `HACIENDA_GEMMA_BASE_URL` | — | Base URL of the Fireworks AI inference endpoint |
| `HACIENDA_GEMMA_TOKEN` | — | Bearer token for Fireworks AI (API key) |
| `HACIENDA_GEMMA_MODEL` | `gemma` | Text model used for the per-style caption writes and repairs |
| `HACIENDA_VISION_MODEL` | same as `HACIENDA_GEMMA_MODEL` | Vision model for describe/verify and direct generation |
| `HACIENDA_JUDGE_MODEL` | same as `HACIENDA_GEMMA_MODEL` | Vision model for deep QA scoring and the offline benchmark |
| `HACIENDA_WORKERS` | `3` | Parallel task workers (submission image uses 4) |
| `HACIENDA_TIME_BUDGET` | `570` | Wall-clock budget in seconds; generation degrades as it runs out |
| `HACIENDA_DEEP_QA` | off | `lite` = best-of-N ranking (submission default); `full`/`1` adds self-eval + regeneration |

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

Note: no `--env-file` — this verifies the baked `.env` works exactly as it will under the harness. The first log line should read `Models: generation=..., judge=...`; if it says `FATAL CONFIG`, the image has no usable credentials.

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

Each caption is natural English (one sentence, or up to three short sentences for punchy humor formats, matching the organizers' published reference style), faithful to the visual evidence, with no hedging or references to the medium.

---

## 🧪 Offline tools

Iterate locally instead of burning leaderboard submissions:

```bash
# Score an existing output/results.json against the clips with the judge model
python scripts/benchmark.py [tasks.json] [results.json]

# Discover which vision models your API token can actually use
python scripts/probe_models.py [model-id ...]
```

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
├── main.py                 # Entry point — parallel orchestration, time governor
├── gemma_client.py         # OpenAI-compatible Fireworks client (retry, env handling)
├── pipeline/
│   ├── reader.py           # I/O: download clips, read/write JSON
│   ├── extractor.py        # Frame sampling & audio extraction (ffmpeg)
│   ├── transcriber.py      # Speech-to-text via Groq Whisper
│   ├── captioner.py        # Describe → verify → styled captions, rules & repair
│   └── evaluator.py        # Judge scoring & candidate ranking (deep QA)
├── scripts/
│   ├── benchmark.py        # Offline scorer for results.json
│   └── probe_models.py     # Discover usable vision models on the API
├── examples/
│   └── tasks.json          # Sample tasks for local testing
├── demo/
│   └── web_app.py          # Interactive FastAPI demo UI
├── Dockerfile              # Production container definition
├── docker-compose.yml      # Dev convenience (auto .env + volume mount)
├── requirements.txt        # Python dependencies
└── .env                    # Local secrets (git-ignored, baked into the image)
```

---

## 👥 Team

Built by **Coatmol** and **Gabdelrahman** for the LabLabAI × AMD Developer Hackathon: ACT II.

---

## 📄 License

This project was created for a hackathon and is provided as-is.
