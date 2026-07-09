import os
import sys
import shutil
import traceback
from uuid import uuid4

from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — allow imports of pipeline.* and gemma_client from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.captioner import DEFAULT_STYLES, fallback_captions, generate_captions
from pipeline.evaluator import evaluate_captions
from pipeline import extractor, reader, transcriber
from gemma_client import GemmaClient

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Hacienda Demo", docs_url="/docs")
client = GemmaClient()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    video_url: str


# ---------------------------------------------------------------------------
# Shared pipeline runner
# ---------------------------------------------------------------------------
def _run_pipeline(task_id: str, video_path: str) -> dict:
    """Run the full captioning pipeline on a single video and return results."""
    temp_root = os.path.join(PROJECT_ROOT, "temp")
    try:
        frame_chunks, duration = extractor.extract_frame_chunks(
            video_path, os.path.join(temp_root, "frames", task_id)
        )
        has_audio = extractor.extract_audio(
            video_path, os.path.join(temp_root, "audio", f"{task_id}.wav")
        )
        transcription = (
            transcriber.transcribe_audio(os.path.join(temp_root, "audio", f"{task_id}.wav"))
            if has_audio
            else None
        )

        task = {"task_id": task_id}
        captions = generate_captions(
            task, frame_chunks, duration, has_audio, client, transcription
        )

        # Self-evaluation
        all_frames = [f for chunk in frame_chunks for f in chunk["frames"]]
        scores = evaluate_captions(all_frames, captions, client)

        # Repair weak styles
        if scores:
            weak_styles = [
                s
                for s, metrics in scores.items()
                if (metrics.get("accuracy", 0) + metrics.get("style_match", 0)) / 2 < 0.6
            ]
            if weak_styles:
                captions = generate_captions(
                    task, frame_chunks, duration, has_audio, client, transcription,
                    focus_styles=weak_styles,
                )

        return {
            "success": True,
            "task_id": task_id,
            "captions": captions,
            "scores": scores,
            "duration": round(duration, 1),
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "success": False,
            "task_id": task_id,
            "error": str(exc),
        }
    finally:
        # Cleanup per-task temp files
        for sub in ("frames", "audio", "clips"):
            path = os.path.join(temp_root, sub, task_id)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
            # Also remove files like <task_id>.wav / <task_id>.mp4
            for ext in (".wav", ".mp4"):
                fpath = os.path.join(temp_root, sub, f"{task_id}{ext}")
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Accept a video URL, download it, and run the full captioning pipeline."""
    task_id = str(uuid4())[:8]
    temp_root = os.path.join(PROJECT_ROOT, "temp")
    video_path = os.path.join(temp_root, "clips", f"{task_id}.mp4")

    try:
        reader.download_video(req.video_url, video_path)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Failed to download video: {exc}"},
        )

    return _run_pipeline(task_id, video_path)


@app.post("/api/upload")
async def api_upload(file: UploadFile):
    """Accept a video file upload and run the full captioning pipeline."""
    task_id = str(uuid4())[:8]
    temp_root = os.path.join(PROJECT_ROOT, "temp")
    video_path = os.path.join(temp_root, "clips", f"{task_id}.mp4")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Failed to save upload: {exc}"},
        )

    return _run_pipeline(task_id, video_path)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gemma_available": client.available,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


# Static file mounts (must be LAST so they don't shadow API routes)
# Serve at both /static (legacy) and / (for relative paths in HTML)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static_legacy")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static_root")
