import json
import math
import os
import subprocess
from typing import Dict, List, Tuple

MAX_FRAMES_PER_CLIP = 8
CHUNK_SECONDS = 60.0
FRAME_WIDTH = 896
FRAME_QUALITY = "3"


def get_duration(video_path: str) -> float:
    """Get clip duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def compute_frame_timestamps(
    duration: float, safety_margin: float = 0.15, max_frames: int = MAX_FRAMES_PER_CLIP
) -> List[float]:
    frame_count = max(8, min(max_frames, int(duration / 3)))
    max_ts = max(duration - safety_margin, 0)

    if frame_count <= 1:
        return [max_ts / 2]

    step = max_ts / (frame_count - 1)
    return [round(min(i * step, max_ts), 2) for i in range(frame_count)]


def compute_frame_chunks(
    duration: float,
    safety_margin: float = 0.15,
    max_frames: int = MAX_FRAMES_PER_CLIP,
    chunk_seconds: float = CHUNK_SECONDS,
) -> List[List[float]]:
    """Return timestamp batches that preserve coverage while keeping requests small."""
    timestamps = compute_frame_timestamps(duration, safety_margin, max_frames)
    chunk_count = max(1, math.ceil(duration / chunk_seconds))
    chunks: List[List[float]] = [[] for _ in range(chunk_count)]

    for ts in timestamps:
        chunk_index = min(int(ts // chunk_seconds), chunk_count - 1)
        chunks[chunk_index].append(ts)

    return [chunk for chunk in chunks if chunk]


def extract_frames(
    video_path: str, out_dir: str
) -> Tuple[List[str], float, List[float]]:
    os.makedirs(out_dir, exist_ok=True)
    duration = get_duration(video_path)
    timestamps = compute_frame_timestamps(duration)

    frame_paths = []
    for idx, ts in enumerate(timestamps):
        out_path = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(ts),
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({FRAME_WIDTH},iw)':-2",
                "-q:v",
                FRAME_QUALITY,
                out_path,
            ],
            capture_output=True,
            check=True,
        )
        frame_paths.append(out_path)

    return frame_paths, duration, timestamps


def extract_frame_chunks(video_path: str, out_dir: str) -> Tuple[List[Dict], float]:
    os.makedirs(out_dir, exist_ok=True)
    duration = get_duration(video_path)
    timestamp_chunks = compute_frame_chunks(duration)

    chunks = []
    for chunk_idx, timestamps in enumerate(timestamp_chunks):
        frame_paths = []
        chunk_dir = os.path.join(out_dir, f"chunk_{chunk_idx:02d}")
        os.makedirs(chunk_dir, exist_ok=True)

        for idx, ts in enumerate(timestamps):
            out_path = os.path.join(chunk_dir, f"frame_{idx:03d}.jpg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(ts),
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale='min({FRAME_WIDTH},iw)':-2",
                    "-q:v",
                    FRAME_QUALITY,
                    out_path,
                ],
                capture_output=True,
                check=True,
            )
            frame_paths.append(out_path)

        chunks.append(
            {
                "index": chunk_idx,
                "start": min(timestamps),
                "end": max(timestamps),
                "timestamps": timestamps,
                "frames": frame_paths,
            }
        )

    return chunks, duration


def extract_audio(video_path: str, out_path: str) -> bool:
    """Extract audio track. Returns False if clip has no audio stream."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    has_audio = bool(json.loads(probe.stdout).get("streams"))
    if not has_audio:
        return False

    folder_path = os.path.dirname(out_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                out_path,
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            "FFMPEG STDERR:",
            e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr,
        )
        raise

    return True
