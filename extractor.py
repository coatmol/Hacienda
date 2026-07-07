import subprocess
import json
import os

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
    duration: float, safety_margin: float = 0.15
) -> list[float]:
    frame_count = max(8, min(24, int(duration / 3)))
    max_ts = max(duration - safety_margin, 0)

    if frame_count <= 1:
        return [max_ts / 2]

    step = max_ts / (frame_count - 1)
    return [round(min(i * step, max_ts), 2) for i in range(frame_count)]


def extract_frames(video_path: str, out_dir: str) -> list[str]:
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
                "-q:v",
                "2",
                out_path,
            ],
            capture_output=True,
            check=True,
        )
        frame_paths.append(out_path)

    return frame_paths, duration, timestamps


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
    return True
