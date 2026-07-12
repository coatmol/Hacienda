import json
import os
import time

import requests


def read_tasks_from_json(file_path):
    with open(file_path, "r") as file:
        tasks = json.load(file)
    return tasks


def download_video(url, output_path, retries=3):
    print(f"Downloading video from {url} to {output_path}...")

    folder_path = os.path.dirname(output_path)
    if os.path.exists(
        output_path
    ):  # Skip download if file already exists for faster testing
        print(f"File {output_path} already exists. Skipping download.")
        return

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    # A single flaky fetch must not cost the whole task: a failed task ships
    # template captions, which score far below even a mediocre real caption.
    last_error = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=90) as response:
                response.raise_for_status()  # Throws error early if 404

                partial_path = output_path + ".part"
                with open(partial_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                os.replace(partial_path, output_path)
                return
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Download failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def resolve_input_path() -> str:
    if os.path.exists("/input/tasks.json"):
        return "/input/tasks.json"
    return "inputs/tasks.json"


def resolve_output_path() -> str:
    if os.path.isdir("/output"):
        return "/output/results.json"
    os.makedirs("output", exist_ok=True)
    return "output/results.json"


def write_results(results, output_path):
    folder_path = os.path.dirname(output_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2, ensure_ascii=True)
