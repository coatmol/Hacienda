from pipeline.evaluator import evaluate_captions
import pipeline.extractor as extractor
import pipeline.reader as reader
import pipeline.transcriber as transcriber
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file (now baked into the Docker image)
load_dotenv()

from pipeline.captioner import DEFAULT_STYLES, fallback_captions, generate_captions
from gemma_client import GemmaClient

if __name__ == "__main__":
    input_path = reader.resolve_input_path()
    output_path = reader.resolve_output_path()
    tasks = reader.read_tasks_from_json(input_path)
    client = GemmaClient()

    results = []
    for task in tasks:
        task_id = task["task_id"]
        video_path = f"temp/clips/{task_id}.mp4"

        evidence = None
        try:
            reader.download_video(task["video_url"], video_path)
            frame_chunks, duration = extractor.extract_frame_chunks(
                video_path, f"temp/frames/{task_id}"
            )
            has_audio = extractor.extract_audio(video_path, f"temp/audio/{task_id}.wav")

            print(
                f"Task ID: {task_id}, Clip duration: {duration:.1f}s, "
                f"Chunks: {len(frame_chunks)}, Has audio: {has_audio}"
            )

            transcription = (
                transcriber.transcribe_audio(f"temp/audio/{task_id}.wav")
                if has_audio
                else None
            )
            if transcription is not None:
                print(f"Transcription for {task_id}: {transcription}")

            captions = generate_captions(
                task, frame_chunks, duration, has_audio, client, transcription
            )

            all_frame_paths = []
            for chunk in frame_chunks:
                all_frame_paths.extend(chunk["frames"])

            scores = evaluate_captions(all_frame_paths, captions, client)

            # 3. If scores are low, trigger a re-generation for specific styles
            if scores:
                weak_styles = [
                    s
                    for s, metrics in scores.items()
                    if (metrics["accuracy"] + metrics["style_match"]) / 2 < 0.6
                ]

                if weak_styles:
                    print(f"  Weak styles detected: {weak_styles}. Regenerating...")
                    captions = generate_captions(
                        task,
                        frame_chunks,
                        duration,
                        has_audio,
                        client,
                        transcription,
                        focus_styles=weak_styles,
                    )
        except Exception as exc:
            print(f"Task {task_id} failed, writing fallback captions: {exc}")
            captions = fallback_captions(task.get("styles") or DEFAULT_STYLES, evidence)

        results.append({"task_id": task_id, "captions": captions})
        reader.write_results(results, output_path)

    reader.write_results(results, output_path)
    shutil.rmtree("temp", ignore_errors=True)
