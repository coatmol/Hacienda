import extractor
import reader

if __name__ == "__main__":
    tasks = reader.read_tasks_from_json("inputs/tasks.json")
    for task in tasks:
        reader.download_video(task["video_url"], f"temp/clips/{task['task_id']}.mp4")

    for task in tasks:
        frames, duration, timestamps = extractor.extract_frames(f"temp/clips/{task['task_id']}.mp4", f"temp/frames/{task['task_id']}")
        has_audio = extractor.extract_audio(f"temp/clips/{task['task_id']}.mp4", f"temp/audio/{task['task_id']}.wav")

        print(f"Task ID: {task['task_id']}, Clip duration: {duration:.1f}s, Has audio: {has_audio}")
