import json
import re
from typing import Any, Dict, List, Optional

from gemma_client import GemmaClient, extract_json_object


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
TECH_WORDS = {
    "algorithm",
    "api",
    "app",
    "bug",
    "cache",
    "code",
    "compiler",
    "cpu",
    "database",
    "debug",
    "gpu",
    "latency",
    "loop",
    "pixel",
    "programming",
    "server",
    "software",
    "thread",
}


def generate_captions(
    task: Dict[str, Any],
    frame_chunks: List[Dict[str, Any]],
    duration: float,
    has_audio: bool,
    client: GemmaClient,
    transcription: Optional[str] = None,
) -> Dict[str, str]:
    styles = task.get("styles") or DEFAULT_STYLES

    try:
        evidence = _collect_visual_evidence(
            task["task_id"], frame_chunks, duration, has_audio, client, transcription
        )
        draft = _generate_from_evidence(evidence, styles, client)

        # --- NEW: Local Self-Evaluation ---
        weak_styles = []
        for style in styles:
            cap = draft.get(style, "")
            if not cap or len(cap.split()) < 5:
                weak_styles.append(style)
            elif style == "humorous_non_tech" and _contains_tech_word(cap):
                weak_styles.append(style)

        # If no weak styles are found, skip the repair step entirely!
        if not weak_styles:
            print(f"  Self-eval: all styles passed for {task['task_id']}.")
            return enforce_caption_rules(draft, styles, evidence)

        print(f"  Self-eval: weak styles detected {weak_styles}, repairing...")
        repaired = _repair_captions(
            evidence, styles, draft, client, focus_styles=weak_styles
        )

        return enforce_caption_rules(repaired, styles, evidence)
    except Exception as exc:
        print(f"Caption pipeline failed for {task.get('task_id')}: {exc}")
        return fallback_captions(styles)


def _collect_visual_evidence(
    task_id: str,
    frame_chunks: List[Dict[str, Any]],
    duration: float,
    has_audio: bool,
    client: GemmaClient,
    transcription: Optional[str] = None,
) -> Dict[str, Any]:
    if not client.available:
        return {
            "summary": "The clip was processed locally, but no Gemma proxy was configured.",
            "setting": "unknown video setting",
            "subjects": ["visible subjects"],
            "actions": ["visible activity"],
            "objects": [],
            "visual_details": [],
            "temporal_changes": [],
            "uncertainty": ["Gemma analysis unavailable"],
            "duration_seconds": round(duration, 1),
            "has_audio": has_audio,
            "transcription": transcription,
        }

    chunk_observations = []
    for chunk in frame_chunks:
        prompt = (
            "Analyze these sampled frames from one video chunk. Return only JSON with keys: "
            "setting, subjects, actions, objects, visual_details, temporal_changes, mood, uncertainty, summary. "
            "Be concrete and literal. Do not invent details that are not visible."
        )
        user_text = f"Task {task_id}, video duration {duration:.1f}s, audio_present={has_audio}. "
        if transcription:
            user_text += f"Audio Transcription: '{transcription}'. "

        user_text += (
            f"This chunk spans {chunk['start']:.1f}s to {chunk['end']:.1f}s. "
            f"Frame timestamps: {chunk['timestamps']}."
        )
        raw = client.chat(
            prompt, user_text, chunk["frames"], max_tokens=900, temperature=0.2
        )
        observation = extract_json_object(raw)
        observation["chunk_start"] = chunk["start"]
        observation["chunk_end"] = chunk["end"]
        chunk_observations.append(observation)

    if len(chunk_observations) == 1:
        merged = chunk_observations[0]
        merged["duration_seconds"] = round(duration, 1)
        merged["has_audio"] = has_audio
        merged["transcription"] = transcription
        return merged

    merge_prompt = (
        "Merge these per-chunk observations into one faithful video evidence brief. "
        "Return only JSON with keys: setting, subjects, actions, objects, visual_details, "
        "temporal_changes, mood, uncertainty, summary. Preserve uncertainty."
    )

    raw = client.chat(
        merge_prompt,
        json.dumps(
            {
                "duration_seconds": round(duration, 1),
                "has_audio": has_audio,
                "chunks": chunk_observations,
            },
            ensure_ascii=True,
        ),
        max_tokens=900,
        temperature=0.2,
    )

    merged = extract_json_object(raw)
    merged["duration_seconds"] = round(duration, 1)
    merged["has_audio"] = has_audio
    merged["transcription"] = transcription
    return merged


def _generate_from_evidence(
    evidence: Dict[str, Any], styles: List[str], client: GemmaClient
) -> Dict[str, str]:
    if not client.available:
        return fallback_captions(styles)

    prompt = (
        "You are Gemma writing captions for a video-captioning benchmark. "
        "Use only the provided visual evidence and audio transcription. Return only JSON where every requested style maps to one caption. "
        "Each caption must be English, one sentence, 10-28 words, and faithful to the evidence. "
        "Style rules: formal is objective and professional; sarcastic is dry and lightly ironic; "
        "humorous_tech is funny with programming or technology references; "
        "humorous_non_tech is funny with everyday humor and no tech jargon."
    )

    # --- NEW: Retry loop for transient JSON errors ---
    for attempt in range(2):
        try:
            raw = client.chat(
                prompt,
                json.dumps(
                    {"requested_styles": styles, "evidence": evidence},
                    ensure_ascii=True,
                ),
                max_tokens=700,
                temperature=0.65,
            )
            data = extract_json_object(raw)
            return {style: str(data.get(style, "")) for style in styles}
        except Exception as e:
            print(f"  Draft generation attempt {attempt + 1} failed: {e}")

    # Fallback if both attempts fail
    return fallback_captions(styles, evidence)


def _repair_captions(
    evidence: Dict[str, Any],
    styles: List[str],
    captions: Dict[str, str],
    client: GemmaClient,
    focus_styles: Optional[List[str]] = None,
) -> Dict[str, str]:
    if not client.available:
        return captions

    styles_note = ""
    if focus_styles:
        styles_note = f" Note: Pay extra attention to completely rewriting and fixing these specific styles: {', '.join(focus_styles)}."

    prompt = (
        "You are a strict LLM judge repairing captions before submission. "
        "Return only JSON with every requested style. Fix missing, unfaithful, off-style, too long, "
        "too short, multi-sentence, or unsafe captions. Do not add facts beyond the evidence."
        f"{styles_note}"
    )

    for attempt in range(2):
        try:
            raw = client.chat(
                prompt,
                json.dumps(
                    {
                        "requested_styles": styles,
                        "evidence": evidence,
                        "draft_captions": captions,
                    },
                    ensure_ascii=True,
                ),
                max_tokens=700,
                temperature=0.25,
            )
            data = extract_json_object(raw)
            return {
                style: str(data.get(style, captions.get(style, ""))) for style in styles
            }
        except Exception as e:
            print(f"  Repair attempt {attempt + 1} failed: {e}")

    return captions


def enforce_caption_rules(
    captions: Dict[str, str], styles: List[str], evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    final = {}
    fallback = fallback_captions(styles, evidence)
    for style in styles:
        caption = _clean_caption(captions.get(style, ""))
        if not caption:
            caption = fallback[style]

        if style == "formal":
            caption = caption.replace("!", ".")
        elif style == "humorous_non_tech" and _contains_tech_word(caption):
            caption = fallback[style]

        final[style] = _limit_words(_ensure_sentence(caption), 28)
    return final


def fallback_captions(
    styles: List[str], evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    subject = _best_subject(evidence)
    action = _best_action(evidence)
    setting = _best_setting(evidence)
    base = f"{subject} {action} in {setting}".strip()

    templates = {
        "formal": f"{base} while the scene remains visually clear and centered on the main activity.",
        "sarcastic": f"{base}, because apparently the day needed a little more drama.",
        "humorous_tech": f"{base} like a tiny production job running straight through the visual pipeline.",
        "humorous_non_tech": f"{base} with the confidence of someone pretending this was the plan.",
    }
    return {style: _ensure_sentence(templates.get(style, templates["formal"])) for style in styles}


def _clean_caption(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().strip("\"'")
    text = re.sub(r"^[\-*: ]+", "", text)
    if not text:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip()


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words]).rstrip(".,;:!?")
    return f"{trimmed}."


def _contains_tech_word(text: str) -> bool:
    words = {word.strip(".,;:!?()[]{}\"'").lower() for word in text.split()}
    return bool(words & TECH_WORDS)


def _best_subject(evidence: Optional[Dict[str, Any]]) -> str:
    if not evidence:
        return "The video"
    subjects = evidence.get("subjects") or []
    if isinstance(subjects, list) and subjects:
        return str(subjects[0]).strip().capitalize()
    return "The video"


def _best_action(evidence: Optional[Dict[str, Any]]) -> str:
    if not evidence:
        return "shows visible activity"
    actions = evidence.get("actions") or []
    if isinstance(actions, list) and actions:
        action = str(actions[0]).strip()
        return action if action.lower().startswith(("is ", "are ", "shows ", "moves ")) else f"shows {action}"
    return "shows visible activity"


def _best_setting(evidence: Optional[Dict[str, Any]]) -> str:
    if not evidence:
        return "the scene"
    setting = str(evidence.get("setting") or "the scene").strip()
    return setting or "the scene"
