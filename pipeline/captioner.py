import json
import re
from typing import Any, Dict, List, Optional

from gemma_client import GemmaClient, extract_json_object


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
TECH_WORDS = {
    "algorithm", "api", "app", "backend", "bandwidth", "binary", "boolean",
    "buffer", "bug", "byte", "cache", "callback", "cli", "cloud", "code",
    "commit", "compiler", "cpu", "css", "data", "database", "debug",
    "deploy", "docker", "endpoint", "firewall", "frontend", "function",
    "git", "gpu", "hash", "html", "http", "instance", "javascript", "json",
    "kernel", "kubernetes", "lambda", "latency", "linux", "loop", "malloc",
    "merge", "microservice", "mutex", "node", "npm", "null", "overclocking",
    "packet", "parse", "pipeline", "pixel", "pointer", "port", "process",
    "programming", "protocol", "python", "query", "ram", "recursion",
    "regex", "repository", "runtime", "script", "segfault", "server",
    "socket", "software", "sql", "stack", "subnet", "subroutine",
    "syntax", "tcp", "thread", "token", "udp", "variable", "vm",
    "webhook", "wifi",
}


def generate_captions(
    task: Dict[str, Any],
    frame_chunks: List[Dict[str, Any]],
    duration: float,
    has_audio: bool,
    client: GemmaClient,
    transcription: Optional[str] = None,
    focus_styles: Optional[List[str]] = None,
) -> Dict[str, str]:
    styles = task.get("styles") or DEFAULT_STYLES

    try:
        evidence = _collect_visual_evidence(
            task["task_id"], frame_chunks, duration, has_audio, client, transcription
        )
        draft = _generate_from_evidence(evidence, styles, client)

        # This will trigger the repair with targeted focus if focus_styles is passed
        repaired = _repair_captions(
            evidence, styles, draft, client, focus_styles=focus_styles
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
        raw = client.vision_chat(
            system_prompt=prompt, user_text=user_text, image_paths=chunk["frames"], max_tokens=900, temperature=0.2
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

    user_text = json.dumps(
        {
            "duration_seconds": round(duration, 1),
            "has_audio": has_audio,
            "chunks": chunk_observations,
        },
        ensure_ascii=True,
    )

    # Try Gemma first, then fallback to minimax-m3 for guaranteed JSON
    for attempt_label, chat_fn in [("Gemma", client.chat), ("Fallback", client.fallback_chat)]:
        try:
            raw = chat_fn(merge_prompt, user_text, max_tokens=900, temperature=0.2)
            merged = extract_json_object(raw)
            merged["duration_seconds"] = round(duration, 1)
            merged["has_audio"] = has_audio
            merged["transcription"] = transcription
            return merged
        except Exception as e:
            print(f"  Chunk merge ({attempt_label}) failed: {e}")

    # If both fail, just use the first chunk's observation
    merged = chunk_observations[0]
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
        "You are an expert video caption writer for a scoring benchmark. "
        "Write exactly ONE caption per requested style. Each caption MUST be a single English sentence, "
        "15-30 words, and grounded ONLY in the provided visual evidence and audio transcription. "
        "Be SPECIFIC — mention concrete details like colors, objects, settings, and actions from the evidence. "
        "Never invent details not present in the evidence.\n\n"
        "STYLE RULES (each caption must sound COMPLETELY DIFFERENT from the others):\n"
        "- formal: Objective, professional, descriptive. State facts. No humor, no opinion, no exclamations.\n"
        "- sarcastic: Dry, ironic, lightly mocking. Use subtle wit to poke fun at what is happening. Must still describe the scene.\n"
        "- humorous_tech: Genuinely funny using a SPECIFIC programming or technology metaphor (e.g. APIs, git, debugging, servers, threads). The humor must come from the tech analogy.\n"
        "- humorous_non_tech: Genuinely funny using everyday humor, wordplay, or absurd observations. ABSOLUTELY ZERO technology or programming words.\n\n"
        "CRITICAL: Return ONLY a raw JSON object with exactly 4 keys. No markdown, no explanation.\n"
        'Format: {"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}'
    )

    user_text = json.dumps(
        {"requested_styles": styles, "evidence": evidence},
        ensure_ascii=True,
    )

    # Try Gemma twice, then fallback to minimax-m3 for guaranteed JSON
    for attempt in range(2):
        try:
            raw = client.chat(prompt, user_text, max_tokens=700, temperature=0.55)
            data = extract_json_object(raw)
            return {style: str(data.get(style, "")) for style in styles}
        except Exception as e:
            print(f"  Draft generation (Gemma) attempt {attempt + 1} failed: {e}")

    # Fallback: use minimax-m3 with guaranteed JSON mode
    try:
        print("  Falling back to minimax-m3 for draft generation...")
        raw = client.fallback_chat(prompt, user_text, max_tokens=700, temperature=0.55)
        data = extract_json_object(raw)
        return {style: str(data.get(style, "")) for style in styles}
    except Exception as e:
        print(f"  Fallback draft generation also failed: {e}")

    # Last resort: evidence-based templates
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
        styles_note = f"\nNote: Pay extra attention to completely rewriting and fixing these specific styles: {', '.join(focus_styles)}."

    prompt = (
        "You are a strict caption editor polishing captions before final submission. "
        "Return ONLY a JSON object with all requested styles. For each caption:\n"
        "1. KEEP all specific visual details (colors, objects, settings, actions) from the evidence.\n"
        "2. FIX any caption that is unfaithful, off-style, too long (>30 words), too short (<12 words), "
        "multi-sentence, or contains hallucinated details.\n"
        "3. Ensure humorous_non_tech has ZERO technology/programming references.\n"
        "4. Ensure humorous_tech includes a SPECIFIC tech/programming metaphor.\n"
        "5. Ensure formal is objective with no humor or opinion.\n"
        "6. Do NOT make captions more generic — preserve specificity."
        f"{styles_note}"
    )

    user_text = json.dumps(
        {
            "requested_styles": styles,
            "evidence": evidence,
            "draft_captions": captions,
        },
        ensure_ascii=True,
    )

    # Try Gemma first, then fallback to minimax-m3
    for attempt_label, chat_fn in [("Gemma", client.chat), ("Fallback", client.fallback_chat)]:
        try:
            raw = chat_fn(prompt, user_text, max_tokens=700, temperature=0.25)
            data = extract_json_object(raw)
            return {style: str(data.get(style, captions.get(style, ""))) for style in styles}
        except Exception as e:
            print(f"  Repair ({attempt_label}) failed: {e}. Trying next...")

    # If both fail, keep the draft captions as-is
    print("  All repair attempts failed. Keeping draft captions.")
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

        final[style] = _limit_words(_ensure_sentence(caption), 40)
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
