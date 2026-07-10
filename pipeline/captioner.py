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
        # Collect all frames from chunks
        all_frames = []
        for chunk in frame_chunks:
            all_frames.extend(chunk.get("frames", []))
            
        draft = _generate_direct_captions(all_frames, duration, has_audio, transcription, styles, client)
        return enforce_caption_rules(draft, styles, None)
    except Exception as exc:
        print(f"Caption pipeline failed for {task.get('task_id')}: {exc}")
        return fallback_captions(styles)


def _generate_direct_captions(
    frame_paths: List[str],
    duration: float,
    has_audio: bool,
    transcription: Optional[str],
    styles: List[str],
    client: GemmaClient
) -> Dict[str, str]:
    if not client.available:
        return fallback_captions(styles)
        
    prompt = (
        "You are an expert video caption writer for a scoring benchmark. "
        "You are analyzing a short video clip using its keyframes (in order).\n"
    )
    if transcription:
        prompt += f"Audio transcript from the clip:\n\"{transcription}\"\n\n"
        
    prompt += (
        "Write exactly ONE caption per requested style, based ONLY on what you see in the frames (and transcript). "
        "Each caption MUST be a single English sentence, 15-30 words. "
        "Be SPECIFIC — mention concrete details like colors, objects, settings, and actions. "
        "Never invent details not present in the evidence.\n\n"
        "STYLE RULES (each caption must sound COMPLETELY DIFFERENT from the others):\n"
        "- formal: Objective, professional, descriptive. State facts. No humor, no opinion, no exclamations.\n"
        "- sarcastic: Dry, ironic, lightly mocking. Use subtle wit to poke fun at what is happening. Must still describe the scene.\n"
        "- humorous_tech: Genuinely funny using a SPECIFIC programming or technology metaphor (e.g. APIs, git, debugging, servers, threads). The humor must come from the tech analogy.\n"
        "- humorous_non_tech: Genuinely funny using everyday humor, wordplay, or absurd observations. ABSOLUTELY ZERO technology or programming words.\n\n"
        "CRITICAL: Return ONLY a raw JSON object with exactly 4 keys. No markdown, no explanation.\n"
        f'Format: {{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}'
    )
    
    user_text = "Analyze these frames and output the JSON captions."

    # Try vision_chat twice
    for attempt in range(2):
        try:
            raw = client.vision_chat(
                system_prompt=prompt,
                user_text=user_text,
                image_paths=frame_paths,
                max_tokens=900,
                temperature=0.55
            )
            data = extract_json_object(raw)
            return {style: str(data.get(style, "")) for style in styles}
        except Exception as e:
            print(f"  Draft generation (Vision) attempt {attempt + 1} failed: {e}")
            
    # Last resort
    return fallback_captions(styles, None)

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
    # Since we don't have evidence dict anymore, fallback is generic
    base = "The video shows visible activity in the scene"

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
