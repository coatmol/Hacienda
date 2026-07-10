import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from gemma_client import GemmaClient, extract_json_object


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
# Styles where the self-judge consistently scores lower and quality varies a
# lot between samples — worth generating multiple candidates for.
HUMOR_STYLES = ["sarcastic", "humorous_tech", "humorous_non_tech"]

# Only words that read as unambiguous tech jargon. Common English words that
# happen to have a tech meaning (loop, stack, process, data, ...) are excluded:
# a false positive costs a repair rewrite of an otherwise good caption.
TECH_WORDS = {
    "algorithm", "api", "app", "backend", "bandwidth", "binary", "boolean",
    "bug", "byte", "cache", "callback", "cli", "commit", "compiler", "cpu",
    "css", "database", "debug", "deploy", "docker", "endpoint", "firewall",
    "frontend", "git", "gpu", "html", "http", "javascript", "json", "kernel",
    "kubernetes", "lambda", "latency", "linux", "malloc", "microservice",
    "mutex", "npm", "null", "overclocking", "parse", "pipeline", "pixel",
    "pointer", "programming", "protocol", "python", "ram", "recursion",
    "regex", "repository", "runtime", "segfault", "server", "software",
    "sql", "subnet", "subroutine", "syntax", "tcp", "udp", "variable",
    "vm", "webhook", "wifi",
}

HEDGE_WORDS = {
    "appears", "appear", "seems", "seem", "seemingly", "likely", "possibly",
    "probably", "perhaps", "maybe", "apparently",
}
META_WORDS = {"video", "clip", "frame", "frames", "footage", "keyframes"}

_STRIP_PUNCT = ".,;:!?()[]{}\"'"

STYLE_RULES = {
    "formal": "Objective, professional, descriptive. State facts. No humor, no opinion, no exclamations.",
    "sarcastic": "Dry, ironic, lightly mocking. Use subtle wit to poke fun at what is genuinely happening on screen — never at invented events.",
    "humorous_tech": "Genuinely funny using a SPECIFIC programming or technology metaphor (e.g. APIs, git, debugging, servers, threads). The caption must first describe the real subject, action, and setting, then attach the tech analogy with 'like' or 'as if'.",
    "humorous_non_tech": "Genuinely funny using everyday humor, wordplay, or absurd observations about what is really shown. ABSOLUTELY ZERO technology or programming words.",
}

# Tone-only few-shot examples; each prompt states explicitly that the content
# is unrelated and must not be reused, so they anchor style without leaking.
STYLE_TONE_EXAMPLES = {
    "formal": "'The subject proceeds through the marked route without deviation.'",
    "sarcastic": "'The pigeon surveys its kingdom of one park bench with the confidence of a landlord.'",
    "humorous_tech": "'404: graceful landing not found.'",
    "humorous_non_tech": "'Confidence level: main character. Execution level: blooper reel.'",
}

GROUNDING_RULE = (
    "GROUNDING RULE (applies to EVERY style, including humorous ones): each caption must still "
    "explicitly name the real subject, their real action, and the real setting from the frames. "
    "Build jokes ON TOP of those facts as comparisons using 'like' or 'as if' — never assert "
    "invented specifics (text on a screen, names, foods, thoughts, outcomes) that are not "
    "directly visible or stated in the transcript.\n"
)


def generate_captions(
    task: Dict[str, Any],
    frame_chunks: List[Dict[str, Any]],
    duration: float,
    has_audio: bool,
    client: GemmaClient,
    transcription: Optional[str] = None,
    focus_styles: Optional[List[str]] = None,
    speed: str = "full",
) -> Dict[str, str]:
    """speed: "full" = describe + verify + per-style writes;
    "no_verify" = skip the verification vision call;
    "direct" = single vision call for all styles (fastest real path)."""
    styles = list(task.get("styles") or DEFAULT_STYLES)
    if focus_styles:
        styles = [s for s in styles if s in focus_styles] or styles

    all_frames: List[str] = []
    for chunk in frame_chunks:
        all_frames.extend(chunk.get("frames", []))

    # Primary path: describe -> self-verify -> write each style from the
    # verified description. Style writers never see the frames, so they cannot
    # assert anything the verification pass did not confirm.
    draft: Dict[str, str] = {}
    if speed != "direct":
        try:
            draft = _grounded_captions(
                all_frames, transcription, styles, client,
                skip_verify=(speed == "no_verify"),
            )
        except Exception as exc:
            print(f"Grounded caption pipeline failed for {task.get('task_id')}: {exc}")

    if not any(_clean_caption(draft.get(style, "")) for style in styles):
        try:
            draft = _generate_direct_captions(
                all_frames, duration, has_audio, transcription, styles, client
            )
        except Exception as exc:
            print(f"Direct caption fallback failed for {task.get('task_id')}: {exc}")

    if not any(_clean_caption(draft.get(style, "")) for style in styles):
        draft = _last_resort_captions(all_frames, transcription, styles, client)

    return enforce_caption_rules(draft, styles, client=client)


def _grounded_captions(
    frame_paths: List[str],
    transcription: Optional[str],
    styles: List[str],
    client: GemmaClient,
    skip_verify: bool = False,
) -> Dict[str, str]:
    if not client.available:
        raise RuntimeError("Gemma proxy is not configured.")

    description = _describe_scene(frame_paths, transcription, client)
    if not skip_verify:
        description = _verify_description(frame_paths, description, client)
    print(f"  Grounding description: {description}")

    captions: Dict[str, str] = {style: "" for style in styles}

    def _write(style: str, prior: List[str]) -> str:
        try:
            return _write_style_caption(style, description, transcription, prior, client)
        except Exception as exc:
            print(f"  Style write failed for {style}: {exc}")
            return ""

    # Write the first (most factual) style alone, then the rest in parallel
    # using it as the sentence-structure reference — one extra round-trip
    # instead of three.
    first, rest = styles[0], styles[1:]
    captions[first] = _write(first, [])
    prior = [captions[first]] if captions[first] else []

    if rest:
        with ThreadPoolExecutor(max_workers=len(rest)) as pool:
            futures = {style: pool.submit(_write, style, prior) for style in rest}
            for style, future in futures.items():
                captions[style] = future.result()
    return captions


def _describe_scene(
    frame_paths: List[str], transcription: Optional[str], client: GemmaClient
) -> str:
    prompt = "These are frames sampled across a short video clip, in order.\n"
    if transcription:
        prompt += f'Audio transcript from the clip: "{transcription}"\n'
    prompt += (
        "Note the setting, the main subjects, the specific action or motion "
        "happening, and any readable on-screen text. Then write 2-4 dense, "
        "factual sentences. Be specific, don't generalize. "
        "Output ONLY the description, no preamble."
    )
    return client.vision_chat(
        system_prompt="You describe video content precisely and factually.",
        user_text=prompt,
        image_paths=frame_paths,
        max_tokens=800,
        temperature=0.2,
        max_images=8,
        json_mode=False,
    ).strip()


def _verify_description(
    frame_paths: List[str], draft: str, client: GemmaClient
) -> str:
    prompt = (
        f'Here is a draft description of these frames: "{draft}"\n\n'
        "Check it against the actual frames. If accurate and specific, repeat it "
        "unchanged. If anything is wrong or too generic, correct it. Output ONLY "
        "the final description."
    )
    verified = client.vision_chat(
        system_prompt="You verify video descriptions against the actual frames.",
        user_text=prompt,
        image_paths=frame_paths,
        max_tokens=800,
        temperature=0.2,
        max_images=8,
        json_mode=False,
    ).strip()
    return verified or draft


def _write_style_caption(
    style: str,
    description: str,
    transcription: Optional[str],
    prior_captions: List[str],
    client: GemmaClient,
) -> str:
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\nCaptions already written for this clip in other styles (for "
            "reference — use a DIFFERENT sentence structure): "
            + " | ".join(prior_captions)
        )
    example = STYLE_TONE_EXAMPLES.get(style, "")
    prompt = (
        f"Style: {style} — {STYLE_RULES[style]}\n"
        "Example of this WRITING STYLE only (unrelated content — do not reuse or "
        f"reference it, it exists only to show the tone): {example}\n\n"
        f"Here is a verified factual description of a video clip:\n{description}\n"
        + (f'Audio transcript from the clip: "{transcription}"\n' if transcription else "")
        + "\nWrite ONE natural caption, usually 12 to 24 words, as a single sentence. "
        "Write as if you personally watched the clip. "
        "Use ONLY facts from the description and transcript. Be confident — never "
        "mention frames, video, models, or uncertainty, and never hedge (appears, "
        "seems, likely). Build any joke on the real subject, action, and setting "
        "using 'like' or 'as if'."
        f"{variety_note}\n"
        'Return ONLY a raw JSON object: {"caption": "..."}'
    )
    raw = client.fallback_chat(
        system_prompt="You write video captions in a requested style. Output raw JSON only.",
        user_text=prompt,
        max_tokens=400,
        temperature=0.3 if style == "formal" else 0.8,
    )
    return _clean_caption(extract_json_object(raw).get("caption", ""))


def _style_block(styles: List[str]) -> str:
    lines = []
    for style in styles:
        line = f"- {style}: {STYLE_RULES[style]}"
        example = STYLE_TONE_EXAMPLES.get(style)
        if example:
            line += f" Tone example (unrelated content, shows tone only): {example}"
        lines.append(line + "\n")
    return "".join(lines)


def _json_format(styles: List[str]) -> str:
    return "{" + ", ".join(f'"{style}": "..."' for style in styles) + "}"


def _generate_direct_captions(
    frame_paths: List[str],
    duration: float,
    has_audio: bool,
    transcription: Optional[str],
    styles: List[str],
    client: GemmaClient,
) -> Dict[str, str]:
    if not client.available:
        raise RuntimeError("Gemma proxy is not configured.")

    prompt = (
        "You are an expert video caption writer for a scoring benchmark. "
        "You are analyzing a short video clip using its keyframes (in order).\n"
    )
    if transcription:
        prompt += f"Audio transcript from the clip:\n\"{transcription}\"\n\n"

    prompt += (
        "Write exactly ONE caption per requested style, based ONLY on what you see in the frames (and transcript). "
        "Each caption should be a natural single English sentence, usually 12-24 words. "
        "Be SPECIFIC — mention concrete details like colors, objects, settings, and actions. "
        "Never invent details not present in the evidence.\n"
        "Never use hedging words (appears, seems, likely, possibly, probably). "
        "Never mention the medium (video, clip, frame, footage) — describe the scene directly.\n"
        f"{GROUNDING_RULE}\n"
        "STYLE RULES (each caption must sound COMPLETELY DIFFERENT from the others):\n"
        f"{_style_block(styles)}\n"
        f"CRITICAL: Return ONLY a raw JSON object with exactly {len(styles)} keys. No markdown, no explanation.\n"
        f"Format: {_json_format(styles)}"
    )

    user_text = "Analyze these frames and output the JSON captions."

    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            raw = client.vision_chat(
                system_prompt=prompt,
                user_text=user_text,
                image_paths=frame_paths,
                max_tokens=2000,
                temperature=0.55,
                max_images=8,
            )
            data = extract_json_object(raw)
            return {style: str(data.get(style, "")) for style in styles}
        except Exception as e:
            last_error = e
            print(f"  Draft generation (Vision) attempt {attempt + 1} failed: {e}")

    raise RuntimeError(f"Vision caption generation failed after retries: {last_error}")


def generate_style_candidates(
    frame_paths: List[str],
    transcription: Optional[str],
    styles: List[str],
    client: GemmaClient,
    candidates_per_style: int = 3,
) -> Dict[str, List[str]]:
    """One vision call producing several alternative captions per style, at a
    higher temperature than the draft pass. Candidates that violate the hard
    rules are dropped here so the selection pool is always compliant."""
    if not client.available or not styles or not frame_paths:
        return {}

    prompt = (
        "You are an expert video caption writer. You are analyzing a short video "
        "clip using its keyframes (in order).\n"
    )
    if transcription:
        prompt += f"Audio transcript from the clip:\n\"{transcription}\"\n\n"

    fmt = "{" + ", ".join(f'"{style}": ["...", "...", "..."]' for style in styles) + "}"
    prompt += (
        f"For EACH requested style, write {candidates_per_style} DIFFERENT candidate captions. "
        "The candidates for a style must take genuinely different comedic angles — "
        "different subjects of the joke, different comparisons — not rewordings of one idea.\n"
        "Each caption should be a natural single English sentence, usually 12-24 words, "
        "based ONLY on what you see in the frames (and transcript). "
        "Never invent details not present in the "
        "evidence. Never use hedging words (appears, seems, likely, possibly, probably). "
        "Never mention the medium (video, clip, frame, footage).\n"
        f"{GROUNDING_RULE}\n"
        "STYLE RULES:\n"
        f"{_style_block(styles)}\n"
        "CRITICAL: Return ONLY a raw JSON object mapping each style to an array of "
        f"{candidates_per_style} strings. No markdown, no explanation.\n"
        f"Format: {fmt}"
    )

    for attempt in range(2):
        try:
            raw = client.vision_chat(
                system_prompt=prompt,
                user_text="Write the JSON candidate captions now.",
                image_paths=frame_paths,
                max_tokens=3000,
                temperature=0.9,
                max_images=8,
            )
            data = extract_json_object(raw)
            pool: Dict[str, List[str]] = {}
            for style in styles:
                values = data.get(style)
                if not isinstance(values, list):
                    continue
                cleaned = [_clean_caption(v) for v in values]
                pool[style] = [
                    c for c in cleaned if c and not _find_violations(style, c)
                ]
            return pool
        except Exception as e:
            print(f"  Candidate generation attempt {attempt + 1} failed: {e}")
    return {}


def _last_resort_captions(
    frame_paths: List[str],
    transcription: Optional[str],
    styles: List[str],
    client: GemmaClient,
) -> Dict[str, str]:
    """Cheaper retry with a single middle frame before giving up on the model.
    Only falls back to canned templates if this also fails."""
    try:
        if client.available and frame_paths:
            middle_frame = frame_paths[len(frame_paths) // 2]
            prompt = (
                "You caption a short video from one representative frame"
                + (f" and this audio transcript:\n\"{transcription}\"\n" if transcription else ".\n")
                + "Write ONE caption per style, each a natural single English sentence of 12-24 words, "
                "describing only what is visible or heard. No hedging words, no mention of the medium.\n\n"
                f"{_style_block(styles)}\n"
                f"Return ONLY a raw JSON object. Format: {_json_format(styles)}"
            )
            raw = client.vision_chat(
                system_prompt=prompt,
                user_text="Write the JSON captions now.",
                image_paths=[middle_frame],
                max_tokens=1200,
                temperature=0.4,
            )
            data = extract_json_object(raw)
            captions = {style: str(data.get(style, "")) for style in styles}
            if any(_clean_caption(text) for text in captions.values()):
                return captions
    except Exception as exc:
        print(f"  Last-resort caption pass failed: {exc}")
    return fallback_captions(styles)


def enforce_caption_rules(
    captions: Dict[str, str],
    styles: List[str],
    evidence: Optional[Dict[str, Any]] = None,
    client: Optional[GemmaClient] = None,
) -> Dict[str, str]:
    """Validate each caption against the hard constraints and repair violations
    via targeted rewrite calls. Never truncates; keeps the best available
    version if repair cannot fully fix a caption."""
    final = {}
    generic = fallback_captions(styles, evidence)
    for style in styles:
        caption = _clean_caption(captions.get(style, ""))
        if not caption:
            final[style] = generic[style]
            continue

        best = caption
        best_problems = _find_violations(style, best)
        attempts = 0
        while best_problems and attempts < 2 and client is not None and client.available:
            attempts += 1
            try:
                revised = _repair_caption(style, best, best_problems, client)
            except Exception as exc:
                print(f"  Repair call failed for {style}: {exc}")
                break
            if not revised:
                break
            problems = _find_violations(style, revised)
            if _severity(revised, problems) < _severity(best, best_problems):
                best, best_problems = revised, problems

        if best_problems:
            print(f"  {style} still violates rules after repair: {best_problems}")
        final[style] = _ensure_sentence(best)
    return final


def _severity(caption: str, problems: List[str]) -> tuple:
    """Orders caption versions during repair: fewer violation types wins, and
    at equal counts, being closer to the relaxed 8-32 word window wins."""
    words = len(caption.split())
    distance = max(0, words - 32) + max(0, 8 - words)
    return (len(problems), distance)


def _find_violations(style: str, caption: str) -> List[str]:
    problems = []
    words = caption.split()
    count = len(words)
    if count < 8:
        problems.append(f"too short: {count} words (must be at least 8 words)")
    elif count > 32:
        problems.append(f"too long: {count} words (must be no more than 32 words)")

    if re.search(r"[.!?]['\")\]]*\s+\S", caption):
        problems.append("contains more than one sentence (must be exactly one sentence)")

    bag = {word.strip(_STRIP_PUNCT).lower() for word in words}

    hedges = sorted(bag & HEDGE_WORDS)
    if hedges:
        problems.append(
            f"remove hedging words ({', '.join(hedges)}) and state directly what happens"
        )

    metas = sorted(bag & META_WORDS)
    if metas:
        problems.append(
            f"do not mention the medium ({', '.join(metas)}); describe the scene itself"
        )

    if style == "formal" and "!" in caption:
        problems.append("formal caption must not contain exclamation marks")

    if style == "humorous_non_tech":
        tech = sorted(bag & TECH_WORDS)
        if tech:
            problems.append(
                f"remove technology words ({', '.join(tech)}) and use everyday humour instead"
            )

    return problems


def _repair_caption(
    style: str, caption: str, problems: List[str], client: GemmaClient
) -> str:
    system_prompt = (
        "You revise a single video caption so it satisfies hard formatting rules. "
        "Keep every factual detail from the original caption and its comedic/tonal intent; "
        "do NOT add any new detail that is not in the original.\n"
        'Return ONLY a raw JSON object: {"caption": "..."}'
    )
    user_text = (
        f"Style: {style} — {STYLE_RULES[style]}\n"
        f'Current caption: "{caption}" ({len(caption.split())} words)\n'
        f"Problems to fix: {'; '.join(problems)}\n"
        "Rewrite the caption as exactly ONE natural sentence, ideally 12-24 words and no more than 30, "
        "fixing every problem. If the original is too long, cut adjectives and secondary "
        "clauses — never add new content. Count your words before answering."
    )
    raw = client.fallback_chat(
        system_prompt=system_prompt,
        user_text=user_text,
        max_tokens=300,
        temperature=0.4,
    )
    return _clean_caption(extract_json_object(raw).get("caption", ""))


def fallback_captions(
    styles: List[str], evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    # Absolute last resort when every model call has failed. Generic, but at
    # least compliant with the hard rules (reasonable length, one sentence, no
    # hedging, no mention of the medium, no tech words in humorous_non_tech).
    templates = {
        "formal": "A central subject carries out a continuous activity in a clearly lit setting while surrounding details remain steady and secondary to the main action throughout.",
        "sarcastic": "Someone commits fully to their ongoing activity in this setting, because the day clearly demanded one more display of unshakable, completely unearned confidence from everyone involved.",
        "humorous_tech": "The main subject keeps grinding through the same routine like a cron job nobody documented, faithfully executing on schedule while the rest of the system watches.",
        "humorous_non_tech": "The main subject powers through the whole activity with the unstoppable determination of someone who skipped breakfast and refuses to admit that was a mistake.",
    }
    return {style: _ensure_sentence(templates.get(style, templates["formal"])) for style in styles}


def _clean_caption(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().strip("\"'")
    return re.sub(r"^[\-*: ]+", "", text)


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _contains_tech_word(text: str) -> bool:
    words = {word.strip(_STRIP_PUNCT).lower() for word in text.split()}
    return bool(words & TECH_WORDS)
