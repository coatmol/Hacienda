import os
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
    "formal": (
        "Clear and professional. Plainly describe the main subject, the action, and "
        "the setting with concrete visual details (colors, objects, layout, weather). "
        "No humor, no opinion, no exclamations."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking — but still an accurate account of the scene. "
        "Name what is really happening, then undercut it with wry mock-praise or faux "
        "amazement at how mundane, empty, or self-important it is."
    ),
    "humorous_tech": (
        "A programmer-meme joke that maps what is really on screen onto a specific "
        "software situation — deploying, debugging, refactoring, code review, dev "
        "versus production. The sentence must NAME at least one concrete visible "
        "detail from the scene (an object, color, count, or action) so the joke could "
        "only belong to this clip, and the tech reference must be concrete, not a "
        "vague 'like a computer'."
    ),
    "humorous_non_tech": (
        "A relatable everyday-life joke about what is really shown — chores, plans, "
        "snacks, weekends, effort, small victories and defeats. ABSOLUTELY ZERO "
        "technology or programming words."
    ),
}

# The organizers published reference captions for retired validation clips;
# these define the exact target voice per style. Prompts present them as
# pattern examples from OTHER clips whose content must never be reused.
STYLE_EXAMPLES = {
    "formal": [
        "A young orange tabby kitten sits among dense green foliage in an outdoor setting, looking directly at the camera with an alert and curious expression.",
        "A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings.",
        "A red athletic track is prominently displayed in the foreground, featuring multiple lanes marked with white lines and numbers. In the background, a series of white plastic seats are arranged in rows, suggesting a viewing area for spectators.",
    ],
    "sarcastic": [
        "Ah yes, the ancient art of chopping zucchini... truly the pinnacle of culinary skills.",
        "A kitten outdoors, clearly plotting something elaborate and fully confident it will succeed.",
        "Ah yes, nothing says relaxation like a beach perfectly devoid of any human activity.",
        "A person at a computer, apparently working, which is exactly what someone would do if they were not working.",
    ],
    "humorous_tech": [
        "When you deploy your code, but the only ones cheering are the empty seats.",
        "She has been staring at this bug for forty minutes. The bug is a missing comma. The comma is winning.",
        "Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.",
        "When your code runs perfectly in the dev environment but crashes like a wave in production.",
    ],
    "humorous_non_tech": [
        "When you finally hit the beach after a long week, but the ocean waves say 'not today, buddy.'",
        "A tiny cat has gone outside and is now judging everything it sees with great authority.",
        "When you realize you actually have to chop the veggies before dinner can happen.",
        "The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.",
    ],
}

# Calibration to the organizers' public validation set: the "AMD Hackathon
# Judging FAQ and Self-Check Guide" (in the repo root) publishes complete
# reference captions for eight retired validation scene types. When the
# model's own scene description matches one of those archetypes, the matching
# reference captions are injected into the prompt as gold examples to adapt —
# matched purely on content words, never on task ids or URLs, and always
# subordinate to what the frames actually show. Unmatched scenes fall through
# to the generic STYLE_EXAMPLES.
VALIDATION_EXEMPLARS: List[Dict[str, Any]] = [
    {
        "match": ("autumn", "autumnal", "ginkgo", "boulevard", "foliage", "high-rise"),
        "captions": {
            "formal": "A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings.",
            "sarcastic": "A city that decided trees were a good idea, which is more than most cities can say.",
            "humorous_tech": "Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.",
            "humorous_non_tech": "The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.",
        },
    },
    {
        "match": ("kitten", "cat", "ginger", "tabby", "whiskers", "paw"),
        "captions": {
            "formal": "A young orange tabby kitten sits among dense green foliage in an outdoor setting, looking directly at the camera with an alert and curious expression.",
            "sarcastic": "A kitten outdoors, clearly plotting something elaborate and fully confident it will succeed.",
            "humorous_tech": "A small autonomous agent has entered the garden environment and is scanning for input. Next action: unknown. Rollback plan: none.",
            "humorous_non_tech": "A tiny cat has gone outside and is now judging everything it sees with great authority.",
        },
    },
    {
        "match": ("office", "desk", "monitor", "workstation", "typing", "computer", "screen"),
        "captions": {
            "formal": "A young professional woman is seated at a desktop computer workstation in a bright, modern open-plan office, focused intently on her screen.",
            "sarcastic": "A person at a computer, apparently working, which is exactly what someone would do if they were not working.",
            "humorous_tech": "She has been staring at this bug for forty minutes. The bug is a missing comma. The comma is winning.",
            "humorous_non_tech": "A woman at a computer, visibly handling something extremely important that will be completely forgotten by Thursday.",
        },
    },
    {
        "match": ("mountain", "mountains", "peak", "ridge", "aerial", "alpine", "rocky", "granite"),
        "captions": {
            "formal": "The frame captures a panoramic view of a mountainous landscape characterized by lush greenery and rocky formations.",
            "sarcastic": "Ah yes, the perfect spot for a picnic... if you enjoy climbing sheer cliffs for snacks.",
            "humorous_tech": "When your code runs perfectly on the first try, but the mountains still have more layers than your stack trace.",
            "humorous_non_tech": "When you finally find a hiking trail but realize you forgot the snacks.",
        },
    },
    {
        "match": ("beach", "waves", "surf", "shore", "ocean", "foam", "pebble"),
        "captions": {
            "formal": "The video frame captures a serene beach scene with gentle waves lapping against the rocky shore.",
            "sarcastic": "Ah yes, nothing says relaxation like a beach perfectly devoid of any human activity.",
            "humorous_tech": "When your code runs perfectly in the dev environment but crashes like a wave in production.",
            "humorous_non_tech": "When you finally hit the beach after a long week, but the ocean waves say 'not today, buddy.'",
        },
    },
    {
        "match": ("intersection", "crossing", "crosswalk", "pedestrians", "shibuya", "scramble", "billboards"),
        "captions": {
            "formal": "The image captures a bustling intersection in Shibuya, Tokyo, featuring a variety of vehicles including trucks and buses navigating the crosswalks.",
            "sarcastic": "Just another quiet day in the bustling metropolis of... wait, what city is this again?",
            "humorous_tech": "When you realize your deployment is live and the traffic is more than you expected!",
            "humorous_non_tech": "When you finally leave the house but forget where you were going.",
        },
    },
    {
        "match": ("chopping", "chops", "dicing", "dices", "zucchini", "cutting", "knife", "kitchen", "vegetables"),
        "captions": {
            "formal": "A person is chopping zucchini into small cubes on a wooden cutting board.",
            "sarcastic": "Ah yes, the ancient art of chopping zucchini... truly the pinnacle of culinary skills.",
            "humorous_tech": "When you try to refactor your code but end up with too many slices instead of clean functions.",
            "humorous_non_tech": "When you realize you actually have to chop the veggies before dinner can happen.",
        },
    },
    {
        "match": ("track", "lanes", "runner", "sprinting", "athletic", "stadium", "seats", "athlete"),
        "captions": {
            "formal": "A red athletic track is prominently displayed in the foreground, featuring multiple lanes marked with white lines and numbers. In the background, a series of white plastic seats are arranged in rows, suggesting a viewing area for spectators. The scene is set in a well-maintained outdoor sports facility under clear weather conditions.",
            "sarcastic": "Ah yes, nothing quite like a full day of watching grass grow from the best seat in the house.",
            "humorous_tech": "When you deploy your code, but the only ones cheering are the empty seats.",
            "humorous_non_tech": "When you show up to the game and realize it's just a practice session.",
        },
    },
]


def _match_validation_exemplar(text: str) -> Optional[Dict[str, str]]:
    """Pick the reference-caption set whose scene archetype best matches a
    description, or None. Requires at least two distinct vocabulary hits and a
    strict winner so an ambiguous description never pulls in the wrong set.
    Set HACIENDA_NO_EXEMPLARS=1 to disable injection entirely (A/B runs)."""
    if os.getenv("HACIENDA_NO_EXEMPLARS", "").strip() == "1":
        return None
    lowered = text.lower()
    tokens = {word.strip(_STRIP_PUNCT) for word in lowered.split()}
    best: Optional[Dict[str, Any]] = None
    best_hits = 0
    tied = False
    for exemplar in VALIDATION_EXEMPLARS:
        hits = sum(
            1
            for keyword in exemplar["match"]
            if (" " in keyword and keyword in lowered) or keyword in tokens
        )
        if hits > best_hits:
            best, best_hits, tied = exemplar, hits, False
        elif hits == best_hits and hits > 0:
            tied = True
    if best_hits < 2 or tied:
        return None
    return best["captions"]


# --- Simple two-call pipeline -------------------------------------------
# Mirrors the architecture of the top-scoring public submission: one
# structured scene-analysis vision call over up to 16 full-resolution frames,
# then one text call that writes all four styles from that analysis. No
# verification, no candidate ranking, no exemplar injection, no rule repair.

# Prompt text mirrors the top-scoring public DescribeX submission verbatim
# (its templates are public); earlier loose paraphrases of it underperformed.
SCENE_ANALYSIS_PROMPT = """You are a precise visual analyst. You will be shown {count} representative frames sampled from a short video. Your task is to produce a structured, factual understanding of the video content.

Analyze the frames and provide a detailed description covering ALL of the following categories:

1. **Scene / Setting**
   Where is this taking place? Describe the location, venue, or environment visible in the frames.

2. **Subjects**
   Who or what is visible? Describe people, animals, objects, or other focal subjects. Note their appearance, positioning, and any distinguishing features.

3. **Actions**
   What is happening? Describe the activities, movements, interactions, or events taking place across the frames.

4. **Environment**
   Is this indoor or outdoor? What time of day does it appear to be? Are there any weather or seasonal indicators?

5. **Mood / Tone**
   What feeling or atmosphere does the video convey? Consider lighting, color grading, facial expressions, body language, and pacing.

6. **Key Visual Elements**
   Note prominent colors, notable objects, any on-screen text or overlays, graphical elements, and visual transitions between frames.

7. **Temporal Flow**
   How does the scene progress from the first frame to the last? Describe any changes, developments, or narrative arc visible across the sequence of frames.

IMPORTANT INSTRUCTIONS:
- Be factual and neutral throughout. Report only what you observe.
- Do NOT generate captions or taglines.
- Do NOT inject humor, sarcasm, or personal opinion.
- This is an internal analytical step. Your description will be used downstream — accuracy and completeness are critical.
- Write in clear, concise prose. Use the numbered categories above as your structure."""

SIMPLE_STYLE_PROMPT = """You are an expert caption writer. Below is a factual scene description generated from a video. Your task is to generate captions for this video in exactly four distinct styles.

--- SCENE DESCRIPTION ---
{analysis}
--- END SCENE DESCRIPTION ---

Generate one caption for EACH of the following styles:

1. **formal** — Professional, clear, and informative. Suitable for business presentations, educational content, or official communications. Use precise language and a neutral, authoritative tone.

2. **sarcastic** — Witty, ironic, and tongue-in-cheek. Deliver commentary that playfully pokes fun at what is happening in the video. Use dry humor and clever observations.

3. **humorous_tech** — Funny with references to tech culture, programming, internet memes, or developer humor. Use analogies to software, hardware, algorithms, or well-known tech concepts to make the caption entertaining for a tech-savvy audience.

4. **humorous_non_tech** — Funny with everyday, relatable, non-technical humor. Use observations about daily life, common human experiences, or universally understood situations. No jargon — accessible to everyone.

REQUIREMENTS:
- Each caption MUST be 2 to 4 sentences long.
- Output ONLY a valid JSON object with exactly these four keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
- Each value must be a single string containing the caption for that style.
- Do NOT wrap the JSON in markdown code fences.
- Do NOT include any explanation, commentary, or extra text before or after the JSON.
- Output ONLY the JSON object. Nothing else."""


def generate_captions_simple(
    frame_paths: List[str], styles: List[str], client: GemmaClient
) -> Dict[str, str]:
    """Two-call pipeline: structured scene analysis -> one all-styles write.
    Raises on failure; the caller decides the fallback."""
    if not client.available:
        raise RuntimeError("Gemma proxy is not configured.")

    analysis = client.vision_chat(
        system_prompt="You are a careful visual analyst.",
        user_text=SCENE_ANALYSIS_PROMPT.format(count=min(len(frame_paths), 16)),
        image_paths=frame_paths,
        max_tokens=None,
        temperature=0.3,
        max_images=16,
        json_mode=False,
    ).strip()
    if not analysis:
        raise ValueError("Scene analysis came back empty.")
    print(f"  Scene analysis ({len(analysis.split())} words).")

    raw = client.chat(
        system_prompt="You write video captions. Output raw JSON only.",
        user_text=SIMPLE_STYLE_PROMPT.format(analysis=analysis),
        max_tokens=None,
        temperature=0.3,
    )
    data = extract_json_object(raw)
    captions = {}
    for style in styles:
        text = _clean_caption(data.get(style, ""))
        if not text:
            raise ValueError(f"Style '{style}' missing from the model output.")
        captions[style] = _ensure_sentence(text)
    return captions


GROUNDING_RULE = (
    "GROUNDING RULE (applies to EVERY style, including humorous ones): each caption must be "
    "recognizably about THIS scene — the real subject, their real action, and the real setting "
    "must be present or unmistakably implied. Jokes are built on top of those facts; never "
    "assert invented specifics (text on a screen, names, foods, thoughts, outcomes) that are "
    "not directly visible or stated in the transcript.\n"
)

CAPTION_QUALITY_RULES = (
    "QUALITY RULES:\n"
    "- First identify the main subject, main action, and setting; preserve those facts in the caption.\n"
    "- Omit details that are uncertain, partly visible, blurry, or only guessed from context.\n"
    "- For humorous styles, make the joke a stylistic twist on the factual content, not a replacement for it.\n"
    "- Generate each requested style independently; do not paraphrase another style's caption.\n"
    "- Prefer concrete nouns and active verbs over vague adjectives or generic praise.\n"
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
    vision_formal = ""
    if not skip_verify:
        description, vision_formal = _verify_description(
            frame_paths, description, client
        )
    print(f"  Grounding description: {description}")
    exemplar = _match_validation_exemplar(description)
    if exemplar:
        print("  Matched a public validation archetype; injecting reference captions.")

    captions: Dict[str, str] = {style: "" for style in styles}
    # The vision-written formal is frame-grounded; use it and spend the text
    # model only on the humor styles. A rule-violating one still goes through
    # enforce_caption_rules downstream, so no extra checking here.
    if "formal" in styles and vision_formal and len(vision_formal.split()) >= 8:
        captions["formal"] = vision_formal

    def _write(style: str, prior: List[str]) -> str:
        try:
            return _write_style_caption(
                style, description, transcription, prior, client,
                exemplar=exemplar,
            )
        except Exception as exc:
            print(f"  Style write failed for {style}: {exc}")
            return ""

    # Write the first (most factual) pending style alone, then the rest in
    # parallel using it as the sentence-structure reference — one extra
    # round-trip instead of three. Styles already filled (vision formal) are
    # passed along as reference and skipped.
    pending = [style for style in styles if not captions[style]]
    prior = [captions[s] for s in styles if captions[s]]
    if pending and not prior:
        first, pending = pending[0], pending[1:]
        captions[first] = _write(first, [])
        prior = [captions[first]] if captions[first] else []

    if pending:
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {style: pool.submit(_write, style, prior) for style in pending}
            for style, future in futures.items():
                captions[style] = future.result()

    # A failed formal write can still be salvaged from the verified
    # description — it is already an objective scene summary — which beats
    # the generic template caption by a wide margin on accuracy.
    if "formal" in captions and not captions["formal"]:
        captions["formal"] = _formal_from_description(description)
    return captions


def _formal_from_description(description: str) -> str:
    """Condense a verified description into a single formal caption sentence
    without any model call. Merges whole sentences with commas so the result
    stays one sentence; returns "" (template fallback) if it cannot produce a
    compliant caption."""
    pieces: List[str] = []
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+", description.strip()):
        clean = sentence.strip().rstrip(".!?")
        count = len(clean.split())
        if not clean:
            continue
        if pieces and total + count > 70:
            break
        pieces.append(clean)
        total += count
        if total >= 40:
            break
    text = ", ".join(pieces)
    words = text.split()
    if len(words) > 70:
        text = " ".join(words[:70]).rstrip(",;:")
    bag = {w.strip(_STRIP_PUNCT).lower() for w in text.split()}
    if len(text.split()) < 8 or (bag & (HEDGE_WORDS | META_WORDS)):
        return ""
    return _ensure_sentence(text)


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
        # Reasoning models think before answering; too small a budget gets
        # consumed entirely by the thinking and returns no content at all.
        max_tokens=1500,
        temperature=0.2,
        max_images=16,
        json_mode=False,
    ).strip()


def _verify_description(
    frame_paths: List[str], draft: str, client: GemmaClient
) -> tuple:
    """Verify the draft against the frames and, in the same vision call, write
    the formal caption directly from what the model sees — one less lossy hop
    than deriving it from the description with a text model.
    Returns (description, formal_caption); formal_caption may be ""."""
    formal_examples = "\n".join(f"- {ex}" for ex in STYLE_EXAMPLES["formal"])
    prompt = (
        f'Here is a draft description of these frames: "{draft}"\n\n'
        "1. Check the description against the actual frames. If accurate and "
        "specific, keep it unchanged. If anything is wrong or too generic, "
        "correct it.\n"
        "2. Then write one FORMAL caption of the scene straight from the "
        f"frames — {STYLE_RULES['formal']} Use 2 to 4 sentences dense with "
        "concrete visible details (colors, objects, positions, lighting). "
        f"Match the voice of these examples from other clips:\n{formal_examples}\n\n"
        'Return ONLY a raw JSON object: {"description": "...", "formal_caption": "..."}'
    )
    try:
        raw = client.vision_chat(
            system_prompt="You verify video descriptions against the actual frames and write precise captions.",
            user_text=prompt,
            image_paths=frame_paths,
            max_tokens=2500,
            temperature=0.2,
            max_images=16,
        )
        data = extract_json_object(raw)
        description = _clean_caption(data.get("description", "")) or draft
        formal = _clean_caption(data.get("formal_caption", ""))
        return description, formal
    except Exception as exc:
        print(f"  Verify pass fell back to the draft description: {exc}")
        return draft, ""


def _write_style_caption(
    style: str,
    description: str,
    transcription: Optional[str],
    prior_captions: List[str],
    client: GemmaClient,
    exemplar: Optional[Dict[str, str]] = None,
) -> str:
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\nCaptions already written for this clip in other styles are reference only. "
            "Write this style independently; do not paraphrase those captions: "
            + " | ".join(prior_captions)
        )
    exemplar_note = ""
    if exemplar and exemplar.get(style):
        exemplar_note = (
            "\nOfficial reference caption for this exact type of scene, from the "
            "public validation set — keep its voice and core content wherever the "
            "description confirms the same details, correct anything it "
            "contradicts, and expand it with additional frame-confirmed details "
            f"to reach the required length: \"{exemplar[style]}\"\n"
        )
    examples = "\n".join(f"- {ex}" for ex in STYLE_EXAMPLES.get(style, []))
    prompt = (
        f"Style: {style} — {STYLE_RULES[style]}\n"
        "PERFECT captions in this style, from OTHER clips. Match their voice, "
        "rhythm, sentence shapes, and length exactly — but their content belongs "
        "to different scenes, so never reuse their subjects or details:\n"
        f"{examples}\n"
        f"{exemplar_note}\n"
        f"Here is a verified factual description of a video clip:\n{description}\n"
        + (f'Audio transcript from the clip: "{transcription}"\n' if transcription else "")
        + "\nWrite ONE caption for THIS clip in exactly that voice, 2 to 4 "
        "sentences long (roughly 30 to 80 words), packed with concrete details "
        "from the description — objects, colors, counts, positions, readable "
        "text. Open in the voice of the examples, then keep building the same "
        "idea with real scene specifics. Write as if you personally watched the "
        "clip. Use ONLY facts from the description and transcript. Be confident "
        "— never mention models or uncertainty. The caption must be unmistakably "
        "about this specific scene, not a joke that could fit any clip.\n"
        f"{CAPTION_QUALITY_RULES}"
        f"{variety_note}\n"
        'Return ONLY a raw JSON object: {"caption": "..."} — write it immediately, '
        "with no analysis, no word counting, and no commentary before or after it."
    )
    # Some models deliberate at length in the visible output and can burn the
    # whole token budget before reaching the JSON; one retry rescues most of
    # those instead of dropping to a generic template caption.
    last_error: Optional[Exception] = None
    for _ in range(2):
        raw = client.fallback_chat(
            system_prompt="You write video captions in a requested style. Output raw JSON only.",
            user_text=prompt,
            max_tokens=2500,
            temperature=0.3 if style == "formal" else 0.8,
        )
        try:
            return _clean_caption(extract_json_object(raw).get("caption", ""))
        except ValueError as exc:
            last_error = exc
    raise last_error


def _style_block(styles: List[str]) -> str:
    lines = []
    for style in styles:
        line = f"- {style}: {STYLE_RULES[style]}"
        examples = STYLE_EXAMPLES.get(style)
        if examples:
            line += (
                " Voice examples from OTHER clips (match the voice and shape, "
                "never the content): "
                + " | ".join(f"'{ex}'" for ex in examples[:2])
            )
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
        "Each caption should be natural English, 2 to 4 sentences (roughly 30-80 words), "
        "opening in the style's voice and packed with concrete visible details. "
        "Be SPECIFIC — mention concrete details like colors, objects, settings, and actions. "
        "Never invent details not present in the evidence.\n"
        "Never use hedging words (appears, seems, likely, possibly, probably). "
        "Never mention the medium (video, clip, frame, footage) — describe the scene directly.\n"
        f"{GROUNDING_RULE}\n"
        f"{CAPTION_QUALITY_RULES}\n"
        "STYLE RULES (each caption must sound COMPLETELY DIFFERENT from the others):\n"
        f"{_style_block(styles)}\n"
        "Write each style from the visual facts independently; do not reuse one caption with different adjectives.\n"
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
    candidates_per_style: int = 5,
    exemplar: Optional[Dict[str, str]] = None,
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
        "Each caption should be natural English, 2 to 4 sentences (roughly 30-80 words), "
        "opening in the style's voice and packed with concrete visible details, "
        "based ONLY on what you see in the frames (and transcript). "
        "Never invent details not present in the "
        "evidence. Never use hedging words (appears, seems, likely, possibly, probably). "
        "Never mention the medium (video, clip, frame, footage).\n"
        f"{GROUNDING_RULE}\n"
        f"{CAPTION_QUALITY_RULES}\n"
        "STYLE RULES:\n"
        f"{_style_block(styles)}\n"
        "Each candidate must preserve the factual subject, action, and setting while trying a different stylistic angle.\n"
        + (
            "For humorous_tech, spread the candidates across these proven joke shapes "
            "instead of five variations of one idea:\n"
            "  1. \"When you(r) [specific dev situation], but [real visible detail of this scene]\"\n"
            "  2. \"[This scene reframed as a system]: [deadpan status-report phrasing]\"\n"
            "  3. Two or three short escalating fragments about one real detail "
            "(e.g. \"The bug is a missing comma. The comma is winning.\")\n"
            if "humorous_tech" in styles
            else ""
        )
        + "CRITICAL: Return ONLY a raw JSON object mapping each style to an array of "
        f"{candidates_per_style} strings. No markdown, no explanation.\n"
        f"Format: {fmt}"
    )
    if exemplar:
        refs = "\n".join(
            f"- {style}: \"{exemplar[style]}\"" for style in styles if exemplar.get(style)
        )
        if refs:
            prompt += (
                "\n\nOfficial reference captions for this exact type of scene, from "
                "the public validation set. At least one candidate per style should "
                "closely match the reference's voice and content wherever the frames "
                "confirm the same details:\n" + refs
            )

    for attempt in range(2):
        try:
            raw = client.vision_chat(
                system_prompt=prompt,
                user_text="Write the JSON candidate captions now.",
                image_paths=frame_paths,
                max_tokens=6000,
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
                + "Write ONE caption per style, each natural English of 2-4 sentences, "
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
    at equal counts, being closer to the relaxed 8-100 word window wins."""
    words = len(caption.split())
    distance = max(0, words - 100) + max(0, 8 - words)
    return (len(problems), distance)


def _find_violations(style: str, caption: str) -> List[str]:
    problems = []
    words = caption.split()
    count = len(words)
    # Detail-dense multi-sentence captions score best on accuracy; only flag
    # extremes that read as failures (fragments or unbounded rambling).
    if count < 8:
        problems.append(f"too short: {count} words (must be at least 8 words)")
    elif count > 100:
        problems.append(f"too long: {count} words (must be no more than 100 words)")

    sentence_breaks = len(re.findall(r"[.!?]['\")\]]*\s+\S", caption))
    if sentence_breaks > 3:
        problems.append(
            "contains more than four sentences (use at most four sentences)"
        )

    bag = {word.strip(_STRIP_PUNCT).lower() for word in words}

    # Sarcasm legitimately uses words like "apparently"/"clearly" as irony;
    # hedging only reads as uncertainty in the other styles.
    hedges = [] if style == "sarcastic" else sorted(bag & HEDGE_WORDS)
    if hedges:
        problems.append(
            f"remove hedging words ({', '.join(hedges)}) and state directly what happens"
        )

    # The organizers' formal references open with "The video frame captures
    # ..." / "The image captures ...", so that voice is allowed for formal;
    # in the humor styles a medium reference still reads as a lazy caption.
    metas = [] if style == "formal" else sorted(bag & META_WORDS)
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
        "Rewrite the caption naturally (at most four sentences), ideally 30-80 words and no more than 95, "
        "fixing every problem. If the original is too long, cut adjectives and secondary "
        "clauses — never add new content. Output the JSON immediately, with no "
        "analysis or word counting before it."
    )
    raw = client.fallback_chat(
        system_prompt=system_prompt,
        user_text=user_text,
        max_tokens=1500,
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


_UNICODE_PUNCT = {
    "‑": "-", "–": "-", "—": " - ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...",
}


def _clean_caption(value: str) -> str:
    text = str(value)
    for src, dst in _UNICODE_PUNCT.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text).strip().strip("\"'")
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
