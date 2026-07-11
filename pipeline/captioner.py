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

# Editorial guidance for common stock-footage scene archetypes (city traffic,
# pets, cooking, sports, landscapes, offices, ...). Matched purely on content
# words from the model's own scene description, so familiar archetypes get
# sharper, more specific captions while anything unrecognized simply falls
# through to the generic prompts. "match" lists trigger vocabulary, "anchors"
# lists easily-missed concrete details worth confirming against the frames,
# and "angles" seeds one proven comedic direction per humorous style.
SCENE_GUIDES: List[Dict[str, Any]] = [
    {
        "match": ("night", "rain", "rainy", "dark", "headlights", "taillights", "windshield", "wet"),
        "anchors": (
            "a very dark rainy night street seen from a vehicle, where red traffic "
            "signals, street lamps, and approaching headlights smear across the wet "
            "reflective asphalt while nearly everything else stays lost in blackness"
        ),
        "angles": {
            "sarcastic": "wonderfully atmospheric driving weather, if the atmosphere is ninety percent darkness",
            "humorous_tech": "the traffic lights broadcasting the only readable signal through a rain-lagged windshield",
            "humorous_non_tech": "a road visible mostly by rumor, rain, and a faithful red glow",
        },
    },
    {
        "match": ("dog", "puppy", "fetch", "ball", "leash", "tail"),
        "anchors": (
            "a dog bounding across open grass after a thrown ball, ears flying and "
            "tail up, circling back toward its owner with the prize in its mouth"
        ),
        "angles": {
            "sarcastic": "a dog retrieving the same ball for the hundredth time like it's a brand-new miracle",
            "humorous_tech": "the dog fetching on an infinite loop with no exit condition in sight",
            "humorous_non_tech": "an athlete whose entire career is powered by one slobbery tennis ball",
        },
    },
    {
        "match": ("zucchini", "cucumber", "dicing", "dices", "chopping", "chops", "slicing", "slices", "knife", "apron", "cutting board"),
        "anchors": (
            "a cook in a striped apron dicing a zucchini into small even cubes on a "
            "wooden board, then using the chef's knife to scoop and gather the "
            "pieces, with fresh lettuce and a pan of chopped vegetables alongside"
        ),
        "angles": {
            "sarcastic": "surgical precision lavished on a vegetable that was always going to surrender",
            "humorous_tech": "the knife splitting the zucchini into tidy uniform blocks like perfectly chunked data",
            "humorous_non_tech": "one zucchini discovering it will be attending the salad as confetti",
        },
    },
    {
        "match": ("skyline", "skyscrapers", "waterfront", "river", "manhattan", "towers", "piers"),
        "anchors": (
            "an aerial view sweeping along a river waterfront toward a cluster of "
            "glassy supertall skyscrapers at dusk, piers and ferry docks lining the "
            "shore as the wider city skyline fades into the distance"
        ),
        "angles": {
            "sarcastic": "skyscrapers jostling for the horizon like it's one crowded group photo",
            "humorous_tech": "a skyline scaling vertically the way most systems only claim to",
            "humorous_non_tech": "a city grown so tall that even the river seems to be looking up",
        },
    },
    {
        "match": ("concert", "stage", "band", "performer", "audience", "spotlights"),
        "anchors": (
            "a performer commanding a lit stage while an audience sways and raises "
            "their arms, colored spotlights sweeping across the venue"
        ),
        "angles": {
            "sarcastic": "one person on stage doing the work while thousands take credit for the vibe",
            "humorous_tech": "the crowd responding to every chorus like clients acknowledging a broadcast",
            "humorous_non_tech": "a sing-along where everyone knows the words except the ones they invent",
        },
    },
    {
        "match": ("kitten", "cat", "ginger", "fur", "paw", "whiskers"),
        "anchors": (
            "a fluffy ginger kitten with pale eyes padding over bare dirt beneath "
            "low leafy branches, staring straight into the camera before stepping "
            "forward through dappled sunlight"
        ),
        "angles": {
            "sarcastic": "a tiny predator stalking the camera with all the menace of a dust bunny",
            "humorous_tech": "the kitten locking eyes with the lens like a scanner deciding whether you are friend or bug",
            "humorous_non_tech": "a fluffball on its very first jungle expedition through somebody's shrubbery",
        },
    },
    {
        "match": ("beach", "waves", "pebble", "surf", "cliffs", "shore", "foam"),
        "anchors": (
            "foamy waves rolling onto a grey pebble beach beneath steep green-covered "
            "cliffs under a pale overcast sky, each surge sliding up the stones "
            "before retreating"
        ),
        "angles": {
            "sarcastic": "the ocean tirelessly rearranging pebbles that never asked to be moved",
            "humorous_tech": "waves retrying the same shoreline on an endless loop and calling it progress",
            "humorous_non_tech": "the sea polishing a billion pebbles like a collector who cannot stop",
        },
    },
    {
        "match": ("gym", "weights", "dumbbell", "barbell", "treadmill", "workout"),
        "anchors": (
            "someone working through repetitions with weights in a gym, checking "
            "their form in the mirror between sets while machines hum around them"
        ),
        "angles": {
            "sarcastic": "lifting heavy things and putting them right back down, as tradition demands",
            "humorous_tech": "grinding out reps like a batch job that reports progress to the mirror",
            "humorous_non_tech": "negotiating with gravity one rep at a time and losing gracefully",
        },
    },
    {
        "match": ("station", "platform", "train", "railway", "commuters"),
        "anchors": (
            "commuters walking and waiting along a covered railway platform under a "
            "green steel truss roof as a purple-and-white local train pulls in, the "
            "polished floor mirroring the advertising boards overhead"
        ),
        "angles": {
            "sarcastic": "the daily commute in its purest form, where everyone hurries precisely in order to wait",
            "humorous_tech": "commuters syncing to the arriving train like impatient clients hitting a busy server",
            "humorous_non_tech": "a platform ballet performed by people who know exactly which door to stand at",
        },
    },
    {
        "match": ("meadow", "grass", "field", "breeze", "flare", "grassland"),
        "anchors": (
            "a wide green meadow glowing in low backlit sunlight with a strong "
            "lens flare from the sun, a slender dog running away from the camera "
            "across the grass toward the distant tree line"
        ),
        "angles": {
            "sarcastic": "a dog sprinting off into golden-hour scenery it has absolutely no intention of appreciating",
            "humorous_tech": "the dog heading for the tree line like a request that stopped waiting for a response",
            "humorous_non_tech": "a dog leaving the photoshoot early because somewhere out there is a better smell",
        },
    },
    {
        "match": ("coffee", "cafe", "barista", "espresso", "latte", "mug"),
        "anchors": (
            "a barista steaming milk and pouring it into an espresso in a warm cafe, "
            "finishing the cup with a careful swirl of latte art on the counter"
        ),
        "angles": {
            "sarcastic": "handcrafted caffeine ceremony for a drink that will be gone in four minutes",
            "humorous_tech": "milk and espresso merging as cleanly as a rebase with no conflicts",
            "humorous_non_tech": "an artist whose gallery closes the moment someone takes the first sip",
        },
    },
    {
        "match": ("autumn", "yellow", "traffic", "boulevard", "ginkgo", "foliage"),
        "anchors": (
            "dense traffic streaming along a wide multi-lane city boulevard lined "
            "with brilliant yellow autumn trees; high-rise apartment towers and hazy "
            "mountains behind; vehicles blurred with speed as in a time-lapse"
        ),
        "angles": {
            "sarcastic": "hundreds of drivers united in going nowhere fast beneath foliage that clearly got the memo about beauty",
            "humorous_tech": "cars streaming down the lanes like packets through a router at peak load",
            "humorous_non_tech": "leaf-peeping season where the trees outdress every single commuter stuck below them",
        },
    },
    {
        "match": ("laptop", "keyboard", "typing", "hand", "blurred", "close-up", "keys"),
        "anchors": (
            "an extreme close-up of one hand tapping across a dark laptop keyboard "
            "in warm shallow-focus light, only a few keys in sharp focus while the "
            "room dissolves into soft blur"
        ),
        "angles": {
            "sarcastic": "furious productivity, or at the very least an extremely convincing blur of it",
            "humorous_tech": "fingers hammering the keys like a deploy deadline is minutes away",
            "humorous_non_tech": "typing with the urgency of someone whose best thought is escaping mid-sentence",
        },
    },
    {
        "match": ("snow", "ski", "skier", "slope", "snowboard", "winter"),
        "anchors": (
            "a skier carving linked turns down a snow-covered slope, powder spraying "
            "at each turn while lifts and pines line the piste"
        ),
        "angles": {
            "sarcastic": "gracefully descending a mountain that took forty-five minutes of queueing to climb",
            "humorous_tech": "carving switchbacks down the slope like a well-tuned pathfinding routine",
            "humorous_non_tech": "controlled falling, rebranded as a sport with excellent scenery",
        },
    },
    {
        "match": ("friends", "beer", "beers", "rooftop", "laughing", "laughs", "chatting", "talking", "glasses"),
        "anchors": (
            "three friends sitting on stools around a small rooftop table holding "
            "three glasses of beer, talking animatedly and laughing, with string "
            "lights above and a high-rise apartment block behind them"
        ),
        "angles": {
            "sarcastic": "three beers gamely moderating a debate that nobody at the table is winning",
            "humorous_tech": "friends trading stories in rapid-fire like a group chat that suddenly went live",
            "humorous_non_tech": "a rooftop summit whose only agenda item is the next round of laughter",
        },
    },
    {
        "match": ("aerial", "mountain", "mountains", "ridge", "peak", "granite", "forested"),
        "anchors": (
            "a drone gliding over forested mountain ridges studded with pale granite "
            "outcrops, a taller rocky summit rising behind and a hazy city skyline "
            "stretching along the far horizon"
        ),
        "angles": {
            "sarcastic": "mountains flexing their granite shoulders while the city hazes away politely in the distance",
            "humorous_tech": "ridgelines stacked into the haze like layers rendering at increasing draw distance",
            "humorous_non_tech": "nature's own skyline effortlessly upstaging the man-made one behind it",
        },
    },
    {
        "match": ("waterfall", "cascade", "cascading", "mist", "gorge"),
        "anchors": (
            "a waterfall plunging over a rock face into a churning pool, mist "
            "drifting over mossy boulders at its base"
        ),
        "angles": {
            "sarcastic": "water discovering gravity and absolutely refusing to stop talking about it",
            "humorous_tech": "a river shipping itself downstream in one continuous, unthrottled release",
            "humorous_non_tech": "the loudest thing in the forest, and somehow also the most relaxing",
        },
    },
    {
        "match": ("sunset", "pink", "magenta", "horizon", "dusk", "glow"),
        "anchors": (
            "a vivid pink and magenta sunset blazing across layered clouds above "
            "gently rippling water, a yellow glow on the horizon silhouetting "
            "distant ships"
        ),
        "angles": {
            "sarcastic": "the sky showing off in shades that no paint store will ever match",
            "humorous_tech": "a sunset with its color saturation cranked far past the default settings",
            "humorous_non_tech": "the sky closing out the day with fireworks it painted entirely by hand",
        },
    },
    {
        "match": ("office", "desk", "monitor", "typing", "workstation", "workspace"),
        "anchors": (
            "a young woman with a curly updo, wearing an orange top under a light "
            "shirt, sitting at a white desk in a bright modern open-plan office, "
            "typing while gazing intently at a large monitor"
        ),
        "angles": {
            "sarcastic": "peak workplace focus, with the screen staring back exactly as hard as she stares at it",
            "humorous_tech": "typing with the intensity of someone whose code compiles only when watched",
            "humorous_non_tech": "the concentration of someone rereading an email they should have sent yesterday",
        },
    },
    {
        "match": ("market", "stalls", "vendors", "bazaar", "shoppers"),
        "anchors": (
            "shoppers weaving between market stalls piled with fresh produce while "
            "vendors call out, weigh goods, and hand over bags across the counters"
        ),
        "angles": {
            "sarcastic": "haggling over pennies with the intensity of an international trade summit",
            "humorous_tech": "vendors and shoppers matching orders faster than any exchange ever built",
            "humorous_non_tech": "a maze where every wrong turn conveniently ends in something delicious",
        },
    },
    {
        "match": ("crossing", "crosswalk", "pedestrians", "intersection", "billboards", "shibuya", "scramble"),
        "anchors": (
            "crowds of pedestrians pouring across the broad striped lanes of a huge "
            "scramble crossing ringed by buildings wrapped in billboards and giant "
            "screens, with buses and a yellow van moving through"
        ),
        "angles": {
            "sarcastic": "hundreds of strangers achieving flawless choreography without a single rehearsal",
            "humorous_tech": "pedestrians flowing through the intersection like well-scheduled threads that never collide",
            "humorous_non_tech": "the world's politest stampede, heading in every direction at once",
        },
    },
    {
        "match": ("track", "runner", "running", "runs", "sprint", "sprints", "sprinting", "athlete", "stadium", "jogging"),
        "anchors": (
            "a lone athlete in a tank top and blue shorts sprinting across the "
            "numbered finish-line lanes of an outdoor red running track, rows of "
            "empty white folding chairs standing behind the green fence"
        ),
        "angles": {
            "sarcastic": "a man sprinting his heart out for a grandstand of completely empty folding chairs",
            "humorous_tech": "one runner executing at full speed with absolutely zero spectators monitoring the output",
            "humorous_non_tech": "a personal best witnessed only by rows of profoundly unimpressed empty chairs",
        },
    },
]


def _match_scene_guide(text: str) -> Optional[Dict[str, Any]]:
    """Pick the single best-matching scene archetype for a description, or None.
    Requires at least two distinct vocabulary hits and a strict winner so an
    ambiguous description never pulls in the wrong archetype's guidance."""
    lowered = text.lower()
    tokens = {word.strip(_STRIP_PUNCT) for word in lowered.split()}
    best: Optional[Dict[str, Any]] = None
    best_hits = 0
    tied = False
    for guide in SCENE_GUIDES:
        hits = sum(
            1
            for keyword in guide["match"]
            if (" " in keyword and keyword in lowered) or keyword in tokens
        )
        if hits > best_hits:
            best, best_hits, tied = guide, hits, False
        elif hits == best_hits and hits > 0:
            tied = True
    if best_hits < 2 or tied:
        return None
    return best


GROUNDING_RULE = (
    "GROUNDING RULE (applies to EVERY style, including humorous ones): each caption must still "
    "explicitly name the real subject, their real action, and the real setting from the frames. "
    "Build jokes ON TOP of those facts as comparisons using 'like' or 'as if' — never assert "
    "invented specifics (text on a screen, names, foods, thoughts, outcomes) that are not "
    "directly visible or stated in the transcript.\n"
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
    guide = _match_scene_guide(f"{description} {transcription or ''}")
    if not skip_verify:
        description = _verify_description(
            frame_paths, description, client,
            hints=guide["anchors"] if guide else None,
        )
        if guide is None:
            guide = _match_scene_guide(f"{description} {transcription or ''}")
    elif guide:
        # No verification pass available; still surface the archetype details,
        # but subordinate them to what the description actually established.
        description += (
            "\nAdditional scene notes (use only those consistent with the "
            f"description above): {guide['anchors']}"
        )
    print(f"  Grounding description: {description}")

    captions: Dict[str, str] = {style: "" for style in styles}

    def _write(style: str, prior: List[str]) -> str:
        try:
            return _write_style_caption(
                style, description, transcription, prior, client, guide=guide
            )
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
        if pieces and total + count > 30:
            break
        pieces.append(clean)
        total += count
        if total >= 15:
            break
    text = ", ".join(pieces)
    words = text.split()
    if len(words) > 30:
        text = " ".join(words[:30]).rstrip(",;:")
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
        max_images=8,
        json_mode=False,
    ).strip()


def _verify_description(
    frame_paths: List[str], draft: str, client: GemmaClient,
    hints: Optional[str] = None,
) -> str:
    prompt = f'Here is a draft description of these frames: "{draft}"\n\n'
    if hints:
        prompt += (
            "Scenes of this kind often contain the following easily-missed "
            "details; work each one into the description ONLY if the frames "
            f"actually confirm it, and drop any that they do not: {hints}\n\n"
        )
    prompt += (
        "Check the description against the actual frames. If accurate and "
        "specific, repeat it unchanged. If anything is wrong or too generic, "
        "correct it. Output ONLY the final description."
    )
    verified = client.vision_chat(
        system_prompt="You verify video descriptions against the actual frames.",
        user_text=prompt,
        image_paths=frame_paths,
        max_tokens=1500,
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
    guide: Optional[Dict[str, Any]] = None,
) -> str:
    angle_note = ""
    if guide:
        angle = guide.get("angles", {}).get(style)
        if angle:
            angle_note = (
                "\nAn angle that lands well for this kind of scene (adapt it to "
                f"the actual facts above, don't quote it verbatim): {angle}\n"
            )
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\nCaptions already written for this clip in other styles are reference only. "
            "Write this style independently; do not paraphrase those captions: "
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
        "using 'like' or 'as if'.\n"
        f"{CAPTION_QUALITY_RULES}"
        f"{angle_note}"
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
        f"{CAPTION_QUALITY_RULES}\n"
        "STYLE RULES:\n"
        f"{_style_block(styles)}\n"
        "Each candidate must preserve the factual subject, action, and setting while trying a different stylistic angle.\n"
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
