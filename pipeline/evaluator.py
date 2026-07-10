import json
from typing import Any, Dict, List, Optional
from gemma_client import GemmaClient, extract_json_object


def _build_eval_prompt() -> str:
    return """You are an evaluator judging video captions against the video content shown in the keyframes.

For EACH of the captions provided, score it on two dimensions (0.0 to 1.0):
1. accuracy: how faithfully the caption reflects what is actually shown in the video frames.
2. style_match: how well the caption matches its intended tone.
   - formal: professional, objective, factual tone (no jokes)
   - sarcastic: dry, ironic, lightly mocking tone
   - humorous_tech: funny, with technology/programming references
   - humorous_non_tech: funny, everyday humour, NO technical jargon

STRICT RULES:
- Output ONLY valid JSON, nothing else. No explanation, no reasoning, no preamble.
- JSON must exactly match this structure:
{"formal": {"accuracy": 0.8, "style_match": 0.9}, "sarcastic": {"accuracy": 0.7, "style_match": 0.8}, "humorous_tech": {"accuracy": 0.6, "style_match": 0.7}, "humorous_non_tech": {"accuracy": 0.9, "style_match": 0.9}}
- Scores must be floats between 0.0 and 1.0.
"""


def metric_score(metrics: Any, accuracy_weight: float = 0.5) -> Optional[float]:
    """Weighted accuracy/style_match from one judge entry, or None if the
    entry is missing or malformed."""
    if not isinstance(metrics, dict):
        return None
    try:
        accuracy = float(metrics.get("accuracy"))
        style_match = float(metrics.get("style_match"))
    except (TypeError, ValueError):
        return None
    return accuracy_weight * accuracy + (1 - accuracy_weight) * style_match


def style_score(scores: Optional[Dict[str, Any]], style: str) -> Optional[float]:
    """Average accuracy/style_match for one style, or None if the judge
    output is missing or malformed for that style."""
    if not isinstance(scores, dict):
        return None
    return metric_score(scores.get(style))


def score_caption_pool(
    frame_paths: List[str],
    pool: Dict[str, List[str]],
    client: GemmaClient,
) -> Optional[Dict[str, List[Dict[str, float]]]]:
    """Score several candidate captions per style against the frames in a
    single vision call. Returns {style: [scores per candidate, in order]}."""
    if not client.available or not pool:
        return None

    system_prompt = """You are an evaluator judging candidate video captions against the video content shown in the keyframes.

For EACH candidate of EACH style, score it on two dimensions (0.0 to 1.0):
1. accuracy: how faithfully the caption reflects what is actually shown in the video frames.
2. style_match: how well the caption matches its intended tone.
   - formal: professional, objective, factual tone (no jokes)
   - sarcastic: dry, ironic, lightly mocking tone
   - humorous_tech: funny, with technology/programming references
   - humorous_non_tech: funny, everyday humour, NO technical jargon

STRICT RULES:
- Output ONLY valid JSON, nothing else. No explanation, no reasoning, no preamble.
- For each style, return an array with ONE score object PER candidate, in the exact order the candidates were given.
- Structure: {"sarcastic": [{"accuracy": 0.8, "style_match": 0.9}, {"accuracy": 0.6, "style_match": 0.7}], ...}
- Scores must be floats between 0.0 and 1.0.
"""

    user_text = "Candidate captions to evaluate:\n"
    for style, candidates in pool.items():
        user_text += f"\nStyle: {style}\n"
        for idx, caption in enumerate(candidates, start=1):
            user_text += f'  candidate {idx}: "{caption}"\n'

    return _judged_json(
        client,
        system_prompt=system_prompt,
        user_text=user_text,
        image_paths=frame_paths,
        max_tokens=1500,
    )


def _judged_json(
    client: GemmaClient,
    system_prompt: str,
    user_text: str,
    image_paths: List[str],
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    """Run a judging call on the judge model, falling back to the generation
    model if the judge model fails or returns unparseable output."""
    for model in (client.judge_model, None):
        try:
            raw_text = client.vision_chat(
                system_prompt=system_prompt,
                user_text=user_text,
                image_paths=image_paths,
                max_tokens=max_tokens,
                temperature=0.2,
                max_images=16,
                model=model,
            )
            return extract_json_object(raw_text)
        except Exception as e:
            label = model or "generation model"
            print(f"  Judge call failed on {label} (non-fatal): {e}")
    return None


def evaluate_captions(
    frame_paths: List[str], captions: Dict[str, str], client: GemmaClient
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Evaluates generated captions against the visual frames using Gemma.
    Returns a dictionary of scores, or None if the evaluation fails.
    """
    if not client.available:
        return None

    system_prompt = _build_eval_prompt()
    user_text = "Captions to evaluate:\n"
    for style, text in captions.items():
        if text:
            user_text += f'{style}: "{text}"\n'

    return _judged_json(
        client,
        system_prompt=system_prompt,
        user_text=user_text,
        image_paths=frame_paths,
        max_tokens=500,
    )
