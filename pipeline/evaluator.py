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

    try:
        raw_text = client.vision_chat(
            system_prompt=system_prompt,
            user_text=user_text,
            image_paths=frame_paths,
            max_tokens=500,
            temperature=0.3,
        )
        return extract_json_object(raw_text)
    except Exception as e:
        print(f"  Self-eval LLM failed (non-fatal): {e}")
        return None
