import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


class GemmaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("HACIENDA_GEMMA_BASE_URL", "").rstrip("/")
        self.token = os.getenv("HACIENDA_GEMMA_TOKEN", "")
        self.model = os.getenv("HACIENDA_GEMMA_MODEL", "gemma")
        self.vision_model = "accounts/fireworks/models/glm-5p2"
        self.timeout = int(os.getenv("HACIENDA_GEMMA_TIMEOUT", "90"))

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.token)

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 900,
        temperature: float = 0.35,
    ) -> str:
        if not self.available:
            raise RuntimeError("Gemma proxy is not configured.")

        # Force use of /completions for the text generation model since it lacks a chat template
        endpoint = self.base_url
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint.replace("/chat/completions", "/completions")
        elif not endpoint.endswith("/completions"):
            endpoint = f"{endpoint}/completions"

        prompt_string = f"<start_of_turn>user\n{system_prompt}\n\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

        payload = {
            "model": self.model,
            "prompt": prompt_string,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["<end_of_turn>"],
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        raw_text = data["choices"][0]["text"].strip()
        print(f"DEBUG {self.model} OUTPUT:\n{raw_text}\n---END DEBUG---", flush=True)
        return raw_text

    def vision_chat(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: List[str],
        max_tokens: int = 900,
        temperature: float = 0.35,
    ) -> str:
        if not self.available:
            raise RuntimeError("Gemma proxy is not configured.")

        # Gemma-4-e4b lacks a chat template on Fireworks, so we use /completions
        endpoint = self.base_url
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint.replace("/chat/completions", "/completions")
        elif not endpoint.endswith("/completions"):
            endpoint = f"{endpoint}/completions"

        # Prevent overwhelming the model with too many image tokens (causes hallucination/loops)
        if len(image_paths) > 3:
            mid = len(image_paths) // 2
            image_paths = [image_paths[0], image_paths[mid], image_paths[-1]]

        # Construct prompt with <image> tokens for the multimodal Gemma model
        images_prompt = " ".join(["<image>"] * len(image_paths))
        full_user_text = f"{images_prompt}\n{user_text}"
        
        prompt_string = f"<start_of_turn>user\n{system_prompt}\n\n{full_user_text}<end_of_turn>\n<start_of_turn>model\n"

        base64_images = [_encode_image(path) for path in image_paths]

        payload = {
            "model": self.model,
            "prompt": prompt_string,
            "images": base64_images,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["<end_of_turn>"],
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["text"].strip()
        print(f"DEBUG {self.model} (VISION) OUTPUT:\n{raw_text}\n---END DEBUG---", flush=True)
        return raw_text


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robust JSON extraction inspired by competitor's fallback logic."""
    if not text:
        raise ValueError("Empty response from model.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in model response.")


def _encode_image(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
