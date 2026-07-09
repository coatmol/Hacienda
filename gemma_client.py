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

        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"},
                }
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> Dict[str, Any]:
    parsed = _extract_json(text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("Expected JSON object.")


def _encode_image(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = re.findall(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in model response.")
