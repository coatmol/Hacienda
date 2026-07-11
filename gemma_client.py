import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env, defending against empty-string presets: an image built with
    `ENV HACIENDA_GEMMA_TOKEN=""` would otherwise shadow the baked .env file
    (python-dotenv never overrides existing keys, even empty ones)."""
    for key in list(os.environ):
        if key.startswith("HACIENDA_") and os.environ[key] == "":
            del os.environ[key]
    load_dotenv()


class GemmaClient:
    def __init__(self) -> None:
        _load_env()
        self.base_url = os.getenv("HACIENDA_GEMMA_BASE_URL", "").rstrip("/")
        self.token = os.getenv("HACIENDA_GEMMA_TOKEN", "")
        self.model = os.getenv("HACIENDA_GEMMA_MODEL", "gemma")
        # Use the configured model for every stage unless the caller
        # explicitly opts into a separate vision/judge model.
        self.vision_model = os.getenv("HACIENDA_VISION_MODEL", self.model)
        self.judge_model = os.getenv("HACIENDA_JUDGE_MODEL", self.model)
        self.timeout = int(os.getenv("HACIENDA_GEMMA_TIMEOUT", "90"))

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.token)

    def _post_with_retry(
        self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], retries: int = 4
    ) -> Dict[str, Any]:
        """POST with exponential backoff on transient failures (429, 5xx,
        network errors). Non-retryable 4xx errors raise immediately."""
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = requests.post(
                    endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                if attempt < retries - 1:
                    wait = min(3 * (2 ** attempt), 20)
                    print(f"  Transient API error ({e}); retrying in {wait}s...")
                    time.sleep(wait)
        raise last_error

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int = 900,
        temperature: float = 0.35,
    ) -> str:
        if not self.available:
            raise RuntimeError("Gemma proxy is not configured.")

        # Standard Fireworks/OpenAI chat completions endpoint
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = self._post_with_retry(endpoint, headers, payload)

        raw_text = _message_content(data)
        print(f"DEBUG {self.model} OUTPUT:\n{raw_text}\n---END DEBUG---", flush=True)
        return raw_text

    def vision_chat(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: List[str],
        max_tokens: int = 900,
        temperature: float = 0.35,
        max_images: int = 8,
        model: Optional[str] = None,
        json_mode: bool = True,
    ) -> str:
        if not self.available:
            raise RuntimeError("Gemma proxy is not configured.")

        vision_model = model or self.vision_model
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        # Prevent overwhelming the model with too many image tokens
        if len(image_paths) > max_images:
            # Evenly sample max_images frames, always keeping the first and last
            count = len(image_paths)
            indices = sorted(
                {round(i * (count - 1) / (max_images - 1)) for i in range(max_images)}
            )
            image_paths = [image_paths[i] for i in indices]

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"},
                }
            )

        payload = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = self._post_with_retry(endpoint, headers, payload)
        raw_text = _message_content(data)
        print(f"DEBUG {vision_model} (VISION) OUTPUT:\n{raw_text}\n---END DEBUG---", flush=True)
        return raw_text

    def fallback_chat(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int = 900,
        temperature: float = 0.35,
    ) -> str:
        """Text-only fallback using the configured model.

        This keeps submission behavior aligned with HACIENDA_GEMMA_MODEL and
        avoids silently using an unapproved model.
        """
        if not self.available:
            raise RuntimeError("Gemma proxy is not configured.")

        fallback_model = self.model
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": fallback_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = self._post_with_retry(endpoint, headers, payload)
        raw_text = _message_content(data)
        print(f"DEBUG {fallback_model} (FALLBACK) OUTPUT:\n{raw_text}\n---END DEBUG---", flush=True)
        return raw_text


def _message_content(data: Dict[str, Any]) -> str:
    """Extract the answer text from a chat completion. Reasoning models can
    exhaust max_tokens while thinking, returning only `reasoning_content` and
    no `content` key at all — surface that as a retryable error instead of a
    KeyError."""
    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if content:
        return content
    if message.get("reasoning_content"):
        raise ValueError(
            "Model spent the whole token budget on reasoning and returned no "
            "answer; raise max_tokens for this call."
        )
    raise ValueError("Model returned an empty message.")


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
