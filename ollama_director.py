import json
import re
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:e4b"
SYSTEM_PROMPT = (
    "You are a professional film director. Based on the user description, create a "
    "shot list as a JSON array. Return ONLY valid JSON, no markdown, no comments, no "
    "explanation. Each element must have: frame_number (int), description (str, "
    "detailed prompt for Stable Diffusion image generation in English), camera_angle "
    "(str, e.g. wide shot / close-up / medium shot / POV), consistency_anchors (list "
    "of str, visual elements that must stay identical to previous frame e.g. character "
    "clothing, location details, lighting)."
)


def _extract_json_array(text: str) -> str | None:
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _validate_shotlist(data: Any) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("Response JSON is not a list.")

    required_keys = {"frame_number", "description", "camera_angle", "consistency_anchors"}
    validated: list[dict] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Element at index {idx} is not an object.")
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Element at index {idx} missing keys: {sorted(missing)}")
        validated.append(item)

    return validated


def generate_shotlist(prompt: str) -> list[dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call Ollama API: {exc}") from exc

    try:
        api_data = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama API returned non-JSON response.") from exc

    llm_text = api_data.get("response")
    if not isinstance(llm_text, str):
        raise RuntimeError("Ollama API response does not contain 'response' text.")

    try:
        parsed = json.loads(llm_text)
    except json.JSONDecodeError:
        extracted = _extract_json_array(llm_text)
        if not extracted:
            raise RuntimeError("Could not parse shot list JSON from model response.")
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Extracted JSON is still invalid.") from exc

    return _validate_shotlist(parsed)


def main() -> None:
    sample_prompt = (
        "A lone astronaut enters an abandoned space station corridor, sees flickering "
        "warning lights, and discovers a mysterious glowing artifact in the control room."
    )
    try:
        shotlist = generate_shotlist(sample_prompt)
        print(json.dumps(shotlist, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
