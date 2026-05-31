import json
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
    "clothing, location details, lighting). For each frame include 'consistency_anchors' "
    "describing EXACTLY: main character appearance (fur color, pattern, size), exact "
    "location in room, lighting direction and color. These must be IDENTICAL across all "
    "frames to ensure visual consistency."
)


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


def generate_shotlist(prompt: str, num_frames: int = 5) -> list[dict]:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": f"{prompt}\n\nGenerate EXACTLY {num_frames} keyframes, no more, no less.",
            "stream": False,
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Failed to call Ollama API: {exc}")
            return []

        try:
            api_data = response.json()
        except ValueError as exc:
            print(f"Ollama API returned non-JSON response: {exc}")
            return []

        llm_text = api_data.get("response", "")
        if not isinstance(llm_text, str):
            llm_text = str(llm_text)

        parsed: Any = None
        try:
            parsed = json.loads(llm_text)
        except Exception:
            first_arr = llm_text.find("[")
            last_arr = llm_text.rfind("]")
            try:
                if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
                    parsed = json.loads(llm_text[first_arr:last_arr + 1])
                else:
                    raise ValueError("Array brackets not found")
            except Exception:
                first_obj = llm_text.find("{")
                last_obj = llm_text.rfind("}")
                try:
                    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
                        parsed = json.loads(f"[{llm_text[first_obj:last_obj + 1]}]")
                    else:
                        raise ValueError("Object braces not found")
                except Exception as exc:
                    print(f"Failed to parse shot list JSON: {exc}")
                    return []

        try:
            return _validate_shotlist(parsed)
        except Exception as exc:
            print(f"Shot list validation failed: {exc}")
            return []
    finally:
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma4:e4b", "keep_alive": 0},
            )
        except requests.RequestException:
            pass


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
