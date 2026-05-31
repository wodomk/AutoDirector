import os
import random
import time
from typing import Any

import requests


COMFYUI_BASE_URL = "http://localhost:8188"
NEGATIVE_PROMPT = "deformed faces, extra limbs, extra fingers, fused fingers, bad anatomy, ugly faces, distorted body, multiple heads, cloned faces, mutation, blurry, low quality, cartoon, anime, unrealistic, watermark, text, signature, out of frame, cropped"
CHECKPOINT_NAME = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
VAE_NAME = "vae-ft-mse-840000-ema-pruned.safetensors"


def _build_workflow(description: str, reference_image_path: str | None = None) -> dict[str, Any]:
    seed = random.randint(0, 2**31 - 1)

    workflow: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": description, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["1", 1]},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 35,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["2", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "codex_frame"},
        },
    }

    if reference_image_path:
        workflow["5"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["9", 0], "vae": ["2", 0]},
        }
        workflow["6"]["inputs"]["denoise"] = 0.75
        workflow["9"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_path, "upload": "image"},
        }
    else:
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 832, "height": 480, "batch_size": 1},
        }

    return workflow


def _extract_first_image_meta(history_item: dict[str, Any]) -> dict[str, str]:
    outputs = history_item.get("outputs", {})
    for node_data in outputs.values():
        images = node_data.get("images")
        if images and isinstance(images, list):
            first = images[0]
            if isinstance(first, dict):
                filename = first.get("filename")
                subfolder = first.get("subfolder", "")
                img_type = first.get("type", "output")
                if filename:
                    return {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": img_type,
                    }
    raise RuntimeError("ComfyUI history does not contain generated image metadata.")


def generate_keyframe(description: str, output_path: str, reference_image_path: str = None) -> str:
    workflow = _build_workflow(description, reference_image_path)

    try:
        submit_resp = requests.post(
            f"{COMFYUI_BASE_URL}/prompt",
            json={"prompt": workflow},
            timeout=120,
        )
        submit_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to submit prompt to ComfyUI: {exc}") from exc

    try:
        submit_data = submit_resp.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI /prompt returned non-JSON response.") from exc

    prompt_id = submit_data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI response does not contain prompt_id.")

    deadline = time.time() + 300
    history_item: dict[str, Any] | None = None

    while time.time() < deadline:
        try:
            history_resp = requests.get(
                f"{COMFYUI_BASE_URL}/history/{prompt_id}",
                timeout=30,
            )
            history_resp.raise_for_status()
            history_data = history_resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed while polling ComfyUI history: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("ComfyUI /history returned non-JSON response.") from exc

        if isinstance(history_data, dict) and prompt_id in history_data:
            history_item = history_data[prompt_id]
            break

        time.sleep(2)

    if history_item is None:
        raise TimeoutError("Timed out waiting for ComfyUI generation (300 seconds).")

    image_meta = _extract_first_image_meta(history_item)

    try:
        image_resp = requests.get(
            f"{COMFYUI_BASE_URL}/view",
            params=image_meta,
            timeout=120,
        )
        image_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download generated image from ComfyUI: {exc}") from exc

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(image_resp.content)

    return output_path


def main() -> None:
    sample_description = (
        "Cinematic wide shot of a futuristic city street at sunset, wet asphalt reflecting "
        "neon signs, a lone detective in a dark trench coat walking toward camera, ultra-detailed, realistic lighting"
    )
    target_path = "test_frame.png"
    try:
        saved_path = generate_keyframe(sample_description, target_path)
        print(f"Saved keyframe to: {saved_path}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
