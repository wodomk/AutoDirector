import os
import random
import time
from typing import Any

import requests


COMFYUI_BASE_URL = "http://localhost:8188"
NEGATIVE_PROMPT = "deformed faces, extra limbs, extra fingers, fused fingers, bad anatomy, ugly faces, distorted body, multiple heads, cloned faces, mutation, blurry, low quality, cartoon, anime, unrealistic, watermark, text, signature, out of frame, cropped"
CHECKPOINT_NAME = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"


def _upload_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        data = {"overwrite": "true", "type": "input"}
        try:
            resp = requests.post(
                f"{COMFYUI_BASE_URL}/upload/image",
                files=files,
                data=data,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to upload image '{image_path}': {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI /upload/image returned non-JSON response.") from exc

    name = payload.get("name") or payload.get("filename")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Upload response missing file name for '{image_path}'.")
    return name


def _build_workflow(description: str, reference_image_name: str | None = None) -> dict[str, Any]:
    seed = random.randint(0, 2**31 - 1)
    if reference_image_name:
        return {
            "ckpt_loader": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CHECKPOINT_NAME},
            },
            "pos_encode": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": description, "clip": ["ckpt_loader", 1]},
            },
            "neg_encode": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": NEGATIVE_PROMPT, "clip": ["ckpt_loader", 1]},
            },
            "clip_vision_loader": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "clip_vision_vit_h.safetensors"},
            },
            "ip_load": {
                "class_type": "IPAdapterUnifiedLoader",
                "inputs": {"model": ["ckpt_loader", 0], "preset": "PLUS (high strength)"},
            },
            "load_ref": {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image_name, "upload": "image"},
            },
            "vae_encode": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["load_ref", 0], "vae": ["ckpt_loader", 2]},
            },
            "ip_apply": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["ip_load", 0],
                    "ipadapter": ["ip_load", 1],
                    "image": ["load_ref", 0],
                    "clip_vision": ["clip_vision_loader", 0],
                    "weight": 0.65,
                    "weight_type": "linear",
                    "start_at": 0.0,
                    "end_at": 1.0,
                    "combine_embeds": "concat",
                },
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 832, "height": 480, "batch_size": 1},
            },
            "ksampler": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 35,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.55,
                    "model": ["ip_apply", 0],
                    "positive": ["pos_encode", 0],
                    "negative": ["neg_encode", 0],
                    "latent_image": ["vae_encode", 0],
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt_loader", 2]},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": ["decode", 0], "filename_prefix": "codex_frame"},
            },
        }

    return {
        "ckpt_loader": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
        },
        "pos_encode": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": description, "clip": ["ckpt_loader", 1]},
        },
        "neg_encode": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["ckpt_loader", 1]},
        },
        "empty_latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 832, "height": 480, "batch_size": 1},
        },
        "ksampler": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 35,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["ckpt_loader", 0],
                "positive": ["pos_encode", 0],
                "negative": ["neg_encode", 0],
                "latent_image": ["empty_latent", 0],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt_loader", 2]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": "codex_frame"},
        },
    }


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
    reference_image_name = None
    if reference_image_path:
        reference_image_name = _upload_image(reference_image_path)

    workflow = _build_workflow(description, reference_image_name)

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
