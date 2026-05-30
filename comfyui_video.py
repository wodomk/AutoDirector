import os
import time
import json
import glob
from typing import Any

import requests


COMFYUI_BASE_URL = "http://localhost:8188"
WAN_VIDEO_MODEL = "WanVideo/fp8_scaled_kj/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"
WAN_VAE_MODEL = "wanvideo/Wan2_1_VAE_bf16.safetensors"
WAN_T5_MODEL = "umt5_xxl_enc_bf16.safetensors"


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

    # Typical fields: name, subfolder, type
    name = payload.get("name") or payload.get("filename")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Upload response missing file name for '{image_path}'.")
    return name


def _build_workflow(uploaded_first_name: str, uploaded_last_name: str, prompt: str) -> dict[str, Any]:
    seed = int(time.time() * 1000) % (2**31 - 1)
    return {
        "1": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": WAN_VIDEO_MODEL,
                "load_device": "main_device",
                "base_precision": "bf16",
                "quantization": "disabled",
            },
        },
        "2": {
            "class_type": "WanVideoVAELoader",
            "inputs": {"model_name": WAN_VAE_MODEL, "precision": "bf16"},
        },
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {"model_name": WAN_T5_MODEL, "precision": "bf16"},
        },
        "4": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "positive_prompt": prompt,
                "negative_prompt": "blurry, low quality, cartoon, anime, unrealistic, bad anatomy, watermark",
                "t5": ["3", 0],
            },
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_first_name, "upload": "image"},
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_last_name, "upload": "image"},
        },
        "7": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "start_image": ["5", 0],
                "end_image": ["6", 0],
                "vae": ["2", 0],
                "width": 832,
                "height": 480,
                "num_frames": 49,
                "force_offload": True,
                "start_latent_strength": 1.0,
                "end_latent_strength": 1.0,
                "noise_aug_strength": 0.0,
                "fun_or_fl2v_model": True,
            },
        },
        "8": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["1", 0],
                "vae": ["2", 0],
                "conditioning": ["4", 0],
                "image_embeds": ["7", 0],
                "steps": 20,
                "frames": 49,
                "width": 832,
                "height": 480,
                "cfg": 6.0,
                "scheduler": "dpm++_sde",
                "riflex_freq_index": 0,
                "shift": 5.0,
                "force_offload": True,
                "seed": seed,
            },
        },
        "9": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["2", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            },
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 24,
                "format": "video/h264-mp4",
                "filename_prefix": "wan_flf2v",
                "loop_count": 0,
                "save_output": True,
                "pingpong": False,
            },
        },
    }


def _extract_video_meta(history_item: dict[str, Any]) -> dict[str, str]:
    outputs = history_item.get("outputs", {})

    def _walk(node: Any) -> dict[str, str] | None:
        if isinstance(node, dict):
            filename = node.get("filename")
            file_type = node.get("type")
            if (
                isinstance(filename, str)
                and filename.lower().endswith(".mp4")
                and str(file_type) == "output"
            ):
                return {
                    "filename": filename,
                    "subfolder": str(node.get("subfolder", "")),
                    "type": "output",
                }
            for value in node.values():
                found = _walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found:
                    return found
        return None

    found_meta = _walk(outputs)
    if found_meta:
        return found_meta

    print("DEBUG: MP4 not found in ComfyUI history outputs. Full outputs structure:")
    print(json.dumps(outputs, indent=2, ensure_ascii=False, default=str))
    raise RuntimeError("ComfyUI history does not contain MP4 metadata.")


def _find_latest_local_mp4() -> str | None:
    candidates = glob.glob(os.path.join("ComfyUI", "output", "**", "*.mp4"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def generate_clip(first_frame_path: str, last_frame_path: str, prompt: str, output_path: str) -> str:
    uploaded_first = _upload_image(first_frame_path)
    uploaded_last = _upload_image(last_frame_path)
    seed = int(time.time() * 1000) % (2**31 - 1)
    workflow = {
        "11": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": "umt5_xxl_enc_bf16.safetensors",
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "disabled",
            },
        },
        "22": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": "WanVideo/fp8_scaled_kj/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                "base_precision": "bf16",
                "load_device": "main_device",
                "quantization": "disabled",
            },
        },
        "38": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": "wanvideo/Wan2_1_VAE_bf16.safetensors",
                "precision": "bf16",
                "use_cpu_cache": False,
                "verbose": False,
            },
        },
        "16": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "positive_prompt": prompt,
                "negative_prompt": "blurry, low quality, cartoon, anime, unrealistic, bad anatomy, watermark",
                "force_offload": True,
                "use_disk_cache": False,
                "device": "gpu",
                "t5": ["11", 0],
            },
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_first, "upload": "image"},
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_last, "upload": "image"},
        },
        "37": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "width": 832,
                "height": 480,
                "num_frames": 49,
                "noise_aug_strength": 0.0,
                "start_latent_strength": 1.0,
                "end_latent_strength": 1.0,
                "force_offload": True,
                "fun_or_fl2v_model": True,
                "start_image": ["5", 0],
                "end_image": ["6", 0],
                "vae": ["38", 0],
            },
        },
        "39": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "blocks_to_swap": 20,
                "offload_txt_in": False,
                "offload_img_in": False,
                "offload_img_emb": False,
                "offload_txt_emb": False,
                "use_non_blocking": False,
                "vace_blocks_to_swap": 0,
            },
        },
        "56": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {"model": ["22", 0], "block_swap_args": ["39", 0]},
        },
        "58": {
            "class_type": "WanVideoSetLoRAs",
            "inputs": {"model": ["22", 0], "lora": ["60", 0]},
        },
        "60": {
            "class_type": "WanVideoLoraSelectMulti",
            "inputs": {
                "lora_0": "None",
                "strength_0": 1.0,
                "lora_1": "None",
                "strength_1": 1.0,
                "lora_2": "None",
                "strength_2": 1.0,
                "lora_3": "None",
                "strength_3": 1.0,
                "lora_4": "None",
                "strength_4": 1.0,
            },
        },
        "55": {
            "class_type": "WanVideoEnhanceAVideo",
            "inputs": {"weight": 2.0, "start_percent": 0.0, "end_percent": 1.0},
        },
        "27": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "steps": 20,
                "cfg": 6.0,
                "shift": 5.0,
                "seed": seed,
                "control_after_generate": "fixed",
                "force_offload": True,
                "scheduler": "dpm++_sde",
                "riflex_freq_index": 0,
                "denoise_strength": 1.0,
                "batched_cfg": False,
                "rope_function": "comfy",
                "start_step": 0,
                "end_step": -1,
                "add_noise_to_samples": False,
                "model": ["56", 0],
                "image_embeds": ["37", 0],
                "text_embeds": ["16", 0],
                "feta_args": ["55", 0],
            },
        },
        "28": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
                "vae": ["38", 0],
                "samples": ["27", 0],
            },
        },
        "30": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "AutoDirector",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
                "images": ["28", 0],
            },
        },
    }

    try:
        submit_resp = requests.post(
            f"{COMFYUI_BASE_URL}/prompt",
            json={"prompt": workflow},
            timeout=120,
        )
        submit_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to submit Wan FLF2V workflow: {exc}") from exc

    try:
        submit_data = submit_resp.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI /prompt returned non-JSON response.") from exc

    prompt_id = submit_data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI response does not contain prompt_id.")

    deadline = time.time() + 1800
    history_item: dict[str, Any] | None = None

    while time.time() < deadline:
        try:
            hist_resp = requests.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}", timeout=60)
            hist_resp.raise_for_status()
            hist_data = hist_resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed while polling ComfyUI history: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("ComfyUI /history returned non-JSON response.") from exc

        if isinstance(hist_data, dict) and prompt_id in hist_data:
            history_item = hist_data[prompt_id]
            break

        time.sleep(3)

    if history_item is None:
        raise TimeoutError("Timed out waiting for video generation (1800 seconds).")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        video_meta = _extract_video_meta(history_item)
        try:
            view_resp = requests.get(
                f"{COMFYUI_BASE_URL}/view",
                params={
                    "filename": video_meta["filename"],
                    "type": "output",
                },
                timeout=300,
            )
            view_resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download MP4 from ComfyUI: {exc}") from exc

        with open(output_path, "wb") as f:
            f.write(view_resp.content)
    except RuntimeError:
        latest_local_mp4 = _find_latest_local_mp4()
        if not latest_local_mp4:
            raise
        with open(latest_local_mp4, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())

    return output_path


def main() -> None:
    first_path = "first.png"
    last_path = "last.png"
    out_path = "test_clip.mp4"
    sample_prompt = (
        "A cinematic movement from dawn to dusk in a futuristic city street, "
        "preserve character identity and environment continuity, realistic lighting."
    )

    if not os.path.isfile(first_path) or not os.path.isfile(last_path):
        print("Brak plikow testowych.")
        print("Aby uruchomic test, umiesc w katalogu roboczym obrazy: first.png i last.png")
        print(
            "Nastepnie uruchom: generate_clip('first.png', 'last.png', 'twoj prompt', 'test_clip.mp4')"
        )
        return

    try:
        saved = generate_clip(first_path, last_path, sample_prompt, out_path)
        print(f"Saved clip to: {saved}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
