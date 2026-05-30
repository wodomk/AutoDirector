import asyncio
import os
import time
from typing import Any, Callable

import comfyui_frames
import comfyui_video
import ollama_director
import video_assembler


class Director:
    def __init__(self, output_root: str = "/home/ai/AutoDirector/output", progress_callback: Callable[..., None] | None = None):
        self.output_root = output_root
        self.progress_callback = progress_callback or (lambda *args, **kwargs: None)

    def _cb(self, stage: str, percent: int, message: str, data: Any = None) -> None:
        self.progress_callback(stage, percent, message, data)

    async def run_job(self, job_id: str, prompt: str, frames: int, width: int, height: int, steps: int) -> str | None:
        job_dir = os.path.join(self.output_root, job_id)
        os.makedirs(job_dir, exist_ok=True)

        try:
            self._cb("scenario", 5, "Generowanie shot listy...")
            shot_list = await asyncio.to_thread(ollama_director.generate_shotlist, prompt)
            self._cb("scenario", 25, "Shot lista gotowa", {"shot_list": shot_list})

            frame_paths: list[str] = []
            frame_urls: list[str] = []
            total_shots = max(1, len(shot_list))
            self._cb("frames", 26, "Generowanie klatek kluczowych...")
            for idx, shot in enumerate(shot_list, start=1):
                description = shot.get("description", prompt) if isinstance(shot, dict) else prompt
                frame_path = os.path.join(job_dir, f"frame_{idx:03d}.png")
                ref = frame_paths[-1] if frame_paths else None
                await asyncio.to_thread(comfyui_frames.generate_keyframe, description, frame_path, ref)
                frame_paths.append(frame_path)
                frame_urls.append(f"/artifacts/{job_id}/frame_{idx:03d}.png")
                progress = 26 + int((idx / total_shots) * 34)
                self._cb(
                    "frames",
                    progress,
                    f"Wygenerowano klatke {idx}/{total_shots}",
                    {"frame_paths": frame_urls},
                )

            if len(frame_paths) < 2:
                raise RuntimeError("Do stworzenia klipu potrzeba co najmniej 2 klatek.")

            time.sleep(10)  # czeka az ComfyUI zwolni VRAM po generowaniu klatek SD

            clip_paths: list[str] = []
            pairs = len(frame_paths) - 1
            self._cb("video", 60, "Generowanie klipow pomiedzy klatkami...")
            for i in range(pairs):
                clip_path = os.path.join(job_dir, f"clip_{i + 1:03d}.mp4")
                clip_prompt = f"{prompt}. Transition segment {i + 1}/{pairs}. {width}x{height}, steps {steps}."
                await asyncio.to_thread(
                    comfyui_video.generate_clip,
                    frame_paths[i],
                    frame_paths[i + 1],
                    clip_prompt,
                    clip_path,
                )
                clip_paths.append(clip_path)
                progress = 60 + int(((i + 1) / pairs) * 30)
                self._cb(
                    "video",
                    progress,
                    f"Wygenerowano klip {i + 1}/{pairs}",
                    {"clip_index": i + 1, "clip_total": pairs, "frame_paths": frame_urls},
                )

            final_path = os.path.join(job_dir, "final.mp4")
            self._cb("assembly", 92, "Skladanie finalnego filmu...")
            await asyncio.to_thread(video_assembler.assemble_clips, clip_paths, final_path)
            self._cb(
                "done",
                100,
                "Film gotowy",
                {"done": True, "final_path": final_path, "frame_paths": frame_urls},
            )
            return final_path

        except Exception as exc:
            name = exc.__class__.__name__
            msg = str(exc)
            if name == "OutOfMemoryError" or "out of memory" in msg.lower() or isinstance(exc, MemoryError):
                self._cb("error", 100, "OOM: zmniejsz frames lub rozdzielczosc", {"error": msg})
            else:
                self._cb("error", 100, msg, {"error": msg})
            return None
