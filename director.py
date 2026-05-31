import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import comfyui_frames
import comfyui_video
import ollama_director
import video_assembler


class Director:
    def __init__(self, output_root: str = "/home/ai/AutoDirector/output", db_path: str = "jobs.db"):
        self.output_root = output_root
        self.db_path = db_path

    def db_update(self, job_id: str, stage: str, progress: int, message: str, data: Any = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        status = "running"
        if stage == "done":
            status = "done"
        elif stage == "error":
            status = "error"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, progress = ?, message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, stage, max(0, min(100, int(progress))), message, now, job_id),
            )

            if isinstance(data, dict):
                shot_list = data.get("shot_list")
                if isinstance(shot_list, list):
                    conn.execute("DELETE FROM job_shots WHERE job_id = ?", (job_id,))
                    for idx, shot in enumerate(shot_list, start=1):
                        conn.execute(
                            "INSERT INTO job_shots (job_id, frame_number, shot_json) VALUES (?, ?, ?)",
                            (job_id, idx, json.dumps(shot, ensure_ascii=False)),
                        )

                frame_paths = data.get("frame_paths")
                if isinstance(frame_paths, list):
                    conn.execute("DELETE FROM job_frames WHERE job_id = ?", (job_id,))
                    for i, path in enumerate(frame_paths, start=1):
                        conn.execute(
                            "INSERT INTO job_frames (job_id, frame_number, path) VALUES (?, ?, ?)",
                            (job_id, i, str(path)),
                        )

                clip_idx = data.get("clip_index")
                clip_path = data.get("clip_path")
                if isinstance(clip_idx, int) and isinstance(clip_path, str):
                    conn.execute(
                        """
                        INSERT INTO job_clips (job_id, clip_number, path)
                        VALUES (?, ?, ?)
                        ON CONFLICT(job_id, clip_number) DO UPDATE SET path=excluded.path
                        """,
                        (job_id, clip_idx, clip_path),
                    )

            conn.commit()

    async def run_job(self, job_id: str, prompt: str, frames: int, width: int, height: int, steps: int) -> str | None:
        job_dir = os.path.join(self.output_root, job_id)
        os.makedirs(job_dir, exist_ok=True)

        try:
            self.db_update(job_id, "scenario", 5, "Generowanie shot listy...")
            shot_list = await asyncio.to_thread(ollama_director.generate_shotlist, prompt)
            self.db_update(job_id, "scenario", 25, "Shot lista gotowa", {"shot_list": shot_list})

            frame_paths: list[str] = []
            frame_urls: list[str] = []
            total_shots = max(1, len(shot_list))
            self.db_update(job_id, "frames", 26, "Generowanie klatek kluczowych...")
            for idx, shot in enumerate(shot_list, start=1):
                description = shot.get("description", prompt) if isinstance(shot, dict) else prompt
                frame_path = os.path.join(job_dir, f"frame_{idx:03d}.png")
                ref = frame_paths[-1] if frame_paths else None
                await asyncio.to_thread(comfyui_frames.generate_keyframe, description, frame_path, ref)
                frame_paths.append(frame_path)
                frame_urls.append(f"/artifacts/{job_id}/frame_{idx:03d}.png")
                progress = 26 + int((idx / total_shots) * 34)
                self.db_update(
                    job_id,
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
            self.db_update(job_id, "video", 60, "Generowanie klipow pomiedzy klatkami...")
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
                self.db_update(
                    job_id,
                    "video",
                    progress,
                    f"Wygenerowano klip {i + 1}/{pairs}",
                    {
                        "clip_index": i + 1,
                        "clip_total": pairs,
                        "clip_path": clip_path,
                        "frame_paths": frame_urls,
                    },
                )

            final_path = os.path.join(job_dir, "final.mp4")
            self.db_update(job_id, "assembly", 92, "Skladanie finalnego filmu...")
            await asyncio.to_thread(video_assembler.assemble_clips, clip_paths, final_path)
            self.db_update(
                job_id,
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
                self.db_update(job_id, "error", 100, "OOM: zmniejsz frames lub rozdzielczosc", {"error": msg})
            else:
                self.db_update(job_id, "error", 100, msg, {"error": msg})
            return None
