import os
import shutil
import subprocess
import tempfile
from typing import Sequence


def assemble_clips(clip_paths: Sequence[str], output_path: str) -> str:
    if not clip_paths:
        raise ValueError("clip_paths is empty")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], output_path)
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copyfile(clip_paths[0], output_path)
        return output_path

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
        list_file = f.name
        for path in clip_paths:
            safe_path = os.path.abspath(path).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        try:
            os.remove(list_file)
        except OSError:
            pass

    return output_path
