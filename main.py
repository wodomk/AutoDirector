import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from director import Director


app = FastAPI(title="WebDirector")

BASE_OUTPUT_DIR = "/home/ai/AutoDirector/output"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    frames: int = Field(default=49, ge=17, le=121)
    width: int = Field(default=832)
    height: int = Field(default=480)
    steps: int = Field(default=20, ge=10, le=30)


jobs: dict[str, dict[str, Any]] = {}
jobs_lock = asyncio.Lock()


def _progress_callback(job_id: str):
    def _cb(stage: str, percent: int, message: str, data: Any = None) -> None:
        job = jobs.get(job_id)
        if not job:
            return
        job["stage"] = stage
        job["progress"] = max(0, min(100, int(percent)))
        job["message"] = message
        if data is not None:
            job["data"] = data
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

    return _cb


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    output_dir = os.path.join(BASE_OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    async with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "prompt": req.prompt,
            "frames": req.frames,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "stage": "queued",
            "progress": 0,
            "message": "Job queued",
            "data": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "output_path": os.path.join(output_dir, "final.mp4"),
        }

    director = Director(
        output_root=BASE_OUTPUT_DIR,
        progress_callback=_progress_callback(job_id),
    )
    asyncio.create_task(
        director.run_job(
            job_id=job_id,
            prompt=req.prompt,
            frames=req.frames,
            width=req.width,
            height=req.height,
            steps=req.steps,
        )
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "stage": job.get("stage", "unknown"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "data": job.get("data", {}),
    }


@app.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    return sorted(jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)


@app.get("/output/{job_id}")
async def output(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    out_path = job.get("output_path") or os.path.join(BASE_OUTPUT_DIR, job_id, "final.mp4")
    if not os.path.isfile(out_path):
        raise HTTPException(status_code=404, detail="Output not ready")

    return FileResponse(out_path, media_type="video/mp4", filename=f"{job_id}.mp4")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/artifacts", StaticFiles(directory=BASE_OUTPUT_DIR), name="artifacts")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
