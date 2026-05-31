import asyncio
import json
import os
import sqlite3
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
DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.db")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    frames: int = Field(default=49, ge=17, le=121)
    width: int = Field(default=832)
    height: int = Field(default=480)
    steps: int = Field(default=20, ge=10, le=30)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_frames (
                job_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (job_id, frame_number)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_clips (
                job_id TEXT NOT NULL,
                clip_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (job_id, clip_number)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_shots (
                job_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                shot_json TEXT NOT NULL,
                PRIMARY KEY (job_id, frame_number)
            )
            """
        )
        conn.execute(
            "UPDATE jobs SET status = 'interrupted', stage = 'interrupted', message = 'Job interrupted by restart', updated_at = ? WHERE status = 'running'",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()


@app.on_event("startup")
async def startup_event() -> None:
    _init_db()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    output_dir = os.path.join(BASE_OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    with _db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, prompt, status, stage, progress, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, req.prompt, "running", "queued", 0, "Job queued", now, now),
        )
        conn.commit()

    director = Director(output_root=BASE_OUTPUT_DIR, db_path=DB_PATH)
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
    with _db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        frames = conn.execute(
            "SELECT frame_number, path FROM job_frames WHERE job_id = ? ORDER BY frame_number ASC",
            (job_id,),
        ).fetchall()
        shots = conn.execute(
            "SELECT frame_number, shot_json FROM job_shots WHERE job_id = ? ORDER BY frame_number ASC",
            (job_id,),
        ).fetchall()

    data: dict[str, Any] = {}
    if shots:
        shot_list: list[Any] = []
        for row in shots:
            raw = row["shot_json"]
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            shot_list.append(parsed)
        data["shot_list"] = shot_list
    if frames:
        data["frame_paths"] = [r["path"] for r in frames]

    return {"stage": job["stage"], "progress": job["progress"], "message": job["message"], "data": data}


@app.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/output/{job_id}")
async def output(job_id: str) -> FileResponse:
    with _db() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    out_path = os.path.join(BASE_OUTPUT_DIR, job_id, "final.mp4")
    if not os.path.isfile(out_path):
        raise HTTPException(status_code=404, detail="Output not ready")

    return FileResponse(out_path, media_type="video/mp4", filename=f"{job_id}.mp4")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/artifacts", StaticFiles(directory=BASE_OUTPUT_DIR), name="artifacts")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
