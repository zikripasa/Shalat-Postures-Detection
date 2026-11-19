"""In-memory background jobs for video analysis (no persistent storage of uploads)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from salat_vision.inference.video import VideoError, analyze_video

from .service import ModelService
from .settings import Settings

log = logging.getLogger("salat_vision.api.jobs")


class TooManyJobs(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    created: float
    annotate: bool
    sample_fps: float | None
    input_path: Path | None
    status: str = "queued"  # queued | running | done | error
    progress: dict = field(default_factory=lambda: {"stage": "queued", "fraction": 0.0})
    result: dict | None = None
    error: str | None = None
    annotated_path: Path | None = None
    finished: float | None = None

    def public(self, include_result: bool = True) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
            "annotated_video_available": bool(self.annotated_path and self.annotated_path.is_file()),
            "result": self.result if include_result else None,
        }


class JobManager:
    def __init__(self, service: ModelService, settings: Settings):
        self.service = service
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="salat-job")

    def submit(self, input_path: Path, *, annotate: bool, sample_fps: float | None) -> Job:
        self.cleanup()
        with self.lock:
            active = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
            if active >= self.settings.max_queued_jobs:
                raise TooManyJobs("Too many videos are being analysed; try again in a minute")
            job = Job(id=uuid.uuid4().hex, created=time.time(), annotate=annotate, sample_fps=sample_fps, input_path=input_path)
            self.jobs[job.id] = job
        self.pool.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.progress = {"stage": "starting", "fraction": 0.0}
        out = self.settings.tmp_dir / f"annotated_{job.id}.mp4" if job.annotate else None
        try:
            predictor = self.service.predictor_for_thread()

            def progress(p: dict) -> None:
                job.progress = {k: v for k, v in p.items() if v is not None}

            job.result = analyze_video(job.input_path, predictor, self.service.cfg, sample_fps=job.sample_fps,
                                       annotate_path=out, progress=progress)
            job.annotated_path = out if out and out.is_file() else None
            job.status = "done"
            job.progress = {"stage": "done", "fraction": 1.0}
        except VideoError as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            log.exception("job %s failed", job.id)
            job.status, job.error = "error", f"Analysis failed: {type(exc).__name__}"
        finally:
            job.finished = time.time()
            if job.input_path:
                job.input_path.unlink(missing_ok=True)  # never keep uploads
                job.input_path = None
            if job.status == "error" and out:
                out.unlink(missing_ok=True)

    def get(self, job_id: str) -> Job | None:
        self.cleanup()
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if not job:
            return False
        for p in (job.annotated_path, job.input_path):
            if p:
                Path(p).unlink(missing_ok=True)
        return True

    def cleanup(self) -> int:
        ttl = self.settings.job_ttl_minutes * 60
        now = time.time()
        expired = [j.id for j in list(self.jobs.values()) if j.finished and now - j.finished > ttl]
        for jid in expired:
            self.delete(jid)
        return len(expired)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        for jid in list(self.jobs):
            self.delete(jid)
