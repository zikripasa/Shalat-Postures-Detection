"""Salat Vision API.

Run locally::

    uvicorn backend.app.main:app --reload --port 8000

Endpoints: ``GET /health``, ``GET /model-info``, ``POST /predict/image``,
``POST /predict/video`` (synchronous), ``POST /jobs/video`` + ``GET /jobs/{id}``
(asynchronous with progress), ``GET /jobs/{id}/video`` (annotated MP4),
``GET /jobs/{id}/report`` (JSON/CSV), ``DELETE /jobs/{id}``, ``GET /demo/video``.
Interactive documentation: ``/docs``.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from salat_vision import __version__
from salat_vision.inference.video import VideoError, analyze_video

from .jobs import JobManager, TooManyJobs
from .schemas import ErrorResponse, HealthResponse, ImagePrediction, JobCreated, JobStatus, VideoAnalysis
from .service import ModelService, ModelUnavailable
from .settings import Settings, get_settings
from .uploads import decode_image, read_limited, save_video_upload

log = logging.getLogger("salat_vision.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

NOTE = ("Skor adalah keyakinan model, bukan probabilitas terkalibrasi. Sistem mengenali postur yang terlihat "
        "dan tidak menilai sah atau tidaknya shalat.")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ModelService(settings)
        if settings.load_model_on_startup:
            await run_in_threadpool(service.load)
        app.state.service = service
        app.state.jobs = JobManager(service, settings)

        async def janitor():
            while True:
                await asyncio.sleep(60)
                app.state.jobs.cleanup()

        task = asyncio.create_task(janitor())
        try:
            yield
        finally:
            task.cancel()
            app.state.jobs.shutdown()
            service.close()

    app = FastAPI(
        title="Salat Vision API",
        version=__version__,
        description="Prayer-posture recognition from images and video. Runs fully locally; uploads are deleted "
                    "after analysis. Recognises visible postures only — it does not judge prayer validity.",
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse, "description": "Model bundle not available"}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    def service_dep(request: Request) -> ModelService:
        return request.app.state.service

    def jobs_dep(request: Request) -> JobManager:
        return request.app.state.jobs

    @app.exception_handler(ModelUnavailable)
    async def _unavailable(_: Request, exc: ModelUnavailable):
        return JSONResponse(status_code=503, content={"detail": f"Model unavailable: {exc}"})

    # ------------------------------------------------------------------ status
    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health(service: ModelService = Depends(service_dep)):
        return HealthResponse(
            model_loaded=service.available,
            model_version=service.bundle.version if service.bundle else None,
            model_error=service.error,
            uptime_seconds=round(time.time() - service.started_at, 1),
            package_version=__version__,
        )

    @app.get("/model-info", tags=["status"])
    def model_info(service: ModelService = Depends(service_dep)) -> dict:
        return service.model_info()

    # ------------------------------------------------------------------ images
    @app.post("/predict/image", response_model=ImagePrediction, tags=["predict"])
    async def predict_image(file: UploadFile = File(..., description="JPEG, PNG or WebP image"),
                            service: ModelService = Depends(service_dep)):
        service.require()
        data = await read_limited(file, settings.max_image_mb)
        rgb, meta = decode_image(data)
        pred = await run_in_threadpool(service.predict_image, rgb)
        d = pred.to_dict(include_landmarks=True)
        thr = float(service.cfg["inference"]["uncertainty_threshold"])
        return ImagePrediction(**d, uncertain=pred.probs is None or pred.confidence < thr, image=meta, note=NOTE)

    # ------------------------------------------------------------------ videos
    @app.post("/predict/video", response_model=VideoAnalysis, tags=["predict"],
              summary="Analyse a video synchronously (for short clips / scripts)")
    async def predict_video(file: UploadFile = File(...), sample_fps: float | None = Query(None, gt=0, le=15),
                            include_frames: bool = Query(False), service: ModelService = Depends(service_dep)):
        service.require()
        path = await save_video_upload(file, settings.max_video_mb, settings.tmp_dir)
        try:
            def work():
                predictor = service.predictor_for_thread()
                return analyze_video(path, predictor, service.cfg, sample_fps=sample_fps)

            result = await run_in_threadpool(work)
        except VideoError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        if not include_frames:
            result["frames"] = None
        return result

    @app.post("/jobs/video", response_model=JobCreated, status_code=202, tags=["jobs"],
              summary="Start an asynchronous video analysis (poll /jobs/{id})")
    async def create_video_job(file: UploadFile = File(...), annotate: bool = Query(True),
                               sample_fps: float | None = Query(None, gt=0, le=15),
                               service: ModelService = Depends(service_dep), jobs: JobManager = Depends(jobs_dep)):
        service.require()
        path = await save_video_upload(file, settings.max_video_mb, settings.tmp_dir)
        try:
            job = jobs.submit(path, annotate=annotate, sample_fps=sample_fps)
        except TooManyJobs as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc
        return JobCreated(job_id=job.id, status=job.status, status_url=f"/jobs/{job.id}")

    def _job_or_404(jobs: JobManager, job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found or expired")
        return job

    @app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
    def get_job(job_id: str, jobs: JobManager = Depends(jobs_dep)):
        return _job_or_404(jobs, job_id).public(include_result=True)

    @app.get("/jobs/{job_id}/video", tags=["jobs"], response_class=FileResponse,
             summary="Annotated video (H.264 MP4)")
    def get_job_video(job_id: str, jobs: JobManager = Depends(jobs_dep)):
        job = _job_or_404(jobs, job_id)
        if not (job.annotated_path and job.annotated_path.is_file()):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No annotated video for this job")
        return FileResponse(job.annotated_path, media_type="video/mp4", filename=f"salat_vision_{job.id[:8]}.mp4")

    @app.get("/jobs/{job_id}/report", tags=["jobs"], summary="Download the analysis as JSON or CSV")
    def get_job_report(job_id: str, format: str = Query("json", pattern="^(json|csv)$"), jobs: JobManager = Depends(jobs_dep)):
        job = _job_or_404(jobs, job_id)
        if job.status != "done" or not job.result:
            raise HTTPException(status.HTTP_409_CONFLICT, "Analysis not finished")
        name = f"salat_vision_{job.id[:8]}"
        if format == "json":
            body = json.dumps(job.result, ensure_ascii=False, indent=2)
            return Response(body, media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["index", "phase", "label", "start_s", "end_s", "duration_s", "confidence", "uncertain", "evidence"])
        for s in job.result["segments"]:
            w.writerow([s["index"], s["phase"], s["label"], s["start"], s["end"], s["duration"], s["confidence"],
                        s["uncertain"], s["evidence"]])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

    @app.delete("/jobs/{job_id}", status_code=204, tags=["jobs"], summary="Delete a job and its files now")
    def delete_job(job_id: str, jobs: JobManager = Depends(jobs_dep)):
        if not jobs.delete(job_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found or expired")
        return Response(status_code=204)

    @app.get("/demo/video", tags=["status"], response_class=FileResponse,
             summary="Synthetic demo sequence assembled from dataset images (if generated locally)")
    def demo_video(service: ModelService = Depends(service_dep)):
        p = service.demo_video_path()
        if p is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Demo video not generated (run scripts/make_demo_video.py)")
        return FileResponse(p, media_type="video/mp4", filename="salat_vision_demo.mp4")

    return app


app = create_app()
