"""Response schemas (also drive the OpenAPI documentation at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_version: str | None = None
    model_error: str | None = None
    uptime_seconds: float
    package_version: str


class ImagePrediction(BaseModel):
    visual_class: str | None = Field(description="Top visual posture: qiyam | ruku | sujud | julus")
    label: str | None = Field(description="Indonesian label of the visual posture")
    confidence: float | None = Field(description="Top class score from the model (not a calibrated guarantee)")
    uncertain: bool = Field(description="True when the score is below the uncertainty threshold or nobody was found")
    probabilities: dict[str, float] | None
    probabilities_pose_model: dict[str, float] | None = None
    probabilities_image_model: dict[str, float] | None = None
    localization: str = Field(description="detector | pose | full_image")
    person_box: list[float] | None = Field(None, description="[x0, y0, x1, y1] normalised to the image")
    pose_detected: bool
    landmarks: list[list[float]] | None = Field(None, description="33 x [x, y, visibility, presence], normalised")
    timings_ms: dict[str, float]
    image: dict[str, Any]
    note: str


class Segment(BaseModel):
    index: int
    phase: str
    label: str
    display_label: str
    visual_class: str
    evidence: str
    start: float
    end: float
    duration: float
    n_frames: int
    confidence: float
    frames_without_person: float
    uncertain: bool
    state: str | None = None


class VideoAnalysis(BaseModel):
    video: dict[str, Any]
    decoder: str
    segments: list[Segment]
    raw_segments: list[Segment]
    summary: dict[str, Any]
    frames: list[dict[str, Any]] | None = None
    performance: dict[str, Any]
    phase_labels: dict[str, str]
    phases: list[str]
    notes: list[str]


class JobCreated(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: dict[str, Any]
    error: str | None = None
    created: float
    finished: float | None = None
    annotated_video_available: bool = False
    result: VideoAnalysis | None = None


class ErrorResponse(BaseModel):
    detail: str
