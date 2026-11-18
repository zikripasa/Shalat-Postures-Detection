"""API settings (environment variables prefixed with ``SALAT_API_``)."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from salat_vision.config import get_config


def _cfg_api(key: str, default):
    try:
        return get_config()["api"].get(key, default)
    except Exception:  # pragma: no cover
        return default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SALAT_API_", env_file=None, extra="ignore")

    model_dir: Path | None = Field(default=None, description="Model bundle directory (default: models/salat_vision_v1)")
    max_image_mb: float = Field(default_factory=lambda: float(_cfg_api("max_image_mb", 10)))
    max_video_mb: float = Field(default_factory=lambda: float(_cfg_api("max_video_mb", 150)))
    allowed_origins: list[str] = Field(default_factory=lambda: list(_cfg_api("allowed_origins", ["http://localhost:5173"])))
    job_ttl_minutes: float = Field(default_factory=lambda: float(_cfg_api("job_ttl_minutes", 30)))
    max_concurrent_jobs: int = Field(default_factory=lambda: int(_cfg_api("max_concurrent_jobs", 2)))
    max_queued_jobs: int = 8
    tmp_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "salat_vision_api")
    load_model_on_startup: bool = True

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json

                return json.loads(v)
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
