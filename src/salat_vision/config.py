"""Central configuration.

All scripts, notebooks, tests and the API read the same YAML file
(``configs/default.yaml``) through :func:`get_config`, so paths and
hyper-parameters are never hard-coded in more than one place.

Resolution order (later wins):

1. ``configs/default.yaml``
2. an optional extra YAML file passed to :func:`load_config` (or ``SALAT_CONFIG``)
3. environment variables listed in :data:`ENV_OVERRIDES`
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent

#: Environment variable -> (section, key) of the YAML configuration.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "SALAT_DATASET_ZIP": ("paths", "dataset_zip"),
    "SALAT_DATASET_DIR": ("paths", "dataset_dir"),
    "SALAT_RAW_DIR": ("paths", "raw_dir"),
    "SALAT_MODELS_DIR": ("paths", "models_dir"),
    "SALAT_PRETRAINED_DIR": ("paths", "pretrained_dir"),
    "SALAT_ARTIFACTS_DIR": ("paths", "artifacts_dir"),
    "SALAT_SEED": ("project", "seed"),
    "SALAT_SAMPLE_FPS": ("inference", "sample_fps"),
    "SALAT_UNCERTAINTY_THRESHOLD": ("inference", "uncertainty_threshold"),
}


def find_project_root() -> Path:
    """Locate the project root (the directory containing ``configs/default.yaml``)."""
    env = os.environ.get("SALAT_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for start in (PACKAGE_DIR, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "configs" / "default.yaml").is_file():
                return candidate
    return Path.cwd().resolve()


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Thin, read-mostly wrapper around the configuration dictionary."""

    def __init__(self, data: dict[str, Any], root: Path):
        self.data = data
        self.root = root

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.data.get(section, {}).get(key, default)

    def path(self, key: str) -> Path | None:
        """Return ``paths.<key>`` as an absolute :class:`Path` (``None`` if unset)."""
        value = self.data.get("paths", {}).get(key)
        if value in (None, ""):
            return None
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def seed(self) -> int:
        return int(self.get("project", "seed", 42))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def load_config(extra_yaml: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    root = find_project_root()
    default_path = root / "configs" / "default.yaml"
    if default_path.is_file():
        data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    else:  # pragma: no cover - only when the package is used outside the repo
        data = {}
    extra_yaml = extra_yaml or os.environ.get("SALAT_CONFIG")
    if extra_yaml:
        data = _deep_merge(data, yaml.safe_load(Path(extra_yaml).read_text(encoding="utf-8")) or {})
    for env_name, (section, key) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            data.setdefault(section, {})[key] = yaml.safe_load(raw)
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, root)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration (call ``get_config.cache_clear()`` after changing env vars)."""
    return load_config()
