"""Dataset discovery and *safe* ZIP extraction.

Nothing inside the archive is ever executed. Extraction refuses absolute paths,
``..`` traversal, symbolic links and suspicious compression ratios, skips file
types that are not dataset content, and strips redundant parent folders so the
dataset always lands at ``<raw_dir>/Salat-Postures-Dataset-main/``.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

DATASET_NAME = "Salat-Postures-Dataset-main"

#: Only these file types are extracted; everything else is reported and skipped.
DATA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mpo", ".bmp", ".webp", ".xml", ".json", ".csv", ".txt", ".md",
                   ".mp4", ".mov", ".avi", ".mkv", ".webm"}

MAX_TOTAL_BYTES = 20 * 1024**3  # 20 GiB
MAX_FILES = 500_000
MAX_COMPRESSION_RATIO = 200.0  # zip-bomb guard (per file)


@dataclass
class ZipReport:
    path: str
    n_entries: int
    n_files: int
    n_dirs: int
    compressed_bytes: int
    uncompressed_bytes: int
    extensions: dict[str, int]
    top_level: list[str]
    common_prefix: str
    suspicious_paths: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    duplicate_names: list[str] = field(default_factory=list)
    high_ratio_entries: list[str] = field(default_factory=list)
    skipped_non_data: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not (self.suspicious_paths or self.symlinks or self.high_ratio_entries)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_safe"] = self.is_safe
        return d


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _is_suspicious(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith(("/", "\\")) or "\\" in name or ":" in parts[0]:
        return True
    return any(part == ".." for part in parts)


def _common_dir_prefix(names: list[str]) -> str:
    """Longest directory prefix shared by every entry (e.g. ``'a/b/'``)."""
    split = [PurePosixPath(n).parts for n in names if n and not n.endswith("/")]
    if not split:
        return ""
    prefix: list[str] = []
    for parts in zip(*[p[:-1] for p in split]):
        if all(x == parts[0] for x in parts):
            prefix.append(parts[0])
        else:
            break
    return "/".join(prefix) + ("/" if prefix else "")


def inspect_zip(zip_path: str | Path) -> ZipReport:
    """List an archive without extracting it and flag extraction risks."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    names = [i.filename for i in infos]
    files = [i for i in infos if not i.is_dir()]
    exts = collections.Counter(Path(i.filename).suffix.lower() or "<none>" for i in files)
    top = sorted({PurePosixPath(n).parts[0] for n in names if n})
    high_ratio = [
        i.filename for i in files
        if i.compress_size > 0 and i.file_size / max(i.compress_size, 1) > MAX_COMPRESSION_RATIO and i.file_size > 10 * 1024**2
    ]
    return ZipReport(
        path=str(zip_path),
        n_entries=len(infos),
        n_files=len(files),
        n_dirs=len(infos) - len(files),
        compressed_bytes=sum(i.compress_size for i in infos),
        uncompressed_bytes=sum(i.file_size for i in infos),
        extensions=dict(sorted(exts.items())),
        top_level=top,
        common_prefix=_common_dir_prefix(names),
        suspicious_paths=[n for n in names if _is_suspicious(n)],
        symlinks=[i.filename for i in infos if _is_symlink(i)],
        duplicate_names=[n for n, c in collections.Counter(names).items() if c > 1],
        high_ratio_entries=high_ratio,
        skipped_non_data=[i.filename for i in files if Path(i.filename).suffix.lower() not in DATA_EXTENSIONS],
    )


def _strip_prefix(report: ZipReport) -> str:
    """Parent folders to strip so the dataset folder itself is the extraction root.

    ``Salat-Postures-Dataset-main/...``           -> strip ``Salat-Postures-Dataset-main/``
    ``outer/Salat-Postures-Dataset-main/...``     -> strip ``outer/Salat-Postures-Dataset-main/``
    ``Sample-images-with-annotations/...`` (flat) -> strip nothing
    """
    prefix = report.common_prefix
    if not prefix:
        return ""
    parts = PurePosixPath(prefix).parts
    if DATASET_NAME in parts:
        return "/".join(parts[: parts.index(DATASET_NAME) + 1]) + "/"
    # A single wrapper folder with another name: keep its content only if it wraps
    # the whole archive; otherwise do not strip anything.
    return prefix if len(report.top_level) == 1 and len(parts) >= 1 and parts[0] == report.top_level[0] else ""


def safe_extract(
    zip_path: str | Path,
    raw_dir: str | Path,
    *,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> tuple[Path, ZipReport]:
    """Safely extract ``zip_path`` to ``raw_dir/dataset_name``.

    Returns the dataset directory and the pre-extraction report. Raises
    :class:`ValueError` when the archive is unsafe.
    """
    zip_path = Path(zip_path)
    report = inspect_zip(zip_path)
    if not report.is_safe:
        raise ValueError(f"Refusing to extract unsafe archive: {json.dumps(report.to_dict(), indent=2)[:2000]}")
    if report.uncompressed_bytes > MAX_TOTAL_BYTES or report.n_files > MAX_FILES:
        raise ValueError("Archive exceeds configured size/file-count limits")

    raw_dir = Path(raw_dir).resolve()
    target = raw_dir / dataset_name
    if target.exists():
        if not overwrite:
            return target, report
        shutil.rmtree(target)
    tmp_target = raw_dir / f".{dataset_name}.partial"
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    tmp_target.mkdir(parents=True)

    strip = _strip_prefix(report)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or _is_symlink(info):
                continue
            name = info.filename
            if Path(name).suffix.lower() not in DATA_EXTENSIONS:
                continue  # reported in report.skipped_non_data
            rel = name[len(strip):] if strip and name.startswith(strip) else name
            if not rel:
                continue
            dest = (tmp_target / rel).resolve()
            if tmp_target.resolve() not in dest.parents:
                raise ValueError(f"Path escapes extraction directory: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.chmod(dest, 0o644)  # never keep executable bits from the archive
            written += 1
            if dest.stat().st_size != info.file_size:
                raise ValueError(f"Size mismatch after extracting {name}")
    tmp_target.rename(target)
    expected = report.n_files - len(report.skipped_non_data)
    if written != expected:
        raise ValueError(f"Extracted {written} files, expected {expected}")
    return target, report


def discover_dataset(extra_roots: list[str | Path] | None = None, dataset_name: str = DATASET_NAME) -> dict[str, list[str]]:
    """Look for an extracted dataset folder or a ZIP archive in common locations."""
    roots: list[Path] = [Path(p) for p in (extra_roots or [])]
    for env in ("SALAT_DATASET_DIR", "SALAT_DATASET_ZIP"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]).expanduser())
    from salat_vision.config import find_project_root

    root = find_project_root()
    roots += [root, root / "data", root / "data" / "raw", Path.home() / "Downloads", Path("/mnt/data"),
              Path.home() / ".claude" / "uploads"]
    found_dirs: list[str] = []
    found_zips: list[str] = []
    seen: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() == ".zip":
            found_zips.append(str(base.resolve()))
            continue
        # shallow search (depth <= 3) to stay fast
        for depth_glob in ("*", "*/*", "*/*/*"):
            for p in base.glob(depth_glob):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                if p.is_dir() and p.name == dataset_name and ".partial" not in p.name:
                    found_dirs.append(str(rp))
                elif p.is_file() and p.suffix.lower() == ".zip" and "salat" in p.name.lower():
                    found_zips.append(str(rp))
    return {"directories": sorted(set(found_dirs)), "zips": sorted(set(found_zips))}
