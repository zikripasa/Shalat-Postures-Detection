"""Upload validation: streamed size limits, extension + magic-byte checks, temp files."""

from __future__ import annotations

import io
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_IMAGE_SIDE = 8000
CHUNK = 1 << 20


def _too_large(limit_mb: float) -> HTTPException:
    return HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds the {limit_mb:g} MB limit")


async def read_limited(upload: UploadFile, limit_mb: float) -> bytes:
    limit = int(limit_mb * 1024 * 1024)
    buf = io.BytesIO()
    while chunk := await upload.read(CHUNK):
        buf.write(chunk)
        if buf.tell() > limit:
            raise _too_large(limit_mb)
    return buf.getvalue()


def decode_image(data: bytes) -> "tuple[object, dict]":
    """Bytes -> upright RGB numpy array (EXIF orientation applied) + metadata."""
    import numpy as np

    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in {"JPEG", "PNG", "WEBP", "MPO"}:
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported image format: {fmt}")
            if max(im.size) > MAX_IMAGE_SIDE:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image larger than {MAX_IMAGE_SIDE}px")
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.asarray(im)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a readable image") from exc
    return arr, {"format": fmt, "width": int(arr.shape[1]), "height": int(arr.shape[0])}


def _video_magic_ok(head: bytes) -> bool:
    if len(head) < 12:
        return False
    if head[4:8] == b"ftyp":  # MP4 / MOV / M4V
        return True
    if head[:4] == b"\x1a\x45\xdf\xa3":  # Matroska / WebM
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    return False


async def save_video_upload(upload: UploadFile, limit_mb: float, tmp_dir: Path) -> Path:
    """Stream an uploaded video to a private temp file, enforcing size and type."""
    ext = Path(upload.filename or "").suffix.lower()
    if ext and ext not in VIDEO_EXTS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported video extension '{ext}'")
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if ctype and not (ctype.startswith("video/") or ctype in {"application/octet-stream"}):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported content type '{ctype}'")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(tmp_dir, 0o700)
    path = tmp_dir / f"upload_{uuid.uuid4().hex}{ext or '.bin'}"
    limit = int(limit_mb * 1024 * 1024)
    size = 0
    head = b""
    try:
        with open(path, "wb") as f:
            while chunk := await upload.read(CHUNK):
                if len(head) < 16:
                    head += chunk[:16]
                size += len(chunk)
                if size > limit:
                    raise _too_large(limit_mb)
                f.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
        if not _video_magic_ok(head):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a supported video container (MP4, MOV, WebM, MKV, AVI)")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def new_temp_path(tmp_dir: Path, suffix: str) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    os.close(fd)
    return Path(name)
