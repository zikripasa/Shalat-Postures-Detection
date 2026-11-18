"""End-to-end video analysis: decode -> sample -> predict -> temporal decoding -> report.

Frames are decoded with FFmpeg (``imageio-ffmpeg``), which handles MP4/MOV/WebM (incl.
browser MediaRecorder output without duration metadata) and applies rotation metadata.
Frames are resampled to ``sample_fps`` by FFmpeg itself, so timestamps are exact.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from salat_vision.config import Config, get_config
from salat_vision.inference.temporal import DECODERS, TemporalConfig, summarize_segments
from salat_vision.labels import PHASE_LABELS_ID, PHASES, VISUAL_CLASSES

MAX_ANALYSIS_SIDE = 1280
# imageio-ffmpeg logs a warning whenever we deliberately rescale frames; it is expected here.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


class VideoError(ValueError):
    pass


def probe_video(path: str | Path) -> dict:
    """Read container metadata without decoding the whole file."""
    import imageio_ffmpeg

    try:
        gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        meta = next(gen)
        gen.close()
    except (RuntimeError, OSError, StopIteration) as exc:
        raise VideoError(f"Cannot decode video: {exc}") from exc
    w, h = meta.get("source_size") or meta.get("size") or (0, 0)
    duration = meta.get("duration")
    return {"width": int(w), "height": int(h), "fps": float(meta.get("fps") or 0.0),
            "duration": float(duration) if duration and math.isfinite(duration) else None}


def _scale_filter(w: int, h: int, max_side: int) -> tuple[str, tuple[int, int]]:
    s = min(1.0, max_side / max(w, h))
    ow, oh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    return f"scale={ow}:{oh}", (ow, oh)


def iter_frames(path: str | Path, fps: float, max_side: int = MAX_ANALYSIS_SIDE):
    """Yield (timestamp_seconds, RGB frame) at exactly ``fps`` frames per second."""
    import imageio_ffmpeg

    info = probe_video(path)
    if info["width"] <= 0 or info["height"] <= 0:
        raise VideoError("Video has no decodable picture stream")
    flt, (ow, oh) = _scale_filter(info["width"], info["height"], max_side)
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-vf", f"fps={fps},{flt}"])
    meta = next(gen)
    w, h = meta["size"]
    k = 0
    try:
        for raw in gen:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            yield k / fps, frame
            k += 1
    finally:
        gen.close()


def analyze_video(path: str | Path, predictor, cfg: Config | None = None, *, sample_fps: float | None = None,
                  decoder: str | None = None, annotate_path: str | Path | None = None,
                  progress: Callable[[dict], None] | None = None, keep_frames: bool = True) -> dict:
    cfg = cfg or get_config()
    inf = cfg["inference"]
    sample_fps = float(sample_fps or inf["sample_fps"])
    decoder = decoder or cfg["temporal"].get("decoder", "hmm")
    info = probe_video(path)
    if info["duration"] is not None and info["duration"] > float(inf["max_video_seconds"]):
        raise VideoError(f"Video is {info['duration']:.0f}s long; the limit is {inf['max_video_seconds']}s")
    expected = int(math.ceil((info["duration"] or 0) * sample_fps)) or None

    t_start = time.perf_counter()
    times, probs, frames = [], [], []
    stage_ms: dict[str, list[float]] = {}
    for i, (ts, rgb) in enumerate(iter_frames(path, sample_fps)):
        if ts > float(inf["max_video_seconds"]):
            raise VideoError(f"Video exceeds {inf['max_video_seconds']}s")
        pred = predictor.predict(rgb)
        times.append(ts)
        probs.append(pred.probs if pred.probs is not None else np.full(len(VISUAL_CLASSES), np.nan))
        for k, v in pred.timings_ms.items():
            stage_ms.setdefault(k, []).append(v)
        if keep_frames:
            frames.append({"t": round(ts, 3), **pred.to_dict(include_landmarks=True)})
        if progress and (i % 5 == 0):
            progress({"stage": "analyzing", "frames_done": i + 1, "frames_expected": expected,
                      "fraction": min(0.95, (i + 1) / expected) * 0.85 if expected else None})
    if not times:
        raise VideoError("No frames could be decoded from the video")
    analysis_seconds = time.perf_counter() - t_start
    P = np.vstack(probs)
    T = np.array(times)
    tc = TemporalConfig.from_cfg(cfg)
    decoded = DECODERS[decoder](P, T, tc)
    raw = DECODERS["raw"](P, T, tc)
    duration = info["duration"] or float(T[-1] + 1.0 / sample_fps)
    segments = decoded["segments"]
    # clip the last segment to the video duration
    if segments:
        segments[-1]["end"] = round(min(segments[-1]["end"], duration), 3)
        segments[-1]["duration"] = round(segments[-1]["end"] - segments[-1]["start"], 3)
    summary = summarize_segments(segments, video_seconds=duration)
    smoothed = decoded.get("smoothed")
    if keep_frames:
        for i, fr in enumerate(frames):
            fr["phase"] = decoded["phase_per_frame"][i]
            fr["raw_visual_class"] = fr.get("visual_class")
            if smoothed is not None and not np.isnan(smoothed[i]).any():
                fr["smoothed_probabilities"] = {c: round(float(p), 4) for c, p in zip(VISUAL_CLASSES, smoothed[i])}
    n = len(T)
    perf = {
        "frames_analyzed": n,
        "analysis_seconds": round(analysis_seconds, 3),
        "analysis_fps": round(n / analysis_seconds, 2) if analysis_seconds > 0 else None,
        "realtime_factor": round(duration / analysis_seconds, 2) if analysis_seconds > 0 else None,
        "mean_ms_per_frame": {k: round(float(np.mean(v)), 2) for k, v in stage_ms.items()},
        "p95_ms_per_frame": {k: round(float(np.percentile(v, 95)), 2) for k, v in stage_ms.items()},
    }
    result = {
        "video": {"duration_seconds": round(duration, 3), "width": info["width"], "height": info["height"],
                  "source_fps": info["fps"], "sample_fps": sample_fps},
        "decoder": decoder,
        "segments": segments,
        "raw_segments": raw["segments"],
        "summary": summary,
        "frames": frames if keep_frames else None,
        "performance": perf,
        "phase_labels": PHASE_LABELS_ID,
        "phases": PHASES,
        "notes": [
            "Postur dikenali secara visual dalam empat kelas. Iktidal, Duduk di antara dua sujud, dan Tasyahud "
            "disimpulkan dari urutan postur, bukan dilihat langsung.",
            "Skor adalah keyakinan model, bukan probabilitas terkalibrasi, dan bukan penilaian sah atau tidaknya shalat.",
        ],
    }
    if annotate_path:
        if progress:
            progress({"stage": "rendering", "fraction": 0.9})
        t_ann = time.perf_counter()
        render_annotated_video(path, annotate_path, T, P, decoded, frames if keep_frames else None, sample_fps)
        perf["annotation_seconds"] = round(time.perf_counter() - t_ann, 3)
    if progress:
        progress({"stage": "done", "fraction": 1.0})
    return result


# ------------------------------------------------------------------------------ annotation
def render_annotated_video(src: str | Path, dst: str | Path, T: np.ndarray, P: np.ndarray, decoded: dict,
                           frames: list[dict] | None, sample_fps: float, out_fps: float = 15.0, max_side: int = 960) -> Path:
    """Re-encode the video (H.264, browser-playable) with posture overlays and a timeline bar."""
    import imageio_ffmpeg

    from salat_vision.features.pose import POSE_EDGES
    from salat_vision.viz import PHASE_COLORS

    dst = Path(dst)
    phases = decoded["phase_per_frame"]
    segs = decoded["segments"]
    info = probe_video(src)
    flt, (ow, oh) = _scale_filter(info["width"], info["height"], max_side)
    bar_h = max(28, oh // 18)
    gen = imageio_ffmpeg.read_frames(str(src), pix_fmt="rgb24", output_params=["-vf", f"fps={out_fps},{flt}"])
    meta = next(gen)
    w, h = meta["size"]
    writer = imageio_ffmpeg.write_frames(str(dst), (w, h + bar_h), fps=out_fps, codec="libx264", pix_fmt_in="rgb24",
                                         pix_fmt_out="yuv420p", quality=None, bitrate=None, macro_block_size=2,
                                         output_params=["-preset", "veryfast", "-crf", "26", "-movflags", "+faststart"])
    writer.send(None)
    total = max(float(T[-1]) + 1.0 / sample_fps, 1e-6)

    def hex2rgb(hx):
        return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))

    # static timeline strip
    strip = np.full((bar_h, w, 3), 245, np.uint8)
    for s in segs:
        x0 = int(s["start"] / total * w)
        x1 = max(x0 + 1, int(s["end"] / total * w))
        col = (200, 198, 190) if s["uncertain"] else hex2rgb(PHASE_COLORS[s["phase"]])
        strip[4:bar_h - 4, x0:x1] = col
        strip[4:bar_h - 4, max(x0 - 1, 0):x0 + 1] = 245  # 2px surface gap between segments
    try:
        k = 0
        for raw in gen:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            ts = k / out_fps
            i = int(min(len(T) - 1, max(0, round(ts * sample_fps))))
            phase = phases[i]
            seg = next((s for s in segs if s["start"] <= ts < s["end"]), segs[-1] if segs else None)
            color = hex2rgb(PHASE_COLORS[phase])
            fr = frames[i] if frames else None
            if fr and fr.get("person_box"):
                x0, y0, x1, y1 = fr["person_box"]
                cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), color, 2, cv2.LINE_AA)
            if fr and fr.get("landmarks"):
                lm = np.array(fr["landmarks"])
                for a, b in POSE_EDGES:
                    if lm[a, 2] > 0.35 and lm[b, 2] > 0.35:
                        cv2.line(frame, (int(lm[a, 0] * w), int(lm[a, 1] * h)), (int(lm[b, 0] * w), int(lm[b, 1] * h)),
                                 (255, 255, 255), 2, cv2.LINE_AA)
            label = (seg["display_label"] if seg else PHASE_LABELS_ID[phase])
            conf = P[i].max() if not np.isnan(P[i]).any() else None
            text = f"{label}" + (f"  {conf:.0%}" if conf is not None else "")
            (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, 10), (10 + tw + 18, 10 + th_ + 18), (15, 31, 61), -1)
            cv2.rectangle(frame, (10, 10), (16, 10 + th_ + 18), color, -1)
            cv2.putText(frame, text, (22, 10 + th_ + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            bar = strip.copy()
            cx = int(ts / total * w)
            bar[:, max(0, cx - 1):cx + 1] = (15, 31, 61)
            writer.send(np.ascontiguousarray(np.vstack([frame, bar])))
            k += 1
    finally:
        gen.close()
        writer.close()
    return dst
