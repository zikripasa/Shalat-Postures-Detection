#!/usr/bin/env python
"""Assemble a *synthetic* demo video from dataset images in prayer order.

Two rakaat + final tasyahud, 0.4 s cross-fades, faces blurred, 1280x720 H.264.
The dataset contains no real prayer videos, so this clip only demonstrates the video
pipeline (decoding, sampling, temporal decoding, annotation, UI). The deployed model was
trained on these images, so results on this clip are NOT an accuracy estimate.

Outputs (git-ignored, regenerate locally):
  artifacts/demo/demo_sequence.mp4
  artifacts/demo/demo_sequence.json   ground-truth phase timeline of the clip
  artifacts/demo/stills/still_<t>.jpg single face-blurred frames (photo mode, end-to-end test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from salat_vision.config import get_config  # noqa: E402
from salat_vision.data.dataset import build_index  # noqa: E402
from salat_vision.data.voc import load_image_rgb  # noqa: E402
from salat_vision.privacy import Anonymizer  # noqa: E402

PLAN = [
    ("berdiri", "ck2upgs2ttpwm07575jjfctev", 5.0), ("rukuk", "ck2ygl8aj78s5072595cehoma", 3.0),
    ("iktidal", "qiam(94)", 2.5), ("sujud", "ck2osjowcmuaf07252lbqno9g", 3.0),
    ("duduk_antara_sujud", "IMG-20191019-WA0062", 2.5), ("sujud", "ck34yexrazmie0757n3kns0l8", 3.0),
    ("berdiri", "ck35sqichhe0n0757m66gwoty", 4.0), ("rukuk", "ruku109", 3.0),
    ("iktidal", "ck2uf5w5a6jk307257lnccvnf", 2.5), ("sujud", "ck2orc96s0em40944v7k1rpc5", 3.0),
    ("duduk_antara_sujud", "SittingMO56", 2.5), ("sujud", "ck2oufih0wy2m0794frm9jrhd", 3.0),
    ("tasyahud", "ck2or3u694ynn0757ptzgbv8t", 6.0),
]
W, H, FPS, FADE = 1280, 720, 25, 0.4
BG = (14, 27, 51)
STILLS_AT = (6.5, 11.5)  # seconds into the clip (Rukuk, first Sujud), saved as JPEG frames


def fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(W / w, (H - 40) / h)
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = BG
    y0, x0 = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    cv2.putText(canvas, "Video demo sintetis - gambar dataset, wajah disamarkan", (16, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 205, 215), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    import imageio_ffmpeg

    cfg = get_config()
    idx = build_index(cfg=cfg)
    inst = idx.labeled_instances
    anon = Anonymizer(cfg.path("pretrained_dir"))
    stills = []
    for phase, image_id, _ in PLAN:
        rows = inst[inst["image_id"] == image_id]
        img = load_image_rgb(idx.dataset_dir / rows["file"].iloc[0])
        boxes = [(r.xmin, r.ymin, r.xmax, r.ymax) for r in rows.itertuples()]
        blurred, _ = anon(img, boxes)
        stills.append(fit(blurred))
    anon.close()

    out_dir = cfg.path("artifacts_dir") / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "demo_sequence.mp4"
    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2, output_params=["-preset", "medium", "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    stills_dir = out_dir / "stills"
    stills_dir.mkdir(exist_ok=True)
    want = {int(round(s * FPS)): s for s in STILLS_AT}
    timeline, t, frame_no = [], 0.0, 0
    fade_n = int(FADE * FPS)
    for k, ((phase, image_id, sec), still) in enumerate(zip(PLAN, stills)):
        n = int(round(sec * FPS))
        for i in range(n):
            frame = still
            if k > 0 and i < fade_n:  # cross-fade from the previous still
                a = (i + 1) / (fade_n + 1)
                frame = cv2.addWeighted(still, a, stills[k - 1], 1 - a, 0)
            writer.send(np.ascontiguousarray(frame))
            if frame_no in want:
                cv2.imwrite(str(stills_dir / f"still_{want[frame_no]}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame_no += 1
        timeline.append({"phase": phase, "image_id": image_id, "start": round(t, 3), "end": round(t + n / FPS, 3)})
        t += n / FPS
    writer.close()
    meta = {"description": __doc__.strip(), "fps": FPS, "size": [W, H], "duration": round(t, 3), "timeline": timeline}
    (out_dir / "demo_sequence.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{path.relative_to(cfg.root)}: {t:.1f}s, {path.stat().st_size / 1e6:.1f} MB")
    print(f"{stills_dir.relative_to(cfg.root)}: {', '.join(sorted(f.name for f in stills_dir.glob('still_*.jpg')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
