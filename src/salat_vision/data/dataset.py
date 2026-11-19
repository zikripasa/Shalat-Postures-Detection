"""Build image- and instance-level tables from a Pascal VOC dataset folder."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from salat_vision.config import Config, get_config
from salat_vision.data.voc import filename_series, parse_voc

_EXIF_ORIENTATION_TAG = 274


@dataclass
class DatasetIndex:
    dataset_dir: Path
    images: pd.DataFrame
    instances: pd.DataFrame
    orphan_annotations: pd.DataFrame
    issues: list[dict] = field(default_factory=list)

    @property
    def labeled_instances(self) -> pd.DataFrame:
        return self.instances[self.instances["valid"]].reset_index(drop=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_kind_from_name(stem: str) -> str:
    """Heuristic capture-source category derived from the file name."""
    if re.fullmatch(r"c[a-z0-9]{24}", stem):
        return "web (Labelbox export)"
    if re.match(r"IMG-\d{8}-WA\d+", stem):
        return "messaging app photo"
    if stem.startswith("IMG_"):
        return "phone camera photo"
    if stem.isdigit():
        return "numbered photo"
    prefix, num = filename_series(stem)
    if num is not None:
        return f"named series ({prefix})"
    return "other"


def _read_path_tail(xml_path: Path) -> str | None:
    """Last two folders of the XML <path> field (drops user-specific prefixes)."""
    import xml.etree.ElementTree as ET

    node = ET.parse(xml_path).getroot().find("path")
    if node is None or not node.text:
        return None
    text = node.text.strip()
    parts = PureWindowsPath(text).parts if "\\" in text else Path(text).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else None


def build_index(dataset_dir: str | Path | None = None, cfg: Config | None = None) -> DatasetIndex:
    cfg = cfg or get_config()
    dataset_dir = Path(dataset_dir or cfg.path("dataset_dir"))
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}. Run scripts/extract_dataset.py first.")
    img_exts = {e.lower() for e in cfg.get("dataset", "image_extensions")}
    pad_min = float(cfg.get("dataset", "min_box_side_px", 16))

    image_files = sorted(p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts)
    xml_files = sorted(p for p in dataset_dir.rglob("*.xml") if p.is_file())
    xml_by_stem = {p.stem: p for p in xml_files}
    img_stems = {p.stem for p in image_files}

    image_rows: list[dict] = []
    inst_rows: list[dict] = []
    issues: list[dict] = []

    for img_path in image_files:
        stem = img_path.stem
        rel = img_path.relative_to(dataset_dir).as_posix()
        row: dict = {
            "image_id": stem,
            "file": rel,
            "ext": img_path.suffix.lower(),
            "bytes": img_path.stat().st_size,
            "sha256": _sha256(img_path),
            "source_kind": source_kind_from_name(stem),
        }
        prefix, num = filename_series(stem)
        row["series_prefix"], row["series_num"] = prefix, num
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                row["format"] = im.format
                row["mode"] = im.mode
                row["width"], row["height"] = im.size
                exif = im.getexif()
                row["exif_orientation"] = int(exif.get(_EXIF_ORIENTATION_TAG, 1)) if exif else 1
                rgb = im.convert("RGB")
                row["phash"] = str(imagehash.phash(rgb))
                arr = np.asarray(rgb.resize((64, 64)))
                row["mean_brightness"] = float(arr.mean())
            row["readable"] = True
        except Exception as exc:  # corrupted / unreadable file
            row.update(readable=False, format=None, mode=None, width=None, height=None, exif_orientation=None,
                       phash=None, mean_brightness=None)
            issues.append({"file": rel, "issue": "unreadable image", "detail": str(exc)})

        xml_path = xml_by_stem.get(stem)
        row["has_annotation"] = xml_path is not None
        row["xml_file"] = xml_path.relative_to(dataset_dir).as_posix() if xml_path else None
        ann = parse_voc(xml_path) if xml_path else None
        row["xml_folder"] = ann.folder_field if ann else None
        row["xml_path_tail"] = _read_path_tail(xml_path) if xml_path else None
        row["xml_filename_field"] = ann.filename_field if ann else None
        row["xml_width"], row["xml_height"] = (ann.width, ann.height) if ann else (None, None)
        if ann and row["readable"]:
            stored = (row["width"], row["height"])
            swapped = (row["height"], row["width"])
            xml_size = (ann.width, ann.height)
            row["xml_size_matches_stored"] = xml_size == stored
            row["xml_size_matches_swapped"] = xml_size == swapped and stored != swapped
            if not row["xml_size_matches_stored"]:
                issues.append({"file": rel, "issue": "XML <size> differs from stored pixel size",
                               "detail": f"xml={xml_size} stored={stored}"})
            fn = (ann.filename_field or "").replace(" ", "")
            row["xml_filename_matches"] = Path(fn).stem == stem.replace(" ", "")
        else:
            row["xml_size_matches_stored"] = None
            row["xml_size_matches_swapped"] = None
            row["xml_filename_matches"] = None
        if xml_path is None:
            issues.append({"file": rel, "issue": "image without annotation", "detail": "excluded from supervised training"})

        labels: list[str] = []
        if ann is not None:
            W = row.get("width") or ann.width
            H = row.get("height") or ann.height
            for k, obj in enumerate(ann.objects):
                x0, y0 = max(0.0, obj.xmin), max(0.0, obj.ymin)
                x1, y1 = min(float(W), obj.xmax), min(float(H), obj.ymax)
                problems = []
                if obj.label is None:
                    problems.append(f"unknown label '{obj.raw_label}'")
                if (x1 - x0) < pad_min or (y1 - y0) < pad_min:
                    problems.append("box too small or empty")
                if obj.xmin < 0 or obj.ymin < 0 or obj.xmax > W or obj.ymax > H:
                    problems.append("box exceeds image bounds (clipped)")
                valid = obj.label is not None and (x1 - x0) >= pad_min and (y1 - y0) >= pad_min and row["readable"]
                inst_rows.append({
                    "instance_id": f"{stem}#{k}",
                    "image_id": stem,
                    "file": rel,
                    "raw_label": obj.raw_label,
                    "label": obj.label,
                    "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                    "box_w": x1 - x0, "box_h": y1 - y0,
                    "img_w": W, "img_h": H,
                    "rel_area": (x1 - x0) * (y1 - y0) / float(W * H) if W and H else None,
                    "aspect_hw": (y1 - y0) / max(x1 - x0, 1e-6),
                    "truncated": obj.truncated,
                    "difficult": obj.difficult,
                    "box_format": obj.box_format,
                    "n_objects_in_image": len(ann.objects),
                    "valid": bool(valid),
                    "issue": "; ".join(problems) if problems else None,
                })
                if problems:
                    issues.append({"file": rel, "issue": "annotation problem", "detail": "; ".join(problems)})
                if obj.label:
                    labels.append(obj.label)
        row["n_objects"] = len(labels)
        row["labels"] = ",".join(sorted(set(labels)))
        row["mixed_labels"] = len(set(labels)) > 1
        image_rows.append(row)

    orphans = []
    for stem, xml_path in sorted(xml_by_stem.items()):
        if stem in img_stems:
            continue
        ann = parse_voc(xml_path)
        orphans.append({
            "xml_file": xml_path.relative_to(dataset_dir).as_posix(),
            "filename_field": ann.filename_field,
            "xml_size": f"{ann.width}x{ann.height}",
            "labels": ",".join(o.raw_label for o in ann.objects),
            "n_objects": len(ann.objects),
        })
        issues.append({"file": xml_path.name, "issue": "annotation without image",
                       "detail": f"<filename>={ann.filename_field}; not paired (no evidence links it to another image)"})

    images = pd.DataFrame(image_rows)
    instances = pd.DataFrame(inst_rows)
    return DatasetIndex(dataset_dir=dataset_dir, images=images, instances=instances,
                        orphan_annotations=pd.DataFrame(orphans), issues=issues)
