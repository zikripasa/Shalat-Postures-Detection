"""Pascal VOC annotation parsing and image loading with EXIF handling."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image, ImageOps

from salat_vision.labels import normalize_label


@dataclass
class VocObject:
    raw_label: str
    label: str | None
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    truncated: int = 0
    difficult: int = 0
    box_format: str = "bndbox"

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class VocAnnotation:
    xml_path: Path
    filename_field: str | None
    folder_field: str | None
    source_folder: str | None  # last directory of the <path> field (no personal path prefix)
    width: int | None
    height: int | None
    depth: int | None
    objects: list[VocObject] = field(default_factory=list)


def _text(node: ET.Element | None, tag: str) -> str | None:
    if node is None:
        return None
    child = node.find(tag)
    return child.text.strip() if child is not None and child.text is not None else None


def _num(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_voc(xml_path: str | Path) -> VocAnnotation:
    """Parse one Pascal VOC XML file. Unknown labels are kept with ``label=None``."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    path_field = _text(root, "path")
    source_folder = None
    if path_field:
        # Windows or POSIX path; keep only the immediate parent folder name.
        parts = PureWindowsPath(path_field).parts if "\\" in path_field else Path(path_field).parts
        source_folder = parts[-2] if len(parts) >= 2 else None
    objects: list[VocObject] = []
    for obj in root.findall("object"):
        raw = _text(obj, "name") or ""
        bb = obj.find("bndbox")
        coords = [_num(_text(bb, k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        fmt = "bndbox"
        if any(c is None for c in coords):
            # Some Labelbox exports store a 4-point <polygon> (x1..xN / y1..yN) instead
            # of a <bndbox>; use its axis-aligned bounding rectangle.
            poly = obj.find("polygon")
            if poly is None:
                continue
            xs = [_num(c.text) for c in poly if re.fullmatch(r"x\d+", c.tag)]
            ys = [_num(c.text) for c in poly if re.fullmatch(r"y\d+", c.tag)]
            xs, ys = [v for v in xs if v is not None], [v for v in ys if v is not None]
            if len(xs) < 2 or len(ys) < 2:
                continue
            coords = [min(xs), min(ys), max(xs), max(ys)]
            fmt = "polygon"
        objects.append(
            VocObject(
                raw_label=raw,
                label=normalize_label(raw),
                xmin=coords[0], ymin=coords[1], xmax=coords[2], ymax=coords[3],
                truncated=int(_num(_text(obj, "truncated")) or 0),
                difficult=int(_num(_text(obj, "difficult")) or 0),
                box_format=fmt,
            )
        )
    w, h, d = (_num(_text(size, k)) for k in ("width", "height", "depth"))
    return VocAnnotation(
        xml_path=xml_path,
        filename_field=_text(root, "filename"),
        folder_field=_text(root, "folder"),
        source_folder=source_folder,
        width=int(w) if w else None,
        height=int(h) if h else None,
        depth=int(d) if d else None,
        objects=objects,
    )


def load_image_rgb(path: str | Path, *, exif_transpose: bool = False) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array.

    ``exif_transpose`` defaults to ``False`` because the VOC boxes in this dataset
    were drawn on the *stored* pixel grid (verified per image in the audit by
    comparing the XML ``<size>`` with the stored and EXIF-rotated sizes).
    """
    with Image.open(path) as im:
        if exif_transpose:
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.asarray(im)


_SERIES_RE = re.compile(r"^(?P<prefix>[A-Za-z_\-]+?)[\s_\-]*\(?(?P<num>\d+)\)?$")


def filename_series(stem: str) -> tuple[str, int | None]:
    """Split a file stem into an alphabetic series prefix and a frame/sequence number.

    ``'sujud31'`` -> ``('sujud', 31)``, ``'qiam(178)'`` -> ``('qiam', 178)``,
    ``'SittingMo58'`` -> ``('sittingmo', 58)``, ``'ck2or3u6...'`` -> ``('ck2or3u6...', None)``.
    """
    m = _SERIES_RE.match(stem.strip())
    if not m:
        return stem.lower(), None
    return m.group("prefix").lower(), int(m.group("num"))
