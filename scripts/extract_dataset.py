#!/usr/bin/env python
"""Locate ``Salat-Postures-Dataset-main`` (folder or ZIP) and extract it safely.

Usage::

    python scripts/extract_dataset.py                       # auto-discover
    python scripts/extract_dataset.py --zip path/to/Salat-Postures-Dataset-main.zip
    python scripts/extract_dataset.py --zip ... --overwrite

Writes an extraction report to ``artifacts/reports/extraction_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salat_vision.config import get_config  # noqa: E402
from salat_vision.data.extract import DATASET_NAME, discover_dataset, inspect_zip, safe_extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=None, help="Path to the dataset ZIP archive")
    ap.add_argument("--search", type=Path, action="append", default=[], help="Extra folder to search")
    ap.add_argument("--overwrite", action="store_true", help="Re-extract even if the folder exists")
    args = ap.parse_args()

    cfg = get_config()
    raw_dir = cfg.path("raw_dir")
    dataset_dir = cfg.path("dataset_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.path("artifacts_dir") / "reports" / "extraction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = args.zip or cfg.path("dataset_zip")
    if zip_path is None:
        if dataset_dir.is_dir() and any(dataset_dir.rglob("*.xml")) and not args.overwrite:
            print(f"Dataset already extracted at {dataset_dir}")
            return 0
        found = discover_dataset(args.search)
        print("Discovery:", json.dumps(found, indent=2))
        if found["zips"]:
            zip_path = Path(found["zips"][0])
        elif found["directories"]:
            print(f"Found an extracted dataset folder: {found['directories'][0]} — point SALAT_DATASET_DIR at it.")
            return 0
        else:
            print("No dataset folder or ZIP found. Pass --zip.", file=sys.stderr)
            return 1

    report = inspect_zip(zip_path)
    print(f"Archive: {zip_path}")
    print(f"  entries={report.n_entries} files={report.n_files} uncompressed={report.uncompressed_bytes/1e6:.2f} MB")
    print(f"  extensions={report.extensions}")
    print(f"  common_prefix='{report.common_prefix}' safe={report.is_safe}")
    if report.skipped_non_data:
        print(f"  skipping non-data files: {report.skipped_non_data}")
    target, _ = safe_extract(zip_path, raw_dir, dataset_name=DATASET_NAME, overwrite=args.overwrite)
    n_files = sum(1 for p in target.rglob("*") if p.is_file())
    print(f"Extracted to {target} ({n_files} files)")
    payload = report.to_dict() | {"extracted_to": str(target.relative_to(cfg.root)) if target.is_relative_to(cfg.root) else str(target),
                                  "files_on_disk": n_files}
    payload["path"] = Path(payload["path"]).name  # do not record private absolute upload paths
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
