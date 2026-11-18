#!/usr/bin/env python
"""Train the selected pipeline on all labelled data and export ``models/<bundle_name>/``.

The selection lives in ``configs/default.yaml`` (section ``final_model``) and was made
from the cross-validated comparison produced by ``scripts/run_experiments.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salat_vision.config import get_config  # noqa: E402
from salat_vision.export import train_and_export  # noqa: E402


def main() -> int:
    out = train_and_export(get_config())
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
