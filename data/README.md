# data/

| Folder | Content | In git? |
|---|---|---|
| `raw/Salat-Postures-Dataset-main/` | the dataset exactly as extracted from the ZIP (never modified) | no — re-create with `scripts/extract_dataset.py` |
| `processed/` | derived caches (if any) | no |
| `splits/` | `cv_splits.json` (3 × stratified group 5-fold, by instance id) and `instance_groups.csv` | yes |

## Getting the data

```bash
python scripts/extract_dataset.py --zip /path/to/Salat-Postures-Dataset-main.zip
```

The extractor lists the archive first, refuses absolute paths, `..` traversal, symlinks and suspicious compression
ratios, extracts only data file types (images, XML, text, video), strips redundant parent folders and verifies sizes.
A record is written to `artifacts/reports/extraction_report.json`.

Use a different location with `SALAT_DATASET_DIR=/path/to/Salat-Postures-Dataset-main` (folder) or
`SALAT_DATASET_ZIP=/path/to/archive.zip`.

## The full dataset

The provided archive is the GitHub **sample** (41 images). The README inside it points to the full release
(1,904 images, 2,156 instances) on Kaggle (`riotulab/salat-postures`). The pipeline is dataset-agnostic: point
`SALAT_DATASET_DIR` at the full VOC folder, re-run `scripts/audit_dataset.py`, review `configs/session_groups.yaml`
(it lists sessions identified in the sample only) and re-run the experiments.

## Privacy

The images show identifiable people. They are not committed to this repository; every figure, screenshot and the demo
clip produced from them blurs faces (`src/salat_vision/privacy.py`).
