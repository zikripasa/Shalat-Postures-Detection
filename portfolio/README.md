# Salat Vision — portfolio deliverables

Everything here is generated from the project's real artefacts (`artifacts/metrics/*`), so numbers can be refreshed by
re-running the scripts instead of editing documents by hand.

| Deliverable | File | Status |
|---|---|---|
| Technical report (PDF, 12 pages, selectable text) | `report/Salat_Vision_Technical_Report.pdf` | done, validated |
| Portfolio deck (PPTX, 14 slides, 16:9, editable) | `presentation/Salat_Vision_Portfolio_Deck.pptx` | done, validated + visually reviewed |
| Visual assets (charts, diagrams, screenshots) | `assets/*.png`, `assets/application_screenshots/` | done |
| Chart source data (CSV/JSON) | `assets/data/` | done |
| Editable diagram sources | `assets/diagrams/*.svg` (vector) and `*.mmd` (Mermaid) | done |
| Brand font (Plus Jakarta Sans, SIL OFL) | `assets/fonts/` | included |
| Canva versions | see [Canva](#canva) | partially done — read before using |

## Personalise

Edit `report/report_config.json` (author, role, contact, repository link — placeholders by default; nothing about you
was invented), then rebuild:

```bash
python portfolio/report/build_report.py                                   # PDF
cd portfolio/presentation && npm ci && NODE_PATH=$PWD/node_modules node build_deck.cjs   # PPTX
```

The deck reads the same `report_config.json`. To apply the deck theme exactly as delivered, set
`PPTX_APPLY_THEME=<path to apply_theme.js>` (the build works without it; colours are also set explicitly).

## Update charts after re-running experiments

```bash
python scripts/run_experiments.py && python scripts/train_final.py
python scripts/benchmark.py && python scripts/benchmark_temporal.py --select
python scripts/make_figures.py          # rewrites assets/*.png and assets/data/*
python portfolio/report/build_report.py
```

`assets/data/deck_data.json` drives every native chart, table and timeline in the PPTX; regenerate it with
`python scripts/export_deck_data.py` (or edit the values) and rebuild the deck. In PowerPoint the native charts are editable
(right-click → *Edit data*).

## Fonts

Headings use **Plus Jakarta Sans** (OFL, files in `assets/fonts/`); deck body text uses **Calibri** (ships with Office).
Install the Plus Jakarta Sans TTFs before editing, or PowerPoint substitutes a fallback for the headings (layouts
leave slack, so nothing should overflow).

## Canva

What was done through the Canva integration (all IDs/links below were returned by Canva itself):

1. **Uploaded to your Canva account** (via Canva's upload tool):
   - deck PDF `Salat_Vision_Portfolio_Deck.pdf` — file id `oAHbUyB3Xe0`
   - report PDF `Salat_Vision_Technical_Report.pdf` — file id `oAHbU7ncngo`
   - 8 real figures as images: model comparison `MAHbU5hLqZY`, confusion matrix `MAHbU0XhwDE`, posture timeline
     `MAHbU1qO3oo`, temporal benchmark `MAHbUyNjApM`, system architecture `MAHbU9J5kuU`, results screenshot
     `MAHbU2nDGqw`, CPU latency `MAHbU_GVTz8`, class distribution `MAHbUw9KjMs`.
   Canva's upload tool rejects `.pptx` ("Unsupported file format PPTX"), so the deck went up as PDF.
2. **Two AI-generated Canva presentations** were created from that content with Canva's design generator and then
   audited against the verified results:

| Canva design | Link | Audit result | Use? |
|---|---|---|---|
| Draft v1 (`DAHbU-wf6F4`) | https://canva.link/16ljig3lz8d8jt0 | Canva **invented** a confusion matrix (e.g. "86/61/91/75") and the claim "bowing → standing is the main ambiguity" (the real main confusion is bowing → sitting); wrong method details ("17×2" landmarks, "median smoothing", "MediaPipe-style", "on-device"); a mock timeline with made-up times; AI figures of people | **No — delete it** |
| Draft v2 (`DAHbU3iKMxE`) | https://canva.link/p0f8sj0x44778ss | No invented numbers, but image frames are **empty placeholders** ("Attached … chart"), most metrics were dropped, and splits are described as "person-grouped" (they are grouped by capture session + near-duplicates) | Only after the fixes below |

**Finishing draft v2** needs edits inside Canva: put the uploaded images into the eight placeholder frames, restore
the key numbers (macro-F1 0.80 [0.68–0.90], per-class F1 0.94 / 0.67 / 0.91 / 0.67, 0.68 on whole images, 84 ms per
frame, 72 tests, temporal F1@50 71 vs 26), replace "person-grouped" with "grouped by capture session". Canva's edit
tool requires your explicit approval before it saves changes, so these were **not** applied automatically.

**Faithful editable versions (recommended)** — the same content as the PDF/PPTX, no regeneration:

- *Presentation:* in Canva open **Uploads → Upload files** and upload `presentation/Salat_Vision_Portfolio_Deck.pptx`,
  then open it. Per Canva Help, text and shapes stay editable; PowerPoint **charts and tables do not import as
  editable** elements, so keep chart edits in PowerPoint (or swap in the PNGs from `assets/`).
- *Report / document:* open the already-uploaded report PDF from **Uploads** in the Canva editor; Canva converts it
  into an editable design (text is real text, images replaceable; some fonts may be substituted).

No Canva export was produced, because neither generated draft is accurate enough to export as a final file.

## Privacy

All images of people derived from the dataset (sample grid, demo clip, screenshots) have faces blurred. The raw
dataset is not redistributed.
