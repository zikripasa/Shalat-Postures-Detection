# Test and validation report

Environment: Linux x86_64, Python 3.12.3, Node 22, Chromium (Playwright 1.63), 4 vCPU, no GPU. Every item below was
actually executed; nothing is reported as passing because the code merely exists.

## Automated tests

| Suite | Command | Result |
|---|---|---|
| Python — extraction safety (traversal, symlinks, nested folders, non-data files) | `pytest tests/test_extract.py` | 5 passed |
| Python — labels, VOC + polygon parsing, leakage-free splits, dataset counts | `pytest tests/test_data.py` | 15 passed |
| Python — pose features, rules, fusion, letterbox, metrics with abstention, ECE | `pytest tests/test_features_models.py` | 11 passed |
| Python — HMM decoding, segment contiguity, uncertainty flags, edit/F1 metrics | `pytest tests/test_temporal.py` | 8 passed |
| Python — real bundle: checksums, tamper detection, image + video inference | `pytest tests/test_inference.py` | 5 passed |
| Python — FastAPI: 503 without model, 400/413/415 validation, job lifecycle, reports, annotated video, CORS | `pytest tests/test_api.py` | 11 passed |
| **Python total** | `pytest -q` | **55 passed** (10 s on an idle machine; 75 s in an earlier run under load) |
| Frontend unit (Vitest + Testing Library): format helpers, PhaseTimeline, pages with a real API fixture, API errors | `npm test` | **12 passed** |
| Frontend type-check (strict) | `npm run typecheck` | passed |
| Frontend production build | `npm run build` | passed (initial JS 302 kB / 96 kB gzip; charts lazy-loaded) |
| End-to-end (Playwright, real FastAPI + model + built app) | `npm run e2e` | **5 passed** — landing status, photo analysis, full video flow (upload → progress → results → timeline seek → CSV download), model page, mobile layout |

## Pipeline and artefact validation

| Check | How | Result |
|---|---|---|
| Dataset extraction integrity | SHA-256 of every extracted file vs. the archive | 82/82 identical |
| ONNX export parity | cosine similarity vs. PyTorch on all 40 crops | min 0.9999999, max abs diff 1.8e-5 |
| Bundle reproducibility | re-export in notebook 02, compare SHA-256 | 3/3 files identical |
| CV reproducibility | notebook 02 re-ran 9 models | identical macro-F1 to the scripted run |
| Notebooks | executed with nbclient, validated with nbformat | 4/4 executed, 0 errors |
| Technical report PDF | pypdf text extraction, page count, metadata | 12 pages, selectable text, metadata set |
| PPTX deck | skill validator (schema, relationships, charts) + LibreOffice render + visual review of all 14 slides | all validations passed; issues found in review were fixed |
| Docker Compose | `docker compose config` | valid |

## Fresh-checkout validation of the package

`dist/salat-vision.zip` (built by `scripts/package_project.sh` from the committed tree) was unpacked into an empty
folder and set up as the README describes. All project code, configuration and the model bundle came from the unpacked
package; only the Python virtual environment (already-installed dependencies) and the provided dataset archive were
reused. The package tested was built from the final code commit; the next commit only adds this section.

| Step | Result |
|---|---|
| `scripts/extract_dataset.py --zip …` | 82 files extracted, archive safety checks passed |
| `scripts/download_models.py` | MediaPipe models (version-pinned URLs) and torchvision weights downloaded; the two models the bundle uses match its recorded SHA-256 |
| `pytest -q` | 55 passed |
| `scripts/make_demo_video.py` | demo clip (42.9 s) and two face-blurred stills written |
| `npm ci` · `npm run typecheck` · `npm test` · `npm run build` | 157 packages installed · passed · 12 passed · built |
| `npm run e2e` | 5 passed, 0 skipped (backend started from the unpacked tree) |

## Not executed / blocked

| Item | Reason |
|---|---|
| Building/running the Docker images | No Docker daemon in the authoring environment (client only). |
| Training/evaluating on the full 1,904-image dataset | Kaggle was not reachable anonymously from the environment; only the provided sample was available. |
| Accuracy on real prayer videos | No labelled prayer videos exist in the dataset; the temporal decoder was evaluated on semi-synthetic sequences. |
| Pushing to GitHub | The session's GitHub App access to the repository was denied (403); commits are local until access is fixed. |

## Bugs found by testing and fixed

* A `numpy.float32` leaked into API JSON (pose-derived boxes) — found by the video pipeline run, fixed at the source.
* Two timelines on the results page shared one accessible name — found by a unit test, given distinct names.
* Download links lost descriptive accessible names after a layout change — found by a unit test, restored with `aria-label`.
* A lazy-loaded page could exceed the default test timeout on a cold transform cache — timeouts raised for that route.
* The photo end-to-end test uploaded a still image that no committed script produced, so it was skipped in a fresh
  checkout — found by the fresh-checkout run; `scripts/make_demo_video.py` now writes the stills.
* Model downloads used MediaPipe's floating `latest` URLs although the bundle verifies third-party checksums, so a
  re-published file could stop the API from loading. Downloads are now pinned to versioned URLs (re-exporting with the
  pinned file reproduced the bundle bit-for-bit), and the download script verifies files against the bundle.
