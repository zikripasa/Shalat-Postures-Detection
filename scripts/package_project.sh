#!/usr/bin/env bash
# Package the committed project as dist/salat-vision.zip (tracked files only:
# no virtual environments, node_modules, raw dataset images, caches or demo renders).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
git archive --format=zip --prefix=salat-vision/ -o dist/salat-vision.zip HEAD
python3 - <<'EOF'
import zipfile
z = zipfile.ZipFile("dist/salat-vision.zip")
names = z.namelist()
size = sum(i.file_size for i in z.infolist())
print(f"dist/salat-vision.zip: {len(names)} entries, {size/1e6:.1f} MB uncompressed")
assert not any("/data/raw/Salat-Postures-Dataset-main/" in n for n in names), "raw dataset must not be packaged"
assert not any("node_modules/" in n or "/.venv/" in n for n in names), "dependencies must not be packaged"
EOF
