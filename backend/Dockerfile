# Salat Vision API — CPU inference image (no PyTorch: the CNN runs in ONNX Runtime).
# Build from the repository root:  docker build -f backend/Dockerfile -t salat-vision-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GLOG_minloglevel=2 \
    SALAT_PROJECT_ROOT=/app

# MediaPipe needs EGL/GLES runtime libraries even on CPU; ffmpeg is bundled by imageio-ffmpeg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements/runtime.txt requirements/runtime.txt
RUN pip install -r requirements/runtime.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
RUN pip install --no-deps -e .

COPY backend ./backend
COPY scripts/download_models.py ./scripts/download_models.py
# Trained bundle + evaluation artefacts shown on the Model page
COPY models/salat_vision_v1 ./models/salat_vision_v1
COPY artifacts/metrics ./artifacts/metrics
# Third-party MediaPipe models (Apache-2.0), verified against the bundle manifest at start-up
RUN python scripts/download_models.py --skip-torch

RUN useradd --create-home --uid 10001 salat && chown -R salat /app
USER salat
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
