import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.settings import Settings

from .conftest import make_video, requires_data, requires_model


def png_bytes(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 130, 140)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(tmp_dir=tmp_path_factory.mktemp("api"), max_image_mb=1, max_video_mb=5)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def no_model_client(tmp_path_factory):
    empty = tmp_path_factory.mktemp("empty_bundle")
    settings = Settings(model_dir=empty, tmp_dir=tmp_path_factory.mktemp("api2"))
    with TestClient(create_app(settings)) as c:
        yield c


# ---------------------------------------------------------------- model unavailable
def test_health_reports_missing_model(no_model_client):
    r = no_model_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False and body["model_error"]


def test_predict_refuses_without_model(no_model_client):
    r = no_model_client.post("/predict/image", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()
    info = no_model_client.get("/model-info").json()
    assert info["available"] is False and info["reason"]


# ---------------------------------------------------------------- validation (no model needed to reject)
@requires_model
def test_rejects_non_image(client):
    r = client.post("/predict/image", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


@requires_model
def test_rejects_oversized_image(client):
    big = np.random.default_rng(0).integers(0, 255, (900, 900, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(big).save(buf, "PNG")  # ~2.4 MB > 1 MB limit
    r = client.post("/predict/image", files={"file": ("big.png", buf.getvalue(), "image/png")})
    assert r.status_code == 413


@requires_model
def test_rejects_unsupported_video(client):
    r = client.post("/jobs/video", files={"file": ("a.txt", b"hello world, not a video", "text/plain")})
    assert r.status_code == 415
    r = client.post("/jobs/video", files={"file": ("a.mp4", b"0123456789abcdef", "video/mp4")})
    assert r.status_code == 415


# ---------------------------------------------------------------- happy paths with the real model
@requires_model
def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h["model_loaded"] is True and h["model_version"]
    info = client.get("/model-info").json()
    assert info["available"] is True
    assert [c["id"] for c in info["classes"]] == ["qiyam", "ruku", "sujud", "julus"]
    assert len(info["phases"]) == 6
    assert info["cv_metrics"]["crop"]["macro_f1_mean"] > 0
    assert info["model_comparison"], "comparison table from real experiments"


@requires_model
@requires_data
def test_predict_image_real_photo(client, sample_image_path):
    r = client.post("/predict/image", files={"file": ("p.jpg", sample_image_path.read_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["visual_class"] in {"qiyam", "ruku", "sujud", "julus"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["image"]["width"] > 0


@requires_model
@requires_data
def test_video_job_lifecycle(client, tmp_path, sample_image_path):
    from salat_vision.data.voc import load_image_rgb
    import cv2

    img = load_image_rgb(sample_image_path)
    frame = cv2.resize(img, (240, 320))
    video = make_video(tmp_path / "v.mp4", [frame] * 20, fps=10)
    r = client.post("/jobs/video?annotate=true&sample_fps=5", files={"file": ("v.mp4", video.read_bytes(), "video/mp4")})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    for _ in range(240):
        st = client.get(f"/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["segments"] and res["performance"]["frames_analyzed"] >= 9
    assert st["annotated_video_available"] is True
    v = client.get(f"/jobs/{job_id}/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and len(v.content) > 1000
    j = client.get(f"/jobs/{job_id}/report?format=json")
    assert j.status_code == 200 and "attachment" in j.headers["content-disposition"]
    c = client.get(f"/jobs/{job_id}/report?format=csv")
    assert c.status_code == 200 and c.text.startswith("index,phase,label")
    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404


@requires_model
@requires_data
def test_sync_video_endpoint(client, tmp_path, sample_image_path):
    from salat_vision.data.voc import load_image_rgb
    import cv2

    frame = cv2.resize(load_image_rgb(sample_image_path), (240, 320))
    video = make_video(tmp_path / "s.mp4", [frame] * 10, fps=10)
    r = client.post("/predict/video?sample_fps=4", files={"file": ("s.mp4", video.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    assert r.json()["frames"] is None and r.json()["segments"]


@requires_model
def test_cors_allows_configured_origin_only(client):
    ok = client.options("/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert bad.headers.get("access-control-allow-origin") is None


@requires_model
def test_unknown_job_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/video").status_code == 404
