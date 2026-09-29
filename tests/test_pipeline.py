"""End-to-end and validation tests for the soft proofing app."""
import io
import struct

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import colorscience, lcms_native
from app.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    data_dir = tmp_path_factory.mktemp("data")
    os.environ["SPD_DATA_DIR"] = str(data_dir)
    # Reload config paths
    from app import config
    config.DATA_DIR = data_dir
    config.PROFILE_DIR = data_dir / "profiles"
    config.JOB_DIR = data_dir / "jobs"
    config.DB_PATH = data_dir / "softproof.db"
    from app import main
    main.DATA_DIR = data_dir
    main.JOB_DIR = data_dir / "jobs"
    main.DB_PATH = data_dir / "softproof.db"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ids(client):
    profs = client.get("/api/profiles").json()
    return {
        "srgb": next(p["profile_id"] for p in profs["source"] if p["color_space"] == "RGB"),
        "cmyk": next(p["profile_id"] for p in profs["press"]),
        "display": next(p["profile_id"] for p in profs["display"]),
    }


def png_bytes(arr: np.ndarray, mode="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format=mode)
    return buf.getvalue()


def sample_img():
    arr = np.zeros((48, 48, 3), np.uint8)
    arr[..., 0] = np.linspace(0, 255, 48, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, 48, dtype=np.uint8)[:, None]
    return png_bytes(arr)


def run(client, ids, data, **overrides):
    form = {
        "source_profile": ids["srgb"],
        "press_profile": ids["cmyk"],
        "display_profile": ids["display"],
        "source_intent": "relative_colorimetric",
        "proof_intent": "relative_colorimetric",
        "threshold": "3.0",
        "use_embedded": "false",
    }
    form.update(overrides)
    return client.post(
        "/api/jobs",
        files={"image": ("img.png", data, "image/png")},
        data=form,
    )


# --------------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_builtin_profiles_present(client):
    p = client.get("/api/profiles").json()
    assert any(x["color_space"] == "CMYK" for x in p["press"])
    assert any(x["color_space"] == "RGB" for x in p["display"])


def test_full_pipeline_and_determinism(client, ids):
    data = sample_img()
    r1 = run(client, ids, data)
    assert r1.status_code == 200, r1.text
    j1 = r1.json()
    r2 = run(client, ids, sample_img())
    j2 = r2.json()
    assert j1["job_id"] == j2["job_id"]
    assert j1["stats"] == j2["stats"]
    assert j1["stats"]["total_pixels"] == 48 * 48
    assert 0.0 <= j1["stats"]["over_ratio"] <= 1.0


def test_artifact_pngs_are_tagged(client, ids):
    j = run(client, ids, sample_img()).json()
    state = client.app.state.state
    display_bytes = state.pm.load_bytes(ids["display"])
    srgb_bytes = state.srgb_bytes
    for nm in ("preview.png", "softproof.png", "heatmap.png", "mask.png", "overlay.png"):
        resp = client.get(f"/api/jobs/{j['job_id']}/images/{nm}")
        assert resp.status_code == 200
        im = Image.open(io.BytesIO(resp.content))
        assert im.info.get("icc_profile") is not None
    # Photographic previews are tagged with the display profile; synthetic
    # overlays carry sRGB.
    preview = client.get(f"/api/jobs/{j['job_id']}/images/preview.png").content
    heat = client.get(f"/api/jobs/{j['job_id']}/images/heatmap.png").content
    assert display_bytes in preview or Image.open(io.BytesIO(preview)).info["icc_profile"] == display_bytes
    assert Image.open(io.BytesIO(heat)).info["icc_profile"] == srgb_bytes


def test_report_binds_hash_profiles_params(client, ids):
    j = run(client, ids, sample_img()).json()
    rep = j["report"]
    assert len(rep["image"]["sha256"]) == 64
    assert rep["profiles"]["press"]["color_space"] == "CMYK"
    assert rep["parameters"]["threshold_delta_e"] == 3.0
    resp = client.get(f"/api/jobs/{j['job_id']}/report")
    assert resp.json()["job_id"] == j["job_id"]


def test_zip_export(client, ids):
    import zipfile
    j = run(client, ids, sample_img()).json()
    z = client.get(f"/api/jobs/{j['job_id']}/export.zip")
    assert z.status_code == 200
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        names = set(zf.namelist())
        assert {"report.json", "softproof.png", "heatmap.png", "mask.png"} <= names


def test_region_inspection(client, ids):
    j = run(client, ids, sample_img()).json()
    r = client.post(
        f"/api/jobs/{j['job_id']}/region",
        json={"x": 5, "y": 5, "width": 10, "height": 8},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["rect"] == {"x": 5, "y": 5, "width": 10, "height": 8}
    assert len(d["lab_source"]["mean"]) == 3
    assert len(d["cmyk"]["mean_percent"]) == 4
    assert len(d["samples_5x5"]["cmyk_percent"]) == 5


def test_reject_rgb_pixels_with_cmyk_source(client, ids):
    r = run(client, ids, sample_img(), source_profile=ids["cmyk"])
    assert r.status_code == 422
    assert "通道" in r.json()["detail"] or "不匹配" in r.json()["detail"]


def test_reject_cmyk_as_display(client, ids):
    r = run(client, ids, sample_img(), display_profile=ids["cmyk"])
    assert r.status_code == 422


def test_reject_srgb_as_press(client, ids):
    r = run(client, ids, sample_img(), press_profile=ids["srgb"])
    assert r.status_code == 422


def test_reject_missing_embedded(client, ids):
    r = run(client, ids, sample_img(), use_embedded="true")
    assert r.status_code == 422
    assert "内嵌" in r.json()["detail"]


def test_reject_garbage_upload(client, ids):
    r = client.post(
        "/api/jobs",
        files={"image": ("x.png", b"not an image at all", "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 422


def test_reject_garbage_icc_upload(client):
    r = client.post(
        "/api/profiles/press",
        files={"file": ("bad.icc", b"\x00" * 300, "application/octet-stream")},
    )
    assert r.status_code == 422


def test_invalid_icc_rejected_on_upload(client):
    # Truncated but correctly-prefixed ICC blob.
    bad = bytearray(128)
    struct.pack_into(">I", bad, 0, 99999)
    bad[36:40] = b"acsp"
    r = client.post(
        "/api/profiles/press",
        files={"file": ("x.icc", bytes(bad), "application/octet-stream")},
    )
    assert r.status_code == 422


def test_cmyk_image_roundtrip(client, ids):
    # CMYK TIFF generated by lcms itself (valid tagged data).
    rgb = Image.new("RGB", (16, 16), (200, 40, 90))
    from PIL import ImageCms
    state_profs = client.app.state.state
    cmyk_prof = state_profs.pm.open_pil(ids["cmyk"])
    cmyk = ImageCms.applyTransform(
        rgb,
        ImageCms.buildTransform(state_profs.srgb_profile, cmyk_prof, "RGB", "CMYK"),
    )
    buf = io.BytesIO()
    cmyk.save(buf, format="TIFF")
    r = client.post(
        "/api/jobs",
        files={"image": ("c.tif", buf.getvalue(), "image/tiff")},
        data={
            "source_profile": ids["cmyk"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "2",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["report"]["image"]["mode"] == "CMYK"


def test_embedded_source_accepted(client, ids):
    from PIL import ImageCms
    srgb_icc = client.app.state.state.srgb_bytes
    arr = np.zeros((16, 16, 3), np.uint8)
    arr[..., 0] = 200
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG", icc_profile=srgb_icc)
    r = run(
        client, ids, buf.getvalue(), use_embedded="true",
        source_profile="unused",
    )
    assert r.status_code == 200, r.text
    assert r.json()["report"]["image"]["embedded_source_sha256"]


def test_lab_values_physically_correct(client, ids):
    """Primary colors must produce the well-known sRGB D50 Lab values.

    This guards against a subtle bug where LAB bytes are reinterpreted with
    the wrong a/b polarity (PIL Image.new vs raw lcms V2 bytes).
    """
    im = Image.new("RGB", (2, 2))
    im.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    j = run(client, ids, buf.getvalue()).json()
    expected = {
        (0, 0): (53, 80, 67),    # red (D50)
        (1, 0): (88, -80, 81),   # green (D50 adapted)
        (0, 1): (29, 68, -112),  # blue
        (1, 1): (100, 0, 0),     # white
    }
    for (x, y), (L, a, b) in expected.items():
        d = client.post(
            f"/api/jobs/{j['job_id']}/region",
            json={"x": x, "y": y, "width": 1, "height": 1},
        ).json()
        got = d["lab_source"]["mean"]
        assert abs(got[0] - L) <= 2, (x, y, got)
        assert abs(got[1] - a) <= 4, (x, y, got)
        assert abs(got[2] - b) <= 4, (x, y, got)
    # White must stay neutral and map to zero ink on the demo press.
    d = client.post(
        f"/api/jobs/{j['job_id']}/region",
        json={"x": 1, "y": 1, "width": 1, "height": 1},
    ).json()
    assert max(d["cmyk"]["mean_percent"]) < 1.0
    assert abs(d["lab_after_press"]["mean"][0] - 100) < 1.0


def test_ciede2000_sharma_rows():
    rows = [
        ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
        ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
        ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0),
        ((50, 0, 0), (50, -1, 2), 2.3669),
        ((50, 2.5, 0), (73, 25, -18), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ]
    for a, b, expected in rows:
        got = float(
            colorscience.ciede2000(np.array([[a]]), np.array([[b]]))[0, 0]
        )
        assert got == pytest.approx(expected, abs=1e-3)


def test_lab_v2_decoding():
    # byte 127 -> +127, byte 128 -> -128, byte 0 -> 0
    lab = np.array([[[255, 127, 128], [0, 0, 255]]], dtype=np.uint8)
    dec = colorscience.decode_lab(lab)
    assert dec[0, 0, 0] == 100.0
    assert dec[0, 0, 1] == 127.0
    assert dec[0, 0, 2] == -128.0
    assert dec[0, 1, 1] == 0.0
    assert dec[0, 1, 2] == -1.0


def test_gray_image_with_gray_profile(client, ids):
    profs = client.get("/api/profiles").json()
    gray_id = next(p["profile_id"] for p in profs["source"] if p["color_space"] == "GRAY")
    g = np.tile(np.linspace(0, 255, 20, dtype=np.uint8), (16, 1))
    buf = io.BytesIO()
    Image.fromarray(g, "L").save(buf, format="TIFF")
    r = client.post(
        "/api/jobs",
        files={"image": ("g.tif", buf.getvalue(), "image/tiff")},
        data={
            "source_profile": gray_id,
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["image"]["mode"] == "L"
    assert rep["profiles"]["source"]["color_space"] == "GRAY"
    # Neutral ramp must stay neutral: sample center region chromaticity ~0
    j = r.json()
    region = client.post(
        f"/api/jobs/{j['job_id']}/region",
        json={"x": 8, "y": 7, "width": 4, "height": 2},
    ).json()
    assert abs(region["lab_source"]["mean"][1]) < 1.0
    assert abs(region["lab_source"]["mean"][2]) < 1.0


def test_demo_cmyk_profile_is_valid():
    data = lcms_native.build_demo_cmyk_profile_bytes()
    header = lcms_native.read_header(data)
    assert header["color_space"] == "CMYK"
    assert header["pcs"] == "Lab"
