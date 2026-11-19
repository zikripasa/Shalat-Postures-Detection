import stat
import zipfile

import pytest

from salat_vision.data.extract import DATASET_NAME, inspect_zip, safe_extract

VOC = b"<annotation><size><width>10</width><height>10</height><depth>3</depth></size></annotation>"


def _zip(path, entries, symlink=None):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return path


def test_extract_standard_layout(tmp_path):
    z = _zip(tmp_path / "d.zip", {f"{DATASET_NAME}/README.md": b"x", f"{DATASET_NAME}/S/a.xml": VOC, f"{DATASET_NAME}/S/a.jpg": b"jpg"})
    out, rep = safe_extract(z, tmp_path / "raw")
    assert out == (tmp_path / "raw" / DATASET_NAME).resolve()
    assert (out / "S" / "a.xml").read_bytes() == VOC
    assert not (out / DATASET_NAME).exists(), "no redundant nesting"
    assert rep.is_safe and rep.n_files == 3


def test_extract_strips_extra_parent_folders(tmp_path):
    z = _zip(tmp_path / "d.zip", {f"outer/{DATASET_NAME}/S/a.xml": VOC, f"outer/{DATASET_NAME}/S/a.jpg": b"j"})
    out, _ = safe_extract(z, tmp_path / "raw")
    assert (out / "S" / "a.jpg").is_file()


def test_extract_rejects_path_traversal(tmp_path):
    z = _zip(tmp_path / "d.zip", {"../evil.txt": b"x", f"{DATASET_NAME}/a.xml": VOC})
    rep = inspect_zip(z)
    assert rep.suspicious_paths == ["../evil.txt"]
    with pytest.raises(ValueError):
        safe_extract(z, tmp_path / "raw")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_symlinks(tmp_path):
    z = _zip(tmp_path / "d.zip", {f"{DATASET_NAME}/a.xml": VOC}, symlink=f"{DATASET_NAME}/link.jpg")
    with pytest.raises(ValueError):
        safe_extract(z, tmp_path / "raw")


def test_extract_skips_non_data_files(tmp_path):
    z = _zip(tmp_path / "d.zip", {f"{DATASET_NAME}/run.sh": b"rm -rf /", f"{DATASET_NAME}/a.xml": VOC})
    out, rep = safe_extract(z, tmp_path / "raw")
    assert f"{DATASET_NAME}/run.sh" in rep.skipped_non_data
    assert not (out / "run.sh").exists()
    assert (out / "a.xml").is_file()
