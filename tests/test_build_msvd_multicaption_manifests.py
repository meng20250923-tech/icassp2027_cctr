import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "build_msvd_multicaption_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_msvd_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_manifest_preserves_split_order_and_caption_mapping():
    manifest = MODULE.build_manifest(
        "test",
        ["video1.avi", "video2"],
        {"video1": ["first caption"], "video2": ["second", "third"]},
        ".avi",
    )
    assert [row["filename"] for row in manifest["videos"]] == ["video1.avi", "video2.avi"]
    assert [row["video_index"] for row in manifest["captions"]] == [0, 1, 1]


def test_build_manifest_rejects_missing_captions():
    with pytest.raises(ValueError, match="No captions"):
        MODULE.build_manifest("test", ["video1"], {}, ".avi")


def test_filename_and_id_accepts_existing_suffix_and_plain_id():
    assert MODULE.filename_and_id("video1.avi", ".avi") == ("video1", "video1.avi")
    assert MODULE.filename_and_id("video2", ".avi") == ("video2", "video2.avi")
