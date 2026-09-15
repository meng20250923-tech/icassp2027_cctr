import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "build_vatex_english_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_vatex_english_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(video_id: str, captions: list[str] | None = None):
    return {"videoID": video_id, "enCap": captions or ["first caption", "second caption"]}


def test_load_rows_and_build_manifest(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        __import__("json").dumps([row("abc_000001_000011"), row("def_000020_000030")]),
        encoding="utf-8",
    )
    rows = MODULE.load_rows(path)
    manifest = MODULE.build_manifest(rows, "test", ".mp4")
    assert manifest["videos"][0]["filename"] == "abc_000001_000011.mp4"
    assert manifest["videos"][0]["youtube_id"] == "abc"
    assert manifest["videos"][0]["clip_start_seconds"] == 1
    assert [caption["video_index"] for caption in manifest["captions"]] == [0, 0, 1, 1]


def test_choose_is_reproducible_and_bounded():
    rows = [{"video_id": f"id{index}"} for index in range(10)]
    assert MODULE.choose(rows, 4, 2027) == MODULE.choose(rows, 4, 2027)
    with pytest.raises(ValueError, match="Requested"):
        MODULE.choose(rows, 11, 2027)


def test_parse_clip_id_rejects_invalid_boundaries():
    with pytest.raises(ValueError, match="boundaries"):
        MODULE.parse_clip_id("abc_000010_000010")
