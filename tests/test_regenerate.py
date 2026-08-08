"""Tests for the offline batch regeneration of already-captured frames.

The single most important guarantee: regenerating a frame rewrites the geometric metadata but
leaves every ground-truth ``"label"`` byte-identical.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from chess_commentator.regenerate import (
    _print_progress,
    iter_frames,
    read_existing_labels,
    regenerate_all,
    regenerate_frame,
    write_variant_csv,
)


def test_print_progress_formats(capsys):
    _print_progress(3, 10, 0)
    out = capsys.readouterr().out
    assert "3/10" in out and "30%" in out

    _print_progress(10, 10, 2)
    out = capsys.readouterr().out
    assert "10/10" in out and "100%" in out and "2 failed" in out


def test_regenerate_all_reports_progress_and_completes(make_v2_setup, capsys):
    setup_dir, frame_dirs, _ = make_v2_setup(n_frames=2)
    regenerate_all(data_root=setup_dir.parent, config_path="config.yaml", max_workers=2)
    out = capsys.readouterr().out
    assert "Regenerating 2 frames" in out
    assert "Done. 2/2 frames regenerated." in out
    for frame_dir in frame_dirs:
        assert (frame_dir / "image_warped.png").exists()


def test_regenerate_frame_preserves_labels_and_updates_geometry(make_v2_setup):
    setup_dir, frame_dirs, labels = make_v2_setup(n_frames=1)
    frame_dir = frame_dirs[0]
    squares_dir = frame_dir / "squares"

    before = read_existing_labels(squares_dir)
    assert len(before) == 64
    # Seeded geometry uses a sentinel so we can prove it actually changes.
    assert json.loads((squares_dir / "e4" / "e4_metadata.json").read_text()) == {
        "top": 999,
        "left": 999,
        "label": "lbl_e4",
    }

    regenerate_frame(str(setup_dir), str(frame_dir), "config.yaml")

    after = read_existing_labels(squares_dir)
    assert after == before  # every label preserved exactly
    assert len(after) == 64

    for sq in ["a1", "e4", "h8", "d5"]:
        meta = json.loads((squares_dir / sq / f"{sq}_metadata.json").read_text())
        assert meta["label"] == labels[sq]                       # label byte-identical
        assert (meta["top"], meta["left"]) != (999, 999)         # geometry rewritten
        assert (squares_dir / sq / f"{sq}_masked.npy").exists()  # crop regenerated


def test_regenerate_frame_writes_warped_and_keeps_original(make_v2_setup):
    setup_dir, frame_dirs, _ = make_v2_setup(n_frames=1)
    frame_dir = frame_dirs[0]

    original_bytes = (frame_dir / "image.png").read_bytes()
    regenerate_frame(str(setup_dir), str(frame_dir), "config.yaml")

    assert (frame_dir / "image_warped.png").exists()
    # The raw distorted capture must be left exactly as it was.
    assert (frame_dir / "image.png").read_bytes() == original_bytes


def test_regenerate_variant_builds_a_self_contained_tree(make_v2_setup, tmp_path):
    """A variant tree must stand on its own and leave the source tree untouched."""
    setup_dir, frame_dirs, labels = make_v2_setup(n_frames=1)
    frame_dir = frame_dirs[0]
    source_before = (frame_dir / "squares" / "e4" / "e4_metadata.json").read_text()

    out_root = tmp_path / "variant"
    regenerate_all(
        data_root=setup_dir.parent,
        config_path="config.yaml",
        max_workers=1,
        variant="none_global",
        out_root=out_root,
    )

    # Board-level eval resolves the calibration by path, so it has to travel with the crops.
    variant_setup_dir = out_root / setup_dir.name
    assert (variant_setup_dir / "calibration_metadata.json").exists()

    variant_squares = variant_setup_dir / frame_dir.name / "squares"
    arr = np.load(variant_squares / "e4" / "e4_masked.npy")
    assert arr.shape == (144, 144, 4)
    assert arr[..., 3].max() == 0  # the none_global arm: the mask channel carries nothing

    # Ground-truth labels are read from the source frame and merged into the variant crops.
    variant_meta = json.loads((variant_squares / "e4" / "e4_metadata.json").read_text())
    assert variant_meta["label"] == labels["e4"]

    # The source tree is not rewritten when a variant is being built.
    assert (frame_dir / "squares" / "e4" / "e4_metadata.json").read_text() == source_before


def test_write_variant_csv_repoints_only_the_path_columns(tmp_path):
    src_csv = tmp_path / "data.csv"
    pl.DataFrame(
        {
            "setup_id": ["s1"],
            "label": ["K"],
            "setup_split": ["train"],
            "square_image_path": ["data/generated/s1/board_1/squares/e4/e4.png"],
            "full_image_path": ["data/generated/s1/board_1/image.png"],
            "calibration_metadata_path": ["data/generated/s1/calibration_metadata.json"],
        }
    ).write_csv(src_csv)

    out = write_variant_csv(
        src_csv,
        Path("data/generated"),
        Path("data/generated_square_global"),
        tmp_path / "variant_data.csv",
    )

    row = pl.read_csv(out).to_dicts()[0]
    assert row["square_image_path"] == "data/generated_square_global/s1/board_1/squares/e4/e4.png"
    assert row["full_image_path"] == "data/generated_square_global/s1/board_1/image.png"
    assert row["calibration_metadata_path"] == "data/generated_square_global/s1/calibration_metadata.json"
    # Everything that is not a location is the same board, so it is copied verbatim.
    assert row["setup_id"] == "s1"
    assert row["label"] == "K"
    assert row["setup_split"] == "train"


def test_write_variant_csv_normalises_legacy_windows_paths(tmp_path):
    """Rows written on Windows before the .as_posix() fix stored backslashes; repoint those too."""
    src_csv = tmp_path / "data.csv"
    pl.DataFrame(
        {"square_image_path": [r"data\generated\s1\board_1\squares\e4\e4.png"]}
    ).write_csv(src_csv)

    out = write_variant_csv(
        src_csv, Path("data/generated"), Path("data/generated_none_global"), tmp_path / "out.csv"
    )
    assert (
        pl.read_csv(out).to_dicts()[0]["square_image_path"]
        == "data/generated_none_global/s1/board_1/squares/e4/e4.png"
    )


def test_iter_frames_discovers_all_setup_frames(make_v2_setup):
    setup_dir, frame_dirs, _ = make_v2_setup(n_frames=3)
    found = list(iter_frames(setup_dir.parent))
    assert len(found) == 3
    assert all(setup == setup_dir for setup, _frame in found)
    assert {frame.name for _setup, frame in found} == {fd.name for fd in frame_dirs}
