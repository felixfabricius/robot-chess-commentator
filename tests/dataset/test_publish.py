"""Tests for the Hugging Face dataset export.

The guarantees that matter: the published images reconstruct the model input exactly, the published
class indices are the ones the model was trained on, and nothing that was deliberately withheld
(raw-frame paths, capture timestamps) survives into the export.
"""

import json

import numpy as np
import polars as pl
import pytest
from datasets import load_dataset

from chess_commentator.board import SQUARES
from chess_commentator.dataset.publish import (
    CORNER_NAMES,
    LABEL_NAMES,
    build,
    build_card,
    build_id_maps,
    counts_from_shards,
    encode_png,
    sanitise_calibration,
)
from chess_commentator.labels import TARGET_MAP, TOP_LEFT_OHE_MAP


def load_export(out_dir, config, split="train"):
    """load_dataset with a cache of its own.

    Without this, datasets can serve a previously cached build of the same path -- so a rebuilt
    export with a new column still reads back with the old schema.
    """
    return load_dataset(
        str(out_dir), config, split=split, cache_dir=str(out_dir.parent / "hfcache")
    )


def test_label_names_track_labels_module():
    """The published ClassLabel order IS the model's target order -- if these ever diverge, every
    published index silently means a different piece."""
    assert LABEL_NAMES == list(TARGET_MAP)
    assert [TARGET_MAP[name] for name in LABEL_NAMES] == list(range(13))
    assert CORNER_NAMES == list(TOP_LEFT_OHE_MAP)


@pytest.mark.parametrize("shape", [(144, 144, 3), (144, 144)])
def test_encode_png_is_lossless(shape):
    """Publishing the 4-channel model input as two ordinary images only works if PNG round-trips."""
    import cv2

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=shape, dtype=np.uint8)

    decoded = cv2.imdecode(np.frombuffer(encode_png(array), np.uint8), cv2.IMREAD_UNCHANGED)
    if array.ndim == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, array)


def test_build_id_maps_is_chronological_and_total():
    data = pl.DataFrame(
        {
            "setup_id": ["2026-07-09_120000", "2026-07-01_090000", "2026-07-01_090000"],
            "image_id": ["board_2026-07-09_120500", "board_2026-07-01_090500", "board_x"],
        }
    )
    setup_map, board_map = build_id_maps(data)

    # Ids are timestamps, so lexicographic order is capture order.
    assert setup_map == {"2026-07-01_090000": "setup_00", "2026-07-09_120000": "setup_01"}
    assert set(board_map) == set(data["image_id"])
    assert len(set(board_map.values())) == 3
    # Nothing anonymised may still contain the original stamp.
    assert not any("2026" in new for new in {**setup_map, **board_map}.values())


def test_sanitise_calibration_drops_raw_frame_and_timestamp():
    raw = {
        "timestamp": "2026-07-01_175452",
        "raw_image_path": "data/generated/2026-07-01_175334/raw.png",
        "height_mm": 8,
        "camera_natural_orientation": {"order": {"tl": "h1"}},
    }
    clean = sanitise_calibration(raw)
    assert "timestamp" not in clean and "raw_image_path" not in clean
    assert clean["height_mm"] == 8
    assert clean["camera_natural_orientation"]["order"]["tl"] == "h1"


# --- end-to-end, against a tiny synthetic data/generated tree -------------------------------------


def _write_square(square_dir, square, label, rng):
    square_dir.mkdir(parents=True, exist_ok=True)
    masked = np.zeros((144, 144, 4), dtype=np.uint8)
    masked[..., :3] = rng.integers(0, 256, size=(144, 144, 3), dtype=np.uint8)
    masked[..., 3] = (rng.random((144, 144)) > 0.5).astype(np.uint8)  # {0, 1}, as on disk
    np.save(square_dir / f"{square}_masked.npy", masked)
    (square_dir / f"{square}_annotated.png").write_bytes(encode_png(masked[..., :3]))
    (square_dir / f"{square}_metadata.json").write_text(
        json.dumps({"top": 3, "left": 5, "label": label}), encoding="utf-8"
    )
    return masked


@pytest.fixture
def generated_tree(tmp_path):
    """A one-setup, one-board data/generated tree with a full 64-square position."""
    rng = np.random.default_rng(0)
    root = tmp_path / "generated"
    setup_id, image_id = "2026-07-01_175334", "board_2026-07-01_175602"
    setup_dir = root / setup_id
    board_dir = setup_dir / image_id
    board_dir.mkdir(parents=True)

    (setup_dir / "calibration_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-01_175452",
                "raw_image_path": f"data/generated/{setup_id}/raw.png",
                "height_mm": 8,
                "camera_natural_orientation": {"order": {"tl": "h1"}},
            }
        ),
        encoding="utf-8",
    )
    (board_dir / "image_warped.png").write_bytes(
        encode_png(rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8))
    )

    labels = {square: ("R" if square == "a1" else "empty") for square in SQUARES}
    originals = {
        square: _write_square(board_dir / "squares" / square, square, labels[square], rng)
        for square in SQUARES
    }
    (board_dir / "metadata.json").write_text(
        json.dumps(
            {
                "valid_game_position": True,
                "legal_move_mode": True,
                "board_fen": "8/8/8/8/8/8/8/R7 w - - 0 1",
                "previous_board_fen": None,
                "move_uci": None,
                "piece_map": labels,
            }
        ),
        encoding="utf-8",
    )

    pl.DataFrame(
        {
            "setup_id": [setup_id] * 64,
            "setup_split": ["train"] * 64,
            "image_id": [image_id] * 64,
            "square": list(SQUARES),
            "label": [labels[square] for square in SQUARES],
            # Deliberately a stale backslash path to a crop that does not exist: the exporter must
            # ignore this column entirely, the way cnn/data.py does.
            "square_image_path": [rf"data\generated\{setup_id}\{image_id}\squares\a1\a1.png"] * 64,
            "full_image_path": [rf"data\generated\{setup_id}\{image_id}\image.png"] * 64,
            "calibration_metadata_path": [""] * 64,
            "valid_game_position": [True] * 64,
            "board_fen": ["8/8/8/8/8/8/8/R7 w - - 0 1"] * 64,
            "previous_board_fen": [""] * 64,
            "move_uci": [""] * 64,
            "created_at": ["2026-07-01T17:56:02"] * 64,
        }
    ).write_csv(root / "data.csv")
    return root, originals


def test_build_round_trips_and_scrubs(tmp_path, generated_tree):
    root, originals = generated_tree
    out_dir = tmp_path / "export"

    counts = build(data_root=root, out_dir=out_dir)
    assert counts == {"squares": {"train": 64}, "boards": {"train": 1}}

    squares = load_export(out_dir, "squares", "train")
    boards = load_export(out_dir, "boards", "train")

    assert squares.features["label"].names == list(TARGET_MAP)
    assert boards.features["labels"].feature.names == list(TARGET_MAP)

    # The two published images must rebuild the model input byte-for-byte.
    for row in squares:
        rebuilt = np.dstack([np.array(row["image"]), np.array(row["mask"]) // 255])
        assert np.array_equal(rebuilt, originals[row["square"]])
        assert row["top"] == 3 and row["left"] == 5
        assert row["setup_id"] == "setup_00" and row["image_id"] == "board_000"
        assert squares.features["top_left_corner"].int2str(row["top_left_corner"]) == "h1"

    by_square = {row["square"]: squares.features["label"].int2str(row["label"]) for row in squares}
    assert by_square["a1"] == "R"
    assert by_square["h8"] == "empty"

    # Board labels are in SQUARES order, not sorted or arbitrary.
    board = boards[0]
    decoded = [boards.features["labels"].feature.int2str(i) for i in board["labels"]]
    assert decoded == [by_square[square] for square in SQUARES]

    # Nothing withheld may survive: no timestamps, no raw-frame paths, in any string column.
    for dataset in (squares, boards):
        for column, feature in dataset.features.items():
            if getattr(feature, "dtype", None) != "string":
                continue
            for value in dataset[column]:
                assert "2026-" not in (value or "")
                assert "image.png" not in (value or "")

    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert list(calibration) == ["setup_00"]
    assert "timestamp" not in calibration["setup_00"]
    assert "raw_image_path" not in calibration["setup_00"]


def test_labels_follow_the_pixels_not_the_fen(tmp_path, generated_tree):
    """data.csv's label column is ground truth for the PIXELS.

    Where the virtual board drifted out of sync with the physical one, that column was
    hand-corrected to match the photograph while board_fen / piece_map were not. A square
    classifier must be trained on what the camera saw, so the CSV wins -- and the board is flagged
    so move evaluation, which needs the FEN chain, can skip it.
    """
    root, _ = generated_tree

    # a1 physically held nothing and h8 held the rook; the FEN still says the old arrangement.
    csv_path = root / "data.csv"
    data = pl.read_csv(csv_path)
    data = data.with_columns(
        pl.when(pl.col("square") == "a1").then(pl.lit("empty"))
        .when(pl.col("square") == "h8").then(pl.lit("R"))
        .otherwise(pl.col("label")).alias("label")
    )
    data.write_csv(csv_path)

    out_dir = tmp_path / "export"
    build(data_root=root, out_dir=out_dir)

    squares = load_export(out_dir, "squares", "train")
    by_square = {r["square"]: squares.features["label"].int2str(r["label"]) for r in squares}
    assert by_square["a1"] == "empty", "exporter used the FEN instead of the corrected pixel label"
    assert by_square["h8"] == "R"

    # Both configs must tell the same story about the same board.
    boards = load_export(out_dir, "boards", "train")
    decoded = [boards.features["labels"].feature.int2str(i) for i in boards[0]["labels"]]
    assert decoded == [by_square[square] for square in SQUARES], "configs disagree on labels"

    # ...and the disagreement with board_fen must be advertised, not hidden.
    assert all(not r["fen_matches_pixels"] for r in squares)
    assert not boards[0]["fen_matches_pixels"]


def test_fen_matches_pixels_is_true_when_they_agree(tmp_path, generated_tree):
    """The flag must not be permanently on: an untouched board is move-evaluable."""
    root, _ = generated_tree
    out_dir = tmp_path / "export"
    build(data_root=root, out_dir=out_dir)

    squares = load_export(out_dir, "squares", "train")
    boards = load_export(out_dir, "boards", "train")
    assert all(r["fen_matches_pixels"] for r in squares)
    assert boards[0]["fen_matches_pixels"]


def test_board_game_fields_come_from_the_csv_not_metadata(tmp_path, generated_tree):
    """metadata.json is stale on valid_game_position for 4 real boards.

    The operator's corrections landed in data.csv only, so both configs must read game fields from
    there -- otherwise they disagree, and move evaluation admits boards the benchmark excludes.
    """
    root, _ = generated_tree
    setup_id, image_id = "2026-07-01_175334", "board_2026-07-01_175602"

    # metadata still claims the position is a valid game position; the CSV knows better.
    meta_path = root / setup_id / image_id / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["valid_game_position"] = True
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    csv_path = root / "data.csv"
    pl.read_csv(csv_path).with_columns(
        pl.lit(False).alias("valid_game_position")  # noqa: FBT003
    ).write_csv(csv_path)

    out_dir = tmp_path / "export"
    build(data_root=root, out_dir=out_dir)

    boards = load_export(out_dir, "boards", "train")
    squares = load_export(out_dir, "squares", "train")
    assert boards[0]["valid_game_position"] is False, "boards config trusted stale metadata.json"
    assert all(r["valid_game_position"] is False for r in squares)


def test_rebuilding_replaces_shards_rather_than_accumulating(tmp_path, generated_tree):
    """A second build into the same directory must not inherit the first build's shards.

    The finished names ("train-00000-of-00001.parquet") and the in-progress ones
    ("train-00000.parquet") share a prefix, so a careless glob republishes stale data as extra
    shards -- silently doubling rows.
    """
    root, _ = generated_tree
    out_dir = tmp_path / "export"

    build(data_root=root, out_dir=out_dir)
    first = sorted(p.name for p in (out_dir / "data" / "squares").glob("*.parquet"))

    build(data_root=root, out_dir=out_dir)
    second = sorted(p.name for p in (out_dir / "data" / "squares").glob("*.parquet"))

    assert first == second, "rebuild changed the shard set"
    assert len(load_export(out_dir, "squares", "train")) == 64


def test_counts_from_shards_matches_the_build(tmp_path, generated_tree):
    """--card-only rewrites the card from the shards, so its counts must equal the build's."""
    root, _ = generated_tree
    out_dir = tmp_path / "export"
    counts = build(data_root=root, out_dir=out_dir)

    assert counts_from_shards(out_dir) == counts
    assert "64" in build_card(counts_from_shards(out_dir))


def test_id_mapping_is_written_outside_the_uploaded_folder(tmp_path, generated_tree):
    """Uploading out_dir must not be able to leak the de-anonymising mapping."""
    root, _ = generated_tree
    out_dir = tmp_path / "export"
    build(data_root=root, out_dir=out_dir)

    assert not (out_dir / "id_mapping.json").exists()
    mapping = json.loads((out_dir.parent / "id_mapping.json").read_text(encoding="utf-8"))
    assert mapping["setups"] == {"2026-07-01_175334": "setup_00"}
    assert mapping["boards"] == {"board_2026-07-01_175602": "board_000"}
