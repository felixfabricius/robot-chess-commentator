"""Build (and optionally push) the Hugging Face dataset repo from ``data/generated``.

What is published, and what deliberately is not:

* **No raw camera frames.** ``board_*/image.png`` and ``<setup>/raw*.png`` show the room the board
  sits in; only the top-down ``image_warped.png`` and the per-square crops leave this machine. The
  cost is that nobody can re-cut alternative mask/crop variants from the published data --
  ``dataset.regenerate`` needs the raw frames.
* **Only ``data/generated``** (the ``default`` hull-mask / per-square-crop tree). The three
  ablation variants with different paddinga and masking stay unpublished.
* **Capture timestamps are anonymised.** ``setup_id`` / ``image_id`` are wall-clock capture times
  and appear in every path, so they are renumbered ``setup_00``.. / ``board_000``.. in capture
  order, and ``created_at`` is dropped. The mapping is written next to the export and is NOT
  uploaded; it is what lets published numbers be joined back to ``evaluation/results.csv``.

Two configs, joined on ``(setup_id, image_id)``:

* ``squares`` -- 23,744 rows, one per square crop. ``image`` + ``mask`` together are exactly the
  model input; ``annotated`` is what the VLM benchmark feeds to Claude.
* ``boards`` -- 371 rows, one per captured position: the warped board plus its full label vector.

``data.csv`` is the row set, not a directory walk: it already excludes the setups with no boards,
the orphan board directories, and the boards that have pixels but no labels. Two of its columns are
deliberately ignored, though:

* ``square_image_path`` locates nothing -- 12,800 of its rows point at plain ``<sq>.png`` crops that
  exist for only 201 of the 371 boards, and 16,256 carry Windows separators. Paths are rebuilt from
  ``(setup_id, image_id, square)`` instead, the way ``cnn/data.py`` rebuilds the ``.npy`` path it
  actually loads.
Labels come from ``data.csv``'s ``label`` column, which is ground truth for the *pixels*: where the
virtual board drifted out of sync with the physical one, that column was hand-corrected to match
what the camera actually saw. ``board_fen`` and ``metadata.json``'s ``piece_map`` were not, so on 28
boards (52 squares) they describe a position that was never photographed.

That split matters because the two are used for different things:

* **Square classification** wants the pixels, so ``label`` is authoritative and the FEN is not.
* **Move estimation** walks the FEN chain (``previous_board_fen`` -> legal moves -> ``move_uci``),
  which describes the *virtual* game. On those 28 boards the chain and the image disagree.

So every row also carries ``fen_matches_pixels``. Filter on it before evaluating move accuracy;
ignore it when training a square classifier. Of the 217 boards that are move-evaluable at all
(``valid_game_position`` and a recorded move), exactly one -- in ``val`` -- is affected.

The same correction pass also flipped ``valid_game_position`` to False on 4 boards, again in the
CSV only. Every game field in BOTH configs therefore comes from ``data.csv``; ``metadata.json``
supplies only the warped image and ``legal_move_mode``. Reading them from ``metadata.json`` would
admit boards the benchmark excludes.

Run with::

    uv run python -m chess_commentator.dataset.publish            # build into dist/huggingface
    uv run python -m chess_commentator.dataset.publish --limit 64 # quick smoke build
    uv run python -m chess_commentator.dataset.publish --push     # build, then upload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import cv2
import numpy as np
import polars as pl

# Eager, unlike publish_model.py's lazy `huggingface_hub` import, and the asymmetry is real:
# `push()` is the only thing there that needs the hub, whereas nothing in this module does
# anything without datasets/pyarrow. The guard only replaces a bare "No module named 'datasets'"
# with the command that fixes it.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import ClassLabel, Features, Image, Sequence, Value
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "chess_commentator.dataset.publish needs the optional 'publish' dependency group "
        "(datasets, pyarrow, huggingface-hub). Install it with: uv sync --group publish"
    ) from exc

from chess_commentator.board import SQUARES
from chess_commentator.labels import TARGET_MAP, TOP_LEFT_OHE_MAP

DATA_ROOT = Path("data") / "generated"
OUT_DIR = Path("dist") / "huggingface" / "squares-dataset"
REPO_ID = "felixfabricius/robot-chess-commentator-squares"

# Both orders come from labels.py so the published class indices cannot drift from the model's.
# ClassLabel columns are int64 on the wire, so label strings are mapped through these on the
# way out; TARGET_MAP's values ARE the indices, so the two cannot disagree.
LABEL_NAMES = list(TARGET_MAP)
CORNER_NAMES = list(TOP_LEFT_OHE_MAP)

# Split names stay as the project uses them (`val`, not HF's conventional `validation`) so they
# match `setup_split` in data.csv, the benchmark's `--splits val test`, and evaluation/results.csv.
SPLITS = ("train", "val", "test")

# Calibration fields that name the unpublished raw frame or a wall-clock capture time.
CALIBRATION_DROP = ("timestamp", "raw_image_path")

# One shard per ~400 MB of encoded rows.
SHARD_BYTES = 400 * 1024 * 1024

SQUARE_FEATURES = Features(
    {
        "setup_id": Value("string"),
        "image_id": Value("string"),
        "square": Value("string"),
        "label": ClassLabel(names=LABEL_NAMES),
        "image": Image(),
        "mask": Image(),
        "annotated": Image(),
        "top": Value("int32"),
        "left": Value("int32"),
        "top_left_corner": ClassLabel(names=CORNER_NAMES),
        "valid_game_position": Value("bool"),
        "fen_matches_pixels": Value("bool"),
        "board_fen": Value("string"),
        "previous_board_fen": Value("string"),
        "move_uci": Value("string"),
    }
)

BOARD_FEATURES = Features(
    {
        "setup_id": Value("string"),
        "image_id": Value("string"),
        "warped_image": Image(),
        "labels": Sequence(ClassLabel(names=LABEL_NAMES), length=64),
        "top_left_corner": ClassLabel(names=CORNER_NAMES),
        "valid_game_position": Value("bool"),
        "legal_move_mode": Value("bool"),
        "fen_matches_pixels": Value("bool"),
        "board_fen": Value("string"),
        "previous_board_fen": Value("string"),
        "move_uci": Value("string"),
    }
)


def build_id_maps(data: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Renumber setups and boards in capture order.

    The ids are ``%Y-%m-%d_%H%M%S`` stamps, so sorting them lexicographically sorts them
    chronologically; the published numbering therefore preserves the order captures happened in
    without publishing the clock times themselves.
    """
    setup_ids = sorted(data["setup_id"].unique().to_list())
    setup_map = {old: f"setup_{i:02d}" for i, old in enumerate(setup_ids)}

    pairs = data.select("setup_id", "image_id").unique().sort(["setup_id", "image_id"]).rows()
    board_map = {image_id: f"board_{i:03d}" for i, (_, image_id) in enumerate(pairs)}
    return setup_map, board_map


def encode_png(array: np.ndarray) -> bytes:
    """PNG-encode an RGB or single-channel uint8 array.

    PNG is lossless, so decoding the result reproduces ``array`` exactly -- which is what lets the
    4-channel model input be published as two ordinary images instead of a raw ``.npy`` blob.
    """
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buffer.tobytes()


def read_top_left_corner(setup_dir: Path) -> str:
    """The board corner that is top-left in the camera image -- the model's only metadata input."""
    with open(setup_dir / "calibration_metadata.json", encoding="utf-8") as f:
        return json.load(f)["camera_natural_orientation"]["order"]["tl"]


def sanitise_calibration(raw: dict) -> dict:
    """Drop the capture timestamp and the pointer to the unpublished raw frame.

    Everything kept is provenance: the pixel coordinates and ``camera_intrinsics`` all describe the
    raw 1920x1080 frame, which is not published, so they document how the warp was derived rather
    than being directly usable against the published images.
    """
    return {key: value for key, value in raw.items() if key not in CALIBRATION_DROP}


def _image_field(data: bytes) -> dict:
    return {"bytes": data, "path": None}


def fen_piece_map(board_fen: str) -> dict[str, str]:
    """The 64 squares implied by a FEN, in the project's 13-class vocabulary."""
    board = chess.Board(board_fen)
    out = {}
    for square in SQUARES:
        piece = board.piece_at(chess.parse_square(square))
        out[square] = piece.symbol() if piece else "empty"
    return out


def build_board_info(data: pl.DataFrame) -> dict[tuple[str, str], dict]:
    """Per-board labels (from the pixels) plus whether the recorded FEN agrees with them.

    Computed once here rather than per row so the ``squares`` and ``boards`` configs cannot end up
    disagreeing about the same board.
    """
    info: dict[tuple[str, str], dict] = {}
    for (setup_id, image_id), group in data.group_by(["setup_id", "image_id"]):
        first = group.row(0, named=True)
        labels = dict(zip(group["square"].to_list(), group["label"].to_list()))
        board_fen = first["board_fen"] or ""
        info[(setup_id, image_id)] = {
            "labels": labels,
            # A board with no FEN records no position at all, so there is nothing for the pixels to
            # agree with; False keeps it out of move evaluation, which is where the flag is used.
            "fen_matches_pixels": bool(board_fen) and labels == fen_piece_map(board_fen),
            # Straight from the CSV, which is where the operator's corrections landed. metadata.json
            # still reports valid_game_position=True on 4 boards the CSV marks False, so reading
            # these from there would make the two configs disagree and would admit boards the
            # benchmark itself excludes.
            "valid_game_position": first["valid_game_position"],
            "board_fen": board_fen,
            "previous_board_fen": first["previous_board_fen"] or "",
            "move_uci": first["move_uci"] or "",
        }
    return info


def iter_square_rows(data: pl.DataFrame, setup_map, board_map, corners, data_root: Path,
                     board_info: dict):
    for row in data.iter_rows(named=True):
        setup_id, image_id, square = row["setup_id"], row["image_id"], row["square"]
        square_dir = data_root / setup_id / image_id / "squares" / square
        info = board_info[(setup_id, image_id)]

        masked = np.load(square_dir / f"{square}_masked.npy")
        if masked.shape != (144, 144, 4):
            raise ValueError(f"{square_dir}: expected (144,144,4), got {masked.shape}")

        with open(square_dir / f"{square}_metadata.json", encoding="utf-8") as f:
            geometry = json.load(f)

        yield {
            "setup_id": setup_map[setup_id],
            "image_id": board_map[image_id],
            "square": square,
            "label": TARGET_MAP[info["labels"][square]],
            "image": _image_field(encode_png(masked[..., :3])),
            # The mask is stored {0,1} on disk; scale to {0,255} so it is a viewable image rather
            # than one that renders as uniform black. Undo with `mask // 255` -- see the card.
            "mask": _image_field(encode_png(masked[..., 3] * 255)),
            "annotated": _image_field((square_dir / f"{square}_annotated.png").read_bytes()),
            "top": geometry["top"],
            "left": geometry["left"],
            "top_left_corner": TOP_LEFT_OHE_MAP[corners[setup_id]],
            "valid_game_position": row["valid_game_position"],
            "fen_matches_pixels": info["fen_matches_pixels"],
            "board_fen": row["board_fen"] or "",
            "previous_board_fen": row["previous_board_fen"] or "",
            "move_uci": row["move_uci"] or "",
        }


def iter_board_rows(data: pl.DataFrame, setup_map, board_map, corners, data_root: Path,
                    board_info: dict):
    boards = data.select("setup_id", "image_id").unique().sort(["setup_id", "image_id"])
    for setup_id, image_id in boards.rows():
        board_dir = data_root / setup_id / image_id
        with open(board_dir / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        info = board_info[(setup_id, image_id)]

        yield {
            "setup_id": setup_map[setup_id],
            "image_id": board_map[image_id],
            "warped_image": _image_field((board_dir / "image_warped.png").read_bytes()),
            # SQUARES order (file-major a1,a2,..,a8,b1,..) so the vector indexes the same way the
            # rest of the project does.
            "labels": [TARGET_MAP[info["labels"][square]] for square in SQUARES],
            "top_left_corner": TOP_LEFT_OHE_MAP[corners[setup_id]],
            "valid_game_position": info["valid_game_position"],
            # legal_move_mode is the one game field the CSV does not carry; it records how the
            # position was entered, not what is in it, so the correction pass never touched it.
            "legal_move_mode": metadata["legal_move_mode"],
            "fen_matches_pixels": info["fen_matches_pixels"],
            "board_fen": info["board_fen"],
            "previous_board_fen": info["previous_board_fen"],
            "move_uci": info["move_uci"],
        }


def write_shards(rows, features: Features, out_dir: Path, split: str, batch_size: int = 128) -> int:
    """Stream ``rows`` into parquet shards of roughly SHARD_BYTES each.

    Written straight through pyarrow rather than via ``Dataset.from_generator`` so the ~1.4 GB of
    images is encoded once instead of also being staged in the datasets cache. ``arrow_schema``
    carries the feature metadata, so the shards stay self-describing: the ClassLabel names and the
    Image type survive into ``load_dataset``.
    """
    schema = features.arrow_schema
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuilds must not inherit the previous run's shards. The final names carry an "-of-NNNNN"
    # suffix that the in-progress names do not, but both start "<split>-<digit>", so a stale
    # "train-00000-of-00002.parquet" would be swept up by the rename below and republished as
    # duplicated data. Clearing first makes a rebuild idempotent.
    for stale in out_dir.glob(f"{split}-*.parquet"):
        stale.unlink()

    written = 0
    batch: list[dict] = []
    writer: pq.ParquetWriter | None = None
    shard = 0
    shard_bytes = 0

    def flush() -> None:
        nonlocal batch, writer, shard, shard_bytes
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / f"{split}-{shard:05d}.parquet", schema)
        writer.write_table(table)
        shard_bytes += table.nbytes
        batch = []
        if shard_bytes >= SHARD_BYTES:
            writer.close()
            writer, shard, shard_bytes = None, shard + 1, 0

    for row in rows:
        batch.append(row)
        written += 1
        if len(batch) >= batch_size:
            flush()
    flush()
    if writer is not None:
        writer.close()

    # Shard filenames have to state how many shards there are, which is only known at the end.
    produced = sorted(out_dir.glob(f"{split}-[0-9][0-9][0-9][0-9][0-9].parquet"))
    for index, path in enumerate(produced):
        path.rename(out_dir / f"{split}-{index:05d}-of-{len(produced):05d}.parquet")
    return written


def _labelled_iterator(config, split_data, setup_map, board_map, corners, data_root, board_info):
    if config == "squares":
        return iter_square_rows(split_data, setup_map, board_map, corners, data_root, board_info)
    return iter_board_rows(split_data, setup_map, board_map, corners, data_root, board_info)


def build(data_root: Path = DATA_ROOT, out_dir: Path = OUT_DIR, limit: int | None = None) -> dict:
    """Build the whole export into ``out_dir``. Returns per-config, per-split row counts."""
    data = pl.read_csv(data_root / "data.csv")
    if limit is not None:
        # Whole boards, so a smoke build still has complete 64-square positions.
        keep = data.select("image_id").unique().sort("image_id").head(limit)["image_id"].to_list()
        data = data.filter(pl.col("image_id").is_in(keep))

    setup_map, board_map = build_id_maps(data)
    board_info = build_board_info(data)
    corners = {
        setup_id: read_top_left_corner(data_root / setup_id)
        for setup_id in data["setup_id"].unique().to_list()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}

    for config, features in (("squares", SQUARE_FEATURES), ("boards", BOARD_FEATURES)):
        counts[config] = {}
        for split in SPLITS:
            split_data = data.filter(pl.col("setup_split") == split)
            if split_data.is_empty():
                continue
            rows = _labelled_iterator(
                config, split_data, setup_map, board_map, corners, data_root, board_info
            )
            written = write_shards(rows, features, out_dir / "data" / config, split)
            counts[config][split] = written
            print(f"  {config:8s} {split:5s} {written:6d} rows")

    calibration = {
        setup_map[setup_id]: sanitise_calibration(
            json.loads(
                (data_root / setup_id / "calibration_metadata.json").read_text(encoding="utf-8")
            )
        )
        for setup_id in sorted(corners)
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Deliberately written OUTSIDE out_dir so that uploading out_dir cannot leak it.
    mapping_path = out_dir.parent / "id_mapping.json"
    mapping_path.write_text(
        json.dumps({"setups": setup_map, "boards": board_map}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"  id mapping (NOT published) -> {mapping_path}")

    (out_dir / "README.md").write_text(build_card(counts), encoding="utf-8")
    return counts


def counts_from_shards(out_dir: Path) -> dict:
    """Row counts read from the parquet footers -- metadata only, so no image bytes are touched."""
    counts: dict[str, dict[str, int]] = {}
    for config in ("squares", "boards"):
        config_dir = out_dir / "data" / config
        if not config_dir.is_dir():
            continue
        counts[config] = {}
        for split in SPLITS:
            shards = sorted(config_dir.glob(f"{split}-*.parquet"))
            if shards:
                counts[config][split] = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    return counts


def _config_yaml(counts: dict) -> str:
    lines = ["configs:"]
    for config in ("squares", "boards"):
        lines.append(f"- config_name: {config}")
        lines.append("  data_files:")
        for split in SPLITS:
            if split in counts.get(config, {}):
                lines.append(f"  - split: {split}")
                lines.append(f"    path: data/{config}/{split}-*.parquet")
    return "\n".join(lines)


def build_card(counts: dict) -> str:
    """The dataset card. Kept in code so the row counts quoted in it are the ones just written."""
    squares = counts.get("squares", {})
    boards = counts.get("boards", {})
    return f"""---
license: cc-by-4.0
task_categories:
- image-classification
tags:
- chess
- robotics
- board-state-recognition
{_config_yaml(counts)}
---

# Robot chess commentator: square dataset

Square crops of a real chess board photographed by a [Reachy
Mini](https://github.com/felixfabricius/robot-chess-commentator), labelled with the piece standing
on each square. {sum(squares.values()):,} labelled squares from {sum(boards.values()):,} board
positions.

The task is reading the move that was played: a board position is photographed, all 64 squares
are classified, and the legal move best explaining the change is chosen. Per-square accuracy is a
diagnostic on the way there, not the goal.

## Configs

### `squares` -- one row per square crop

| Split | Rows |
| --- | --- |
{chr(10).join(f"| `{s}` | {squares.get(s, 0):,} |" for s in SPLITS if s in squares)}

| Column | Notes |
| --- | --- |
| `image` | 144x144 RGB crop. |
| `mask` | 144x144 crop mask, `{{0, 255}}`. |
| `annotated` | The crop with its hull outline drawn on, as fed to the VLM baseline. |
| `label` | 13 classes: `empty`, `KQRBNP` (white), `kqrbnp` (black). |
| `top`, `left` | Crop origin within the padded warped board. |
| `top_left_corner` | Which board corner is top-left in the camera image. |
| `board_fen`, `previous_board_fen`, `move_uci` | Position context; empty for free-placement edits. |
| `valid_game_position` | False where the position was set up by hand rather than reached by legal play. |
| `fen_matches_pixels` | False on the 28 boards where the recorded FEN disagrees with the photograph. Filter on it for move evaluation; ignore it for classification. |

**Reconstructing the model input.** The model takes 4 channels, RGB plus the mask:

```python
import numpy as np
from datasets import load_dataset

row = load_dataset("{REPO_ID}", "squares", split="test")[0]
rgb = np.array(row["image"])                    # (144, 144, 3) uint8
mask = np.array(row["mask"]) // 255             # (144, 144)    uint8, {{0, 1}}
model_input = np.dstack([rgb, mask])            # (144, 144, 4) uint8
```

Scale to `[0, 1]` by dividing the RGB channels by 255. There is **no mean/std normalisation**.

### `boards` -- one row per captured position

`warped_image` is the top-down rectified board; `labels` is the 64-way label vector in file-major
order (`a1, a2, ..., a8, b1, ...`). Join to `squares` on `(setup_id, image_id)`.

## Splits are by setup, not by square

A *setup* is one physical calibration -- a camera pose, a board position, a lighting condition --
and contributes many positions. The 64 squares of one capture share all of that, so splitting by
square would leak. Every split here is a disjoint set of whole setups, and any re-split should
group by `setup_id` too.

## Labels

Labels come from a human driving a virtual board alongside the physical one, not from an engine and
not from the pixels. `valid_game_position=False` marks positions reached by free placement of chess pieces, which
is why some are not legal chess positions.

### `label` describes the pixels; `board_fen` describes the virtual game

Occasionally the virtual board drifted out of sync with the physical one -- a piece set down on the
wrong square, two pieces transposed at setup. When that was noticed, the labels were corrected
to match the photograph; the FEN was not. On 28 of 371 boards (52 squares) the two therefore
disagree, and `fen_matches_pixels` is `False`.

Which one you want depends on the task:

- **Square classification** -- use `label` and ignore the flag. It is what the camera saw, which is
  the only thing a classifier can learn.
- **Move estimation** -- you need `previous_board_fen` -> legal moves -> `move_uci`, and that chain
  describes the virtual game. Filter to `fen_matches_pixels` first:

```python
boards = load_dataset("felixfabricius/robot-chess-commentator-squares", "boards", split="test")
evaluable = boards.filter(
    lambda r: r["fen_matches_pixels"] and r["valid_game_position"] and r["move_uci"]
)
```

Of the 217 boards that carry a recorded move at all, exactly one (in `val`) is affected, so
this costs almost nothing in practice.

## License

CC-BY-4.0. Copyright 2026 Felix Fabricius. The code that produced it is
[GPL-3.0-or-later]({"https://github.com/felixfabricius/robot-chess-commentator"}); the trained model
is at [`felixfabricius/robot-chess-commentator-cnn`](https://huggingface.co/felixfabricius/robot-chess-commentator-cnn).
"""


def push(out_dir: Path = OUT_DIR, repo_id: str = REPO_ID) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    print(f"pushed -> https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--limit", type=int, help="only the first N boards, for a smoke build")
    parser.add_argument("--push", action="store_true", help="upload after building")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="rewrite README.md from the shards already in --out-dir, without rebuilding them",
    )
    args = parser.parse_args()

    if args.card_only:
        counts = counts_from_shards(args.out_dir)
        (args.out_dir / "README.md").write_text(build_card(counts), encoding="utf-8")
        print(f"rewrote {args.out_dir / 'README.md'} from existing shards: {counts}")
    else:
        print(f"building {args.out_dir} from {args.data_root}")
        counts = build(args.data_root, args.out_dir, args.limit)
        total = sum(sum(splits.values()) for splits in counts.values())
        print(f"built {total} rows")

    if args.push:
        push(args.out_dir, args.repo_id)


if __name__ == "__main__":
    main()
