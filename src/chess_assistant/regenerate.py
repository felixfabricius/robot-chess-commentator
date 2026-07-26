"""Offline batch regeneration of warped images + per-square crops (Phase 2).

Two-phase workflow:

* **Phase 1 — relabelling** (``chess_assistant.calibration.relabel_existing_setups``): the
  interactive session that opens each existing setup's ``raw.png`` undistorted and writes an
  updated, versioned ``calibration_metadata.json`` (adds the centre point + camera intrinsics).
* **Phase 2 — regeneration** (this module): for each setup, build its :class:`Processor` once
  (freezing the undistortion maps, homography, vanishing point, magnitude field and all 64
  square geometries) and reuse it across every frame; per frame only undistort -> warp ->
  cutout -> merge the preserved label back in.

**Labels are never derivable from pixels** — they were written by the live capture session and
live only in each square's ``_metadata.json`` (and the CSV). So each square's existing
``"label"`` is read *before* cutout overwrites its metadata, then merged back afterwards.

Reads camera intrinsics from each setup's metadata, so the batch never imports ``reachy_mini``
and the worker is a plain top-level function (importable by name under Windows "spawn").

Run with ``uv run python -m chess_assistant.regenerate [--data-root DIR] [--config FILE]
[--workers N]``; the defaults regenerate everything under ``data/generated`` using ``config.yaml``
and one worker process per CPU.

**Mask/crop ablation variants.** ``--variant <name>`` (see
``image_processing.MASK_VARIANTS``) instead writes a self-contained tree at
``<data-root>_<variant>`` — the same boards cut a deliberately worse way — leaving the original
untouched. Each variant tree gets the per-setup calibration metadata and its own ``data.csv``
with the path columns repointed, so a training run only needs
``data.csv_path=data/generated_<variant>/data.csv``::

    uv run python -m chess_assistant.regenerate --variant square_global
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from chess_assistant.image_processing import MASK_VARIANTS, Processor

DATA_ROOT = Path("data") / "generated"

# The CSV columns that hold file locations, and so have to be repointed when a variant tree is
# written next to the original one.
PATH_COLUMNS = ("square_image_path", "full_image_path", "calibration_metadata_path")

# Per-worker-process cache: one Processor per (setup, mask/crop mode), built the first time a
# worker sees that combination and reused for all of the setup's frames handled by the worker.
# The modes are part of the key because they change the geometry the Processor freezes.
_PROCESSOR_CACHE: dict[tuple, Processor] = {}


def _get_processor(
    setup_dir: Path,
    config_path: str | None,
    mask_mode: str | None = None,
    crop_mode: str | None = None,
) -> Processor:
    key = (str(setup_dir), mask_mode, crop_mode)
    processor = _PROCESSOR_CACHE.get(key)
    if processor is None:
        processor = Processor(
            setup_dir / "calibration_metadata.json",
            config_path,
            mask_mode=mask_mode,
            crop_mode=crop_mode,
        )
        _PROCESSOR_CACHE[key] = processor
    return processor


def write_variant_csv(src_csv: Path, src_root: Path, dst_root: Path, dst_csv: Path | None = None) -> Path:
    """Copy ``src_csv`` with every path column repointed from ``src_root`` to ``dst_root``.

    Labels, splits and FENs are copied untouched: a variant tree holds exactly the same boards
    cut a different way, so only the file locations change. Legacy rows written on Windows with
    backslash separators are normalised to posix on the way through (matching model/data.py).
    """
    dst_csv = Path(dst_csv) if dst_csv else Path(dst_root) / "data.csv"
    src_prefix = Path(src_root).as_posix().rstrip("/") + "/"
    dst_prefix = Path(dst_root).as_posix().rstrip("/") + "/"

    data = pl.read_csv(src_csv)
    data = data.with_columns(
        [
            pl.col(column)
            .str.replace_all("\\", "/", literal=True)
            .str.replace(src_prefix, dst_prefix, literal=True)
            .alias(column)
            for column in PATH_COLUMNS
            if column in data.columns
        ]
    )
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(dst_csv)
    return dst_csv


def read_existing_labels(squares_dir: Path) -> dict[str, str]:
    """Collect the ``"label"`` field from every existing per-square metadata JSON."""
    labels: dict[str, str] = {}
    if not squares_dir.exists():
        return labels
    for square_dir in squares_dir.iterdir():
        if not square_dir.is_dir():
            continue
        square = square_dir.name
        meta_path = square_dir / f"{square}_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "label" in meta:
                labels[square] = meta["label"]
    return labels


def merge_preserved_labels(squares_dir: Path, labels: dict[str, str]) -> None:
    """Re-attach preserved labels to the freshly written per-square metadata JSONs."""
    for square, label in labels.items():
        meta_path = squares_dir / square / f"{square}_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["label"] = label
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def regenerate_frame(
    setup_dir_str: str,
    frame_dir_str: str,
    config_path_str: str | None,
    mask_mode: str | None = None,
    crop_mode: str | None = None,
    out_frame_dir_str: str | None = None,
) -> str:
    """Regenerate one frame's warped image + square crops, preserving labels.

    Writes in place unless ``out_frame_dir_str`` points elsewhere, which is how the mask ablation
    builds a variant tree without disturbing the original crops. Labels are always read from the
    *source* frame and merged into whichever squares dir was written.

    Top-level and picklable so it can be submitted to a ``ProcessPoolExecutor`` under spawn.
    """
    setup_dir = Path(setup_dir_str)
    frame_dir = Path(frame_dir_str)
    out_frame_dir = Path(out_frame_dir_str) if out_frame_dir_str else frame_dir

    # Capture the ground-truth labels BEFORE cutout overwrites the per-square metadata.
    preserved_labels = read_existing_labels(frame_dir / "squares")

    processor = _get_processor(setup_dir, config_path_str, mask_mode, crop_mode)
    warped_path = processor.warp(
        frame_dir / "image.png", out_path=out_frame_dir / "image_warped.png"
    )
    processor.cutout(warped_path)

    merge_preserved_labels(out_frame_dir / "squares", preserved_labels)
    return frame_dir_str


def iter_frames(data_root: Path):
    """Yield ``(setup_dir, frame_dir)`` for every regenerable frame under ``data_root``."""
    for setup_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not (setup_dir / "calibration_metadata.json").exists():
            continue
        for frame_dir in sorted(setup_dir.glob("board_*")):
            if (frame_dir / "image.png").exists():
                yield setup_dir, frame_dir


def _print_progress(done: int, total: int, failures: int) -> None:
    """Overwrite a single-line progress bar in place (dependency-free)."""
    width = 30
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    pct = (100 * done / total) if total else 100.0
    suffix = f"  {failures} failed" if failures else ""
    print(f"\r  [{bar}] {done}/{total} ({pct:3.0f}%){suffix}", end="", flush=True)


def regenerate_all(
    data_root=DATA_ROOT,
    config_path="config.yaml",
    max_workers=None,
    variant: str | None = None,
    out_root=None,
) -> None:
    """Regenerate every frame under ``data_root`` using one persistent process pool.

    With no ``variant`` this rewrites the crops in place (the original behaviour). With one, it
    builds a **self-contained variant tree** instead and leaves ``data_root`` untouched: crops cut
    the variant's way, each setup's ``calibration_metadata.json`` copied across (board-level eval
    reads it by path), and a ``data.csv`` whose path columns point into the new tree. Train
    against it with ``data.csv_path=<out_root>/data.csv``, which is also what evaluate() derives
    its board-level data root from.
    """
    data_root = Path(data_root)
    mask_mode, crop_mode = None, None
    if variant is not None:
        if variant not in MASK_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(MASK_VARIANTS)}. Got {variant!r}.")
        mask_mode, crop_mode = MASK_VARIANTS[variant]

    if out_root is None:
        out_root = (
            data_root if variant is None
            else data_root.with_name(f"{data_root.name}_{variant}")
        )
    out_root = Path(out_root)
    in_place = out_root == data_root

    tasks = list(iter_frames(data_root))
    total = len(tasks)
    suffix = f" [{variant}: mask={mask_mode}, crop={crop_mode}] -> {out_root}" if variant else ""
    print(f"Regenerating {total} frames{suffix}...")
    if total == 0:
        return

    if not in_place:
        # Copy the per-setup calibration across first: the dataset resolves crops from the CSV,
        # but evaluate()'s board-level pass also loads each setup's calibration metadata by path.
        for setup_dir in sorted({setup for setup, _frame in tasks}):
            dst_setup_dir = out_root / setup_dir.name
            dst_setup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                setup_dir / "calibration_metadata.json",
                dst_setup_dir / "calibration_metadata.json",
            )

    completed = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                regenerate_frame,
                str(setup),
                str(frame),
                str(config_path),
                mask_mode,
                crop_mode,
                None if in_place else str(out_root / setup.name / frame.name),
            ): frame
            for setup, frame in tasks
        }
        for future in as_completed(futures):
            frame = futures[future]
            completed += 1
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                print(f"\n  FAILED {frame}: {exc}")
            _print_progress(completed, total, failures)
    print()  # finish the progress line
    print(f"Done. {total - failures}/{total} frames regenerated.")

    if not in_place:
        src_csv = data_root / "data.csv"
        if src_csv.exists():
            dst_csv = write_variant_csv(src_csv, data_root, out_root)
            print(f"Wrote {dst_csv} (path columns repointed to {out_root}).")
        else:
            print(f"  NOTE: {src_csv} not found, so no variant data.csv was written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=sorted(MASK_VARIANTS),
        default=None,
        help=(
            "Mask/crop ablation variant. Omit to regenerate in place; pass one to build a "
            "self-contained variant tree (default <data-root>_<variant>) and leave the "
            "original untouched."
        ),
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Where to write a variant tree. Defaults to <data-root>_<variant>.",
    )
    args = parser.parse_args()
    regenerate_all(args.data_root, args.config, args.workers, args.variant, args.out_root)
