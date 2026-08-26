"""Shared test fixtures — synthetic v2 calibration setups (no pygame/robot hardware)."""

import json

import cv2
import numpy as np
import pytest

from chess_commentator.board import SQUARES


def _intersect(p1, p2, p3, p4):
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    d1, d2 = p2 - p1, p4 - p3
    t = np.linalg.solve(np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]]), p3 - p1)
    return p1 + t[0] * d1


def build_v2_metadata(apex=(960.0, -600.0), s=0.18, with_intrinsics=False):
    """A synthetic v2 calibration dict: a perspective trapezoid whose extension rays converge
    at ``apex`` (so V == H(apex) exactly). ``with_intrinsics`` folds in the real Reachy Lite
    K/D via :func:`get_lite_camera_KD`."""
    corner_map = {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}
    actual = {
        "a8": [700.0, 300.0],
        "h8": [1220.0, 300.0],
        "h1": [1500.0, 800.0],
        "a1": [420.0, 800.0],
    }
    apex_arr = np.array(apex)
    extended = {
        label: (np.array(px) + s * (apex_arr - np.array(px))).tolist()
        for label, px in actual.items()
    }
    center_base = _intersect(actual["a8"], actual["h1"], actual["h8"], actual["a1"])
    extended_center = (center_base + s * (apex_arr - center_base)).tolist()

    metadata = {
        "calibration_version": 2,
        "camera_natural_orientation": {"order": corner_map},
        "actual_corner_order": ["a1", "a8", "h8", "h1"],
        "actual_corners_px": actual,
        "extended_corner_order": ["a1", "a8", "h8", "h1"],
        "extended_corners_px": extended,
        "extended_center_px": extended_center,
    }
    if with_intrinsics:
        from chess_commentator.perception.undistort import get_lite_camera_KD

        K, D = get_lite_camera_KD((1920, 1080))
        metadata["camera_intrinsics"] = {
            "K": K.tolist(),
            "D": D.reshape(-1).tolist(),
            "image_size": [1920, 1080],
        }
    return metadata


@pytest.fixture
def make_v2_setup(tmp_path):
    """Factory building a synthetic v2 setup on disk with labelled per-square metadata.

    Returns ``(setup_dir, [frame_dir, ...], labels)``; ``labels`` maps each square to the
    distinct sentinel label seeded into every frame's per-square metadata JSON, so a test can
    assert the labels survive regeneration byte-for-byte.
    """

    def _make(n_frames=1, with_intrinsics=True):
        setup_dir = tmp_path / "setup"
        setup_dir.mkdir()
        (setup_dir / "calibration_metadata.json").write_text(
            json.dumps(build_v2_metadata(with_intrinsics=with_intrinsics))
        )
        labels = {sq: f"lbl_{sq}" for sq in SQUARES}
        frame_dirs = []
        for i in range(n_frames):
            frame_dir = setup_dir / f"board_frame{i}"
            frame_dir.mkdir()
            frame = np.random.default_rng(i).integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
            cv2.imwrite(str(frame_dir / "image.png"), frame)
            for sq in SQUARES:
                sq_dir = frame_dir / "squares" / sq
                sq_dir.mkdir(parents=True)
                (sq_dir / f"{sq}_metadata.json").write_text(
                    json.dumps({"top": 999, "left": 999, "label": labels[sq]})
                )
            frame_dirs.append(frame_dir)
        return setup_dir, frame_dirs, labels

    return _make
