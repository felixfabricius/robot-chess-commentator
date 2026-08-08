"""Tests for the pure calibration helpers (no reachy hardware, no cv2 GUI).

Importing ``chess_commentator.calibration`` must not require ``reachy_mini`` — the robot import is
lazy — so these run even without the ``robot`` group (though the suite installs it anyway).
"""

import numpy as np
import pytest

from chess_commentator.calibration import (
    build_calibration_metadata,
    build_inspector_calibration,
    compute_inspector_results,
    derive_center_px,
    render_full_overlay,
)


def _base_and_extended_points(s=0.15, apex=(960.0, -600.0)):
    actual = {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]}
    apex_arr = np.array(apex)
    extended = {
        k: (np.array(v, float) + s * (apex_arr - np.array(v, float))).tolist()
        for k, v in actual.items()
    }
    order = {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}
    center_base = np.array(derive_center_px(actual, order))
    extended["center"] = (center_base + s * (apex_arr - center_base)).tolist()
    return actual, extended, center_base


def test_derive_center_px_square_board():
    actual = {"a8": [0, 0], "h8": [400, 0], "h1": [400, 400], "a1": [0, 400]}
    order = {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}
    assert derive_center_px(actual, order, board_size=400) == pytest.approx([200.0, 200.0], abs=1e-3)


def test_derive_center_px_is_diagonal_intersection_for_trapezoid():
    # Perspective trapezoid; the board centre maps to the intersection of the quad's diagonals.
    actual = {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]}
    order = {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}
    assert derive_center_px(actual, order, board_size=400) == pytest.approx([960.0, 462.5], abs=1e-2)


def test_build_calibration_metadata_preserves_and_adds():
    existing = {"height_mm": 8, "pitch_deg": 26, "timestamp": "t0", "keep_me": 123}
    actual = {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]}
    extended = {"a8": [710, 260], "h8": [1210, 260], "h1": [1470, 760], "a1": [440, 760]}

    md = build_calibration_metadata(
        existing=existing,
        actual_corners_px=actual,
        extended_corners_px=extended,
        extended_center_px=[960, 380],
        K=np.eye(3),
        D=np.zeros(12),
        image_size=(1920, 1080),
        raw_image_path="setup/raw.png",
    )

    # Pre-existing fields survive untouched.
    assert md["keep_me"] == 123
    assert md["height_mm"] == 8
    assert md["timestamp"] == "t0"
    # New v2 fields added.
    assert md["calibration_version"] == 2
    assert md["extended_center_px"] == [960, 380]
    assert md["extended_corners_px"] == extended
    assert "center_px" in md and len(md["center_px"]) == 2
    assert md["camera_intrinsics"]["image_size"] == [1920, 1080]
    assert len(md["camera_intrinsics"]["D"]) == 12
    assert md["camera_natural_orientation"]["order"]["tl"] in {"a1", "a8", "h8", "h1"}
    assert md["raw_image_path"] == "setup/raw.png"
    assert md["center_measured"] is True  # defaults to measured


def test_build_calibration_metadata_records_unmeasured_center():
    actual = {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]}
    extended = {"a8": [690, 250], "h8": [1230, 250], "h1": [1470, 760], "a1": [440, 760]}
    interpolated_center = list(np.mean([extended[k] for k in extended], axis=0))
    md = build_calibration_metadata(
        existing={},
        actual_corners_px=actual,
        extended_corners_px=extended,
        extended_center_px=interpolated_center,
        K=np.eye(3),
        D=np.zeros(12),
        image_size=(1920, 1080),
        center_measured=False,
    )
    assert md["center_measured"] is False
    assert md["extended_center_px"] == interpolated_center


def test_build_calibration_metadata_round_trips_through_processor():
    from chess_commentator.image_processing import Processor

    # Extended points converge at an apex above the board (so V is well-defined).
    apex = np.array([960.0, -600.0])
    actual = {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]}
    s = 0.15
    extended = {
        k: (np.array(v, float) + s * (apex - np.array(v, float))).tolist()
        for k, v in actual.items()
    }
    center_base = np.array(derive_center_px(actual, {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}))
    extended_center = (center_base + s * (apex - center_base)).tolist()

    md = build_calibration_metadata(
        existing={},
        actual_corners_px=actual,
        extended_corners_px=extended,
        extended_center_px=extended_center,
        K=np.eye(3),
        D=np.zeros(12),
        image_size=(1920, 1080),
    )
    processor = Processor(md, None)
    assert processor.is_v2 is True
    assert processor.square_geometry is not None
    assert len(processor.square_geometry) == 64
    assert processor.vp_residual < 1.0


# --------------------------------------------------------------------------------------------
# Headless calibration-UI helpers (the interactive event loop is validated by hand)
# --------------------------------------------------------------------------------------------

def test_build_inspector_calibration_keys():
    base, extended, _ = _base_and_extended_points()
    calib = build_inspector_calibration(base, extended)
    assert set(calib) == {
        "camera_natural_orientation",
        "actual_corners_px",
        "extended_corners_px",
        "extended_center_px",
    }
    assert "center" not in calib["extended_corners_px"]
    assert calib["extended_center_px"] == extended["center"]


def test_compute_inspector_results_camera_frame_shape():
    base, extended, _ = _base_and_extended_points()
    calib = build_inspector_calibration(base, extended)
    results = compute_inspector_results(calib, None)
    assert len(results) == 64
    for r in results:
        assert r.floor_cam.shape == (4, 2)
        assert r.ceiling_cam.shape == (4, 2)
        assert r.bbox_cam.shape == (4, 2)


def test_render_full_overlay_produces_frame_shaped_image():
    base, extended, center_base = _base_and_extended_points()
    calib = build_inspector_calibration(base, extended)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    results = compute_inspector_results(calib, None)

    plain = render_full_overlay(frame, base, center_base, extended, highlight=None)
    assert plain.shape == frame.shape

    highlighted = render_full_overlay(frame, base, center_base, extended, highlight=results[10])
    assert highlighted.shape == frame.shape
    # The highlight actually draws something on the frame.
    assert not np.array_equal(plain, highlighted)
