"""Tests for camera undistortion helpers.

These incorporate the real ``reachy_mini`` intrinsics path (requires the ``robot`` dependency
group: ``uv sync --group robot``); the constants module is pure numpy and imports headlessly.
"""

import numpy as np
import pytest

from chess_commentator.camera_utils import (
    build_undistort_maps,
    get_lite_camera_KD,
    undistort,
)


def test_get_lite_camera_KD_scales_to_capture_resolution():
    K, D = get_lite_camera_KD((1920, 1080))

    assert K.shape == (3, 3)
    assert D.reshape(-1).shape == (12,)

    # Factory K is defined at 3840x2592; captured frames are 1920x1080 with crop_factor 1.115.
    # fx/fy scale by (target/ref) * crop; cx/cy scale by (target/ref).
    fx_expected = 2001.8076426486707 * (1920 / 3840) * 1.115
    fy_expected = 2003.0778885944105 * (1080 / 2592) * 1.115
    cx_expected = (1905.876059826701 / 3840) * 1920
    cy_expected = (1328.3239717935594 / 2592) * 1080

    assert K[0, 0] == pytest.approx(fx_expected, rel=1e-6)
    assert K[1, 1] == pytest.approx(fy_expected, rel=1e-6)
    assert K[0, 2] == pytest.approx(cx_expected, rel=1e-6)
    assert K[1, 2] == pytest.approx(cy_expected, rel=1e-6)


def test_get_lite_camera_KD_defaults_to_lite_resolution():
    K_default, D_default = get_lite_camera_KD()
    K_explicit, D_explicit = get_lite_camera_KD((1920, 1080))
    assert np.allclose(K_default, K_explicit)
    assert np.allclose(D_default, D_explicit)


def test_build_undistort_maps_shapes():
    K, D = get_lite_camera_KD((1920, 1080))
    map1, map2 = build_undistort_maps(K, D, (1920, 1080))
    # CV_16SC2 fixed-point maps: map1 holds (x, y) pairs, map2 the interpolation table.
    assert map1.shape == (1080, 1920, 2)
    assert map1.dtype == np.int16
    assert map2.shape == (1080, 1920)


def test_undistort_identity_is_noop():
    # Zero distortion with newCameraMatrix == K must reproduce the input exactly.
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    map1, map2 = build_undistort_maps(K, D, (1920, 1080))
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    out = undistort(frame, map1, map2)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)


def test_undistort_real_distortion_changes_frame():
    K, D = get_lite_camera_KD((1920, 1080))
    map1, map2 = build_undistort_maps(K, D, (1920, 1080))
    frame = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    out = undistort(frame, map1, map2)
    assert out.shape == frame.shape
    # Strong lens distortion (k1 ~= -1.47) must actually move pixels.
    assert not np.array_equal(out, frame)
