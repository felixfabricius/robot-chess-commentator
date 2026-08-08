"""Camera lens undistortion helpers.

The Reachy Mini Lite camera ships factory intrinsics/distortion (``K``, ``D``) that are
calibrated at a 3840x2592 reference resolution.  Captured frames are 1920x1080, so ``K`` must
be scaled to the capture resolution (``D`` is dimensionless and is used unchanged).

The pure functions here (:func:`build_undistort_maps`, :func:`undistort`) take ``K``/``D`` as
arguments and never import ``reachy_mini`` — they are cheap to unit-test and are what the
offline batch uses (reading the scaled ``K``/``D`` cached in ``calibration_metadata.json``).
Only :func:`get_lite_camera_KD` reaches into ``reachy_mini`` (lazily), and is called once at
calibration time to fetch + scale the factory constants.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``cv2.remap`` lookup maps for lens undistortion.

    Computed once per process/setup (not per frame).  ``newCameraMatrix`` is ``K`` itself so
    the undistorted frame keeps the same intrinsics — the homography built from clicks on the
    undistorted image then absorbs any framing choice.

    Args:
        K: 3x3 intrinsic matrix, scaled to ``image_size``.
        D: distortion coefficients (supports OpenCV's 4/5/8/12/14-length models).
        image_size: ``(width, height)`` of the frames the maps will be applied to.

    Returns:
        ``(map1, map2)`` suitable for :func:`undistort` / ``cv2.remap``.
    """
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    width, height = int(image_size[0]), int(image_size[1])
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (width, height), cv2.CV_16SC2
    )
    return map1, map2


def undistort(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """Apply precomputed undistortion maps to a frame (thin ``cv2.remap`` wrapper)."""
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def get_lite_camera_KD(
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the Reachy Mini Lite factory intrinsics, scaled to the capture resolution.

    Mirrors ``reachy_mini.media.camera_base.set_resolution``: the factory ``K`` is defined at
    the 3840x2592 reference, and is rescaled to the target resolution with that resolution's
    ``crop_factor``.  ``D`` is returned unchanged.

    Args:
        image_size: ``(width, height)`` of the captured frames.  Defaults to the Lite's
            default capture resolution (1920x1080).

    Returns:
        ``(K, D)`` — a 3x3 float64 matrix scaled to ``image_size`` and the raw distortion
        coefficients.
    """
    # Lazy import: reachy_mini lives in the optional `robot` dependency group.
    from reachy_mini.media.camera_constants import (
        CameraResolution,
        ReachyMiniLiteCamSpecs,
    )
    from reachy_mini.media.camera_utils import scale_intrinsics

    spec = ReachyMiniLiteCamSpecs
    reference = CameraResolution.R3840x2592at30fps.value
    reference_size = (reference[0], reference[1])

    default = spec.default_resolution.value  # (width, height, fps, crop_factor)
    if image_size is None:
        target_size = (default[0], default[1])
        crop_scale = default[3]
    else:
        target_size = (int(image_size[0]), int(image_size[1]))
        crop_scale = _crop_factor_for_size(target_size, spec, default)

    K = scale_intrinsics(
        np.asarray(spec.K, dtype=np.float64),
        reference_size,
        target_size,
        crop_scale,
    )
    D = np.asarray(spec.D, dtype=np.float64)
    return K, D


def _crop_factor_for_size(target_size, spec, default) -> float:
    """crop_factor of the camera resolution matching ``target_size`` (falls back to default)."""
    if (default[0], default[1]) == tuple(target_size):
        return default[3]
    for resolution in getattr(spec, "available_resolutions", []):
        value = resolution.value
        if (value[0], value[1]) == tuple(target_size):
            return value[3]
    # Unknown resolution: default crop_factor keeps behaviour sane for the standard capture.
    return default[3]
