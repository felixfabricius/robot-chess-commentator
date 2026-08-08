"""Live calibration monitor.

Shows Reachy's camera feed — undistorted with the *exact* same maps the processing pipeline
uses (see ``Processor.__init__`` / ``camera_utils``), i.e. the same correction applied to the
frames shown for annotation and to the frames captured during gameplay — with the calibrated
board corners (``actual_corners_px``) and the quadrilateral connecting them drawn on top.

Because the overlay is fixed (it comes from the calibration metadata) while the video is live,
the corners will sit exactly on the physical board corners *only while the camera is at the pose
the board was calibrated at*. If the head/camera drifts, you will see the green quad peel away
from the real board edges — an at-a-glance check for the pose-repeatability problem.

Launched from ``main()`` right after ``setup()`` via :func:`launch_calibration_monitor`, which
runs the viewer in its own daemon process so the game loop is never blocked.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import cv2
import numpy as np

from chess_assistant.camera_utils import build_undistort_maps, undistort

WINDOW_NAME = "Reachy calibration monitor (press q to close)"
LABEL_ORDER = ["a1", "a8", "h8", "h1"]
_MARKER_COLOR = (0, 0, 255)   # red corner dots + labels
_LINE_COLOR = (0, 255, 0)     # green connecting quadrilateral
_TEXT_COLOR = (0, 255, 255)   # yellow status text


def _corner_polygon_order(metadata: dict) -> list[str]:
    """Corner labels in a non-crossing cyclic order (tl -> tr -> br -> bl when available).

    Falls back to ``LABEL_ORDER`` if the natural-orientation mapping is missing so the overlay
    still works on older metadata."""
    order = metadata.get("camera_natural_orientation", {}).get("order")
    corners = metadata.get("actual_corners_px", {})
    if isinstance(order, dict) and all(k in order for k in ("tl", "tr", "br", "bl")):
        seq = [order["tl"], order["tr"], order["br"], order["bl"]]
        if all(label in corners for label in seq):
            return seq
    return [label for label in LABEL_ORDER if label in corners]


def _load_overlay(metadata_path: str):
    """Read the calibration metadata into (corner labels, Nx2 int points, undistort maps).

    The undistort maps are built identically to ``Processor.__init__`` (v2 calibrations only);
    when absent (v1 metadata) they are ``None`` and the raw frame is shown, exactly as
    ``Processor.warp`` would behave."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    order = _corner_polygon_order(metadata)
    corners = metadata["actual_corners_px"]
    points = np.array([corners[label] for label in order], dtype=np.int32)

    map1 = map2 = None
    intrinsics = metadata.get("camera_intrinsics")
    if intrinsics:
        width, height = intrinsics["image_size"]
        map1, map2 = build_undistort_maps(
            np.asarray(intrinsics["K"], dtype=np.float64),
            np.asarray(intrinsics["D"], dtype=np.float64),
            (width, height),
        )
    return order, points, map1, map2


def _draw_overlay(frame: np.ndarray, order: list[str], points: np.ndarray, status: str) -> np.ndarray:
    cv2.polylines(frame, [points.reshape(-1, 1, 2)], isClosed=True, color=_LINE_COLOR, thickness=2)
    for label, (x, y) in zip(order, points):
        cv2.circle(frame, (int(x), int(y)), 6, _MARKER_COLOR, -1)
        cv2.putText(frame, label, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _MARKER_COLOR, 2)
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 2)
    return frame


def _monitor_worker(metadata_path: str, media_backend: str, fps: int) -> None:
    """Process entry point: open a camera connection and stream the annotated feed."""
    try:
        order, points, map1, map2 = _load_overlay(metadata_path)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never take the game down
        print(f"[calibration-monitor] could not load calibration metadata: {exc}")
        return

    try:
        from reachy_mini import ReachyMini
    except Exception as exc:  # noqa: BLE001
        print(f"[calibration-monitor] reachy_mini unavailable: {exc}")
        return

    period = 1.0 / max(fps, 1)
    undistorted = map1 is not None
    frames = 0
    started = time.monotonic()
    try:
        with ReachyMini(media_backend=media_backend) as mini:
            print("[calibration-monitor] live. Press 'q' in the window to close.")
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(period)
                    continue
                frame = undistort(frame, map1, map2) if undistorted else frame.copy()
                frames += 1
                elapsed = time.monotonic() - started
                status = (
                    f"{'undistorted' if undistorted else 'RAW (no intrinsics)'} | "
                    f"frame {frames} | {frames / elapsed:.1f} fps"
                )
                _draw_overlay(frame, order, points, status)
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(max(int(period * 1000), 1)) & 0xFF) == ord("q"):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break  # user closed the window
    except Exception as exc:  # noqa: BLE001
        print(f"[calibration-monitor] stopped: {exc}")
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:  # noqa: BLE001
            pass


def launch_calibration_monitor(metadata_path, *, media_backend: str = "default", fps: int = 10):
    """Spawn the live calibration monitor in a separate daemon process; returns the handle.

    Non-blocking — ``main()`` continues immediately. The process is a daemon, so it dies with
    the parent. It opens its **own** ``ReachyMini`` connection to pull frames (the game loop's
    ``mini`` cannot be shared across processes).

    NOTE: this means a second simultaneous camera consumer. If your media backend does not allow
    two consumers, the monitor prints a message and exits without disturbing the game; in that
    case check the calibration on its own instead, with the game stopped, by running this module
    directly (see ``__main__`` below).
    """
    proc = mp.Process(
        target=_monitor_worker,
        args=(str(metadata_path), media_backend, fps),
        daemon=True,
        name="calibration-monitor",
    )
    proc.start()
    return proc


if __name__ == "__main__":
    # Standalone viewer, for checking a calibration with the game loop stopped (the same window
    # launch_calibration_monitor spawns, minus the second camera consumer):
    #
    #   uv run python -m chess_assistant.calibration_monitor data/<setup>/calibration_metadata.json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m chess_assistant.calibration_monitor <calibration_metadata.json>"
        )
    _monitor_worker(sys.argv[1], media_backend="default", fps=10)
