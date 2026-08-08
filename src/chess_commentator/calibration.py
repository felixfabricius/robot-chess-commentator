"""Interactive calibration: pin the head, click the board, write ``calibration_metadata.json``.

A calibration ties one physical setup (where the board is, where the robot's head was) to the
geometry the rest of the pipeline needs. The v2 metadata records, all measured on the
*undistorted* frame: the four board corners; the four "extended" corners — the top of a piece
standing on each corner square, which is what tells us how far a piece overhangs its square —
plus an extended centre; the head pose the frame was captured at; and the camera intrinsics,
cached so the offline batch never has to import ``reachy_mini``.

The pose matters as much as the clicks: the homography is only valid while the camera is where
it was when the corners were clicked. So the head is made rigid before the capture frame is
frozen (``make_head_rigid``) and driven back to the same pose before every gameplay frame
(``move_to_capture_pose``); ``calibration_monitor`` is the live check that it has not drifted.
This has revealed that the ``input.source: keyboard`` (see ``config.yaml``) is preferable to 
``input.source: robot`` for sake of image stability.

Entry points (see ``__main__``): ``calibrate`` needs the live robot; ``annotate_existing`` and
``relabel_existing_setups`` re-click a stored ``raw.png`` and do not.
"""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from chess_commentator.camera_utils import build_undistort_maps, get_lite_camera_KD, undistort
from chess_commentator.config import SQUARES

# reachy_mini (the `robot` dependency group) is imported lazily inside the functions that need
# the live robot, so the pure calibration helpers below stay importable without it.

# Safety limits on the head pose. Every requested pose goes through make_safe_pose, which clamps
# into this box, so the interactive keyboard controls below can never command the head outside
# the range that keeps the board in view and the Stewart platform within its usable travel.
MIN_HEIGHT_MM = -40
MAX_HEIGHT_MM = 21

MIN_PITCH_DEG = -28
MAX_PITCH_DEG = 28

# How far one keypress moves the head in the interactive loop (w/s: height, i/k: pitch).
HEIGHT_STEP_MM = 2
PITCH_STEP_DEG = 2

# Where the head starts: the pose the board tends to frame up well from.
OPT_HEIGHT_MM = 8
OPT_PITCH_DEG = 26

MOVE_DURATION = 0.25

# How many consecutive empty camera polls in the interactive loop mean the video pipeline is
# dead rather than merely slow to warm up. At the loop's ~1ms waitKey this is a few seconds.
NO_FRAME_LIMIT = 300


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def make_safe_pose(height_mm: float, pitch_deg: float):
    """Build a head pose from a height/pitch request, clamped into the safe range above.

    Returns ``(pose, clamped_height_mm, clamped_pitch_deg)``. Callers compare the clamped values
    against what they asked for to notice that a request was trimmed.
    """
    from reachy_mini.utils import create_head_pose

    height_mm = clamp(height_mm, MIN_HEIGHT_MM, MAX_HEIGHT_MM)
    pitch_deg = clamp(pitch_deg, MIN_PITCH_DEG, MAX_PITCH_DEG)

    pose = create_head_pose(
        z=height_mm,
        pitch=pitch_deg,
        mm=True,
        degrees=True,
    )

    return pose, height_mm, pitch_deg


def position_robot(mini, height, pitch):
    """Move the head to a (clamped) capture pose with a smooth, fixed-duration motion."""
    pose = make_safe_pose(height, pitch)[0]
    mini.goto_target(pose, duration=MOVE_DURATION)


# Settle tolerances for confirming the head has physically reached the capture pose.
SETTLE_POS_TOL_M = 0.001       # 1 mm
SETTLE_ROT_TOL_RAD = 0.0052    # ~0.3 deg
SETTLE_TIMEOUT_S = 1.5
SETTLE_POLL_S = 0.03


def make_head_rigid(mini) -> None:
    """Put the head into a stiff, non-drifting position hold so every capture is taken from
    the exact same pose. The head is a Stewart platform whose daemon default can be
    compliant/gravity-compensated (backdrivable — it does not spring back when pushed) and
    can wobble with audio; both perturb the camera pose and de-register the board from the
    calibration homography. Each command is guarded because they are daemon-side and may be
    absent/no-ops on some backends.

    Ordering note: ``enable_motors()`` pins targets to the *present* pose before enabling
    torque, so it must run here (before the ``set_target`` that drives to the capture pose),
    never after it."""
    # (method_name, args). Looked up by name inside the guard so a backend that lacks the
    # command (AttributeError) is tolerated just like one that raises when it runs.
    commands = (
        ("disable_wobbling", ()),               # stop audio-reactive head motion
        ("disable_gravity_compensation", ()),   # leave compliant/backdrivable mode
        ("set_automatic_body_yaw", (False,)),   # pin the body-yaw DOF
        ("enable_motors", ()),                  # rigid torque hold
    )
    for name, cmd_args in commands:
        try:
            getattr(mini, name)(*cmd_args)
        except Exception as exc:  # noqa: BLE001 - best-effort; backend may lack the command
            print(f"make_head_rigid: {name} skipped ({exc})")


def _head_pose_error(current, target) -> tuple[float, float]:
    """Return (translation error in metres, rotation error in radians) between two 4x4 poses."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pos_err = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    r_err = current[:3, :3].T @ target[:3, :3]
    cos_angle = (np.trace(r_err) - 1.0) / 2.0
    rot_err = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return pos_err, rot_err


def move_to_capture_pose(mini, height_mm, pitch_deg):
    """Snap the head to the stored calibration pose and wait until it has physically arrived.

    Uses ``set_target`` (immediate) with a pinned ``body_yaw`` so the camera pose is fully
    determined, then polls ``get_current_head_pose()`` until the head is within tolerance of
    the target (or a timeout elapses) instead of blindly sleeping a fixed duration."""
    pose = make_safe_pose(height_mm, pitch_deg)[0]
    mini.set_target(head=pose, body_yaw=0.0)

    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    pos_err = float("nan")
    rot_err = float("nan")
    while time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_S)
        try:
            current = mini.get_current_head_pose()
        except Exception as exc:  # noqa: BLE001 - fall back to a plain settle sleep
            print(f"move_to_capture_pose: get_current_head_pose unavailable ({exc}); using sleep")
            time.sleep(MOVE_DURATION)
            return
        pos_err, rot_err = _head_pose_error(current, pose)
        if pos_err <= SETTLE_POS_TOL_M and rot_err <= SETTLE_ROT_TOL_RAD:
            return
    print(
        f"move_to_capture_pose: head not settled within {SETTLE_TIMEOUT_S}s "
        f"(pos_err={pos_err*1000:.1f}mm, rot_err={np.rad2deg(rot_err):.2f}deg)"
    )


def _log_calibration_summary(calibration_data: dict, config_path) -> None:
    """Build a Processor from the fresh calibration and print the vanishing-point residual.

    Purely a fit-quality read-out — the square geometry does not use the vanishing point (see
    ``Processor.__init__``). A large residual usually means an extended point was mis-clicked.
    """
    try:
        from chess_commentator.image_processing import Processor

        processor = Processor(calibration_data, config_path)
        print(f"Vanishing-point residual: {processor.vp_residual:.3f} px")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not compute calibration summary: {exc})")


def calibrate(
    mini,
    setup_dir: Path = Path("data") / "raw_images",
    config_path="config.yaml",
    annotate_center: bool = False,
) -> dict | None:
    """Frame the board with the live head, then collect a v2 calibration from a frozen frame.

    Shows the camera feed with keyboard control of the pose: ``w``/``s`` raise/lower the head,
    ``i``/``k`` pitch it up/down, SPACE freezes the current frame and hands it to ``CalibrationUI``
    for clicking, ``q`` aborts. The pose held at the moment of SPACE is the capture pose gameplay
    will replay, and is stored alongside the clicks.

    Writes ``<setup_dir>/calibration_metadata.json`` (plus ``raw.png`` and ``raw_undistorted.png``)
    and returns the calibration dict, or ``None`` if the user aborted. ``annotate_center`` asks for
    a clicked extended centre rather than interpolating it from the four extended corners.
    """
    height_mm = OPT_HEIGHT_MM
    pitch_deg = OPT_PITCH_DEG

    # Stiffen the head so the frame the corners are clicked on is captured from the same rigid,
    # non-drifting pose gameplay will later reproduce (see move_to_capture_pose / make_head_rigid).
    make_head_rigid(mini)

    # Move to the initial pose right away so the live view — and any capture taken before the
    # user nudges the head — matches the pose we store. Without this the robot sits at its rest
    # pose while the metadata would claim the (untouched) OPT_* values.
    pose, height_mm, pitch_deg = make_safe_pose(height_mm, pitch_deg)
    mini.set_target(head=pose, body_yaw=0.0)
    last_sent_height = height_mm
    last_sent_pitch = pitch_deg

    # A camera whose GStreamer pipeline failed to start returns None forever. Without this guard
    # the loop below spins silently -- no window is ever shown, so waitKey has nothing to read
    # keys from and the whole pipeline looks hung. Fail loudly instead. The usual cause is
    # another process (a notebook kernel holding a `ReachyMini()`, a previous run) still owning
    # the camera: the daemon's video stream is single-consumer.
    empty_frames = 0

    while True:
        frame = mini.media.get_frame()

        if frame is None:
            empty_frames += 1
            if empty_frames > NO_FRAME_LIMIT:
                raise RuntimeError(
                    f"Camera delivered no frame in {NO_FRAME_LIMIT} consecutive polls. "
                    "The video pipeline is not running -- check for a GStreamer error above, "
                    "and make sure no other process (notebook kernel, earlier run, dashboard) "
                    "still holds the camera."
                )
        else:
            empty_frames = 0
            cv2.imshow("Reachy board view", frame)

        key = cv2.waitKey(1) & 0xFF

        new_height_mm = height_mm
        new_pitch_deg = pitch_deg

        if key == ord("w"):
            new_height_mm += HEIGHT_STEP_MM
        elif key == ord("s"):
            new_height_mm -= HEIGHT_STEP_MM
        elif key == ord("i"):
            new_pitch_deg += PITCH_STEP_DEG
        elif key == ord("k"):
            new_pitch_deg -= PITCH_STEP_DEG
        elif key == ord(" "):
            if frame is None:
                continue

            frozen_frame = frame.copy()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            setup_dir.mkdir(parents=True, exist_ok=True)
            raw_image_path = setup_dir / "raw.png"
            metadata_path = setup_dir / "calibration_metadata.json"

            # Save the original distorted capture untouched, then do all clicks + geometry
            # on the undistorted frame.
            cv2.imwrite(str(raw_image_path), frozen_frame)
            undistorted, K, D, image_size = undistort_reference_frame(frozen_frame, setup_dir)
            cv2.destroyWindow("Reachy board view")

            collected = CalibrationUI(
                undistorted, config_path, annotate_center=annotate_center
            ).run()
            if collected is None:
                cv2.destroyAllWindows()
                return None
            base_points, extended_points = collected

            # Store the pose the robot is actually holding right now (kept in sync with
            # last_sent_* below); this is the chosen capture position that gameplay replays.
            calibration_data = build_calibration_metadata(
                existing={
                    "height_mm": height_mm,
                    "pitch_deg": pitch_deg,
                    "timestamp": timestamp,
                },
                actual_corners_px=base_points,
                extended_corners_px={k: v for k, v in extended_points.items() if k != "center"},
                extended_center_px=extended_points["center"],
                K=K,
                D=D,
                image_size=image_size,
                raw_image_path=raw_image_path,
                center_measured=annotate_center,
            )

            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(calibration_data, f, indent=2)

            _log_calibration_summary(calibration_data, config_path)
            print(f"Saved raw image: {raw_image_path}")
            print(f"Saved calibration metadata: {metadata_path}")

            cv2.destroyAllWindows()
            return calibration_data

        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None
        else:
            continue

        # Clamp and move robot
        pose, safe_height, safe_pitch = make_safe_pose(new_height_mm, new_pitch_deg)

        if safe_height != new_height_mm or safe_pitch != new_pitch_deg:
            print(
                "Requested pose outside safe range. "
                f"Clamped to height={safe_height}, pitch={safe_pitch}"
            )

        if safe_height != last_sent_height or safe_pitch != last_sent_pitch:
            print(f"Moving to height={safe_height}, pitch={safe_pitch}")
            try:
                mini.set_target(head=pose, body_yaw=0.0)
                height_mm = safe_height
                pitch_deg = safe_pitch
                last_sent_height = safe_height
                last_sent_pitch = safe_pitch
                time.sleep(MOVE_DURATION)
            except Exception as e:
                print("Move failed:", e)
                print(f"Keeping previous pose: height={height_mm}, pitch={pitch_deg}")


def infer_camera_natural_corner_order(corners_px: dict[str, list[int]]) -> dict:
    """
    Infer which semantic board corner sits in each visual quadrant (top-left,
    top-right, bottom-right, bottom-left) using pixel-coordinate comparisons.

    Tries all four cyclic orderings of the four corners and scores each by
    counting how many expected image-coordinate inequalities hold.
    Smaller x → further left; smaller y → higher up (image convention).
    """
    candidates = [
        ["a8", "h8", "h1", "a1"],
        ["h8", "h1", "a1", "a8"],
        ["h1", "a1", "a8", "h8"],
        ["a1", "a8", "h8", "h1"],
    ]

    def x(label: str) -> int:
        return corners_px[label][0]

    def y(label: str) -> int:
        return corners_px[label][1]

    scored = []
    for candidate in candidates:
        tl, tr, br, bl = candidate
        score = sum([
            x(tl) < x(tr),
            x(bl) < x(br),
            y(tl) < y(bl),
            y(tr) < y(br),
            x(tl) < x(br),
            x(bl) < x(tr),
            y(tl) < y(br),
            y(tr) < y(bl),
        ])
        scored.append({"order": candidate, "score": score})

    scored.sort(key=lambda e: e["score"], reverse=True)
    best = scored[0]
    tl, tr, br, bl = best["order"]

    return {
        "order": {
            "tl": tl,
            "tr": tr,
            "br": br,
            "bl": bl,
        },
        "score": best["score"],
        "all_scores": scored,
        "ambiguous": scored[0]["score"] == scored[1]["score"],
    }


LABEL_ORDER = ["a1", "a8", "h8", "h1"]


def derive_center_px(actual_corners_px: dict, order: dict, board_size: int = 400) -> list:
    """Camera-pixel position of the board centre (board coord (0.5, 0.5)), derived from H.

    ``order`` maps ``tl/tr/br/bl`` to the corresponding square label (from
    ``camera_natural_orientation``). Derived from the homography rather than clicked, which is
    more precise than any hand click at the centre.
    """
    last = board_size - 1
    src = np.array(
        [actual_corners_px[order[pos]] for pos in ["tl", "tr", "br", "bl"]], dtype=np.float32
    )
    dst = np.array([[0, 0], [last, 0], [last, last], [0, last]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    center_warped = np.array([[[last / 2.0, last / 2.0]]], dtype=np.float32)
    center_px = cv2.perspectiveTransform(center_warped, np.linalg.inv(homography)).reshape(2)
    return [float(center_px[0]), float(center_px[1])]


def build_calibration_metadata(
    *,
    existing: dict,
    actual_corners_px: dict,
    extended_corners_px: dict,
    extended_center_px,
    K,
    D,
    image_size,
    board_size: int = 400,
    raw_image_path=None,
    center_measured: bool = True,
) -> dict:
    """Assemble versioned (v2) calibration metadata, preserving any pre-existing fields.

    Mirrors the ``annotate_existing`` idiom: spread ``**existing`` first, then add/override
    only the new fields. ``center_px`` is derived from the homography (not a click); the scaled
    camera intrinsics are cached so the batch/gameplay never re-fetch them from reachy_mini.
    ``center_measured`` records whether ``extended_center_px`` was clicked (a real central piece)
    or interpolated from the corners; either way it is stored so ``Processor`` reads one field.
    """
    order_info = infer_camera_natural_corner_order(actual_corners_px)
    metadata = {
        **existing,
        "calibration_version": 2,
        "actual_corner_order": LABEL_ORDER,
        "actual_corners_px": actual_corners_px,
        "camera_natural_orientation": order_info,
        "extended_corner_order": LABEL_ORDER,
        "extended_corners_px": extended_corners_px,
        "extended_center_px": list(extended_center_px),
        "center_measured": bool(center_measured),
        "center_px": derive_center_px(actual_corners_px, order_info["order"], board_size),
        "camera_intrinsics": {
            "K": np.asarray(K, dtype=float).tolist(),
            "D": np.asarray(D, dtype=float).reshape(-1).tolist(),
            "image_size": [int(image_size[0]), int(image_size[1])],
        },
    }
    if raw_image_path is not None:
        metadata["raw_image_path"] = Path(raw_image_path).as_posix()
    return metadata


def undistort_reference_frame(frame, setup_dir: Path):
    """Undistort a setup's reference frame in memory and persist it for later visual diffing.

    Returns ``(undistorted, K, D, image_size)`` and writes ``raw_undistorted.png`` next to
    ``raw.png`` (straight real-world edges should look straighter there). The original
    distorted ``raw.png`` is written/kept by the caller and never modified here.
    """
    height, width = frame.shape[:2]
    image_size = (width, height)
    K, D = get_lite_camera_KD(image_size)
    map1, map2 = build_undistort_maps(K, D, image_size)
    undistorted = undistort(frame, map1, map2)
    cv2.imwrite(str(setup_dir / "raw_undistorted.png"), undistorted)
    return undistorted, K, D, image_size


# --------------------------------------------------------------------------------------------
# Interactive calibration UI (runs on the undistorted frame): 5-point collection, a quick
# review with drag-to-correct, and a background per-square mask inspector.
# --------------------------------------------------------------------------------------------

BASE_CORNER_LABELS = ["a1", "a8", "h8", "h1"]
EXTENDED_LABELS = ["a1", "a8", "h8", "h1", "center"]

_BASE_COLOR = (0, 0, 255)           # red   - clicked base corners
_CENTER_BASE_COLOR = (0, 255, 255)  # yellow - derived centre base (not clicked)
_EXTENDED_COLOR = (255, 128, 0)     # orange - clicked extended points
_LINK_COLOR = (0, 255, 0)           # green  - base->extended links
_CEILING_COLOR = (0, 200, 200)      # cyan   - ceiling quad / cuboid ceiling
_WALL_COLOR = (200, 200, 0)         # cuboid walls
_DRAG_RADIUS = 14


@dataclass
class SquareResult:
    """One square's inspector geometry, inverse-warped to the camera (undistorted) frame."""

    label: str
    floor_cam: np.ndarray    # (4, 2)
    ceiling_cam: np.ndarray  # (4, 2)
    bbox_cam: np.ndarray     # (4, 2) crop bounding-box corners


def build_inspector_calibration(base_points: dict, extended_points: dict) -> dict:
    """Minimal geometry-only calibration dict from in-progress clicks, for the inspector."""
    return {
        "camera_natural_orientation": infer_camera_natural_corner_order(base_points),
        "actual_corners_px": base_points,
        "extended_corners_px": {k: v for k, v in extended_points.items() if k != "center"},
        "extended_center_px": extended_points["center"],
    }


def compute_inspector_results(calibration_dict, config_path="config.yaml") -> list:
    """Per-square cuboid + crop box, inverse-warped to the camera frame (64, in SQUARES order).

    Pure/headless (no cv2 GUI); runs in a background thread during review so each square's box
    can be drawn directly on the full undistorted frame the user clicked on.
    """
    from chess_commentator.image_processing import Processor

    processor = Processor(calibration_dict, config_path)
    inv = np.linalg.inv(processor.matrix)

    def to_camera(pts):
        return cv2.perspectiveTransform(
            np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), inv
        ).reshape(-1, 2)

    results = []
    for label in SQUARES:
        geom = processor.square_geometry[label]
        x_min, y_min, x_max, y_max = geom.bbox
        bbox = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64
        )
        results.append(
            SquareResult(
                label=label,
                floor_cam=to_camera(geom.floor_pts),
                ceiling_cam=to_camera(geom.ceiling_pts),
                bbox_cam=to_camera(bbox),
            )
        )
    return results


def _draw_cuboid(
    canvas,
    floor,
    ceiling,
    floor_color=_BASE_COLOR,
    ceiling_color=_CEILING_COLOR,
    wall_color=_WALL_COLOR,
    thickness=1,
):
    """Thin cuboid wireframe (floor quad + ceiling quad + 4 walls), no fill."""
    floor = np.asarray(floor).astype(np.int32)
    ceiling = np.asarray(ceiling).astype(np.int32)
    cv2.polylines(canvas, [floor], True, floor_color, thickness)
    cv2.polylines(canvas, [ceiling], True, ceiling_color, thickness)
    for f, c in zip(floor, ceiling):
        cv2.line(canvas, tuple(f), tuple(c), wall_color, thickness)
    return canvas


def _wheel_direction(flags) -> int:
    """+1 / -1 / 0 from an EVENT_MOUSEWHEEL ``flags`` value (this OpenCV build lacks
    ``cv2.getMouseWheelDelta``, so fall back to the signed high 16 bits of ``flags``)."""
    getter = getattr(cv2, "getMouseWheelDelta", None)
    if getter is not None:
        delta = getter(flags)
    else:
        delta = (int(flags) >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
    return 1 if delta > 0 else -1 if delta < 0 else 0


def render_review_overlay(frame, base_points, center_base, extended_points) -> np.ndarray:
    """Quick review overlay: 10 points, base->extended links, and the ceiling quad."""
    img = frame.copy()

    def pt(p):
        return (int(round(p[0])), int(round(p[1])))

    for label in BASE_CORNER_LABELS:
        if label in base_points and label in extended_points:
            cv2.line(img, pt(base_points[label]), pt(extended_points[label]), _LINK_COLOR, 1)
    if center_base is not None and "center" in extended_points:
        cv2.line(img, pt(center_base), pt(extended_points["center"]), _LINK_COLOR, 1)

    if len(base_points) == 4:
        order = infer_camera_natural_corner_order(base_points)["order"]
        if all(order[p] in extended_points for p in ["tl", "tr", "br", "bl"]):
            ceiling = np.array(
                [extended_points[order[p]] for p in ["tl", "tr", "br", "bl"]], dtype=np.int32
            )
            cv2.polylines(img, [ceiling], True, _CEILING_COLOR, 1)

    for p in base_points.values():
        cv2.circle(img, pt(p), 5, _BASE_COLOR, -1)
    if center_base is not None:
        cv2.circle(img, pt(center_base), 5, _CENTER_BASE_COLOR, -1)
    for p in extended_points.values():
        cv2.circle(img, pt(p), 5, _EXTENDED_COLOR, -1)
    return img


def render_full_overlay(frame, base_points, center_base, extended_points, highlight=None):
    """Review overlay, plus (if given) one square's cuboid + crop box highlighted on the frame."""
    img = render_review_overlay(frame, base_points, center_base, extended_points)
    if highlight is not None:
        _draw_cuboid(
            img,
            highlight.floor_cam,
            highlight.ceiling_cam,
            floor_color=(0, 255, 255),
            ceiling_color=(255, 0, 255),
            wall_color=(255, 255, 0),
            thickness=2,
        )
        cv2.polylines(img, [highlight.bbox_cam.astype(np.int32)], True, (255, 255, 255), 1)
        anchor = highlight.floor_cam.mean(axis=0)
        cv2.putText(
            img, highlight.label, (int(anchor[0]) - 12, int(anchor[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    return img


class CalibrationUI:
    """Interactive calibration collector on an already-undistorted frame.

    Flow: click 4 base corners -> click the extended points (4 corners, plus the centre only if
    ``annotate_center``) -> a review step, drawn on the full frame, with drag-to-correct and a
    scrollable per-square box inspector (`i` toggles it). ``run()`` returns
    ``(base_points, extended_points_with_center)`` or ``None`` on abort. When ``annotate_center``
    is False the extended centre is the bilinear interpolation (average) of the 4 extended
    corners and is not independently draggable — it tracks the corners.
    """

    def __init__(self, frame, config_path="config.yaml", window_name="Calibration",
                 annotate_center=False):
        self.frame = frame
        self.config_path = config_path
        self.window_name = window_name
        self.annotate_center = annotate_center
        self.base_points: dict[str, list[int]] = {}
        self.extended_points: dict[str, list[int]] = {}
        self.selected = None
        self.generation = 0
        self.results: list = [None] * 64
        self.results_lock = threading.Lock()
        self.inspector_index = 0
        self.show_inspector = False

    def run(self):
        ext_labels = EXTENDED_LABELS if self.annotate_center else BASE_CORNER_LABELS
        ext_title = (
            "extended points (corners then centre)" if self.annotate_center
            else "extended board corners"
        )
        while True:
            self.base_points, self.extended_points = {}, {}
            self.show_inspector, self.inspector_index = False, 0
            if not self._collect(self.base_points, BASE_CORNER_LABELS, "actual board corners", _BASE_COLOR):
                cv2.destroyWindow(self.window_name)
                return None
            if not self._collect(self.extended_points, ext_labels, ext_title, _EXTENDED_COLOR):
                cv2.destroyWindow(self.window_name)
                return None
            outcome = self._review()
            if outcome == "retry":
                continue
            cv2.destroyWindow(self.window_name)
            return outcome  # (base_points, extended_with_center) or None

    def _collect(self, store, labels, title, color) -> bool:
        display = self.frame.copy()
        store.clear()

        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN or len(store) >= len(labels):
                return
            label = labels[len(store)]
            store[label] = [x, y]
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.putText(display, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(self.window_name, display)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, display)
        cv2.setMouseCallback(self.window_name, on_click)
        print(f"\nClick {title} in order: {', '.join(labels)}. ESC to abort.")
        while len(store) < len(labels):
            if (cv2.waitKey(20) & 0xFF) == 27:  # ESC
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return False
        cv2.setMouseCallback(self.window_name, lambda *a: None)
        return True

    def _center_base(self):
        order = infer_camera_natural_corner_order(self.base_points)["order"]
        return derive_center_px(self.base_points, order)

    def _extended_center(self):
        """Extended centre: the clicked point if measured, else the average of the 4 corners."""
        if "center" in self.extended_points:
            return self.extended_points["center"]
        corners = [self.extended_points[label] for label in BASE_CORNER_LABELS
                   if label in self.extended_points]
        if len(corners) == 4:
            return list(np.mean(np.array(corners, dtype=float), axis=0))
        return None

    def _extended_with_center(self):
        """The 4 extended corners plus the (measured or interpolated) centre."""
        extended = {k: v for k, v in self.extended_points.items() if k != "center"}
        center = self._extended_center()
        if center is not None:
            extended["center"] = center
        return extended

    def _launch_inspector(self):
        self.generation += 1
        generation = self.generation
        calibration = build_inspector_calibration(self.base_points, self._extended_with_center())
        config_path = self.config_path
        with self.results_lock:
            self.results = [None] * 64

        def work():
            try:
                results = compute_inspector_results(calibration, config_path)
            except Exception as exc:  # noqa: BLE001 - a bad in-progress click shouldn't crash the UI
                print(f"inspector computation failed: {exc}")
                return
            with self.results_lock:
                if generation == self.generation:  # discard stale computations
                    self.results = results

        threading.Thread(target=work, daemon=True).start()

    def _find_marker(self, x, y):
        # Only actually-clicked points are draggable (the interpolated centre tracks the corners).
        best, best_dist = None, _DRAG_RADIUS ** 2
        markers = [("base", lbl, p) for lbl, p in self.base_points.items()]
        markers += [("extended", lbl, p) for lbl, p in self.extended_points.items()]
        for which, label, p in markers:
            dist = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if dist <= best_dist:
                best, best_dist = (which, label), dist
        return best

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = self._find_marker(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selected is not None:
            which, label = self.selected
            store = self.base_points if which == "base" else self.extended_points
            store[label] = [x, y]
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected is not None:
                self.selected = None
                self._launch_inspector()  # recompute from the corrected points
        elif event == cv2.EVENT_MOUSEWHEEL and self.show_inspector:
            self._scroll(_wheel_direction(flags))

    def _scroll(self, step):
        if step:
            self.inspector_index = (self.inspector_index + step) % 64
            self._redraw()

    def _redraw(self):
        highlight = None
        if self.show_inspector:
            with self.results_lock:
                highlight = self.results[self.inspector_index]
        img = render_full_overlay(
            self.frame, self.base_points, self._center_base(), self._extended_with_center(),
            highlight,
        )
        if self.show_inspector and highlight is None:
            cv2.putText(
                img, f"{SQUARES[self.inspector_index]}: computing...",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        cv2.imshow(self.window_name, img)

    def _review(self):
        self._launch_inspector()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()
        print(
            "Review: drag any marker to correct. 'i' toggles the per-square box overlay; "
            "mouse-wheel or ,/. steps through squares. ENTER/SPACE accept, 'r' retry, 'q'/ESC abort."
        )
        while True:
            key = cv2.waitKey(30) & 0xFF
            if self.show_inspector:
                self._redraw()  # pick up freshly computed inspector results
            if key in (13, ord(" ")):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return self.base_points, self._extended_with_center()
            if key == ord("i"):
                self.show_inspector = not self.show_inspector
                self._redraw()
            elif key in (ord("."), ord("]")):
                self._scroll(1)
            elif key in (ord(","), ord("[")):
                self._scroll(-1)
            elif key == ord("r"):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return "retry"
            elif key in (ord("q"), 27):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return None


def annotate_existing(
    image_path: Path, config_path="config.yaml", annotate_center: bool = False
) -> dict | None:
    """
    Load a stored raw image, undistort it, collect the v2 calibration (4 base corners + 4
    extended corners, plus the extended centre only if ``annotate_center``) interactively, and
    write (or overwrite) calibration_metadata.json in the same directory.

    Pre-existing metadata fields (height_mm, pitch_deg, timestamp, …) are preserved; the
    corner/centre/intrinsics fields are (re)written on the undistorted frame.

    Usage:
        python -m chess_commentator.calibration data/generated/<setup>/raw.png
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return None

    setup_dir = image_path.parent
    metadata_path = setup_dir / "calibration_metadata.json"

    existing: dict = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded existing metadata from: {metadata_path}")

    undistorted, K, D, image_size = undistort_reference_frame(frame, setup_dir)

    collected = CalibrationUI(
        undistorted, config_path, annotate_center=annotate_center
    ).run()
    if collected is None:
        cv2.destroyAllWindows()
        return None
    base_points, extended_points = collected

    calibration_data = build_calibration_metadata(
        existing=existing,
        actual_corners_px=base_points,
        extended_corners_px={k: v for k, v in extended_points.items() if k != "center"},
        extended_center_px=extended_points["center"],
        K=K,
        D=D,
        image_size=image_size,
        raw_image_path=image_path,
        center_measured=annotate_center,
    )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    _log_calibration_summary(calibration_data, config_path)
    print(f"Saved calibration metadata: {metadata_path}")
    cv2.destroyAllWindows()
    return calibration_data


def relabel_existing_setups(
    data_root: Path = Path("data") / "generated",
    config_path="config.yaml",
    annotate_center: bool = False,
) -> None:
    """Phase 1: re-run the v2 calibration UI over every existing setup's stored ``raw.png``.

    Opens each ``<setup>/raw.png`` undistorted in the improved UI (re-clicking all corners fresh)
    and writes updated versioned metadata in place. ``annotate_center`` defaults to False — the
    central square is usually empty, so its extended point is interpolated from the corners
    rather than clicked. Only touches the setups, not the captured frames — Phase 2
    (``regenerate.py``) regenerates those afterwards.
    """
    data_root = Path(data_root)
    setups = sorted(
        p for p in data_root.iterdir() if p.is_dir() and (p / "raw.png").exists()
    )
    print(f"Relabelling {len(setups)} setups under {data_root}.")
    print("Per setup: ENTER/SPACE accept, 'r' retry, 'q'/ESC abort (stops the session).")
    for i, setup_dir in enumerate(setups, 1):
        print(f"\n[{i}/{len(setups)}] {setup_dir.name}")
        if annotate_existing(setup_dir / "raw.png", config_path, annotate_center) is None:
            print("Aborted; stopping relabelling session.")
            break


if __name__ == "__main__":
    # Three modes:
    #   (no args)            fresh calibration on the live robot
    #   relabel              re-click every stored setup under data/generated
    #   <path/to/raw.png>    re-click one stored setup
    # `--center` opts into clicking the extended centre (default: interpolate it from corners).
    import sys

    args = sys.argv[1:]
    annotate_center = "--center" in args
    args = [a for a in args if a != "--center"]

    if args:
        # Both relabelling modes work off a stored raw.png, so they never touch the robot.
        if args[0] == "relabel":
            relabel_existing_setups(annotate_center=annotate_center)
        else:
            annotate_existing(Path(args[0]), annotate_center=annotate_center)
    else:
        # Only a fresh calibration needs the live camera and head, so open the robot here rather
        # than at import time (reachy_mini is an optional dependency group).
        from reachy_mini import ReachyMini

        with ReachyMini(media_backend="default") as mini:
            calibrate(mini, annotate_center=annotate_center)
