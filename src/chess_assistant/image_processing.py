"""Turn a raw camera frame into the 64 masked per-square crops the CNN sees.

``Processor`` freezes everything that depends only on a setup's calibration (undistortion maps,
the padded board homography, and all 64 mask polygons + crop boxes) at construction time, so
gameplay and the offline batch only pay for undistort -> warp -> cutout per frame.

A square's crop is deliberately *not* the square itself: it is the convex hull of the square's
floor quad and the quad the same corners project to at piece height ("ceiling"). That hull is
the 3D column of space above the square — which is where the piece actually is — so a crop
contains its own piece even though pieces lean away from the camera and overhang their
neighbours. ``Processor.__init__`` explains how the ceiling is derived from calibration.
"""

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
import json

from chess_assistant.camera_utils import build_undistort_maps, undistort

def letterbox(
    img: np.ndarray,
    target_size: tuple[int, int],
    pad_value_img: int | float = 0,
    pad_value_mask: int | float = 0,
) -> np.ndarray:
    """
    Letterbox a 4-channel image to target size.

    Args:
        img: np.ndarray of shape (H, W, 4).
             Channels 0:3 are image channels.
             Channel 3 is a binary mask with values 0 or 1.
        target_size: (target_h, target_w).
        pad_value_img: padding value for image channels.
        pad_value_mask: padding value for mask channel.

    Returns:
        np.ndarray of shape (target_h, target_w, 4).
    """
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected image shape (H, W, 4), got {img.shape}")

    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Split image and mask so we can resize them differently
    image_channels = img[:, :, :3]
    mask_channel = img[:, :, 3]

    resized_image = cv2.resize(
        image_channels,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    resized_mask = cv2.resize(
        mask_channel,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Create padded output
    out = np.empty((target_h, target_w, 4), dtype=img.dtype)
    out[:, :, :3] = pad_value_img
    out[:, :, 3] = pad_value_mask

    # Center the resized image
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    out[top:top + new_h, left:left + new_w, :3] = resized_image
    out[top:top + new_h, left:left + new_w, 3] = resized_mask

    return out


def estimate_vanishing_point(
    base_pts: np.ndarray, ext_pts: np.ndarray
) -> tuple[np.ndarray, float]:
    """Least-squares intersection of the N base->ext rays.

    ``base_pts``, ``ext_pts``: ``(N, 2)`` in the same frame. Returns ``(V, mean_residual)`` —
    the point minimising the summed squared perpendicular distance to every ray, and the mean
    perpendicular distance from ``V`` to each ray (a fit-quality diagnostic; one ray far larger
    than the others usually means a bad click, not a modelling problem).
    """
    base_pts = np.asarray(base_pts, dtype=np.float64)
    ext_pts = np.asarray(ext_pts, dtype=np.float64)
    dirs = ext_pts - base_pts
    dirs_unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in zip(base_pts, dirs_unit):
        d = d.reshape(2, 1)
        M = np.eye(2) - d @ d.T
        A += M
        b += M @ p
    V = np.linalg.solve(A, b)
    residuals = [
        np.linalg.norm((np.eye(2) - d.reshape(2, 1) @ d.reshape(1, 2)) @ (V - p))
        for p, d in zip(base_pts, dirs_unit)
    ]
    return V, float(np.mean(residuals))


def bilinear_interp(m_tl, m_tr, m_br, m_bl, u, v):
    """Bilinear value over a unit patch. ``(u, v)`` in ``[0,1]^2`` (u: left->right, v: top->bottom).

    Corners map exactly: ``(0,0)->m_tl``, ``(1,0)->m_tr``, ``(1,1)->m_br``, ``(0,1)->m_bl``.
    """
    top = m_tl * (1 - u) + m_tr * u
    bottom = m_bl * (1 - u) + m_br * u
    return top * (1 - v) + bottom * v


class QuadrantField:
    """Bilinear field over board coords ``(u, v) in [0,1]^2`` from 4 corner + 1 centre values.

    Node values may be scalars or numpy vectors (used here for 2D extension displacements —
    ``bilinear_interp`` operates componentwise). The four edge midpoints are derived as the mean
    of their two adjacent corners (matching what a plain bilinear patch predicts along an edge).
    The board is split into NW/NE/SE/SW quadrant patches (each bounded by one real corner, the
    centre, and two derived edge midpoints); a query ``(u, v)`` is dispatched to its quadrant,
    remapped to that quadrant's local ``[0,1]^2`` and bilinearly interpolated. When the centre
    value equals the corner average, the field reduces exactly to a plain bilinear patch.
    """

    def __init__(self, m_tl, m_tr, m_br, m_bl, m_center):
        self.m_tl = m_tl
        self.m_tr = m_tr
        self.m_br = m_br
        self.m_bl = m_bl
        self.m_center = m_center
        # Derived edge midpoints.
        self.m_tm = (m_tl + m_tr) / 2  # top
        self.m_rm = (m_tr + m_br) / 2  # right
        self.m_bm = (m_br + m_bl) / 2  # bottom
        self.m_lm = (m_tl + m_bl) / 2  # left

    def __call__(self, u, v):
        u = min(max(float(u), 0.0), 1.0)
        v = min(max(float(v), 0.0), 1.0)
        if u <= 0.5 and v <= 0.5:  # NW
            vals = (self.m_tl, self.m_tm, self.m_center, self.m_lm)
            lu, lv = u * 2, v * 2
        elif u > 0.5 and v <= 0.5:  # NE
            vals = (self.m_tm, self.m_tr, self.m_rm, self.m_center)
            lu, lv = (u - 0.5) * 2, v * 2
        elif u > 0.5 and v > 0.5:  # SE
            vals = (self.m_center, self.m_rm, self.m_br, self.m_bm)
            lu, lv = (u - 0.5) * 2, (v - 0.5) * 2
        else:  # SW
            vals = (self.m_lm, self.m_center, self.m_bm, self.m_bl)
            lu, lv = u * 2, (v - 0.5) * 2
        return bilinear_interp(vals[0], vals[1], vals[2], vals[3], lu, lv)


def mask_bounding_box(mask_points: np.ndarray, margin: float = 1.05):
    """Axis-aligned box of ``mask_points``, expanded by ``margin`` about their centroid.

    Returns ``(x_min, y_min, x_max, y_max)`` as ints (floor/ceil). The caller is expected to
    clamp this to the canvas.
    """
    mask_points = np.asarray(mask_points, dtype=np.float64)
    centroid = mask_points.mean(axis=0)
    expanded = centroid + (mask_points - centroid) * margin
    x_min, y_min = expanded.min(axis=0)
    x_max, y_max = expanded.max(axis=0)
    return int(np.floor(x_min)), int(np.floor(y_min)), int(np.ceil(x_max)), int(np.ceil(y_max))


# --- Mask / crop ablation knobs -------------------------------------------------------------
# The 4th channel and the crop geometry are the two things the mask ablation varies. Both default
# to None everywhere, meaning "follow the calibration": v2 metadata gets the convex-hull mask and
# the tight per-square crop, v1 falls back to the square mask and the global padding box. Setting
# them explicitly forces a combination, which is what the ablation runs need (a v2 setup can then
# be cut the "bad" way on purpose).
#
#   mask "hull"   - convex hull of floor + ceiling: the column of space the piece occupies.
#   mask "square" - only the square's own floor quad (axis-aligned in the rectified warp).
#   mask "none"   - constant 0. Keeps the 4-channel input shape, but the channel carries no
#                   information, so it is equivalent to a 3-channel model (a conv over an
#                   all-zero channel contributes nothing but its bias).
#   crop "per_square" - tight bounding box of that square's own hull (varies per square).
#   crop "global"     - the square plus the same global padding box on every side, for all 64.
MASK_MODES = ("hull", "square", "none")
CROP_MODES = ("per_square", "global")

# Named ablation variants -> (mask_mode, crop_mode). "default" is what ships.
MASK_VARIANTS = {
    "default": ("hull", "per_square"),
    "square_per_square": ("square", "per_square"),
    "square_global": ("square", "global"),
    "none_global": ("none", "global"),
}


def compute_square_mask(mask_polygon: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
    """Hard 0/1 mask in the crop's own local coordinates (polygon already offset to the crop).

    Isolated on purpose: a future soft-mask version replaces only this function's body, so no
    caller needs to change. ``crop_shape`` is ``(height, width)``.
    """
    mask = np.zeros(crop_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, mask_polygon.astype(np.int32), 1)
    return mask


@dataclass
class SquareGeometry:
    """Precomputed per-square geometry in the padded warped frame (frozen per setup)."""

    label: str
    floor_pts: np.ndarray      # (4, 2) tl, tr, br, bl — the square's own corners
    ceiling_pts: np.ndarray    # (4, 2) vanishing-point-projected corners, matching floor order
    mask_polygon: np.ndarray   # (K, 2) convex hull of floor + ceiling
    bbox: tuple                # (x_min, y_min, x_max, y_max), clamped to the canvas


class Processor:
    def __init__(
        self,
        metadata_source,
        config_path: Path | None = None,
        mask_mode: str | None = None,
        crop_mode: str | None = None,
    ) -> None:
        """Freeze the per-setup geometry: padding, warp matrix, undistort maps, square geometry.

        The warped canvas is the board plus padding on each side. The padding is not decorative:
        pieces lean away from the camera, so their tops project *outside* the board quad, and the
        canvas has to be big enough to still contain them.

        ``metadata_source`` is either a path/str to a ``calibration_metadata.json`` file or an
        already-loaded calibration dict. The dict form lets the calibration UI build the same
        geometry from in-progress clicks without touching disk.

        ``mask_mode`` / ``crop_mode`` are the ablation knobs (see MASK_MODES / CROP_MODES). Left
        at ``None`` they follow the calibration, which is the shipped behaviour; the mask ablation
        sets them explicitly to cut a v2 setup the deliberately worse way.
        """
        if mask_mode is not None and mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}. Got {mask_mode!r}.")
        if crop_mode is not None and crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}. Got {crop_mode!r}.")
        self.mask_mode = mask_mode
        self.crop_mode = crop_mode
        # Get board size of transformed image (excl. padding) from config
        board_size = None
        square_cutout_size = None
        if config_path:
            config = OmegaConf.load(config_path)
            board_size = config.get("image_processing", OmegaConf.create()).get("board_size")
            square_cutout_size = config.get("image_processing", OmegaConf.create()).get("square_cutout_size")
        board_size = board_size if board_size else 400
        square_cutout_size = square_cutout_size if square_cutout_size else 144
        last_coordinate = board_size - 1

        # Load metadata (file path or already-loaded dict)
        if isinstance(metadata_source, dict):
            metadata = metadata_source
        else:
            with open(metadata_source, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Which board square sits in each visual corner of the frame, e.g. {"tl": "a8"}. The
        # robot can be placed on any side of the board, so this (inferred at calibration time)
        # is what pins the image's orientation to the board's own coordinates.
        self.corner_map = metadata["camera_natural_orientation"]["order"]

        src = np.array(
            [
                metadata["actual_corners_px"][corner_square] 
                for corner_square in 
                [self.corner_map[square_position] for square_position in ["tl", "tr", "br", "bl"]]
            ],
            dtype=np.float32
        )

        dst_initial = np.array(
            [
                [0, 0], # top-left
                [last_coordinate, 0], # top_right
                [last_coordinate, last_coordinate],
                [0, last_coordinate]
            ],
            dtype=np.float32
        )

        matrix_initial = cv2.getPerspectiveTransform(src, dst_initial)

        # The "extended" corners are the clicked piece-tops above each board corner. Push them
        # through the unpadded homography to see how far outside the board quad they land — that
        # overhang is exactly the padding the canvas needs in each direction.
        src_extended_corners = np.array(
            [
                metadata["extended_corners_px"][corner_square]
                for corner_square in
                [self.corner_map[square_position] for square_position in ["tl", "tr", "br", "bl"]]
            ],
            dtype=np.float32
        ) # Shape: (4, 2)

        dst_extended_corners_initial = (
            cv2.perspectiveTransform(
                src_extended_corners.reshape(4, 1, 2),  # cv2 wants (N, 1, 2) here
                matrix_initial
            )
            .reshape(4, 2)
        )

        # One (dx, dy) overhang per corner, in tl, tr, br, bl order. Sign convention: dx > 0
        # means the extended corner projects further right than the board corner it sits above
        # (so the canvas must grow to the right), dx < 0 further left; dy > 0 further down,
        # dy < 0 further up. The padding in a direction is the largest overhang in that direction.
        pixel_differences = dst_extended_corners_initial - dst_initial

        # v2 calibration adds a measured "center" extension point (its base is the board
        # centre, derived from the homography at board coord (0.5, 0.5); only its extended
        # position is clicked). Correctness fix vs the old 4-corner-only padding: the measured
        # centre can exceed every corner, so the canvas must be sized over all 5 measured
        # extension points, not just the 4 corners.
        is_v2 = "extended_center_px" in metadata
        differences_for_padding = pixel_differences
        if is_v2:
            dst_extended_center_initial = cv2.perspectiveTransform(
                np.array([[metadata["extended_center_px"]]], dtype=np.float32),
                matrix_initial,
            ).reshape(2)
            center_base_initial = np.array([last_coordinate / 2.0, last_coordinate / 2.0])
            differences_for_padding = np.vstack(
                [pixel_differences, dst_extended_center_initial - center_base_initial]
            )

        # p_up: 'padding_up'. Max signed extension across the measured points in each
        # direction, floored at 0, with a 5% margin.
        p_up = round(1.05 * max(np.max(-differences_for_padding, axis=0)[1], 0))
        p_down = round(1.05 * max(np.max(differences_for_padding, axis=0)[1], 0))
        p_left = round(1.05 * max(np.max(-differences_for_padding, axis=0)[0], 0))
        p_right = round(1.05 * max(np.max(differences_for_padding, axis=0)[0], 0))

        padding = {
            "up": p_up,
            "down": p_down,
            "left": p_left,
            "right": p_right
        }

        # Given self.padding, modify size of image and create new matrix
        size_x = board_size + padding["left"] + padding["right"]
        size_y = board_size + padding["up"] + padding["down"]

        # Modify destination coordinates of the board corners to account for padding
        dst = np.array(
            [
                [padding["left"], padding["up"]], # top-left
                [padding["left"] + board_size, padding["up"]], # top-right
                [padding["left"] + board_size, padding["up"] + board_size], # bottom-right
                [padding["left"], padding["up"] + board_size]
            ],
            dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        
        # Store attributes
        self.padding = padding
        self.board_size = board_size
        self.square_cutout_size = square_cutout_size
        self.image_size = (size_x, size_y)
        self.matrix = matrix

        # Lens undistortion maps, built once per setup from the intrinsics cached in the
        # calibration metadata (v2 calibrations only). When absent (v1 metadata), warp()
        # behaves exactly as before — no undistortion.
        self.undistort_map1 = None
        self.undistort_map2 = None
        camera_intrinsics = metadata.get("camera_intrinsics")
        if camera_intrinsics:
            width, height = camera_intrinsics["image_size"]
            self.undistort_map1, self.undistort_map2 = build_undistort_maps(
                np.asarray(camera_intrinsics["K"], dtype=np.float64),
                np.asarray(camera_intrinsics["D"], dtype=np.float64),
                (width, height),
            )

        # Perspective geometry (v2 calibration only): the per-square extension is a field of
        # measured base->extended displacement VECTORS, interpolated across the board. Working
        # directly with the measured displacements (rather than pointing each corner toward a
        # single least-squares vanishing point) reproduces the clicked piece-tops exactly at the
        # 5 measured points and cannot flip when the piece-height rays are near-parallel/noisy.
        # V/vp_residual are still computed as a fit-quality diagnostic (logged), not for geometry.
        self.is_v2 = is_v2
        self.V = None
        self.vp_residual = None
        self.extension_field = None
        self.square_geometry = None
        self.square_labels = self._compute_square_labels()
        if is_v2:
            base_center = np.array(
                [padding["left"] + board_size / 2.0, padding["up"] + board_size / 2.0]
            )
            base_pts = np.vstack([dst.astype(np.float64), base_center])  # (5, 2), tl,tr,br,bl,c

            ext_corners = (
                cv2.perspectiveTransform(src_extended_corners.reshape(4, 1, 2), matrix)
                .reshape(4, 2)
                .astype(np.float64)
            )
            ext_center = (
                cv2.perspectiveTransform(
                    np.array([[metadata["extended_center_px"]]], dtype=np.float32), matrix
                )
                .reshape(2)
                .astype(np.float64)
            )
            ext_pts = np.vstack([ext_corners, ext_center])  # (5, 2)

            self.V, self.vp_residual = estimate_vanishing_point(base_pts, ext_pts)

            # Measured extension displacement vectors (padded warped frame), tl, tr, br, bl.
            corner_disp = ext_corners - dst.astype(np.float64)  # (4, 2)
            self.extension_field = QuadrantField(
                m_tl=corner_disp[0],
                m_tr=corner_disp[1],
                m_br=corner_disp[2],
                m_bl=corner_disp[3],
                m_center=ext_center - base_center,
            )
            self.square_geometry = self._build_square_geometry()

    @property
    def effective_mask_mode(self) -> str:
        """The mask actually used: the explicit override, else what the calibration supports."""
        if self.mask_mode is not None:
            return self.mask_mode
        return "hull" if self.square_geometry is not None else "square"

    @property
    def effective_crop_mode(self) -> str:
        """The crop actually used: the explicit override, else what the calibration supports."""
        if self.crop_mode is not None:
            return self.crop_mode
        return "per_square" if self.square_geometry is not None else "global"

    def _compute_square_labels(self) -> dict:
        """Map each ``(i, j)`` grid cell (row from top, col from left) to its square label.

        Same mapping the v1 cutout computed inline; depends only on which board corner sits at
        the image's top-left (``self.corner_map["tl"]``).
        """
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = [str(i) for i in range(1, 9)]
        label_map = {
            "a8": [{i: ranks[-(i + 1)] for i in range(8)}, {j: files[j] for j in range(8)}],
            "a1": [{i: files[i] for i in range(8)}, {j: ranks[j] for j in range(8)}],
            "h1": [{i: ranks[i] for i in range(8)}, {j: files[-(j + 1)] for j in range(8)}],
            "h8": [{i: files[-(i + 1)] for i in range(8)}, {j: ranks[-(j + 1)] for j in range(8)}],
        }
        tl = self.corner_map["tl"]
        is_reverse = tl in ["a1", "h8"]
        lm = label_map[tl]
        return {
            (i, j): (lm[1][j] + lm[0][i]) if not is_reverse else (lm[0][i] + lm[1][j])
            for i in range(8)
            for j in range(8)
        }

    def _build_square_geometry(self) -> dict:
        """Precompute every square's mask polygon + crop box once, in the padded warped frame.

        Per floor corner: ceiling = corner + the extension field evaluated at that corner's board
        coord ``(u, v)`` — the interpolated *measured* base->extended displacement, deliberately
        not a ray aimed at ``self.V`` (see ``__init__``). The mask polygon is the convex hull of
        the 4 floor + 4 ceiling corners, so it spans the whole column of space above the square,
        which is where the piece standing on it actually appears. The crop box is that hull's
        (margin-expanded) bounding box, clamped to the canvas.
        """
        square_size = self.board_size // 8
        pl, pu = self.padding["left"], self.padding["up"]
        board_size = self.board_size
        size_x, size_y = self.image_size

        geometry: dict[str, SquareGeometry] = {}
        for i in range(8):
            for j in range(8):
                floor = np.array(
                    [
                        [pl + j * square_size, pu + i * square_size],
                        [pl + (j + 1) * square_size, pu + i * square_size],
                        [pl + (j + 1) * square_size, pu + (i + 1) * square_size],
                        [pl + j * square_size, pu + (i + 1) * square_size],
                    ],
                    dtype=np.float64,
                )
                ceiling = np.empty_like(floor)
                for k, corner in enumerate(floor):
                    u = (corner[0] - pl) / board_size
                    v = (corner[1] - pu) / board_size
                    # Extension = interpolated measured displacement (toward the clicked tops).
                    ceiling[k] = corner + self.extension_field(u, v)

                hull = cv2.convexHull(
                    np.vstack([floor, ceiling]).astype(np.float32)
                ).reshape(-1, 2)
                x_min, y_min, x_max, y_max = mask_bounding_box(hull)
                bbox = (
                    max(0, x_min),
                    max(0, y_min),
                    min(size_x, x_max),
                    min(size_y, y_max),
                )
                label = self.square_labels[(i, j)]
                geometry[label] = SquareGeometry(
                    label=label,
                    floor_pts=floor,
                    ceiling_pts=ceiling,
                    mask_polygon=hull,
                    bbox=bbox,
                )
        return geometry

    def warp(self, image_path: Path, out_path: Path | None = None) -> Path:
        image = cv2.imread(image_path)
        # Undistort in memory before warping. The raw frame on disk is never modified, and no
        # undistorted copy of a full frame is written out (only the warped result is saved).
        if self.undistort_map1 is not None:
            image = undistort(image, self.undistort_map1, self.undistort_map2)
        warped_image = cv2.warpPerspective(image, self.matrix, self.image_size)
        if not out_path:
            warped_image_path = image_path.parent / (str(image_path.stem) + "_warped.png")
        else:
            warped_image_path = out_path
            out_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(warped_image_path), warped_image)
        # No colour conversion here: the image stays BGR end to end. It is only converted to RGB
        # at the point where it is handed to something that expects RGB (see the .npy crops).
        return warped_image_path
    
    def cutout(self, warped_image_path):
        """Write the 64 per-square crops (masked ``.npy`` + annotated PNG + metadata) to disk.

        Returns the ``squares/`` directory next to the warped image. Squares are addressed by
        their ``(i, j)`` grid position — ``i`` counting rows down from the top of the image,
        ``j`` counting columns left to right — and the map from ``(i, j)`` to a label like "e2"
        depends on which board corner the camera happens to see in the top left; see
        ``_compute_square_labels`` / ``self.corner_map``.

        Dispatches on ``effective_crop_mode``: the tight per-square hull crop when the
        calibration carried the geometry to build it (and the ablation has not forced otherwise),
        else the global padding box. v1 metadata has no geometry, so it always takes the global
        path.
        """
        warped_image = cv2.imread(warped_image_path)
        squares_dir = warped_image_path.parent / "squares"

        if self.effective_crop_mode == "per_square":
            if self.square_geometry is None:
                raise ValueError(
                    "crop_mode='per_square' needs the v2 calibration geometry, which this setup "
                    "does not have."
                )
            return self._cutout_v2(warped_image, squares_dir)
        return self._cutout_global(warped_image, squares_dir)

    def _cutout_global(self, warped_image, squares_dir):
        """Crop every square as the square plus the *same* global padding box on all four sides.

        The original (pre-v2) geometry. Kept both so v1 setups still process and as the "worse
        padding" arm of the mask ablation: the crop is not steered toward the direction the piece
        actually leans, so a tall piece can fall outside its own crop. The 4th channel is whatever
        ``effective_mask_mode`` asks for (by default the square's own footprint).

        Walk the grid from the top-left square: the outer loop steps i down the rows (adding one
        square height to y each time), the inner loop steps j across the columns (adding one
        square width to x). (i, j) therefore uniquely identifies a square in the image, and
        ``self.square_labels`` maps it to a label like "e2" (which depends on the corner the
        camera happens to see in the top left).
        """
        mask_mode = self.effective_mask_mode
        if mask_mode == "hull" and self.square_geometry is None:
            raise ValueError(
                "mask_mode='hull' needs the v2 calibration geometry, which this setup does not "
                "have."
            )
        square_size = self.board_size // 8 # board_size is expected to be a multiple of 8

        # Top left coordinates of top-left square
        tl_y = self.padding["up"]
        tl_x = self.padding["left"]

        squares_dir.mkdir(exist_ok=True) # exist_ok=True for debugging purposes

        for i in range(8):
            for j in range(8):
                top = tl_y + i * square_size - self.padding["up"]
                bottom = tl_y + (i + 1) * square_size + self.padding["down"]
                left = tl_x + j * square_size - self.padding["left"]
                right = tl_x + (j + 1) * square_size + self.padding["right"]

                square_label = self.square_labels[(i, j)]

                # Crop by numpy slicing: axis 0 is y, axis 1 is x, axis 2 is colour.
                square_cutout = warped_image[top:bottom, left:right]

                # Where the square itself sits *inside* its crop. Because every crop is the
                # square plus the same global padding, this is the same box for all 64 of them.
                square_left_cutout = self.padding["left"]
                square_right_cutout = self.padding["left"] + square_size
                square_top_cutout = self.padding["up"]
                square_bottom_cutout = self.padding["up"] + square_size

                # The fourth channel tells the model which part of the crop is the square it is
                # being asked about. "square" is the original behaviour (1 over the square's own
                # footprint, 0 over the surrounding padding); the ablation arms weaken it.
                mask = np.zeros_like(square_cutout[:,:,0])
                if mask_mode == "square":
                    mask[square_top_cutout:square_bottom_cutout, square_left_cutout:square_right_cutout] = 1
                elif mask_mode == "hull":
                    mask = compute_square_mask(
                        self.square_geometry[square_label].mask_polygon - np.array([left, top]),
                        mask.shape,
                    )
                # "none" leaves the channel all-zero, i.e. carrying no information at all.
                square_cutout_masked = np.concatenate([square_cutout, np.expand_dims(mask, mask.ndim)], axis=2)

                # Corners of the actual board square in the full warped image
                square_left = tl_x + j * square_size
                square_right = tl_x + (j + 1) * square_size
                square_top = tl_y + i * square_size
                square_bottom = tl_y + (i + 1) * square_size

                # Letterbox pad the square cutout so it has size self.square_size x self.square_size
                square_cutout_masked =letterbox(
                    square_cutout_masked,
                    (self.square_cutout_size, self.square_cutout_size)
                )

                square_cutout_annotated = square_cutout.copy()

                # The square's own corners, in the full warped image, converted below into the
                # crop's local coordinates so they can be dotted onto the annotated crop.
                corners_global = [
                    (square_left, square_top),
                    (square_left, square_bottom),
                    (square_right, square_bottom),
                    (square_right, square_top),
                ]

                for x_global, y_global in corners_global:
                    x_local = x_global - left
                    y_local = y_global - top
                    cv2.circle(
                        square_cutout_annotated,
                        (int(x_local), int(y_local)),
                        6,
                        (0, 0, 255),
                        -1,
                    )
                
                # On-disk layout: one directory per setup (raw.png + calibration metadata),
                # containing one directory per captured frame, each with its warped image and a
                # squares/<label>/ directory like this one.
                square_dir = squares_dir / square_label
                square_dir.mkdir(exist_ok=True) # exist_ok=True for debugging purposes

                square_metadata = {
                    # Position of the crop in the warped image. Kept so the model can pick up on
                    # e.g. pieces nearer the camera appearing larger.
                    "top": top,
                    "left": left
                }

                # Save cutout (annotated PNG + masked npy; the plain PNG is intentionally dropped)
                cv2.imwrite(str(square_dir / f"{square_label}_annotated.png"), square_cutout_annotated)
                # Save uncompressed numpy array which also contains the mask
                square_cutout_masked[:, :, :3] = cv2.cvtColor(
                    square_cutout_masked[:, :, :3],
                    cv2.COLOR_BGR2RGB,
                )
                np.save(str(square_dir / f"{square_label}_masked.npy"), square_cutout_masked)

                # Save square metadata
                with open(square_dir / f"{square_label}_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(square_metadata, f, indent=2)

        return squares_dir

    def _cutout_v2(self, warped_image, squares_dir):
        """Per-square crop + hard mask from the precomputed square geometry.

        Each square's crop is the tight bounding box of its own mask polygon (variable size
        per square); the polygon is translated into the crop's local frame before being filled.
        The letterbox step (unchanged) handles the variably-sized crops. Which polygon gets
        filled is ``effective_mask_mode``'s call -- the hull by default, the square's own floor
        quad or nothing at all for the ablation arms.
        """
        squares_dir.mkdir(exist_ok=True)
        mask_mode = self.effective_mask_mode
        for label, geom in self.square_geometry.items():
            x_min, y_min, x_max, y_max = geom.bbox
            square_cutout = warped_image[y_min:y_max, x_min:x_max]

            offset = np.array([x_min, y_min])
            local_polygon = geom.mask_polygon - offset
            if mask_mode == "hull":
                mask = compute_square_mask(local_polygon, square_cutout.shape[:2])
            elif mask_mode == "square":
                # "Bad" mask: only the square's own floor quad. The warp rectifies the board, so
                # that quad is an axis-aligned square -- it stops short of the column of space the
                # piece actually leans into, which is the whole point of the hull.
                mask = compute_square_mask(geom.floor_pts - offset, square_cutout.shape[:2])
            else:  # "none": constant channel, carrying no information at all
                mask = np.zeros(square_cutout.shape[:2], dtype=np.uint8)
            square_cutout_masked = np.concatenate([square_cutout, mask[:, :, None]], axis=2)
            square_cutout_masked = letterbox(
                square_cutout_masked, (self.square_cutout_size, self.square_cutout_size)
            )

            # Annotated crop (BGR): convex-hull outline + the square's own floor-corner dots.
            square_cutout_annotated = square_cutout.copy()
            cv2.polylines(
                square_cutout_annotated,
                [local_polygon.astype(np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=1,
            )
            for x_local, y_local in geom.floor_pts - offset:
                cv2.circle(
                    square_cutout_annotated,
                    (int(round(x_local)), int(round(y_local))),
                    4,
                    (0, 0, 255),
                    -1,
                )

            square_dir = squares_dir / label
            square_dir.mkdir(exist_ok=True)
            square_metadata = {"top": int(y_min), "left": int(x_min)}

            cv2.imwrite(str(square_dir / f"{label}_annotated.png"), square_cutout_annotated)
            square_cutout_masked[:, :, :3] = cv2.cvtColor(
                square_cutout_masked[:, :, :3], cv2.COLOR_BGR2RGB
            )
            np.save(str(square_dir / f"{label}_masked.npy"), square_cutout_masked)
            with open(square_dir / f"{label}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(square_metadata, f, indent=2)

        return squares_dir


if __name__ == "__main__":
    # Warp one frame and write its 64 square crops next to it. Useful for eyeballing a
    # calibration (the annotated crops show the hull and the square's corners) without the robot.
    #
    #   uv run python -m chess_assistant.image_processing \
    #       <calibration_metadata.json> <config.yaml> <image.png>
    #
    # e.g. uv run python -m chess_assistant.image_processing \
    #       data/raw_images/calibration_metadata.json config.yaml data/raw_images/raw.png
    import sys
    assert len(sys.argv) == 4
    processor = Processor(sys.argv[1], sys.argv[2])
    warped_path = processor.warp(Path(sys.argv[3]))
    processor.cutout(warped_path)
