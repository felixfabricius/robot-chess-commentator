"""Tests for the pure geometry helpers in image_processing and the v2 Processor geometry.

Import-light: cv2/numpy only, no reachy/pygame/torch.
"""

import json

import cv2
import numpy as np
import pytest

from chess_commentator.perception.image_processing import (
    CROP_MODES,
    MASK_MODES,
    MASK_VARIANTS,
    Processor,
    QuadrantField,
    bilinear_interp,
    compute_square_mask,
    estimate_vanishing_point,
    mask_bounding_box,
)


# --------------------------------------------------------------------------------------------
# estimate_vanishing_point
# --------------------------------------------------------------------------------------------

def test_estimate_vanishing_point_recovers_apex():
    apex = np.array([320.0, -450.0])
    base_pts = np.array([[100.0, 500.0], [540.0, 500.0], [300.0, 480.0], [200.0, 520.0],
                         [420.0, 470.0]])
    # Each extended point lies on the ray from its base toward the apex.
    ext_pts = base_pts + 0.3 * (apex - base_pts)
    V, residual = estimate_vanishing_point(base_pts, ext_pts)
    assert np.allclose(V, apex, atol=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_estimate_vanishing_point_residual_flags_bad_ray():
    apex = np.array([320.0, -450.0])
    base_pts = np.array([[100.0, 500.0], [540.0, 500.0], [300.0, 480.0], [420.0, 470.0]])
    ext_pts = base_pts + 0.3 * (apex - base_pts)
    # Corrupt one ray so it no longer points at the apex.
    ext_pts[0] = ext_pts[0] + np.array([40.0, 0.0])
    V, residual = estimate_vanishing_point(base_pts, ext_pts)
    assert residual > 1.0  # a bad click shows up as a large mean residual


# --------------------------------------------------------------------------------------------
# bilinear_interp
# --------------------------------------------------------------------------------------------

def test_bilinear_interp_corners():
    assert bilinear_interp(1, 2, 3, 4, 0, 0) == 1  # tl
    assert bilinear_interp(1, 2, 3, 4, 1, 0) == 2  # tr
    assert bilinear_interp(1, 2, 3, 4, 1, 1) == 3  # br
    assert bilinear_interp(1, 2, 3, 4, 0, 1) == 4  # bl


def test_bilinear_interp_midpoints():
    # centre = mean of all four; edge midpoints = mean of the two adjacent corners.
    assert bilinear_interp(0, 0, 0, 0, 0.5, 0.5) == 0
    assert bilinear_interp(2, 4, 8, 6, 0.5, 0.5) == pytest.approx(5.0)
    assert bilinear_interp(2, 4, 8, 6, 0.5, 0.0) == pytest.approx(3.0)  # top edge midpoint


# --------------------------------------------------------------------------------------------
# QuadrantField
# --------------------------------------------------------------------------------------------

def test_quadrant_field_returns_measured_values_at_nodes():
    f = QuadrantField(m_tl=10, m_tr=20, m_br=30, m_bl=40, m_center=100)
    assert f(0, 0) == pytest.approx(10)
    assert f(1, 0) == pytest.approx(20)
    assert f(1, 1) == pytest.approx(30)
    assert f(0, 1) == pytest.approx(40)
    assert f(0.5, 0.5) == pytest.approx(100)


def test_quadrant_field_edge_midpoints_are_corner_means():
    f = QuadrantField(m_tl=10, m_tr=20, m_br=30, m_bl=40, m_center=100)
    assert f(0.5, 0.0) == pytest.approx(15)  # top mid = (tl+tr)/2
    assert f(1.0, 0.5) == pytest.approx(25)  # right mid = (tr+br)/2
    assert f(0.5, 1.0) == pytest.approx(35)  # bottom mid = (br+bl)/2
    assert f(0.0, 0.5) == pytest.approx(25)  # left mid = (tl+bl)/2


def test_quadrant_field_continuous_across_boundary():
    f = QuadrantField(m_tl=10, m_tr=20, m_br=30, m_bl=40, m_center=100)
    eps = 1e-6
    # Straddle the u=0.5 seam in the top half.
    assert f(0.5 - eps, 0.25) == pytest.approx(f(0.5 + eps, 0.25), abs=1e-3)
    # Straddle the v=0.5 seam in the left half.
    assert f(0.25, 0.5 - eps) == pytest.approx(f(0.25, 0.5 + eps), abs=1e-3)


def test_quadrant_field_supports_vectors():
    # Node values are 2D displacement vectors; interpolation is componentwise.
    f = QuadrantField(
        m_tl=np.array([1.0, 2.0]),
        m_tr=np.array([3.0, 4.0]),
        m_br=np.array([5.0, 6.0]),
        m_bl=np.array([7.0, 8.0]),
        m_center=np.array([4.0, 5.0]),  # == corner average -> reduces to plain bilinear
    )
    assert np.allclose(f(0, 0), [1, 2])
    assert np.allclose(f(1, 1), [5, 6])
    assert np.allclose(f(0.5, 0.5), [4, 5])  # plain-bilinear centre


# --------------------------------------------------------------------------------------------
# Processor v2 geometry (synthetic calibration)
# --------------------------------------------------------------------------------------------

def _intersect(p1, p2, p3, p4):
    """Intersection of line p1-p2 with line p3-p4."""
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    d1, d2 = p2 - p1, p4 - p3
    t = np.linalg.solve(np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]]), p3 - p1)
    return p1 + t[0] * d1


def _write_v2_metadata(tmp_path, apex=(960.0, -600.0), s=0.18):
    corner_map = {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}
    actual = {
        "a8": [700.0, 300.0],
        "h8": [1220.0, 300.0],
        "h1": [1500.0, 800.0],
        "a1": [420.0, 800.0],
    }
    apex = np.array(apex)
    extended = {
        label: (np.array(px) + s * (apex - np.array(px))).tolist()
        for label, px in actual.items()
    }
    # The board centre (board coord (0.5, 0.5)) maps to the intersection of the corner-quad's
    # diagonals — that is the true base of the centre extension. Seeding it here makes all five
    # rays converge exactly at the camera apex, so the warped V equals H(apex).
    center_base = _intersect(actual["a8"], actual["h1"], actual["h8"], actual["a1"])
    extended_center = (center_base + s * (apex - center_base)).tolist()

    metadata = {
        "calibration_version": 2,
        "camera_natural_orientation": {"order": corner_map},
        "actual_corner_order": ["a1", "a8", "h8", "h1"],
        "actual_corners_px": actual,
        "extended_corner_order": ["a1", "a8", "h8", "h1"],
        "extended_corners_px": extended,
        "extended_center_px": extended_center,
    }
    path = tmp_path / "calibration_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_processor_v2_builds_vanishing_point_and_field(tmp_path):
    apex = (960.0, -600.0)
    processor = Processor(_write_v2_metadata(tmp_path, apex=apex), None)

    assert processor.is_v2 is True
    assert processor.V is not None and processor.V.shape == (2,)
    assert np.all(np.isfinite(processor.V))
    # All five rays converge exactly at the camera apex, so V == H(apex) and residual ~ 0.
    assert processor.vp_residual < 0.5
    expected_V = cv2.perspectiveTransform(
        np.array([[apex]], dtype=np.float32), processor.matrix
    ).reshape(2)
    assert np.allclose(processor.V, expected_V, atol=1.0)

    # The extension field returns finite 2D displacement vectors.
    for u, v in [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]:
        disp = processor.extension_field(u, v)
        assert np.asarray(disp).shape == (2,)
        assert np.all(np.isfinite(disp))


def test_processor_v1_metadata_has_no_v2_geometry():
    # Legacy v1 metadata: no extended_center_px / camera_intrinsics -> no undistortion and no
    # v2 per-square geometry (behaves exactly as before).
    v1 = {
        "camera_natural_orientation": {"order": {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}},
        "actual_corners_px": {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]},
        "extended_corners_px": {"a8": [690, 250], "h8": [1230, 250], "h1": [1470, 760], "a1": [440, 760]},
    }
    processor = Processor(v1, None)
    assert processor.is_v2 is False
    assert processor.V is None
    assert processor.extension_field is None
    assert processor.square_geometry is None
    assert processor.undistort_map1 is None


def test_extension_follows_clicks_even_when_vanishing_point_flips():
    # Divergent piece-height rays: the least-squares V lands on the wrong side, so the OLD
    # `unit(V - corner)` direction would point away from the clicked tops. The displacement
    # field must still reproduce the measured extensions exactly at the corners.
    from chess_commentator.perception.calibration import infer_camera_natural_corner_order

    actual = {"a8": [500, 800], "h8": [1400, 800], "h1": [1400, 300], "a1": [500, 300]}
    extended = {"a8": [380, 720], "h8": [1520, 720], "h1": [1520, 230], "a1": [380, 230]}
    meta = {
        "actual_corners_px": actual,
        "extended_corners_px": extended,
        "extended_center_px": [950, 515],
        "camera_natural_orientation": infer_camera_natural_corner_order(actual),
    }
    p = Processor(meta, None)
    order = meta["camera_natural_orientation"]["order"]

    dst = np.array(
        [
            [p.padding["left"], p.padding["up"]],
            [p.padding["left"] + p.board_size, p.padding["up"]],
            [p.padding["left"] + p.board_size, p.padding["up"] + p.board_size],
            [p.padding["left"], p.padding["up"] + p.board_size],
        ],
        dtype=np.float64,
    )
    src_ext = np.array([extended[order[q]] for q in ["tl", "tr", "br", "bl"]], np.float32)
    ext_w = cv2.perspectiveTransform(src_ext.reshape(4, 1, 2), p.matrix).reshape(4, 2)
    measured = ext_w - dst  # measured warped displacements, tl, tr, br, bl

    # This config genuinely triggers the flip: V is on the opposite vertical side of the clicks.
    assert np.sign((p.V - dst[0])[1]) != np.sign(measured[0][1])

    # The field reproduces the measured displacement (direction + magnitude) at each corner.
    for idx, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        assert np.allclose(p.extension_field(u, v), measured[idx], atol=1e-6)


# --------------------------------------------------------------------------------------------
# mask_bounding_box / compute_square_mask
# --------------------------------------------------------------------------------------------

def test_mask_bounding_box_margin():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)  # centroid (5, 5)
    assert mask_bounding_box(pts, margin=1.0) == (0, 0, 10, 10)
    # 20% expansion about the centroid pushes each corner out by 1 px.
    assert mask_bounding_box(pts, margin=1.2) == (-1, -1, 11, 11)


def test_compute_square_mask_fills_polygon():
    poly = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float64)
    mask = compute_square_mask(poly, (10, 10))
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask[5, 5] == 1  # inside the hull
    assert mask[0, 0] == 0  # outside the hull


# --------------------------------------------------------------------------------------------
# Processor.cutout (v2)
# --------------------------------------------------------------------------------------------

def test_processor_v2_cutout_outputs(tmp_path):
    processor = Processor(_write_v2_metadata(tmp_path), None)

    frame = np.random.default_rng(2).integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    img_path = tmp_path / "image.png"
    cv2.imwrite(str(img_path), frame)

    warped_path = processor.warp(img_path)
    squares_dir = processor.cutout(warped_path)

    assert sum(1 for d in squares_dir.iterdir() if d.is_dir()) == 64

    tops = {}
    for sq in ["a1", "e4", "h8", "d5"]:
        sq_dir = squares_dir / sq
        arr = np.load(sq_dir / f"{sq}_masked.npy")
        assert arr.shape == (144, 144, 4)  # letterboxed RGBA
        assert set(np.unique(arr[..., 3])).issubset({0, 1})  # hard 0/1 mask
        assert arr[..., 3].max() == 1  # the mask actually covers something
        meta = json.loads((sq_dir / f"{sq}_metadata.json").read_text())
        assert "top" in meta and "left" in meta
        tops[sq] = (meta["top"], meta["left"])

    # Per-square tight crops: different squares have different crop origins (unlike the old
    # fixed global padding, which gave every square an identical origin/size).
    assert len(set(tops.values())) > 1


# --------------------------------------------------------------------------------------------
# Mask / crop ablation variants
# --------------------------------------------------------------------------------------------

def _cut_with(tmp_path, mask_mode=None, crop_mode=None):
    """Warp + cut one synthetic frame under the given ablation modes; returns its squares dir."""
    metadata_path = _write_v2_metadata(tmp_path)
    work = tmp_path / f"cut_{mask_mode}_{crop_mode}"
    work.mkdir()
    processor = Processor(metadata_path, None, mask_mode=mask_mode, crop_mode=crop_mode)
    frame = np.random.default_rng(2).integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    img_path = work / "image.png"
    cv2.imwrite(str(img_path), frame)
    return processor.cutout(processor.warp(img_path))


def _mask_of(squares_dir, square):
    return np.load(squares_dir / square / f"{square}_masked.npy")[..., 3]


def test_mask_variants_are_well_formed():
    assert MASK_VARIANTS["default"] == ("hull", "per_square")
    for mask_mode, crop_mode in MASK_VARIANTS.values():
        assert mask_mode in MASK_MODES
        assert crop_mode in CROP_MODES


def test_modes_default_to_what_the_calibration_supports(tmp_path):
    v2 = Processor(_write_v2_metadata(tmp_path), None)
    assert (v2.effective_mask_mode, v2.effective_crop_mode) == ("hull", "per_square")

    v1 = Processor(
        {
            "camera_natural_orientation": {"order": {"tl": "a8", "tr": "h8", "br": "h1", "bl": "a1"}},
            "actual_corners_px": {"a8": [700, 300], "h8": [1220, 300], "h1": [1500, 800], "a1": [420, 800]},
            "extended_corners_px": {"a8": [690, 250], "h8": [1230, 250], "h1": [1470, 760], "a1": [440, 760]},
        },
        None,
    )
    assert (v1.effective_mask_mode, v1.effective_crop_mode) == ("square", "global")


def test_processor_rejects_unknown_modes(tmp_path):
    metadata_path = _write_v2_metadata(tmp_path)
    with pytest.raises(ValueError, match="mask_mode"):
        Processor(metadata_path, None, mask_mode="soft")
    with pytest.raises(ValueError, match="crop_mode"):
        Processor(metadata_path, None, crop_mode="tight")


def test_mask_mode_none_leaves_the_channel_empty(tmp_path):
    squares_dir = _cut_with(tmp_path, "none", "global")
    for sq in ["a1", "e4", "h8"]:
        # Constant channel -> carries no information, so the 4-channel input is equivalent to a
        # 3-channel one. That is the point of the "no mask at all" arm.
        assert _mask_of(squares_dir, sq).max() == 0


def test_square_mask_is_a_strict_subset_of_the_hull_mask(tmp_path):
    hull = _cut_with(tmp_path, "hull", "per_square")
    square = _cut_with(tmp_path, "square", "per_square")
    for sq in ["a1", "e4", "h8", "d5"]:
        hull_mask, square_mask = _mask_of(hull, sq), _mask_of(square, sq)
        assert square_mask.max() == 1
        assert np.all(square_mask <= hull_mask)     # never covers what the hull does not
        assert square_mask.sum() < hull_mask.sum()  # stops short of the piece's column of space


def test_mask_mode_does_not_change_the_crop(tmp_path):
    """The crop box comes from the hull bbox regardless of which mask gets filled.

    This is what makes the ablation cheap: at a fixed crop mode every mask arm shares the very
    same RGB pixels, and only the 4th channel differs.
    """
    hull = _cut_with(tmp_path, "hull", "per_square")
    square = _cut_with(tmp_path, "square", "per_square")
    for sq in ["a1", "e4", "h8"]:
        hull_rgb = np.load(hull / sq / f"{sq}_masked.npy")[..., :3]
        square_rgb = np.load(square / sq / f"{sq}_masked.npy")[..., :3]
        assert np.array_equal(hull_rgb, square_rgb)


def test_crop_mode_global_gives_every_square_the_same_box(tmp_path):
    """The "worse padding" arm: one padding box for all 64 squares.

    The annotated PNG is the raw (un-letterboxed) crop, so its shape is the crop box.
    """
    def crop_shapes(squares_dir):
        return {
            d.name: cv2.imread(str(d / f"{d.name}_annotated.png")).shape[:2]
            for d in squares_dir.iterdir()
            if d.is_dir()
        }

    global_shapes = crop_shapes(_cut_with(tmp_path, "square", "global"))
    per_square_shapes = crop_shapes(_cut_with(tmp_path, "square", "per_square"))

    assert len(global_shapes) == 64 and len(set(global_shapes.values())) == 1
    # Per-square crops are steered toward where each square's own piece leans, so they vary.
    assert len(set(per_square_shapes.values())) > 1
