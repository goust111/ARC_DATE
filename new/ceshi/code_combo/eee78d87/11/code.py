# Combo variant of eee78d87 (index 11)
# combo_spec: D4/3+D11/2
# Name: 像素放大印章+旋转拼接

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for combo D4/3 + D11/2
# spec_variant:
#   preserved:
#   - input stays a 6x6 orange grid containing one tight 3x3 contour glyph in a single non-orange color
#   - the 3x3 glyph still decodes the same 3x3 orange/black phase tile, and that tile still fills a 16x16 output
#   - glyph vocabulary remains {plus, H, X, 4corners}; no extra objects or colors are introduced
#   changed_core:
#   - the center is no longer a 6x6 repeated motif and no longer a single 6x6 pixel-enlarged motif
#   - instead, build four rotated versions of the 3x3 glyph (0°, 90° clockwise, 180°, 90° counterclockwise),
#     enlarge each rotated glyph by 2x2 pixel replication to get four 6x6 blocks, then stitch those blocks into
#     one centered 12x12 maroon stamp
#   rotation_layout:
#   - the stitched 12x12 stamp uses a cyclic quadrant order so the rotations read consistently around the center:
#       top-left     = original
#       top-right    = 90° clockwise
#       bottom-right = 180°
#       bottom-left  = 90° counterclockwise
#   concrete_trace:
#   - example glyph H = 101/111/101
#   - original task center: np.tile(H, (2, 2)) gives a 6x6 pattern with repeated 3x3 seams
#   - D11 alone: enlarging H gives one 6x6 coarse mask
#   - this combo: enlarge H, enlarge rot90(H), enlarge rot180(H), enlarge rot270(H), then assemble them into a 12x12
#     pinwheel-like stamp; for H the horizontal and vertical forms alternate by quadrant, so the rotation rule becomes visible
#   complexity_calibration:
#   - the scale is fixed at 2 and the layout is fixed at 2x2 quadrants, so there is only one new spatial rule to infer
#   - the output remains 16x16, which keeps the lattice familiar while making the new center visually obvious
#   - generator biases toward H because plus/X/4corners are rotation-invariant and would hide the D4 component in too many samples
#   implicit_assumption_audit:
#   - _decode_phase_tile depends only on the 3x3 boolean glyph, so it is still valid unchanged
#   - the old assumption center.shape == (6, 6) is invalidated; the combo stamp is now always 12x12
#   - geometric centering still works because 12x12 fits safely inside 16x16, yielding a fixed window (rows 2:14, cols 2:14)
#   - maroon overwrite remains safe because only the centered stamp is recolored; the background lattice underneath is irrelevant
#   placement_legality:
#   - input placement is unchanged: a 3x3 glyph with top-left in 0..3 always fits in the 6x6 input
#   - the centered 12x12 stamp occupies rows 2..13 and cols 2..13 of the 16x16 output, so it never exceeds bounds
#   visual_observability:
#   - every sample shows a much larger 12x12 coarse stamp, so the enlargement component is always visible
#   - H samples also show quadrant-wise rotation alternation, which disambiguates "rotation stitching" from simple 2x2 copying
#   visual_cleanliness:
#   - only one compact maroon object is overlaid on the lattice, so the output remains readable despite the larger center
#   color_inference:
#   - output colors stay {black, orange, maroon}; the fusion only changes shape construction, not palette complexity
#   generator_validation:
#   - oracle validation still uses main(inp)
#   - anti-degeneration checks compare the 12x12 combo stamp against the legacy 6x6 centers and record how many distinct
#     enlarged quadrants appear; this confirms the generator is not silently collapsing back to the old behavior
#   new_main_assumptions:
#   - the combo helper assumes motif.shape == (3, 3) and fixed scale == 2
#   - generate_input guarantees these assumptions by constructing only valid original-task glyph instances

STAMP_SCALE = 2
OUT_SHAPE = (16, 16)
GLYPH_NAMES = ["H", "plus", "X", "4corners"]
GLYPH_PROBS = np.array([0.55, 0.15, 0.15, 0.15], dtype=float)


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 orange phase mask."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edge_centers = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)
    phase = np.zeros((3, 3), dtype=bool)

    if not bool(corners.any()):
        assert bool(motif[1, 1])
        assert bool(edge_centers.all())
        phase[1:, 1:] = True
        return phase

    assert bool(corners.all())
    assert not bool(motif[0, 1])
    assert not bool(motif[2, 1])
    assert bool(motif[1, 0]) == bool(motif[1, 2])

    phase[1:, 0] = True
    if bool(motif[1, 0]):
        assert bool(motif[1, 1])
    else:
        phase[0, 1:] = True

    if not bool(motif[1, 1]):
        phase[0, 0] = True

    return phase


def _pixel_enlarge(mask: np.ndarray, scale: int) -> np.ndarray:
    """Expand each boolean pixel into a scale x scale solid block."""
    assert mask.ndim == 2
    assert scale >= 1
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _build_combo_stamp(motif: np.ndarray) -> np.ndarray:
    """Build the 12x12 rotation-stitched stamp from one 3x3 motif."""
    assert motif.shape == (3, 3)
    enlarged = [_pixel_enlarge(np.rot90(motif, -k), STAMP_SCALE) for k in range(4)]
    top = np.hstack([enlarged[0], enlarged[1]])
    bottom = np.hstack([enlarged[3], enlarged[2]])
    stamp = np.vstack([top, bottom])
    assert stamp.shape == (12, 12)
    return stamp


def _legacy_tiled_stamp(motif: np.ndarray) -> np.ndarray:
    return np.tile(motif, (2, 2)).astype(bool)


def _legacy_enlarged_stamp(motif: np.ndarray) -> np.ndarray:
    return _pixel_enlarge(motif, STAMP_SCALE)


def _quadrant_signature_count(stamp: np.ndarray) -> int:
    """Count distinct 6x6 quadrant masks inside the 12x12 combo stamp."""
    assert stamp.shape == (12, 12)
    h = stamp.shape[0] // 2
    w = stamp.shape[1] // 2
    quadrants = [
        stamp[:h, :w],
        stamp[:h, w:],
        stamp[h:, :w],
        stamp[h:, w:],
    ]
    return len({q.astype(int).tobytes() for q in quadrants})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into the original periodic lattice and place a larger rotation-stitched maroon seal at the center.

    Input structure:
    - The input is a 6x6 orange grid with exactly one non-orange color.
    - Those non-orange cells occupy one tight 3x3 bounding box.
    - The 3x3 object is one of the contour glyphs plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same 3x3 orange/black phase tile as in the base task and repeat it over a 16x16 canvas.
    - Build four rotated versions of the glyph: original, 90 degrees clockwise, 180 degrees, and 90 degrees counterclockwise.
    - Enlarge each rotated glyph by turning every pixel into a 2x2 block, stitch the four 6x6 results into a centered 12x12 seal, and recolor its true cells to maroon.

    Key insight:
    - The input glyph still controls the lattice phase, but the center emblem is no longer a direct copy or a simple enlargement: it is a composite assembled from enlarged rotated copies of the same 3x3 code.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill(OUT_SHAPE, phase_colors)

    center = _build_combo_stamp(motif)
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(GLYPH_NAMES, p=GLYPH_PROBS))
    motif = GLYPHS[glyph_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    combo_stamp = _build_combo_stamp(motif)
    if combo_stamp.shape != (12, 12):
        return None

    legacy_tile = _legacy_tiled_stamp(motif)
    legacy_enlarged = _legacy_enlarged_stamp(motif)
    if legacy_tile.shape != (6, 6) or legacy_enlarged.shape != (6, 6):
        return None

    maroon = out == Color.MAROON
    maroon_box = bbox(maroon)
    if maroon_box != (2, 2, 13, 13):
        return None

    # The combo must never collapse to the old 6x6 center footprint.
    if int(maroon.sum()) <= int(legacy_enlarged.sum()):
        return None

    # H should expose the rotation stitching with at least two distinct enlarged quadrants.
    if glyph_name == "H" and _quadrant_signature_count(combo_stamp) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for combo variant eee78d87/11")
