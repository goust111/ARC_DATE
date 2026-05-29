# Variant of eee78d87
# Dimension: D3 — 目标画布的结构 (dst)
# Change: 16x16黑色画布用周期3的橙黑相位瓦片铺满，中心(5,5)-(10,10)区域放置2x2平铺的栗色字形印章... → 画布尺寸由字形类型决定

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D3 variant
# ----------------------------------
# spec_variant:
#   unchanged_input:
#     grid: 6x6
#     background: orange
#     object: one 3x3 contour glyph in a single non-orange color
#     glyph_types: plus, H, X, 4corners
#   changed_output:
#     lattice: still a black canvas tiled by the glyph's decoded 3x3 orange phase tile
#     stamp: still a 2x2 tiling of the original glyph recolored to maroon
#     canvas_size_map:
#       plus: 12x12
#       X: 15x15
#       H: 18x18
#       4corners: 21x21
#
# behavior_trace:
#   original:
#     plus -> decode phase -> always build a 16x16 lattice -> stamp the 6x6 maroon motif at rows 5..10, cols 5..10
#   variant:
#     plus -> decode phase -> build a 12x12 lattice -> center the same 6x6 maroon stamp at rows 3..8, cols 3..8
#     H    -> decode phase -> build an 18x18 lattice -> center the same 6x6 maroon stamp at rows 6..11, cols 6..11
#   difference:
#     the motif still controls the orange/black phase pattern, but now it also selects the canvas size class.
#
# complexity_calibration:
#   - Only one dimension changes: the destination canvas size.
#   - There are four discrete size classes, one per glyph, so the variation is obvious in output silhouettes.
#   - The mapping is exact and low-entropy; it does not depend on color ids or input position.
#
# hidden_assumption_review:
#   - `g.shape == (6, 6)` remains valid because the input format is unchanged.
#   - `motif.shape == (3, 3)` remains valid because each glyph still occupies a tight 3x3 bbox.
#   - The original `out_shape = (3*g.shape[0]-2, 3*g.shape[1]-2)` assumed output size was derived only from the fixed input size.
#     That assumption is no longer valid, so the new solver must classify the glyph before constructing the canvas.
#   - Center stamping still works only if the chosen canvas is at least 6x6; the smallest new size is 12x12, so placement is legal.
#
# placement_legality:
#   - Input placement is unchanged: the 3x3 motif is placed at any top-left (r0, c0) in 0..3.
#   - Output placement is always legal because every canvas size is square and >= 12, while the maroon stamp is fixed at 6x6.
#   - The stamp is centered by integer floor division, which is deterministic for both even and odd canvas sizes.
#
# visual_observability:
#   - Human solvers still see the same contour glyph in the input and the same phase decoding in the output.
#   - The new rule is visible because the whole output frame expands or shrinks by glyph type while the central maroon motif stays 6x6.
#   - No invisible feature such as the foreground color id is used in the mapping.
#
# cleanliness:
#   - Only three output colors remain: black, orange, maroon.
#   - The maroon stamp stays in a dedicated centered block, separated from the outer lattice by several cells on every size class.


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

CANVAS_SIZE_BY_GLYPH = {
    "plus": 12,
    "X": 15,
    "H": 18,
    "4corners": 21,
}


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


def _glyph_name(motif: np.ndarray) -> str:
    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    raise AssertionError("Unknown contour glyph")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a glyph-sized periodic lattice and place a centered maroon stamp of the motif.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form a tight 3x3 contour glyph.
    - The glyph is one of four known code shapes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 contour glyph into the same period-3 orange/black phase tile as in the base task.
    - Choose the output canvas size from the glyph type itself: plus->12, X->15, H->18, 4corners->21.
    - Fill that square canvas by tiling the decoded phase tile, then center a 2x2 maroon tiling of the original glyph on top.

    Key insight:
    - The input glyph now carries two pieces of information at once: its contour still specifies the repeating orange phase, and its identity also selects the overall size of the destination canvas.
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

    glyph_name = _glyph_name(motif)
    canvas_size = CANVAS_SIZE_BY_GLYPH[glyph_name]

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((canvas_size, canvas_size), phase_colors)

    center = np.tile(motif, (2, 2))
    assert center.shape == (6, 6)
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    stamp = out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]]
    stamp[center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
GLYPH_NAMES = tuple(GLYPHS.keys())


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(GLYPH_NAMES))
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

    expected_size = CANVAS_SIZE_BY_GLYPH[glyph_name]
    if out.shape != (expected_size, expected_size):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D3 variant")
