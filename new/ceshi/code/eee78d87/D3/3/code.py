# Variant of eee78d87
# Dimension: D3 — 目标画布的结构 (dst)
# Change: 16x16黑色画布用周期3的橙黑相位瓦片铺满，中心(5,5)-(10,10)区域放置2x2平铺的栗色字形印章... → 中心区域为空洞的环形格栅

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input stays identical to the source task: a 6x6 orange field with one non-orange
#      3x3 contour glyph (plus / H / X / four corners).
#    - Output still starts from the same 16x16 orange/black period-3 lattice decoded
#      from that glyph.
#    - The changed part is the overlay layout:
#        source task: center 6x6 receives a 2x2 maroon tiling of the glyph
#        this variant: center 6x6 is forced to pure black, and one 3x3 maroon copy of
#        the glyph is stamped in each output corner.
#    - Example with the plus glyph:
#        old flow  : decode plus -> periodic lattice -> maroon plus tiled at center
#        new flow  : decode plus -> periodic lattice -> erase center 6x6 to black
#                    -> stamp maroon plus at (0,0), (0,13), (13,0), (13,13)
#      The lattice code is unchanged; only the destination structure differs.
#
# b) Complexity calibration
#    - The glyph still controls the lattice phase, so the task is not reduced to a fixed
#      template. Four glyph categories remain, and each yields a distinct orange pattern.
#    - The new D3 signal is visually strong: the center is always empty while all four
#      corners carry the same maroon glyph. This is easy to compare across examples.
#
# c) Hidden assumptions from the original main()
#    - Input assertions remain valid: 6x6 size, orange background, exactly one non-orange
#      color, tight 3x3 bbox.
#    - The old code assumed the only maroon overlay lived in a centered 6x6 block made by
#      np.tile(motif, (2, 2)). That assumption is no longer valid and must be removed.
#    - The lattice fill can no longer be left untouched in the center; we must explicitly
#      clear the center slice to black, otherwise the "hole" would leak orange cells from
#      the phase tile.
#
# d) Placement legality
#    - Each corner stamp is exactly 3x3, so anchors (0,0), (0,13), (13,0), (13,13) fit
#      inside the 16x16 output without overlap.
#    - The center hole is rows 5..10 and cols 5..10, disjoint from all corner stamps.
#    - The generator still only has to place one 3x3 motif in a 6x6 input, which remains
#      valid for every top-left in 0..3.
#
# e) Visual observability
#    - From a few examples, a human can see two stable facts: the middle 6x6 is always
#      black, and the maroon glyph migrates to all four corners. Because the lattice still
#      varies with glyph type, the rule is not a trivial fixed output.
#
# f) Visual cleanliness
#    - Layers are separated: black/orange lattice everywhere, black hole only in the
#      center, maroon stamps only in corners. No region carries multiple competing marks.
#
# g) Color inference
#    - Colors remain black/orange/maroon only in the output. Maroon still signals the
#      decoded glyph, but now by corner placement rather than center emphasis.


OUT_SHAPE = (16, 16)
CENTER_HOLE = (5, 11, 5, 11)  # half-open slice: rows 5:11, cols 5:11
CORNER_ANCHORS = ((0, 0), (0, 13), (13, 0), (13, 13))


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


def _stamp_corner_glyphs(out: np.ndarray, motif: np.ndarray) -> None:
    for r0, c0 in CORNER_ANCHORS:
        out[r0 : r0 + 3, c0 : c0 + 3][motif] = Color.MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose center becomes a black void and whose corners carry maroon glyph stamps.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one tight 3x3 contour glyph.
    - The glyph is one of the task's four valid code shapes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile used by the source task.
    - Repeat that phase tile across a 16x16 black canvas.
    - Erase the centered 6x6 zone to pure black, then stamp one maroon copy of the glyph into each of the four 3x3 corners.

    Key insight:
    - The input glyph still acts as a phase code for the lattice, but the highlighted decoded glyph no longer occupies the center; the center is a hole and the corners become the emphasized locations.
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

    r0, r1, c0, c1 = CENTER_HOLE
    out[r0:r1, c0:c1] = Color.BLACK
    _stamp_corner_glyphs(out, motif)
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6                        # input grid dimensions
MOTIF_SIZE = 3                               # glyph bounding box side length
MOTIF_POS_MAX = 4                            # exclusive upper bound for top-left position (0..3)
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]           # valid foreground colors (exclude 0, 7, 9)
MAX_RETRIES = 300                            # oracle-validation retry budget

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    # Variant-specific oracle checks: the center must be a pure black hole and each
    # corner must contain at least one maroon pixel from the glyph stamp.
    if not np.all(out[5:11, 5:11] == Color.BLACK):
        return None
    corner_views = (
        out[0:3, 0:3],
        out[0:3, 13:16],
        out[13:16, 0:3],
        out[13:16, 13:16],
    )
    if any(not np.any(view == Color.MAROON) for view in corner_views):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D3 variant")
