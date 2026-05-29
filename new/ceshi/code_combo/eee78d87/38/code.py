# Combo variant of eee78d87 (index 38)
# combo_spec: D1/3+D3/0+D9/0
# Name: 位置编码偏移+类型决定画布+位置映射印章

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Input stays a 6x6 orange grid with exactly one 3x3 contour glyph in one non-orange color.
#    - Original task used three fixed choices after reading the glyph: fixed 16x16 canvas, zero lattice
#      offset, and centered 6x6 maroon stamp. Only the glyph shape changed the background tile.
#    - Combo behavior splits control across two visible signals:
#        1. glyph identity selects canvas size,
#        2. glyph bbox top-left selects both lattice phase offset and stamp position.
#    - Example: a plus glyph at input top-left (3,1) decodes to the original plus base tile, but now:
#        plus -> 12x12 canvas,
#        offset -> (3 mod 3, 1 mod 3) = (0,1),
#        stamp origin -> scale (3,1) from input range 0..3 onto output range 0..6, giving (6,2).
#      The same plus at (0,0) would still make a 12x12 canvas, but with offset (0,0) and stamp at (0,0).
#      So location changes two output attributes while shape changes a third.
# b) Complexity calibration:
#    - Canvas size has 4 discrete classes, one per glyph: plus->12, X->15, H->18, 4corners->21.
#    - Phase offset has 9 modulo-3 classes, but these collapse from the same visible 4x4 placement grid.
#    - Stamp placement has 16 possible positions because the 3x3 bbox top-left ranges over 0..3 on each axis.
#    - This is richer than the base task but still low-entropy: every control variable comes from a small,
#      repeated, human-visible category instead of hidden numeric ids.
# c) Implicit assumption review:
#    - `g.shape == (6, 6)` still holds; the generator keeps the original input frame.
#    - `bbox(fg)` must isolate one object and `crop_bbox(...).shape == (3,3)` must hold; the generator still
#      places exactly one known glyph with a tight bbox.
#    - The original assumption `out_shape = (16,16)` is no longer valid, so the solver must classify the glyph
#      before building the canvas.
#    - The original assumption that stamp origin is centered is also invalid; it now depends on bbox position and
#      therefore on the chosen canvas size because the legal destination range changes with size.
#    - Using `tile_fill(..., offset=...)` remains safe because modulo-3 offsets are always legal for a 3x3 tile.
# d) Placement legality:
#    - Input glyph top-left is sampled in 0..3 for both row and col, so the 3x3 motif always fits inside 6x6.
#    - The 2x2 stamp is always 6x6. All canvas sizes are at least 12x12, so every scaled origin stays in-bounds.
#    - Scaling is done independently on rows and columns from source limit 3 to destination limit `canvas_size-6`,
#      which guarantees the full stamp remains inside the output.
# e) Visual observability:
#    - Human solvers can see output size change by glyph class, so D3 is globally obvious.
#    - They can also compare same-size samples or mentally discount size and notice that glyph position shifts the
#      lattice phase and the maroon stamp together, but by different formulas: modulo-3 for background, scaling for
#      stamp. Both depend only on visible placement, not on color ids.
# f) Visual cleanliness:
#    - Output still uses only black, orange, and maroon.
#    - There are only two layers: a tiled background and one 6x6 stamp. The stamp may appear near edges, but it
#      never leaves the canvas and remains visually distinct from the orange/black lattice.
# g) Change inferability:
#    - No new color mapping is introduced. The only cues are glyph type and glyph position, both directly visible.
#    - Because the canvas size classes are distinct and the stamp positions are far apart after scaling, 2-4
#      examples are enough to separate the three rules instead of conflating them into one opaque transform.


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

GLYPH_NAMES = tuple(GLYPHS.keys())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the original period-3 orange phase mask."""
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


def _scale_origin(src: int, src_limit: int, dst_limit: int) -> int:
    """Map one discrete legal origin range onto another by proportional scaling."""
    src = int(src)
    src_limit = int(src_limit)
    dst_limit = int(dst_limit)
    assert 0 <= src <= src_limit
    assert dst_limit >= 0
    if src_limit == 0:
        return 0
    return int(round(src * dst_limit / src_limit))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a variable-sized periodic lattice whose phase and maroon stamp placement are both controlled by the glyph's input placement.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one tight 3x3 contour glyph.
    - The glyph is one of the four task code shapes, and its bbox top-left position is meaningful.

    Rule:
    - Decode the 3x3 glyph into the same base period-3 orange/black tile as in the original task.
    - Identify the glyph type and choose the square canvas size from it: plus->12, X->15, H->18, 4corners->21.
    - Use the glyph bbox top-left modulo 3 as the lattice phase offset when tiling the background.
    - Tile the original 3x3 glyph 2x2 into a 6x6 stamp, recolor its true cells to maroon, and place that stamp at a position obtained by scaling the input bbox top-left into the legal output stamp-origin range.

    Key insight:
    - The small contour code now carries three separable signals at once: shape selects canvas size, while placement selects both the repeating background phase and the stamp destination.
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
    phase_offset = (int(b[0]) % 3, int(b[1]) % 3)
    out = tile_fill((canvas_size, canvas_size), phase_colors, offset=phase_offset)

    stamp = np.tile(motif, (2, 2))
    assert stamp.shape == (6, 6)
    in_r_limit = g.shape[0] - motif.shape[0]
    in_c_limit = g.shape[1] - motif.shape[1]
    out_r_limit = out.shape[0] - stamp.shape[0]
    out_c_limit = out.shape[1] - stamp.shape[1]
    r0 = _scale_origin(int(b[0]), in_r_limit, out_r_limit)
    c0 = _scale_origin(int(b[1]), in_c_limit, out_c_limit)
    out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]][stamp] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 38")
