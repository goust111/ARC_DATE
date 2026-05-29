# Combo variant of eee78d87 (index 16)
# combo_spec: D3/3+D9/1
# Name: 中心空洞环形+角落印章

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Input remains the source-task format: one 3x3 contour glyph inside a 6x6 orange
#      field, using one non-orange color.
#    - The original output pipeline had two stages:
#        1. decode glyph -> 3x3 orange/black phase tile -> repeat over 16x16
#        2. tile the 3x3 glyph 2x2 into a 6x6 maroon center stamp
#    - The combo keeps stage 1 intact but changes stage 2 in two linked ways:
#        1. the center 6x6 is turned into a pure black hole
#        2. the maroon glyph is moved to exactly one corner, chosen by glyph type
#    - Example with a plus glyph:
#        source: decode plus -> lattice -> 6x6 maroon plus-tiling at rows/cols 5..10
#        combo : decode plus -> lattice -> force rows/cols 5..10 to black ->
#                place one 3x3 maroon plus at the top-left corner
#      The lattice code still comes from the glyph shape; the highlighted glyph moves
#      away from the center and its corner now also carries information.
#
# b) Complexity calibration
#    - The task still has four glyph classes, each producing a different lattice phase.
#    - A second visible attribute is added by a four-way glyph->corner mapping:
#        plus -> top-left, X -> top-right, 4corners -> bottom-left, H -> bottom-right
#    - This is enough variation to make the fusion visible without adding fragile
#      degrees of freedom such as random corner choice or color-dependent branching.
#
# c) Hidden assumptions from the original main()
#    - Shape / color assertions still hold: 6x6 input, orange background, one non-orange
#      color, tight 3x3 bbox, motif from the known contour family.
#    - The original centered `np.tile(motif, (2, 2))` overlay assumption is invalid:
#      the center must stay empty, so no center stamp can be drawn.
#    - A naive D9-only merge that keeps the 6x6 stamp and moves it to a corner would
#      overlap the 6x6 center hole on a 16x16 canvas. That would either break the
#      "pure black hole" requirement or clip the corner stamp asymmetrically.
#    - The fused rule therefore uses a single 3x3 maroon glyph in one corner. This
#      preserves both component intents without introducing overlap artifacts.
#
# d) Placement legality
#    - The center hole uses rows 5:11 and cols 5:11.
#    - The four legal 3x3 corner anchors are (0,0), (0,13), (13,0), (13,13).
#    - Those corner regions are fully disjoint from the center hole and always fit
#      inside the 16x16 output. Input placement is unchanged and valid for any top-left
#      row/col in 0..3.
#
# e) Visual observability
#    - Across 2-4 examples, a human can see two stable facts:
#        1. the center 6x6 is always a black void
#        2. exactly one corner contains the maroon glyph, and which corner it is varies
#           with the glyph class
#    - Because the lattice phase also changes with the same contour, the input clearly
#      acts as a code rather than a literal copy instruction.
#
# f) Visual cleanliness
#    - Layers are cleanly separated:
#        orange/black lattice across the canvas,
#        black overwrite only in the center hole,
#        maroon overwrite only in one 3x3 corner.
#    - No area carries competing overlays, so the hole and the corner stamp stay easy
#      to distinguish.
#
# g) Change inferability
#    - The output uses only black, orange, and maroon.
#    - There is no hidden dependence on numeric color ids; only the visible contour
#      class matters. The generator cycles through glyph classes so small sample sets
#      expose multiple corner outcomes and make the mapping learnable.


OUT_SHAPE = (16, 16)
CENTER_HOLE = (5, 11, 5, 11)  # half-open slice: rows 5:11, cols 5:11

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

STAMP_ANCHORS = {
    "plus": (0, 0),
    "X": (0, OUT_SHAPE[1] - 3),
    "4corners": (OUT_SHAPE[0] - 3, 0),
    "H": (OUT_SHAPE[0] - 3, OUT_SHAPE[1] - 3),
}

GLYPH_ORDER = ["plus", "H", "X", "4corners"]
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
_GLYPH_COUNTER = 0


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


def _stamp_glyph(out: np.ndarray, motif: np.ndarray, glyph_name: str) -> None:
    r0, c0 = STAMP_ANCHORS[glyph_name]
    out[r0 : r0 + 3, c0 : c0 + 3][motif] = Color.MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice with a black center void and a glyph-specific maroon corner stamp.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other visible color.
    - The non-orange cells form a tight 3x3 contour glyph.
    - The glyph family is the same four-shape codebook: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the source task.
    - Repeat that phase tile across a 16x16 black canvas to build the lattice.
    - Overwrite the centered 6x6 area with pure black, then place one maroon 3x3 copy of the glyph into the corner assigned to that glyph class.

    Key insight:
    - The input contour now controls two output attributes at once: it still determines the orange lattice phase, and it also selects which corner receives the only maroon glyph while the center stays empty.
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
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill(OUT_SHAPE, phase_colors)

    r0, r1, c0, c1 = CENTER_HOLE
    out[r0:r1, c0:c1] = Color.BLACK
    _stamp_glyph(out, motif, glyph_name)
    return out


def _choose_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_COUNTER
    offset = int(rng.integers(0, len(GLYPH_ORDER))) if _GLYPH_COUNTER == 0 else 0
    name = GLYPH_ORDER[(_GLYPH_COUNTER + offset) % len(GLYPH_ORDER)]
    _GLYPH_COUNTER += 1
    return name


def _corner_views(out: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "plus": out[0:3, 0:3],
        "X": out[0:3, 13:16],
        "4corners": out[13:16, 0:3],
        "H": out[13:16, 13:16],
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = _choose_glyph_name(rng)
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

    if not np.all(out[5:11, 5:11] == Color.BLACK):
        return None

    corner_views = _corner_views(out)
    maroon_corners = [name for name, view in corner_views.items() if np.any(view == Color.MAROON)]
    if maroon_corners != [glyph_name]:
        return None

    if not np.array_equal(corner_views[glyph_name] == Color.MAROON, motif):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 16")
