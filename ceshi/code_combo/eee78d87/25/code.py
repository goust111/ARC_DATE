# Combo variant of eee78d87 (index 25)
# combo_spec: D6/0+D15/2
# Name: 双模式交替瓦片+可变周期

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Input remains the seed task's 6x6 orange field with exactly one 3x3 contour glyph in one non-orange color.
#    - Output remains a 16x16 black/orange lattice plus the centered 6x6 maroon 2x2 stamp of the original glyph.
#    - The fused change applies at the lattice stage:
#      1. glyph type chooses a period p in {2, 4, 5, 6}
#      2. the same glyph chooses an asymmetric base tile A of shape p x p
#      3. B is forced to be A.T
#      4. tile rows alternate A, B, A, B...
#    - Example trace with X:
#      seed task    : decode a fixed 3x3 phase tile and repeat that same tile everywhere.
#      D6 alone     : X changes the tile size to 5x5, but every tile row still uses the same phase.
#      D15 alone    : tile rows alternate A/B, but the period would still be fixed at 3.
#      combo here   : X produces a 5x5 asymmetric hook; rows 0..4 use A, rows 5..9 use A.T, rows 10..14 use A,
#                     and the final partial band also follows the same parity. The maroon center stamp is unchanged.
#    - Important conflict resolution: using the seed task's plus-phase would make A == A.T, so the D15 half would
#      disappear on plus examples. The combo therefore uses four asymmetric phase templates, including plus.
#
# b) Complexity calibration
#    - Four periods are enough to make density differences obvious while keeping the mapping learnable.
#    - Reusing one tile family per glyph keeps the rule small: the solver only needs to infer "which glyph class"
#      and then apply both consequences of that class.
#    - Avoiding a symmetric tile for any glyph prevents accidental collapse back to a single-mode lattice.
#
# c) Implicit-assumption audit against the seed main()
#    - `motif.shape == (3, 3)` is still valid; the encoded object is unchanged.
#    - The seed `_decode_phase_tile()` assumed a fixed 3x3 result. That assumption no longer holds because D6 makes
#      the period glyph-dependent, so decoding must now return both glyph name and period-sized tile.
#    - The seed `tile_fill(out_shape, phase_colors)` assumed one global tile for the whole canvas. D15 breaks that:
#      the fill must alternate A and A.T by tile-row band.
#    - The output size formula and the centered 6x6 maroon stamp remain valid because they depend only on the fixed
#      16x16 canvas and the unchanged 2x2 tiling of the 3x3 motif.
#
# d) Placement legality
#    - The generator still places exactly one 3x3 glyph at top-left coordinates in 0..3, so the bbox is always 3x3
#      and never crosses the 6x6 boundary.
#    - There are no multiple input objects, so overlap is impossible.
#    - In the output, the center stamp overwrites only colors, not geometry checks; the background lattice is fully
#      determined before stamping.
#
# e) Visual observability
#    - A human can see two simultaneous regularities:
#      1. the lattice spacing changes with the glyph class
#      2. neighboring tile rows visibly swap between a pattern and its transpose
#    - The generator cycles through the four glyph classes, so a small sample set exposes all four period choices.
#    - No hidden numeric property such as the color id or input position affects the rule.
#
# f) Visual cleanliness
#    - The output still has only two layers: background lattice and central maroon stamp.
#    - Alternation happens in full horizontal bands, so the pattern is structured rather than noisy even when a
#      partial final band appears at the bottom of the 16x16 canvas.
#
# g) Color-rule audit / inference
#    - Input foreground color still serves only to mark the glyph; its numeric value is ignored.
#    - Output still uses only black, orange, and maroon, so the new reasoning burden is spatial, not chromatic.
#
# Post-change input assumptions to re-check
#    - The solver assumes the non-orange foreground is exactly one of the four known contour glyphs.
#    - The generator guarantees this by sampling only from `GLYPHS`.
#    - The solver also assumes the decoded tile is asymmetric so that A.T is visibly different; the custom phase
#      templates below enforce that for all four glyph classes.

PERIOD_BY_GLYPH = {
    "plus": 2,
    "H": 4,
    "X": 5,
    "4corners": 6,
}

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
_GLYPH_ORDER = tuple(GLYPHS.keys())
_GEN_RNG = np.random.default_rng()
_GLYPH_QUEUE: list[str] = []


def _classify_glyph(motif: np.ndarray) -> str:
    """Classify the 3x3 contour glyph into one of the four legal motif names."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edge_centers = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)

    if not bool(corners.any()):
        assert bool(motif[1, 1])
        assert bool(edge_centers.all())
        return "plus"

    assert bool(corners.all())
    assert not bool(motif[0, 1])
    assert not bool(motif[2, 1])
    assert bool(motif[1, 0]) == bool(motif[1, 2])

    if bool(motif[1, 0]):
        assert bool(motif[1, 1])
        return "H"
    if bool(motif[1, 1]):
        return "X"
    return "4corners"


def _decode_period_and_phase_tile(motif: np.ndarray) -> tuple[str, int, np.ndarray]:
    """
    Decode the contour glyph into its glyph class, period, and asymmetric base phase tile A.

    The phase templates are intentionally asymmetric so that A.T is always a visibly different companion tile B.
    """
    glyph = _classify_glyph(motif)
    period = PERIOD_BY_GLYPH[glyph]
    phase = np.zeros((period, period), dtype=bool)

    if glyph == "plus":
        phase[-1, :] = True
    elif glyph == "H":
        phase[1:, 0] = True
    elif glyph == "X":
        phase[1:, 0] = True
        phase[0, 2:] = True
    else:
        phase[2:, 0] = True
        phase[0, 1:] = True

    assert not np.array_equal(phase, phase.T)
    return glyph, period, phase


def _fill_alternating_row_bands(shape: tuple[int, int], tile_a: np.ndarray, tile_b: np.ndarray) -> np.ndarray:
    """Repeat tile rows across the canvas, alternating A then B=A.T by tile-row band."""
    h, w = map(int, shape)
    th, tw = tile_a.shape
    assert tile_a.shape == tile_b.shape
    assert th == tw

    out = np.full((h, w), Color.BLACK, dtype=int)
    for band_index, band_r0 in enumerate(range(0, h, th)):
        tile = tile_a if band_index % 2 == 0 else tile_b
        hh = min(th, h - band_r0)
        for c0 in range(0, w, tw):
            ww = min(tw, w - c0)
            out[band_r0 : band_r0 + hh, c0 : c0 + ww] = tile[:hh, :ww]
    return out


def _alternating_lattice_from_motif(motif: np.ndarray, bg: int, out_shape: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Build the black/orange alternating lattice for the given motif."""
    _glyph, period, phase_a = _decode_period_and_phase_tile(motif)
    phase_b = phase_a.T
    tile_a = np.where(phase_a, bg, Color.BLACK).astype(int)
    tile_b = np.where(phase_b, bg, Color.BLACK).astype(int)
    return period, _fill_alternating_row_bands(out_shape, tile_a, tile_b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a variable-period alternating lattice, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange cells form a single contour glyph inside a tight 3x3 bounding box.
    - The glyph is one of four legal code shapes: plus, H, X, or four corners.

    Rule:
    - Identify which glyph class the 3x3 contour belongs to.
    - Use that glyph class to choose both a lattice period and an asymmetric base phase tile A.
    - Form B by transposing A, then fill the fixed 16x16 output so odd tile rows use A and even tile rows use B.
    - Tile the original 3x3 glyph 2x2 and recolor its true cells to maroon in the centered 6x6 region.

    Key insight:
    - The input motif now encodes two coupled pieces of structure at once: the repeat spacing and the alternating
      pair of background tiles. Solving the task requires combining the glyph-to-period mapping with the A/A.T band rule.
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

    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    _period, out = _alternating_lattice_from_motif(motif, bg, out_shape)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _next_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_QUEUE
    if not _GLYPH_QUEUE:
        _GLYPH_QUEUE = list(rng.permutation(_GLYPH_ORDER))
    return _GLYPH_QUEUE.pop()


def _combo_behavior_is_visible(motif: np.ndarray) -> bool:
    """Guard against accidental regression to a single-tile lattice."""
    glyph, period, phase_a = _decode_period_and_phase_tile(motif)
    assert period == PERIOD_BY_GLYPH[glyph]

    out_shape = (3 * GRID_H - 2, 3 * GRID_W - 2)
    phase_b = phase_a.T
    if np.array_equal(phase_a, phase_b):
        return False

    tile_a = np.where(phase_a, Color.ORANGE, Color.BLACK).astype(int)
    alt = _fill_alternating_row_bands(out_shape, tile_a, np.where(phase_b, Color.ORANGE, Color.BLACK).astype(int))
    mono = tile_fill(out_shape, tile_a)
    return not np.array_equal(alt, mono)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[_next_glyph_name(rng)]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
        assert out.shape == (16, 16)
        assert _combo_behavior_is_visible(motif)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(_GEN_RNG)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 25")
