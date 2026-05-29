# Combo variant of eee78d87 (index 32)
# combo_spec: D6/1+D12/1
# Name: 行列不同周期+非正方形输入

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Seed task:
#      1. Read one 3x3 contour glyph from a fixed 6x6 orange square.
#      2. Decode it into one 3x3 orange/black phase tile.
#      3. Repeat that tile on a fixed 16x16 square, then overlay one centered 6x6
#         maroon 2x2 stamp of the original glyph.
#    - Combo task:
#      1. Input is rectangular, not square: examples such as 5x7, 6x8, 7x9.
#      2. The glyph is still exactly 3x3 and still comes from the original four
#         contour families.
#      3. First decode the original 3x3 phase tile exactly as before.
#      4. Then derive different row/column periods from the glyph:
#         middle-row side arms present  -> (vertical, horizontal) = (2, 3)
#         middle-row side arms absent   -> (vertical, horizontal) = (3, 2)
#      5. Convert the original 3x3 phase tile into that rectangular period by
#         keeping the bottom two rows when the vertical period is 2 and keeping
#         the left two columns when the horizontal period is 2.
#      6. Output size becomes rectangular as well:
#           out_h = vertical_period * input_h - (vertical_period - 1)
#           out_w = horizontal_period * input_w - (horizontal_period - 1)
#      7. The 6x6 maroon center stamp is still a 2x2 tiling of the input glyph.
#    - Example trace:
#      H on a 6x8 input used to yield a 16x16 period-3 square lattice.
#      In this combo variant, H has both side arms, so it uses period (2,3),
#      the phase tile becomes the bottom two rows of the original H tile, and
#      the output becomes 11x22 before the centered 6x6 maroon H-stamp is added.
#
# b) Complexity calibration
#    - The solver must now separate three things by axis: input height, input width,
#      and which of the two unequal period orientations the glyph selects.
#    - I kept only two period pairs, (2,3) and (3,2), so the rule stays learnable
#      from a few ARC examples instead of becoming an arbitrary lookup table.
#    - The original four glyph families are preserved so the task does not collapse
#      into a trivial "read the canvas size only" variant.
#
# c) Implicit-assumption audit of the original main()
#    - `assert g.shape == (6, 6)` breaks: the combo requires rectangular inputs.
#    - `out_shape = (3*h-2, 3*w-2)` breaks: each axis now needs its own period.
#    - `_decode_phase_tile(motif)` returning a square 3x3 tile is no longer enough;
#      we now need a second step that anisotropically reduces the coarse phase tile.
#    - Center stamping still works because it depends only on the final output shape
#      and the fixed 6x6 2x2 motif copy.
#
# d) Placement legality
#    - The generator chooses only rectangles with both sides at least 4, and only
#      from a pool whose two possible anisotropic output sizes are both rectangular.
#      This keeps the motif legal and prevents accidental regression to square output.
#    - There is still exactly one foreground object, so no overlap logic is needed.
#    - The motif top-left is sampled from legal ranges `0..h-3` and `0..w-3`.
#
# e) Visual observability
#    - Rectangular input and rectangular output are directly visible.
#    - The lattice is also visibly anisotropic: some samples repeat every 2 rows and
#      3 columns, others every 3 rows and 2 columns.
#    - Because the generator cycles through all four glyphs and multiple rectangles,
#      small example sets are unlikely to hide one axis or regress to the seed task.
#
# f) Visual cleanliness
#    - The output still has only two layers: one orange/black lattice and one centered
#      maroon stamp. No extra markers are introduced.
#    - Using only periods 2 and 3 keeps the lattice bold enough to read on medium-size
#      rectangular canvases.
#
# g) Color-rule audit
#    - Input foreground color remains a pure carrier and has no semantic effect.
#    - Output still uses only black, orange, and maroon.
#
# New solver assumptions that the generator must satisfy
#    - The foreground mask must be exactly one of the four known 3x3 contour glyphs.
#    - Inputs must be rectangular (`h != w`) and both sides must be at least 4.
#    - The variant further assumes the decoded output shape is rectangular as well;
#      the generator guarantees this by using only vetted input aspect ratios.
#    - The period branch assumes the motif's middle row distinguishes the two
#      anisotropic period orientations; the generator guarantees valid glyphs only.


def _decode_phase_tile_3(motif: np.ndarray) -> np.ndarray:
    """Decode the original 3x3 contour glyph into the original period-3 phase mask."""
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


def _decode_rectangular_phase(motif: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Decode the glyph into unequal row/column periods and the corresponding phase tile."""
    base_phase = _decode_phase_tile_3(motif)
    has_side_arms = bool(motif[1, 0]) and bool(motif[1, 2])

    if has_side_arms:
        vertical_period, horizontal_period = 2, 3
        row_idx = np.array([1, 2], dtype=int)
        col_idx = np.array([0, 1, 2], dtype=int)
    else:
        vertical_period, horizontal_period = 3, 2
        row_idx = np.array([0, 1, 2], dtype=int)
        col_idx = np.array([0, 1], dtype=int)

    phase = base_phase[np.ix_(row_idx, col_idx)]
    assert phase.shape == (vertical_period, horizontal_period)
    return vertical_period, horizontal_period, phase


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a rectangular anisotropic lattice, then stamp the glyph in maroon at the output center.

    Input structure:
    - The grid is rectangular rather than square, with orange background and exactly one non-orange color.
    - The non-orange pixels form one tight 3x3 contour glyph.
    - The glyph is still one of the original four families: plus, H, X, or four corners.

    Rule:
    - Decode the glyph into the original 3x3 orange/black phase tile.
    - Use the glyph's middle-row structure to choose unequal axis periods: side arms present gives period (2,3), side arms absent gives period (3,2).
    - Convert the coarse 3x3 phase tile into that rectangular period, tile it across an output whose height and width each use their own period formula, and then place a centered 6x6 maroon 2x2 stamp of the original glyph.

    Key insight:
    - The glyph now controls the lattice separately by axis: one visible feature picks which axis gets period 2 versus 3, while the rectangular input size separately determines the final canvas extent in height and width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 4
    assert h != w

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)

    vertical_period, horizontal_period, phase = _decode_rectangular_phase(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (
        vertical_period * h - (vertical_period - 1),
        horizontal_period * w - (horizontal_period - 1),
    )
    assert out_shape[0] != out_shape[1]
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
INPUT_SHAPES = (
    (4, 6),
    (5, 8),
    (6, 4),
    (6, 8),
    (8, 5),
    (9, 7),
)
MOTIF_SIZE = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
_GLYPH_ORDER = tuple(GLYPHS.keys())
_SHAPE_ORDER = INPUT_SHAPES
_GEN_RNG = np.random.default_rng()
_GLYPH_QUEUE: list[str] = []
_SHAPE_QUEUE: list[tuple[int, int]] = []


def _next_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_QUEUE
    if not _GLYPH_QUEUE:
        _GLYPH_QUEUE = list(rng.permutation(_GLYPH_ORDER))
    return _GLYPH_QUEUE.pop()


def _next_input_shape(rng: np.random.Generator) -> tuple[int, int]:
    global _SHAPE_QUEUE
    if not _SHAPE_QUEUE:
        perm = rng.permutation(len(_SHAPE_ORDER))
        _SHAPE_QUEUE = [_SHAPE_ORDER[int(i)] for i in perm]
    return _SHAPE_QUEUE.pop()


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h, w = _next_input_shape(rng)
    motif = GLYPHS[_next_glyph_name(rng)]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, h - MOTIF_SIZE + 1))
    c0 = int(rng.integers(0, w - MOTIF_SIZE + 1))

    inp = np.full((h, w), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(_GEN_RNG)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for combo variant eee78d87/32")
