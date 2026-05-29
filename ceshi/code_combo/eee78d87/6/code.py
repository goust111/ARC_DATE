# Combo variant of eee78d87 (index 6)
# combo_spec: D5/0+D6/2
# Name: 完整瓦片裁剪+周期4大瓦片

from __future__ import annotations

# Design analysis for this combo variant
# --------------------------------------
# spec_variant:
#   input:
#     grid: 6x6
#     background: orange (7)
#     object:
#       count: exactly 1 monochrome motif
#       bbox: exactly 4x4
#       family: one of four 4x4 contour glyphs formed by expanding the original
#         3x3 contour families with row/col band widths [1, 2, 1]
#       color: any single color except black(0), orange(7), maroon(9)
#       position: top-left row/col each in [0, 2]
#   output:
#     canvas: 20x20
#     lattice: black/orange tiling of one decoded 4x4 phase tile
#     tile_count: exactly 5x5 complete tiles, with no partial fringe
#     stamp: the original 4x4 motif tiled 2x2 into a centered 8x8 maroon block
#     colors_used: black(0), orange(7), maroon(9)
#   decode:
#     1. Match the 4x4 motif against the four canonical expanded glyphs.
#     2. Collapse that family back to its original 3x3 contour meaning.
#     3. Decode the original period-3 phase rule on the coarse glyph.
#     4. Re-expand the decoded phase mask with the same [1, 2, 1] banding,
#        yielding a period-4 tile.
#     5. Fill exactly (input_size - 1) x (input_size - 1) full tiles, so
#        6x6 input -> 5x5 tiles -> 20x20 output.
#
# a) Concrete behavior trace:
#    - Base task plus glyph:
#        motif       = .#./###/.#.
#        phase tile  = ... /.OO /.OO
#        output      = 16x16, so the last row/col are partial copies of a 3x3 tile.
#    - D6-only style variant:
#        motif       = .##./####/####/.##.
#        phase tile  = .... /.OOO /.OOO /.OOO
#        output      = 16x16 = 4x4 full 4x4 tiles.
#    - This combo:
#        same 4x4 motif and same period-4 phase tile as D6/2,
#        but tile count follows the D5 tight-crop idea: use 5x5 full tiles.
#        Therefore output = 20x20, and the 8x8 maroon center stamp sits inside a
#        wider 6-cell border instead of the 4-cell border of the D6-only version.
#
# b) Complexity calibration:
#    - The contour family count stays at four. That is enough variety to make the
#      decoding non-trivial, but still learnable from a few ARC examples.
#    - The two changed dimensions remain visually independent and both observable:
#      bbox grows from 3x3 to 4x4, and output grows from 16x16 to 20x20 while
#      preserving a strict 4-cell periodic lattice.
#
# c) Implicit assumption audit against the original main():
#    - `motif.shape == (3, 3)` breaks; it must become `(4, 4)`.
#    - Direct coordinate logic inside `_decode_phase_tile()` on a 3x3 mask no longer
#      applies to the visible input motif; we must first recognize the expanded glyph.
#    - `out_shape = (3 * g.shape[0] - 2, ...)` encoded both the old period and the old
#      partial-fringe geometry. The combo replaces it with `4 * (size - 1)`, which
#      means "period 4" plus "exactly 5 complete tiles" for a 6x6 input.
#    - `np.tile(motif, (2, 2))` still works, but the centered stamp is now 8x8.
#
# d) Placement legality:
#    - A 4x4 motif inside 6x6 fits only at top-left rows/cols 0..2.
#    - There is only one object, so no overlap logic is needed beyond bounds.
#    - The generator samples only canonical motifs and validates them through `main()`.
#
# e) Visual observability:
#    - Human solvers can see the larger 4x4 contour in the input.
#    - They can also see the output side length is 20, not 16, and that it is an exact
#      multiple of the 4-cell lattice period. This makes the D5 crop change visible.
#    - Different glyph families still induce different orange/black phase tiles, so the
#      examples can support decoding instead of collapsing to one repetitive pattern.
#
# f) Visual cleanliness:
#    - The output still has only two layers: a repeated orange/black lattice plus one
#      centered maroon stamp. The larger 20x20 canvas actually increases separation.
#    - The stamp is centered on tile boundaries because 20 and 8 are both even, so the
#      maroon block does not introduce awkward off-center clipping.
#
# g) Change inferability:
#    - The input foreground color remains irrelevant; only the binary contour matters.
#    - Output colors remain restricted to black/orange/maroon, so the new rule is about
#      geometry and periodicity rather than hidden color remapping.
#
# New solver assumptions introduced here:
#   - The foreground mask must match one of the four canonical 4x4 expanded glyphs.
#   - The output tile count is derived from the fixed 6x6 input extent as `size - 1`.
#   - The generator guarantees both assumptions by construction.

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 4
PHASE_PERIOD = 4
OUTPUT_TILES = GRID_H - 1
OUTPUT_H, OUTPUT_W = PHASE_PERIOD * OUTPUT_TILES, PHASE_PERIOD * OUTPUT_TILES
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
_BANDS = (1, 2, 1)


def _expand_121(mask: np.ndarray) -> np.ndarray:
    """Expand a 3x3 boolean mask to 4x4 using row/col band widths [1,2,1]."""
    assert mask.shape == (3, 3)
    out = np.zeros((4, 4), dtype=bool)
    r0 = 0
    for r, rh in enumerate(_BANDS):
        c0 = 0
        for c, cw in enumerate(_BANDS):
            if bool(mask[r, c]):
                out[r0 : r0 + rh, c0 : c0 + cw] = True
            c0 += cw
        r0 += rh
    return out


COARSE_GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPHS = {name: _expand_121(mask) for name, mask in COARSE_GLYPHS.items()}


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


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 4x4 expanded contour glyph into a period-4 orange phase mask."""
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            coarse_phase = _decode_phase_tile_3(COARSE_GLYPHS[name])
            return _expand_121(coarse_phase)
    raise AssertionError("Unrecognized 4x4 contour glyph")


def _output_shape_for_input(shape: tuple[int, int]) -> tuple[int, int]:
    """Return the exact full-tile output size for a given input shape."""
    h, w = map(int, shape)
    assert h == GRID_H and w == GRID_W
    return PHASE_PERIOD * (h - 1), PHASE_PERIOD * (w - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 4x4 contour motif into a tightly tiled 20x20 period-4 lattice and stamp a centered maroon 2x2 copy of that motif.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels have a tight 4x4 bbox containing one expanded contour glyph.
    - The glyph is one of four canonical 4x4 forms obtained by stretching the original contour families with [1,2,1] row/col bands.

    Rule:
    - Identify which 4x4 contour family the motif belongs to, decode its coarse 3x3 meaning, and expand the corresponding phase tile into a 4x4 orange/black tile.
    - Fill the output with exactly 5x5 complete copies of that period-4 tile, producing a 20x20 canvas with no partial border tiles.
    - Tile the original 4x4 glyph 2x2 and recolor its true pixels to maroon in the centered 8x8 region.

    Key insight:
    - The colored input object is a compact code: one part of the task is still decoding which lattice phase to repeat, and the new part is noticing that the canvas is cropped to a whole-number count of those larger 4x4 tiles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill(_output_shape_for_input(g.shape), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


MOTIF_POS_MAX = GRID_H - MOTIF_SIZE + 1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
