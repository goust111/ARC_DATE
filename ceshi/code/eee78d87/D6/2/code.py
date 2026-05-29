# Variant of eee78d87
# Dimension: D6 — 铺贴周期 (meta)
# Change: 固定周期3（3x3相位瓦片），每个瓦片单元为3x3像素... → 周期4的4x4相位瓦片

from __future__ import annotations

# Design analysis for this D6 variant
# -----------------------------------
# spec_variant:
#   input:
#     grid: 6x6
#     background: orange (7)
#     object:
#       count: exactly 1 monochrome motif
#       bbox: exactly 4x4
#       family: one of four 4x4 contour glyphs obtained by inflating the original
#         3x3 coarse glyphs with row/col band widths [1, 2, 1]
#       color: any single color except black(0), orange(7), maroon(9)
#       position: top-left row/col each in [0, 2]
#   output:
#     canvas: 16x16
#     lattice: black/orange tiling of one decoded 4x4 phase tile
#     stamp: the original 4x4 motif tiled 2x2 into an 8x8 maroon center block
#     colors_used: black(0), orange(7), maroon(9)
#   decode:
#     1. Match the 4x4 motif against the four canonical inflated contour glyphs.
#     2. Collapse the matched family back to its original 3x3 coarse contour code.
#     3. Decode that coarse code with the original task's logic.
#     4. Inflate the resulting 3x3 phase tile with the same [1, 2, 1] banding,
#        producing a period-4 tile.
#
# a) Concrete behavior trace:
#    - Original plus glyph:
#        input motif  = .#./###/.#.
#        phase tile   = ... /.OO /.OO
#        output       = repeat that 3x3 tile, then stamp a 6x6 maroon 2x2 copy.
#    - Variant plus glyph:
#        input motif  = .##./####/####/.##.  (the coarse 3x3 plus expanded by [1,2,1])
#        phase tile   = .... /.OOO /.OOO /.OOO
#        output       = repeat that 4x4 tile on 16x16, then stamp an 8x8 maroon 2x2 copy.
#    - The changed stage is the glyph decode + lattice period. Background color logic,
#      single-object extraction, and centered stamping remain the same in spirit.
#
# b) Complexity calibration:
#    - I deliberately did not introduce an arbitrary larger lookup table of unrelated
#      glyphs, because ARC examples would not support learning that from 2-4 pairs.
#    - The motif surface still expands from 3x3 to 4x4, and the solver must now read
#      a larger contour code and produce a period-4 lattice. The four family branches
#      remain visually distinct and repeatedly supported across samples.
#
# c) Implicit assumption audit against the original main():
#    - `motif.shape == (3, 3)` is no longer valid; it must become `(4, 4)`.
#    - `_decode_phase_tile()` previously inspected exact 3x3 coordinates. In the
#      variant that direct indexing would be wrong, so the code now matches a 4x4
#      canonical glyph and delegates the actual logical decode to the old 3x3 rule.
#    - `out_shape = (3 * g.shape[0] - 2, ...)` relied on period 3. Keeping that
#      formula would accidentally change the whole canvas geometry for reasons not
#      tied to D6. The variant instead fixes the canvas at 16x16 and changes only
#      the repeated tile period and the central stamp size.
#    - `np.tile(motif, (2, 2))` still holds, but the stamped region becomes 8x8.
#
# d) Placement legality:
#    - A 4x4 motif inside a 6x6 grid fits only at top-left rows/cols 0..2.
#    - There is still exactly one object, so no overlap checks beyond bounds are needed.
#    - The generator validates each sample with `main()`, so only legal motifs survive.
#
# e) Visual observability:
#    - The input object's bbox is visibly larger (4x4 instead of 3x3).
#    - The background lattice period is visibly 4: each branch repeats every 4 cells,
#      not every 3, and the center stamp is correspondingly larger at 8x8.
#    - The rule remains inferable because the four contour families are the same four
#      familiar cases, only stretched into a larger code surface.
#
# f) Visual cleanliness:
#    - The output keeps exactly two layers: a black/orange lattice and one centered
#      maroon stamp. No extra markers or overlapping annotation channels are added.
#    - A 16x16 canvas leaves a clean 4-cell border around the 8x8 center stamp.
#
# g) Color inference:
#    - The input foreground color remains irrelevant except for marking the motif.
#      Output colors stay restricted to black/orange/maroon, exactly as in the base task.
#
# New solver assumptions introduced here:
#   - The foreground mask must match one of the four canonical 4x4 inflated glyphs.
#   - The generator guarantees this by sampling only from that dictionary.

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 4
OUTPUT_H, OUTPUT_W = 16, 16
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
    """Decode the 4x4 inflated contour glyph into a period-4 orange phase mask."""
    assert motif.shape == (4, 4)

    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            coarse_phase = _decode_phase_tile_3(COARSE_GLYPHS[name])
            return _expand_121(coarse_phase)
    raise AssertionError("Unrecognized 4x4 contour glyph")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 4x4 contour motif into a 16x16 period-4 lattice and stamp a centered maroon 2x2 copy of that motif.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels have a tight 4x4 bbox containing one expanded contour glyph.
    - The glyph is one of four canonical 4x4 forms obtained by stretching the original contour families with [1,2,1] row/col bands.

    Rule:
    - Identify which 4x4 contour family the motif belongs to, decode its coarse 3x3 meaning, and expand the corresponding phase tile into a 4x4 orange/black tile.
    - Fill a black 16x16 canvas by repeating that period-4 phase tile.
    - Tile the original 4x4 glyph 2x2 and recolor its true pixels to maroon in the centered 8x8 region.

    Key insight:
    - The colored object is still a small code rather than something copied from its input position; the variant only stretches that code and its decoded lattice from period 3 to period 4.
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
    out = tile_fill((OUTPUT_H, OUTPUT_W), phase_colors)

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
    raise RuntimeError("Failed to generate input for eee78d87 D6 variant")
