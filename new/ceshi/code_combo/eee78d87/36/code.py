# Combo variant of eee78d87 (index 36)
# combo_spec: D6/2+D10/2
# Name: 三色格栅+周期4大瓦片

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
#       family: one of four canonical inflated contour glyphs obtained by expanding
#         the original 3x3 contour families with row/col band widths [1, 2, 1]
#       color: any single color except black(0), grey(5), orange(7), maroon(9)
#       position: top-left row/col each in [0, 2]
#   output:
#     canvas: 16x16
#     lattice: one repeated 4x4 tile using black(0), grey(5), and orange(7)
#     stamp: the original 4x4 motif tiled 2x2 into a centered 8x8 maroon block
#     colors_used: black(0), grey(5), orange(7), maroon(9)
#   decode:
#     1. Match the 4x4 motif against the four canonical inflated contour glyphs.
#     2. Collapse that family back to its coarse 3x3 contour code.
#     3. Decode the coarse code with the base task's phase logic and expand the
#        resulting 3x3 phase mask to a 4x4 period tile with the same [1,2,1] bands.
#     4. Fuse the expanded phase mask with the observed 4x4 motif mask cellwise:
#        both false -> black, exactly one true -> grey, both true -> orange.
#
# a) Concrete behavior trace:
#    - Base task plus glyph:
#        coarse motif = .#./###/.#.
#        coarse phase = ... /.## /.##
#        output lattice = black/orange period-3 repetition of that phase tile.
#    - D6-only variant plus glyph:
#        4x4 motif  = .##./####/####/.##.
#        4x4 phase  = .... /.### /.### /.###
#        output lattice = black/orange period-4 repetition.
#    - This combo keeps that 4x4 phase tile but fuses it with the 4x4 motif mask:
#        overlap cells become orange, disagreement cells become grey, joint-empty
#        cells stay black. The same motif therefore controls both the phase layout
#        and the orange/grey split inside the repeated tile.
#
# b) Complexity calibration:
#    - I avoided introducing a large arbitrary lookup table of unrelated 4x4 color
#      patterns. The only families are still the four contour glyphs humans can track.
#    - The third color is structurally derived from the relation between two visible
#      boolean layers, so examples can support it. This is richer than the D6-only
#      variant but still learnable from a handful of samples.
#
# c) Implicit assumption audit against the original main():
#    - `motif.shape == (3, 3)` no longer holds; the object must be a tight 4x4 box.
#    - The original `_decode_phase_tile()` read hard-coded 3x3 coordinates directly.
#      Here the solver must first recognize one of four canonical 4x4 glyphs, then
#      delegate the logical meaning back through the coarse 3x3 contour family.
#    - The base code assumed a binary orange/black tile. That assumption is broken by
#      D10, so the lattice builder must now emit three colors via a cellwise fusion.
#    - `out_shape = (3 * g.shape[0] - 2, ...)` encoded the old period-3 geometry.
#      The combo fixes the canvas at 16x16 and repeats a 4x4 tile instead.
#    - `np.tile(motif, (2, 2))` still applies, but the stamp region is now 8x8.
#
# d) Placement legality:
#    - A 4x4 motif fits inside 6x6 only at top-left rows/cols 0..2.
#    - There is still exactly one object, so no inter-object overlap logic is needed.
#    - The generator samples only canonical glyphs and oracle-validates with `main()`.
#
# e) Visual observability:
#    - The motif bbox visibly grows from 3x3 to 4x4, and the background repetition
#      visibly grows from period 3 to period 4.
#    - Grey is not sprinkled arbitrarily: it consistently marks cells where the 4x4
#      motif mask and the decoded 4x4 phase mask disagree, while orange marks overlap.
#    - The centered maroon stamp exposes the exact 4x4 motif geometry, giving the
#      solver a direct visual clue for how the tricolor background is organized.
#
# f) Visual cleanliness:
#    - The output still has only two layers: a repeated lattice and one centered stamp.
#    - The lattice uses just three background colors with a single deterministic fusion
#      rule, so the grid remains readable rather than becoming a patchwork of markers.
#    - A 16x16 canvas leaves a 4-cell border around the 8x8 stamp, keeping the center
#      object distinct from the periodic backdrop.
#
# g) Change inferability:
#    - Color usage is deliberately capped at black/grey/orange for the lattice and
#      maroon for the stamp. The third lattice color always comes from the same logical
#      condition (phase xor motif), not from the input foreground hue.
#    - Because the four motif families recur and the center stamp reveals the motif,
#      humans can compare multiple examples and infer the orange/grey partition rule.
#
# New solver assumptions introduced here:
#   - The foreground mask must match one of the four canonical 4x4 inflated glyphs.
#   - The generator guarantees a single foreground color and one tight 4x4 object.
#   - The tricolor lattice is defined only through the relation between that motif mask
#     and its decoded phase mask; random 4x4 shapes would not be valid inputs.

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 6, 6
OUTPUT_H, OUTPUT_W = 16, 16
MOTIF_SIZE = 4
FG_COLORS = [1, 2, 3, 4, 6, 8]
MAX_RETRIES = 300
_BANDS = (1, 2, 1)


def _expand_121(mask: np.ndarray) -> np.ndarray:
    """Expand a 3x3 mask to 4x4 using row/col band widths [1,2,1]."""
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
MOTIF_POS_MAX = GRID_H - MOTIF_SIZE + 1


def _decode_phase_tile_3(motif: np.ndarray) -> np.ndarray:
    """Decode the original 3x3 contour glyph into the original phase mask."""
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


def _match_glyph_name(motif: np.ndarray) -> str:
    """Return the canonical glyph family name for a 4x4 inflated contour motif."""
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)
    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    raise AssertionError("Unrecognized 4x4 contour glyph")


def _decode_phase_mask(motif: np.ndarray) -> np.ndarray:
    """Decode a 4x4 inflated contour glyph into its period-4 boolean phase mask."""
    name = _match_glyph_name(motif)
    coarse_phase = _decode_phase_tile_3(COARSE_GLYPHS[name])
    return _expand_121(coarse_phase)


def _compose_tricolor_tile(motif: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Fuse motif and phase masks into a 4x4 black/grey/orange tile."""
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)
    assert phase.shape == (MOTIF_SIZE, MOTIF_SIZE)

    tile = np.full((MOTIF_SIZE, MOTIF_SIZE), Color.BLACK, dtype=int)
    tile[np.logical_xor(motif, phase)] = Color.GREY
    tile[np.logical_and(motif, phase)] = Color.ORANGE
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a 4x4 contour code, expand it into a period-4 tricolor lattice, and stamp a centered maroon 2x2 copy of the motif.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels form a tight 4x4 bounding box matching one of four canonical inflated contour glyphs.
    - The foreground hue is only an occupancy marker; the 4x4 shape is the meaningful code.

    Rule:
    - Recognize the 4x4 contour family, collapse it to the corresponding coarse 3x3 glyph, decode the original phase logic, and expand that phase back to a 4x4 mask.
    - Fuse the 4x4 motif mask with the 4x4 phase mask cellwise: overlap becomes orange, disagreement becomes grey, and joint-empty cells remain black.
    - Repeat that 4x4 tricolor tile over a 16x16 canvas, then place a 2x2 maroon tiling of the original motif in the centered 8x8 region.

    Key insight:
    - The motif now controls two linked structures at once: the hidden phase tile inherited from the base task and the visible 4x4 occupancy mask. The third lattice color appears exactly where those two structures disagree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_mask(motif)
    tile = _compose_tricolor_tile(motif, phase)
    out = tile_fill((OUTPUT_H, OUTPUT_W), tile)

    center = np.tile(motif, (2, 2))
    r0 = (OUTPUT_H - center.shape[0]) // 2
    c0 = (OUTPUT_W - center.shape[1]) // 2
    center_view = out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]]
    center_view[center] = Color.MAROON
    return out


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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 36")
