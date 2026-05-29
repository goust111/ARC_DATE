# Variant of eee78d87
# Dimension: D2 — 源对象的形状集合 (src)
# Change: 4种固定的3x3轮廓字形：plus(.#./#11/.#.)、H(#.#/###/#.#)、X(#.#/.#./#.#)... → 字形从网格边缘色条中读取

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the D2 variant
# a) Concretization and trace:
#    - The input stays 6x6 with orange background, but the glyph is no longer a 3x3
#      island inside the field. Instead, the outer ring is a legend:
#      top edge (0,1:4)   = glyph top row
#      left edge (1:4,0)  = glyph left column
#      right edge (1:4,5) = glyph right column
#      bottom edge (5,1:4)= glyph bottom row
#      cell (0,5)         = glyph center
#    - Example: for plus(010/111/010), the old task placed those five pixels as a tight
#      3x3 object. The new task shows 010 on the top edge, 010 on the left edge, 010 on
#      the right edge, 010 on the bottom edge, and a single center marker at (0,5).
#      The solver must first reassemble the 3x3 motif from that legend, then continue
#      with the same phase decode and centered maroon stamp.
# b) Complexity calibration:
#    - The four glyphs are still the same semantic cases, so the downstream phase rule is
#      unchanged. The visible innovation is that the source "shape family" is now border
#      legends rather than interior 3x3 blobs.
#    - The legend is structural, not an arbitrary category code: every edge bar matches a
#      side of the reconstructed glyph, and the top-right marker supplies the center bit.
# c) Implicit assumption audit against the original main():
#    - `bbox(fg)` and `crop_bbox(...).shape == (3,3)` are no longer valid because the
#      foreground spans multiple border strips.
#    - `len(colors_present(...)) == 1` remains valid; a single non-orange legend color is
#      still enough because the information is carried by positions, not color ids.
#    - The phase decoder still assumes one of the four contour glyphs. The new parser
#      explicitly reconstructs a 3x3 boolean motif before calling it.
# d) Placement legality:
#    - The generator uses only reserved border cells, so nothing overlaps and nothing can
#      go out of bounds. The interior 4x4 area remains empty, making the variant visually
#      distinct from the original embedded-motif task.
# e) Visual observability:
#    - Humans can compare the edge bars with the maroon 6x6 center stamp in the output and
#      see that the border legend is being "read" into a motif, then decoded into the
#      orange/black lattice. This is inferable from a few examples without any hidden
#      dependence on color ids.
# f) Visual cleanliness:
#    - The border legend and the output stamp occupy different roles and different regions.
#      No layered clutter is introduced in the input.
# g) Color-change inferability:
#    - Input still uses orange + one foreground color only. Output still uses black,
#      orange, and maroon only. Colors are role-based and unchanged from the original
#      downstream rule.
# h) Variant metric to check against degeneration:
#    - Interior non-orange count should be exactly 0 for generated inputs, while border
#      non-orange count varies by glyph (plus=5, X=9, 4corners=8, H=11 under this legend).
#      This confirms the source object really moved to the edge legend.
# i) New main() assumptions and generator guarantees:
#    - Assumption: the four edge bars are the reserved legend carriers.
#      Guarantee: generator writes only into those cells.
#    - Assumption: duplicated corner facts are consistent, e.g. top-left of the glyph is
#      encoded by both the top bar and the left bar.
#      Guarantee: generator derives every bar from the same motif array.
#    - Assumption: the reconstructed motif is one of the supported contour glyphs.
#      Guarantee: generator samples only from that fixed set and oracle-validates with main().


GRID_H, GRID_W = 6, 6
OUT_H, OUT_W = 16, 16
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

TOP_SLICE = (0, slice(1, 4))
LEFT_SLICE = (slice(1, 4), 0)
RIGHT_SLICE = (slice(1, 4), 5)
BOTTOM_SLICE = (5, slice(1, 4))
CENTER_POS = (0, 5)

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
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


def _read_legend_motif(g: np.ndarray, bg: int) -> np.ndarray:
    top = g[TOP_SLICE] != bg
    left = g[LEFT_SLICE] != bg
    right = g[RIGHT_SLICE] != bg
    bottom = g[BOTTOM_SLICE] != bg
    center = bool(g[CENTER_POS] != bg)

    # These pairs redundantly encode the same glyph corners.
    assert bool(top[0]) == bool(left[0])
    assert bool(top[2]) == bool(right[0])
    assert bool(bottom[0]) == bool(left[2])
    assert bool(bottom[2]) == bool(right[2])

    motif = np.array(
        [
            [top[0], top[1], top[2]],
            [left[1], center, right[1]],
            [bottom[0], bottom[1], bottom[2]],
        ],
        dtype=bool,
    )

    assert any(np.array_equal(motif, glyph) for glyph in GLYPHS.values())
    return motif


def _stamp_legend(inp: np.ndarray, motif: np.ndarray, fg: int) -> None:
    inp[TOP_SLICE][motif[0]] = fg
    inp[LEFT_SLICE][motif[:, 0]] = fg
    inp[RIGHT_SLICE][motif[:, 2]] = fg
    inp[BOTTOM_SLICE][motif[2]] = fg
    if bool(motif[1, 1]):
        inp[CENTER_POS] = fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a border legend that encodes a contour glyph, then expand it into the same 16x16 phase lattice and centered maroon stamp as the original task family.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange legend color.
    - The glyph is not embedded as a 3x3 object; instead its top row, left column, right column, bottom row, and center bit are stored on reserved outer-ring cells.
    - Those legend strips reconstruct one of the four contour glyphs: plus, H, X, or four corners.

    Rule:
    - Reassemble the 3x3 motif from the edge legend.
    - Decode that motif into the same period-3 orange phase tile as in the original task.
    - Tile the phase over a black 16x16 canvas, then stamp a 2x2 maroon tiling of the reconstructed glyph into the centered 6x6 region.

    Key insight:
    - The input foreground no longer marks the glyph directly inside the field; it is a border template that must be read first, and only the reconstructed motif controls both the lattice phase and the central maroon stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    motif = _read_legend_motif(g, bg)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((OUT_H, OUT_W), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
    fg = int(rng.choice(FG_COLORS))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    _stamp_legend(inp, motif, fg)

    # Variant guard: the motif must live on the border only.
    assert not bool(np.any(inp[1:5, 1:5] != Color.ORANGE))

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
    raise RuntimeError("Failed to generate input for eee78d87 D2 variant")
