# Combo variant of eee78d87 (index 42)
# combo_spec: D14/1+D15/1
# Name: 颜色编码格栅线宽+加粗格栅线

from __future__ import annotations

import numpy as np

from solvers.arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task decodes a 3x3 contour glyph into a 3x3 orange/black phase tile,
#    where the first phase row/col act like 1-pixel black grid lines and the remaining
#    two phase rows/cols form the orange fill area. The fusion keeps the glyph->phase
#    decoding unchanged, but replaces the row/col band sizes [1,1,1] with [w,1,1].
#    For blue motifs w=2, so the lattice period becomes 4; for red motifs w=3, the
#    period becomes 5. Example: plus -> original phase .../.OO/.OO. After expansion:
#    blue gives a 4x4 tile with a 2-pixel black top band and left band; red gives the
#    same structure with 3-pixel bands. The center 6x6 maroon stamp stays unchanged,
#    so the visible difference is entirely in the surrounding lattice thickness.
# b) Complexity calibration:
#    Only two width levels are used (2 and 3). That is enough to make the fusion
#    visually obvious, while avoiding an over-fragmented color->width codebook.
#    The generator uses only blue and red so a human can infer the mapping from hue,
#    not from hidden numeric color ids.
# c) Hidden assumptions review:
#    `_decode_phase_tile` still assumes the foreground mask is exactly one of the four
#    original 3x3 contour glyphs; the generator still guarantees that. `main()` now
#    adds a new assumption that the lone foreground color is one of the supported
#    semantic colors in `FG_TO_LINE_WIDTH`; the generator enforces this explicitly.
#    The old assumption that a 3x3 phase tile can be tiled directly no longer holds,
#    so the code expands the phase tile before `tile_fill`.
# d) Placement legality:
#    The input still contains exactly one 3x3 motif placed at a top-left in [0,3]^2,
#    so it cannot go out of bounds or overlap anything else. The center stamp remains
#    a 6x6 overlay placed inside the fixed 16x16 output canvas.
# e) Visual observability:
#    The rule is visible because blue inputs always yield a period-4 lattice with
#    2-pixel black bands, while red inputs yield a period-5 lattice with 3-pixel
#    bands. No invisible ordering or parity of color ids is used.
# f) Visual cleanliness:
#    The lattice is still a single repeated background pattern plus one maroon center
#    stamp, so regions remain readable. There is no extra object overlap beyond the
#    original central overwrite.
# g) Learnability of the color change:
#    Color does not recolor the output directly; it only toggles a single geometric
#    parameter. Restricting the generator to two canonical colors keeps the mapping
#    compact enough to infer from a few examples.


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the original 3x3 orange phase mask."""
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


def _expand_phase_tile(phase: np.ndarray, line_width: int) -> np.ndarray:
    """
    Expand a 3x3 phase tile by thickening the first row/column band.

    The original lattice uses row/col band sizes [1,1,1]. This fused variant uses
    [w,1,1], where w is determined by the foreground color and already includes the
    D15 thickening baseline.
    """
    assert phase.shape == (3, 3)
    assert line_width in (2, 3)

    band_sizes = [line_width, 1, 1]
    out = np.zeros((line_width + 2, line_width + 2), dtype=bool)

    r0 = 0
    for pr, rh in enumerate(band_sizes):
        c0 = 0
        for pc, cw in enumerate(band_sizes):
            if bool(phase[pr, pc]):
                out[r0 : r0 + rh, c0 : c0 + cw] = True
            c0 += cw
        r0 += rh
    return out


FG_TO_LINE_WIDTH = {
    Color.BLUE: 2,
    Color.RED: 3,
}

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

FG_COLORS = [Color.BLUE, Color.RED]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose black grid bands are thickened and color-modulated.

    Input structure:
    - The grid is 6x6 with orange background and exactly one semantic foreground color.
    - The non-orange pixels occupy a tight 3x3 bounding box containing one of the four original contour glyphs.
    - The foreground hue is part of the code: blue selects the baseline thick grid, red selects an even thicker one.

    Rule:
    - Read the 3x3 foreground mask exactly as in the base task to recover the original 3x3 phase tile.
    - Convert that phase tile into a thicker lattice tile by expanding the first row/column band from width 1 to width 2 or 3 according to the foreground color.
    - Fill the 16x16 output by tiling this expanded orange/black lattice, then stamp a 2x2 repetition of the input glyph in maroon inside the centered 6x6 region.

    Key insight:
    - The glyph still determines the phase of the lattice, but the input color is no longer decorative: it controls how thick the repeated black grid bands are around the same decoded phase pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    cols = colors_present(g, exclude={bg})
    assert len(cols) == 1
    fg_color = int(cols[0])
    assert fg_color in FG_TO_LINE_WIDTH

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    expanded_phase = _expand_phase_tile(phase, FG_TO_LINE_WIDTH[fg_color])
    phase_colors = np.where(expanded_phase, bg, Color.BLACK).astype(int)

    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 42")
