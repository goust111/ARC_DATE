# Variant of eee78d87
# Dimension: D3 — 目标画布的结构 (dst)
# Change: 16x16黑色画布用周期3的橙黑相位瓦片铺满，中心(5,5)-(10,10)区域放置2x2平铺的栗色字形印章... → 多个印章区域等间距分布

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, paste, tile_fill


# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   preserved:
#     - input is still a 6x6 orange grid with exactly one 3x3 contour glyph
#     - the glyph family is unchanged: plus / H / X / 4corners
#     - the glyph still decodes a 3x3 orange/black phase tile
#     - the output canvas is still 16x16 because it is derived as 3*6-2
#   changed:
#     - the old output had one 6x6 maroon stamp at the center
#     - the new output stamps the same 6x6 2x2-tiled motif into a regular 2x2 lattice of stamp slots
#     - for a 16x16 canvas and a 6x6 stamp, the clean symmetric slot origins are
#       rows [1, 9] and cols [1, 9], yielding four stamps at:
#       (1,1), (1,9), (9,1), (9,9)
#     - this leaves a 1-cell outer border and a 2-cell gutter between stamp blocks
# behavior_trace:
#   example_glyph: plus = [[0,1,0],[1,1,1],[0,1,0]]
#   original_flow:
#     - decode plus -> phase tile with orange at rows/cols [1:,1:]
#     - tile that phase across 16x16
#     - place one 6x6 maroon plus-stamp at center top-left (5,5)
#   variant_flow:
#     - decode plus exactly the same way
#     - tile the same 16x16 phase lattice
#     - place the same 6x6 maroon plus-stamp four times at the quadrant-style slot origins
#   visible_difference:
#     - the phase background is unchanged
#     - the maroon foreground is no longer a single central event; it forms a regular 2x2 arrangement
# complexity_calibration:
#   - this is visibly different from the source task in every valid sample
#   - the new rule adds only one discrete degree of freedom: repeated slot placement
#   - it does not introduce hidden indexing by color id or ambiguous per-sample branching
# implicit_assumption_review:
#   - get_bg / colors_present assumptions remain valid because input format is unchanged
#   - bbox/crop_bbox still rely on one tight 3x3 glyph bbox; generator preserves that
#   - _decode_phase_tile is unchanged because the glyph vocabulary is unchanged
#   - the old center-placement code depended on "exactly one stamp"; that assumption is removed
#   - the new slot computation depends on:
#       1. output shape being 16x16
#       2. stamp shape being 6x6
#       3. two non-overlapping slots fitting per axis with 1-cell margins
#     these are guaranteed by the unchanged 6x6 input and 3x3 motif size
# placement_legality:
#   - each stamp occupies a 6x6 region
#   - origins [1,9] on each axis give intervals [1,6] and [9,14]
#   - stamps never overlap, stay in bounds, and leave rows/cols 0 and 15 as border
# visual_observability:
#   - four maroon copies are obvious in 2-3 examples even when the glyph is symmetric
#   - the regular spacing is human-visible without needing any invisible metadata
# visual_cleanliness:
#   - maroon only overlays the phase lattice and never another stamp
#   - the 2-cell horizontal/vertical gutters keep the four stamp zones distinct
# color_inference:
#   - colors remain structurally simple: orange/black lattice plus maroon stamps
#   - no new color mapping is introduced


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


def _stamp_origins(axis_len: int, stamp_len: int) -> list[int]:
    """Return the regular slot origins for the repeated stamp lattice on one axis."""
    axis_len = int(axis_len)
    stamp_len = int(stamp_len)
    outer_margin = 1
    gap = axis_len - 2 * outer_margin - 2 * stamp_len
    assert gap >= 0
    step = stamp_len + gap
    origins = list(range(outer_margin, axis_len - stamp_len, step))
    assert len(origins) == 2
    assert origins[0] == outer_margin
    assert origins[1] + stamp_len == axis_len - outer_margin
    return origins


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped multiple times in maroon.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange color.
    - The non-orange pixels form a tight 3x3 contour glyph bbox.
    - The glyph is one of the task's four period-3 contour codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the original task.
    - Fill a black 16x16 canvas by repeating that orange/black phase tile.
    - Tile the original 3x3 glyph into a 6x6 stamp and place that maroon stamp at every slot of a regular 2x2 stamp lattice.

    Key insight:
    - The input glyph still serves two roles at once: it specifies the orange phase pattern for the whole lattice, and it also supplies the small stamp template that must be repeated at evenly spaced output positions.
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
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    stamp_mask = np.tile(motif, (2, 2))
    stamp = np.where(stamp_mask, Color.MAROON, Color.BLACK).astype(int)
    row_origins = _stamp_origins(out.shape[0], stamp.shape[0])
    col_origins = _stamp_origins(out.shape[1], stamp.shape[1])
    for r0 in row_origins:
        for c0 in col_origins:
            paste(out, stamp, r0, c0, transparent=Color.BLACK)

    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

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
    raise RuntimeError("Failed to generate input for eee78d87 D3 variant")
