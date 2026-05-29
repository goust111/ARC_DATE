# Variant of eee78d87
# Dimension: D7 — 字形在输入网格中的定位方式 (src)
# Change: 字形可放置在6x6网格中的任意合法位置（左上角行0-3、列0-3），通过检测非橙色像素的外接矩形定位... → 字形位置由角落标记像素指示
from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the D7 variant
# ----------------------------------
# concrete_behavior:
#   input remains 6x6 orange with one 3x3 contour glyph and one detached control pixel.
#   The glyph still uses one solid foreground color and still occupies a legal 3x3 bbox with top-left (r0,c0) in 0..3.
#   The new control pixel uses a second non-orange color and is placed in a 2x2 corner block opposite the glyph quadrant:
#     glyph TL quadrant -> marker in bottom-right 2x2
#     glyph TR quadrant -> marker in bottom-left 2x2
#     glyph BL quadrant -> marker in top-right 2x2
#     glyph BR quadrant -> marker in top-left 2x2
#   Within that opposite 2x2 block, the marker's local row/col directly encode the fine offset inside the quadrant.
#   Example:
#     original logic on a glyph at (0,3): bbox(all non-orange) is the glyph bbox itself, so crop->decode works.
#     new logic on the same glyph: marker sits at (4,1), so bbox(all non-orange) becomes large and no longer isolates the glyph.
#     solver must first decode (4,1) as "top half + right half" -> glyph top-left (0,3), then read the 3x3 patch there.
# complexity_calibration:
#   There are still only 4 glyph classes and 16 legal placements.
#   The marker adds exactly one extra pixel and four geometric branches, enough to be visible without making the rule combinatorial.
# implicit_assumption_audit:
#   The original `len(colors_present(..., exclude={bg})) == 1` is no longer valid because control and glyph colors differ.
#   The original `bbox(g != bg)` assumes every non-bg pixel belongs to one tight 3x3 object; the detached marker breaks that.
#   The original `crop_bbox(...).shape == (3,3)` is therefore invalid and must be replaced by marker-decoded slicing.
# placement_legality:
#   Opposite-corner placement makes the singleton control pixel disjoint from the 3x3 glyph for every (r0,c0).
#   The marker always lands inside one of the four 2x2 corner blocks, never out of bounds.
# visual_observability:
#   Across examples, humans can see the detached corner pixel move in the opposite corner block as the glyph moves.
#   The relation is geometric, not based on hidden color ids or arbitrary sorting.
# visual_cleanliness:
#   Only two input objects exist: one 3x3 glyph and one isolated marker. They never overlap, so the scene stays readable.
# color_inference:
#   Color semantics stay minimal: orange is background, one color is the glyph, one singleton color is the marker.
#   The output color rule is unchanged from the base task, so the new challenge is purely source-location decoding.


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CTRL_BLOCK = 2
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

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


def _decode_marker_position(marker: tuple[int, int]) -> tuple[int, int]:
    """Map the detached corner marker back to the glyph top-left position."""
    r, c = map(int, marker)

    if 0 <= r < CTRL_BLOCK and 0 <= c < CTRL_BLOCK:
        return r + 2, c + 2
    if 0 <= r < CTRL_BLOCK and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r + 2, c - 4
    if GRID_H - CTRL_BLOCK <= r < GRID_H and 0 <= c < CTRL_BLOCK:
        return r - 4, c + 2
    if GRID_H - CTRL_BLOCK <= r < GRID_H and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r - 4, c - 4
    raise AssertionError("Control pixel must lie in a 2x2 corner block")


def _encode_marker_position(r0: int, c0: int) -> tuple[int, int]:
    """Inverse of `_decode_marker_position` used by the generator."""
    r0 = int(r0)
    c0 = int(c0)

    if r0 < 2 and c0 < 2:
        return r0 + 4, c0 + 4
    if r0 < 2 and c0 >= 2:
        return r0 + 4, c0 - 2
    if r0 >= 2 and c0 < 2:
        return r0 - 2, c0 + 4
    return r0 - 2, c0 - 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background, one multi-pixel glyph color, and one singleton control color.
    - The glyph is still one of the task's 3x3 contour forms, but its location is no longer found by taking the bbox of all non-orange pixels.
    - A detached control pixel sits in a 2x2 corner block; its corner-block position encodes the glyph's top-left offset.

    Rule:
    - Find the singleton non-orange control pixel and decode its corner-block coordinates into the glyph's 3x3 location.
    - Read the glyph from that decoded 3x3 window and convert it into the same period-3 orange/black phase tile as in the base task.
    - Fill the 16x16 canvas with that phase tile, then stamp a 2x2 maroon tiling of the glyph into the centered 6x6 region.

    Key insight:
    - The detached corner pixel is not part of the motif itself; it is a location code. The solver must interpret that code first, then extract the real 3x3 contour.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    nonbg_colors = colors_present(g, exclude={bg})
    assert len(nonbg_colors) == 2

    ctrl_candidates = [int(col) for col in nonbg_colors if int(np.count_nonzero(g == int(col))) == 1]
    assert len(ctrl_candidates) == 1
    ctrl = int(ctrl_candidates[0])
    fg = int(next(col for col in nonbg_colors if int(col) != ctrl))

    marker_cells = np.argwhere(g == ctrl)
    assert marker_cells.shape == (1, 2)
    marker = tuple(map(int, marker_cells[0]))

    r0, c0 = _decode_marker_position(marker)
    motif_patch = g[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    assert motif_patch.shape == (MOTIF_SIZE, MOTIF_SIZE)
    assert not bool(np.any(motif_patch == ctrl))
    assert set(map(int, np.unique(motif_patch))) <= {bg, fg}

    motif = motif_patch == fg
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r_center = (out.shape[0] - center.shape[0]) // 2
    c_center = (out.shape[1] - center.shape[1]) // 2
    out[r_center : r_center + center.shape[0], c_center : c_center + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    ctrl_choices = [c for c in FG_COLORS if int(c) != fg]
    ctrl = int(rng.choice(ctrl_choices))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))
    mr, mc = _encode_marker_position(r0, c0)

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg
    assert inp[mr, mc] == Color.ORANGE
    inp[mr, mc] = ctrl

    try:
        solved = main(inp)
    except Exception:
        return None

    assert solved.shape == (16, 16)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D7 variant")
