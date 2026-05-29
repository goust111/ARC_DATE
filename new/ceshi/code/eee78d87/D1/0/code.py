# Variant of eee78d87
# Dimension: D1 — 字形到相位瓦片的解码规则 (match_cond)
# Change: 4种轮廓字形（plus/H/X/4corners）各对应一种固定的3x3相位瓦片，通过角点与边中心像素的布尔组合确定橙色... → 字形颜色影响相位偏移

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D1 variant
# - Concrete behavior:
#   The 3x3 contour still selects which period-3 orange/black tile is repeated,
#   but the glyph color now selects where that period starts on the 16x16 output.
#   Red keeps the original phase, blue shifts the lattice by one row, and green
#   shifts it by one column. The central 6x6 maroon stamp is unchanged.
# - Trace against the original:
#   Originally a red/blue/green plus would all produce the same lattice because
#   only the boolean mask was read. In this variant the same plus shape yields
#   the same 3x3 tile content, but `tile_fill(..., offset=...)` changes from
#   `(0, 0)` to `(1, 0)` or `(0, 1)` depending on the visible input color.
# - Complexity calibration:
#   Three color branches are enough to make the phase rule visible without
#   introducing too many cases for a human solver to infer from few examples.
# - Old hidden assumptions to replace:
#   The original solver could safely discard the foreground color after taking
#   `g != bg`. That is no longer valid because color carries one axis of the
#   output rule, so the new solver keeps both the mask and the unique fg color.
# - Placement legality:
#   The generator still places exactly one 3x3 glyph inside a 6x6 orange grid at
#   top-left positions 0..3, so the bbox is always valid and the solver's crop is
#   well-defined.
# - Visual observability and cleanliness:
#   Phase shifts affect the whole background lattice while the central maroon
#   stamp stays fixed, making the color-controlled change easy to notice without
#   adding any extra overlapping structures.


PHASE_SHIFT_BY_COLOR = {
    Color.RED: (0, 0),
    Color.BLUE: (1, 0),
    Color.GREEN: (0, 1),
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


def _phase_offset_for_color(color: int) -> tuple[int, int]:
    assert color in PHASE_SHIFT_BY_COLOR
    return PHASE_SHIFT_BY_COLOR[color]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn one colored 3x3 contour glyph into a 16x16 periodic orange/black lattice whose phase depends on both shape and color.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one non-orange foreground color.
    - The foreground pixels occupy a tight 3x3 bounding box that forms one contour glyph.
    - The glyph shape is one of plus, H, X, or four corners, and the glyph color is red, blue, or green.

    Rule:
    - Read the 3x3 contour shape to recover the corresponding period-3 orange phase tile.
    - Read the glyph color to choose the lattice offset: red -> (0, 0), blue -> (1, 0), green -> (0, 1).
    - Repeat that phase tile over a black 16x16 canvas using the chosen offset, then tile the original glyph 2x2 and recolor its true cells to maroon in the centered 6x6 region.

    Key insight:
    - The small motif now encodes two independent signals: its silhouette determines which tile is repeated, while its visible color determines how that repeated tile is phase-shifted across the large canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])
    phase_offset = _phase_offset_for_color(fg_color)

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors, offset=phase_offset)

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
FG_COLORS = list(PHASE_SHIFT_BY_COLOR.keys())
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
    raise RuntimeError("Failed to generate input for eee78d87 D1 variant")
