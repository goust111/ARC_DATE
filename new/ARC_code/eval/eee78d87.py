from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange color.
    - The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the period-3 orange phase tile.
    - Fill a black 16x16 canvas by repeating that orange phase tile.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The colored object is not copied at its input location; it is a small contour code whose phase tells where orange appears in the repeated lattice.
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

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out

# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6                        # input grid dimensions
MOTIF_SIZE = 3                                # glyph bounding box side length
MOTIF_POS_MAX = 4                             # exclusive upper bound for top-left position (0..3)
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]            # valid foreground colors (exclude 0, 7, 9)
MAX_RETRIES = 300                             # oracle-validation retry budget

GLYPHS = {
    'plus':     np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    'H':        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    'X':        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    '4corners': np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0:r0 + MOTIF_SIZE, c0:c0 + MOTIF_SIZE][motif] = fg

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
    raise RuntimeError("Failed to generate input for eee78d87")
