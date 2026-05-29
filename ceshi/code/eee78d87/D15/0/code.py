# Variant of eee78d87
# Dimension: D15 — 格栅的空间方向 (meta)
# Change: 格栅为正交方向（水平/垂直线交织），铺贴沿行列方向周期重复... → 对角线方向的格栅

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg

# Design analysis
# - concretization:
#   The input stays identical to the seed task: a 6x6 orange board with one tight 3x3 contour glyph.
#   The output keeps the 16x16 size and the centered 6x6 maroon 2x2 stamp, but the orange/black carrier
#   lattice is no longer tiled by (row mod 3, col mod 3). Instead it is sampled in diagonal coordinates
#   u = (r + c) mod 3 and v = (r - c) mod 3, so the same 3x3 phase tile repeats along two 45-degree axes.
#   Example trace: for the H glyph, the decoded phase tile is 000/100/100. In the seed task that creates
#   vertical orange bars every 3 columns; here it creates offset NE-SW diagonal chains because both phase
#   indices depend on both row and column.
# - complexity calibration:
#   Keep the original four glyph classes and the original decoder, so there is still one small discrete
#   phase code to infer. Only the spatial embedding changes; this makes the D15 change visually obvious
#   without adding extra branches, colors, or object types.
# - implicit assumptions review:
#   `_decode_phase_tile()` relies on the old contour catalog only; that catalog is unchanged.
#   `bbox(fg)` and the 3x3 crop assumption still hold and are enforced by the generator.
#   The old `tile_fill()` call assumed orthogonal periodicity; that is the one line replaced.
#   The centered maroon overwrite remains safe because it intentionally sits on top of the lattice in both
#   the original task and this variant.
# - placement legality:
#   The generator still places exactly one 3x3 motif wholly inside the 6x6 input, so there is no overlap,
#   no clipping, and `main()` always sees one connected contour code in a 3x3 bbox.
# - visual observability:
#   The diagonal embedding changes the same decoded tile into rhombic / slanted repetition. Multiple random
#   glyph types are still sampled, so across a few examples the solver must notice that periodicity follows
#   diagonal coordinates rather than row/column coordinates.
# - cleanliness:
#   The output still has only two layers: a black/orange background lattice and a maroon central stamp.
#   That keeps the scene legible even though the carrier pattern is now diagonal.
# - color inference:
#   No new color mapping is introduced. The only color behavior remains orange/black for the lattice and
#   maroon for the center stamp, exactly as in the seed task.


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


def _tile_fill_diagonal(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    """Repeat `tile` using diagonal coordinates instead of row/column coordinates."""
    h, w = tile.shape
    rr, cc = np.indices(shape)
    return tile[(rr + cc) % h, (rr - cc) % w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a 16x16 diagonal lattice with the motif stamped in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange color.
    - The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the seed task.
    - Fill a black 16x16 canvas by repeating that phase tile in diagonal coordinates, using
      u=(row+col) mod 3 and v=(row-col) mod 3 instead of orthogonal row/column tiling.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The colored contour still acts as a tiny phase code, but the periodic structure it controls now lives
      on two 45-degree lattice axes rather than on horizontal and vertical repeats.
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
    out = _tile_fill_diagonal(out_shape, phase_colors)

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
    raise RuntimeError("Failed to generate input for eee78d87 D15")
