# Variant of eee78d87
# Dimension: D4 — 栗色重着色变换 (transform)
# Change: 中心6x6区域中字形像素位置统一重着色为栗色(9)，非字形位置保持格栅原样
#   -> 栗色仅出现在格栅线与字形的交叉处

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis for the D4 variant
# spec:
#   unchanged_core:
#     input_grid: 6x6 orange background with one 3x3 contour glyph in one non-orange color
#     glyph_types: plus, H, X, 4corners
#     lattice: decode the glyph into the same 3x3 orange/black phase tile and repeat it on 16x16
#     center_box: the tiled motif still occupies the centered 6x6 region at rows 5..10 and cols 5..10
#   variant_core:
#     old_center_rule: every true pixel of the 2x2 tiled glyph became maroon
#     new_center_rule: maroon appears only where the tiled glyph overlaps the 3-period lattice seams
#     seam_definition: inside the centered 6x6 box, a seam is any pixel whose local row or local col is divisible by 3
#   behavior_trace:
#     original: for plus, all 20 true pixels of the 6x6 tiled glyph were recolored
#     variant: for plus, only the 8 true pixels on rows {0,3} or cols {0,3} inside that 6x6 box become maroon
#     difference: the center pattern is now an intersection mask, so some glyph pixels stay orange/black
#   complexity_check:
#     avoiding phase-orange AND: with the original four glyphs, glyph pixels and phase-orange pixels in the center are disjoint,
#       which would collapse the stamp to zero maroon pixels
#     chosen seam mask: always non-empty for all four glyphs, but never equals the full glyph, so the D4 change is visible
#   legality:
#     input placement is unchanged: one 3x3 glyph fits anywhere in the 6x6 input without overlap issues
#     output placement is unchanged: the 6x6 center box is fixed and fully inside the 16x16 lattice
#   new_solver_assumptions:
#     the centered stamp remains a 2x2 tiling of a 3x3 motif, so a period-3 seam mask is meaningful
#     generate_input guarantees exactly the same 3x3 motif forms as the original task, so decode assertions still hold

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


def _lattice_seam_mask(shape: tuple[int, int]) -> np.ndarray:
    """Rows/cols at multiples of 3 inside the centered 6x6 stamp box."""
    rows, cols = np.indices(shape)
    return ((rows % 3) == 0) | ((cols % 3) == 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour cue into the same 16x16 periodic lattice, then tint only the glyph pixels that land on lattice seams at the center.

    Input structure:
    - The input is a 6x6 orange grid containing exactly one non-orange contour glyph.
    - That glyph occupies a tight 3x3 bounding box and uses one foreground color.
    - The 3x3 form is one of the four supported contour codes: plus, H, X, or four corners.

    Rule:
    - Convert the 3x3 contour code into the task's period-3 orange/black phase tile.
    - Repeat that tile to build the full 16x16 lattice exactly as in the original task.
    - Tile the 3x3 glyph 2x2 into the centered 6x6 box, but recolor to maroon only where those glyph pixels also lie on the box's period-3 seam rows or seam columns.

    Key insight:
    - The center mark is no longer the whole tiled glyph; it is the boolean intersection between the glyph mask and the lattice's visible 3-step seam structure.
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
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    seam_mask = _lattice_seam_mask(center.shape)
    maroon_mask = center & seam_mask

    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][maroon_mask] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    center = out[5:11, 5:11]
    maroon_count = int((center == Color.MAROON).sum())
    if maroon_count <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D4")
