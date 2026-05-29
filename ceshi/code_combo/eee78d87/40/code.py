# Combo variant of eee78d87 (index 40)
# combo_spec: D3/2+D4/0
# Name: 保留原色印章+字形色背景

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the fused variant:
# - Concrete trace:
#   A green plus still decodes to the same 3x3 phase tile as in the source task.
#   Originally that produced an orange/black lattice and a maroon center stamp.
#   Here the lattice becomes orange/green, and the central 6x6 2x2 stamp also stays
#   green. The center therefore stands out only because its green pixels follow the
#   tiled glyph mask, not because they use a unique color.
# - Complexity calibration:
#   The original four glyph classes, 6x6 input, and 16x16 output are preserved.
#   Only two coloring decisions change, so the variant is visibly different without
#   adding extra object types or unstable case splits.
# - Assumption audit:
#   `_decode_phase_tile()` still depends only on the 3x3 boolean contour shape, so
#   it remains valid. The old assumptions that "non-phase cells are black" and
#   "center glyph pixels are maroon" are both invalid now and must be replaced with
#   the single input foreground color. A naive fusion would only recolor the glyph
#   pixels in the center, but that collapses the stamp into the lattice because the
#   lattice complement is also foreground-colored. To keep the fused task visible,
#   the center must replace the full 6x6 patch with an orange/foreground motif
#   rendering, and the periodic lattice needs a fixed nonzero phase offset so the
#   background pattern does not coincide exactly with the stamp.
# - Placement legality:
#   The generator still places exactly one 3x3 motif inside a 6x6 orange field with
#   top-left coordinates in 0..3, so the glyph never clips or overlaps anything.
# - Visual observability and cleanliness:
#   The lattice and stamp share one color, so observability has to come from shape.
#   Replacing the whole centered 6x6 patch with the 2x2 tiled motif on orange keeps
#   the stamp visible as a structured region instead of letting it vanish into the
#   periodic background. The scene stays tidy: one repeating background and one
#   centered overwritten area.
# - New generator guarantees:
#   Foreground colors exclude black, orange, and maroon, so the fused output cannot
#   collapse back into the source palette or hide the orange/foreground contrast.

LATTICE_OFFSET = (1, 1)


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
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose two colors are orange and the glyph's own foreground color, with the center stamp kept in that same foreground color.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels form one contour glyph with a tight 3x3 bounding box.
    - The glyph shape is one of the task's period-3 contour codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph shape into the same period-3 phase tile used by the source task.
    - Repeat that tile across a 16x16 canvas with a fixed phase offset, coloring phase cells orange and the complementary cells with the input foreground color instead of black.
    - Tile the original 3x3 glyph mask 2x2 into the centered 6x6 region, then replace that whole 6x6 patch with an orange/foreground rendering of the motif so the stamp keeps its original color without disappearing into the lattice.

    Key insight:
    - The input color serves two roles at once: it replaces black in the repeated lattice and it also colors the centered stamp, so the center must be identified from a full motif-pattern replacement rather than from a unique stamp color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, fg_color).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors, offset=LATTICE_OFFSET)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]] = np.where(center, fg_color, bg)
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
    fg_color = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg_color

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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
