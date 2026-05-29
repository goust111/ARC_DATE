# Variant of eee78d87
# Dimension: D10 — 格栅的颜色角色分配 (transform)
# Change: 相位瓦片中True位置为橙色(7)、False位置为黑色(0)，固定两色；中心印章为栗色(9)
# ... → 格栅线和填充色互换

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D10 variant
# ===================================
# variant_spec:
#   preserved_structure:
#     input_grid: 6x6 orange background with exactly one non-orange 3x3 contour glyph
#     glyph_catalog: plus, H, X, 4corners
#     output_grid: 16x16 periodic lattice plus centered 6x6 maroon 2x2 stamp of the glyph
#   core_change:
#     original: decoded phase=True cells are orange and phase=False cells are black
#     variant: decoded phase=True cells are black blocks while phase=False cells are orange grid lines
#   concrete_pixel_effect:
#     plus glyph still decodes to the same boolean phase tile:
#       ...
#       .##
#       .##
#     but its colored 3x3 tile changes from:
#       000 / 077 / 077? no, original used 0 for False and 7 for True:
#       000
#       077
#       077
#     to:
#       777
#       700
#       700
#     After tiling, the old output looks like orange filled squares on black gaps, while the new output
#     looks like black square holes separated by orange lattice lines. The 6x6 maroon center stamp is unchanged.
#   pipeline_impact:
#     input_construction: unchanged, because the clue glyph itself does not encode colors of the output lattice
#     decode_logic: unchanged, because the glyph-to-phase mapping is the same structural code
#     colorization_step: changed from np.where(phase, orange, black) to np.where(phase, black, orange)
#     center_stamp: unchanged and still overlays maroon on top of either orange or black cells
#   complexity_check:
#     The task still has only four glyph classes and one deterministic output per input.
#     Visual distinction remains strong because the periodic field flips foreground/background everywhere,
#     not just in a tiny local patch. The center stamp remains an anchor so the output is not ambiguous.
#   implicit_assumption_review:
#     - bg = get_bg(g): still valid because the input background remains orange.
#     - colors_present(...)=1: still valid because the input still uses one foreground color.
#     - bbox/crop to 3x3: still valid because generation still places exactly one glyph in a tight 3x3 box.
#     - _decode_phase_tile(motif): still valid because D10 does not alter the glyph vocabulary.
#     - tile_fill(out_shape, phase_colors): valid after inversion because the lattice remains a repeated 3x3 tile.
#     - center overwrite with maroon: still safe because maroon is a dedicated overlay color and the center region
#       may cover either orange or black lattice cells without causing ambiguity.
#   placement_legality:
#     The generator still chooses r0,c0 in 0..3, so the 3x3 motif never exceeds the 6x6 input.
#     There is only one placed object, so overlap constraints are unchanged.
#   observability:
#     In 2-3 examples, a human can spot that the same motif class now produces the photographic negative
#     of the old lattice: rows/columns that used to be orange become black blocks, and vice versa.
#     This is visible across the whole 16x16 field, so the rule is not reliant on color ids or hidden ordering.
#   cleanliness:
#     The output still uses three colors with one clear role each:
#       orange = lattice lines/background,
#       black = repeated phase blocks,
#       maroon = centered stamp.
#     No new overlaps or clutter are introduced.


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 phase mask."""
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
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose orange/black roles are inverted, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one additional color.
    - The non-orange pixels form a single contour motif with a tight 3x3 bounding box.
    - The motif is one of the four glyph codes: plus, H, X, or four corners.

    Rule:
    - Interpret the 3x3 contour motif as the same period-3 boolean phase tile used in the original task.
    - Repeat that tile across a 16x16 canvas, but color phase=True cells black and phase=False cells orange so the lattice polarity is reversed.
    - Tile the original 3x3 motif 2x2 into a centered 6x6 region and recolor the motif pixels to maroon.

    Key insight:
    - The input glyph still acts as a code for a periodic pattern, but this variant asks for the negative of the usual orange mask: orange becomes the grid framework and black becomes the filled phase blocks.
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
    phase_colors = np.where(phase, Color.BLACK, bg).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

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
    motif_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    # Non-degeneration guard: D10 must flip the lattice polarity relative to the original task.
    phase = _decode_phase_tile(motif)
    original_phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    original_like = tile_fill(out.shape, original_phase_colors)
    center = np.tile(motif, (2, 2))
    rr = (out.shape[0] - center.shape[0]) // 2
    cc = (out.shape[1] - center.shape[1]) // 2
    original_like[rr : rr + center.shape[0], cc : cc + center.shape[1]][center] = Color.MAROON
    assert np.any(out != original_like)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D10 variant")
