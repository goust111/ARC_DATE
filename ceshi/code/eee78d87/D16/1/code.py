# Variant of eee78d87
# Dimension: D16 — 符号编码方式 (symbol_instructions)
# Change: 字形本身即为符号编码——3x3布尔模式编码3x3相位瓦片的橙黑分布，一种隐式的形状到铺贴的查找表... → 输入中增加独立的颜色图例区

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_ROOT) not in sys.path:
    sys.path.append(str(SOLVER_ROOT))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis
# - concretization:
#   - Input becomes 7x6. Row 0 is a dedicated legend strip; rows 1:7 keep the original 6x6
#     motif field. The legend is fixed to the top edge to keep detection simple and visible.
#   - The strip uses two 3-cell groups: [black, line_color, black, orange, fill_color, orange].
#     The bracket colors are the canonical source roles from the base task, and the middle cells
#     are the actual lattice colors to use in the output.
#   - The lower 6x6 field still contains exactly one 3x3 contour glyph in a tight bbox, but its
#     own color is now independent from the lattice colors. That is the core D16 change.
#   - Small trace:
#     original flow: motif shape -> decode phase -> hard-code {False:black, True:orange}.
#     variant flow: legend -> decode {black->line_color, orange->fill_color} -> motif shape ->
#     phase -> tile with remapped colors. The glyph still selects the phase tile, but no longer
#     selects the palette by itself.
# - complexity calibration:
#   - Only two semantic roles are remapped, so the legend stays learnable from 2-3 examples.
#   - Actual line/fill colors are chosen from a small visible set and are forced away from
#     black/orange to prevent degeneration back into the base task.
# - implicit-assumption audit of the original main():
#   - `assert g.shape == (6, 6)`: no longer valid; solver must split legend row from body.
#   - `bg == Color.ORANGE`: no longer valid; the motif field background can vary because output
#     palette is now supplied externally by the legend.
#   - `len(colors_present(g, exclude={bg})) == 1`: no longer valid on the full input because the
#     legend adds extra colors; the single-foreground assumption only applies to the 6x6 body.
#   - `phase_colors = np.where(phase, bg, Color.BLACK)`: unsafe in the variant because `bg` is
#     not the intended lattice fill color anymore. The fill/line colors must come from the legend.
#   - `out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)`: must use body size, not total input
#     size, or the added legend row would incorrectly inflate the output.
# - placement legality:
#   - The legend occupies row 0 only, so it cannot overlap the motif.
#   - The motif still uses a 3x3 bbox placed with top-left in [0,3]x[0,3] inside the 6x6 body.
#   - Generator validates every sample through `main()` to catch any accidental invalid layout.
# - observability / cleanliness / color inference:
#   - The legend is visually isolated at the edge and uses a repeated bracket pattern, so the
#     remapping cue is localized and easy to compare against the output lattice.
#   - The body background and motif color are kept distinct from legend/output colors to avoid
#     clutter and accidental "same color means same role" shortcuts.
#   - Output still has the same geometric structure as the base task, so the only new inference
#     burden is reading the legend strip.
# - new solver assumptions and generator guarantees:
#   - Assumption: row 0 exactly matches [0, L, 0, 7, F, 7] with L != F and both outside {0,7,9}.
#     Generator enforces this directly.
#   - Assumption: rows 1:7 contain exactly one non-background glyph with a tight 3x3 bbox and one
#     non-background color. Generator constructs only the four canonical glyphs and validates them.


CANON_LINE = Color.BLACK
CANON_FILL = Color.ORANGE
INPUT_BODY_H, INPUT_BODY_W = 6, 6
INPUT_H, INPUT_W = INPUT_BODY_H + 1, INPUT_BODY_W
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
MAX_RETRIES = 300
NON_SPECIAL_COLORS = [1, 2, 3, 4, 5, 6, 8]

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


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


def _read_legend(g: np.ndarray) -> tuple[int, int]:
    legend = g[0, :]
    assert legend.shape == (6,)
    assert int(legend[0]) == CANON_LINE
    assert int(legend[2]) == CANON_LINE
    assert int(legend[3]) == CANON_FILL
    assert int(legend[5]) == CANON_FILL

    line_color = int(legend[1])
    fill_color = int(legend[4])
    assert line_color != fill_color
    assert line_color not in {CANON_LINE, CANON_FILL, Color.MAROON}
    assert fill_color not in {CANON_LINE, CANON_FILL, Color.MAROON}
    return line_color, fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read an edge legend that remaps the lattice palette, then decode the 3x3 contour glyph into the same periodic tiling as the base task.

    Input structure:
    - The grid is 7x6: a 1x6 legend strip on the top edge and a 6x6 motif field below it.
    - The legend strip is `[black, line_color, black, orange, fill_color, orange]`, so the canonical line/fill roles are shown independently from the glyph.
    - The lower 6x6 field has one non-background color whose tight 3x3 bbox is one of the four contour glyphs: plus, H, X, or four corners.

    Rule:
    - Read the actual lattice line color from the black-bracketed legend triplet and the actual lattice fill color from the orange-bracketed triplet.
    - Decode the 3x3 glyph into the same period-3 phase tile used by the original task.
    - Fill a 16x16 canvas by tiling that phase with the legend-selected colors, then stamp a 2x2 maroon copy of the glyph in the centered 6x6 region.

    Key insight:
    - The glyph still determines which phase tile to use, but the output palette no longer comes from fixed colors or the glyph color itself; it must be read from the separate legend strip first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_H, INPUT_W)

    line_color, fill_color = _read_legend(g)
    body = g[1:, :]
    body_bg = int(get_bg(body))
    assert len(colors_present(body, exclude={body_bg})) == 1

    fg = body != body_bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, fill_color, line_color).astype(int)
    out_shape = (3 * body.shape[0] - 2, 3 * body.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
    line_color, fill_color, body_bg, fg_color = map(
        int, rng.choice(NON_SPECIAL_COLORS, size=4, replace=False).tolist()
    )
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    body = np.full((INPUT_BODY_H, INPUT_BODY_W), body_bg, dtype=int)
    body[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg_color
    legend = np.array([CANON_LINE, line_color, CANON_LINE, CANON_FILL, fill_color, CANON_FILL], dtype=int)
    inp = np.vstack([legend[np.newaxis, :], body])

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
    raise RuntimeError("Failed to generate input for eee78d87 D16 variant")
