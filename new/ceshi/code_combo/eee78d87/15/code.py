# Combo variant of eee78d87 (index 15)
# combo_spec: D2/3+D16/1
# Name: 边缘色条字形+颜色图例区

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_ROOT) not in sys.path:
    sys.path.append(str(SOLVER_ROOT))

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the combo variant
# a) Concretization and trace:
#    - The cleanest fusion is a 7x6 input split into two visible bands:
#      row 0 is a dedicated palette legend, rows 1:7 form a 6x6 body whose outer ring
#      carries the glyph legend.
#    - Example trace:
#      original task:
#        1. detect a tight 3x3 contour glyph inside a 6x6 field
#        2. decode that glyph into the period-3 lattice phase
#        3. render the lattice with fixed {line=black, fill=orange}
#      fused task:
#        1. read row 0 as `[black, L, black, orange, F, orange]`
#        2. read the 6x6 body's border markers and reconstruct the 3x3 glyph
#        3. decode the glyph into the same phase tile
#        4. render the lattice with `{line=L, fill=F}`
#      For a plus glyph, the body still shows the 010/111/010 structure indirectly on
#      its border, while the top row separately says which two colors replace black and
#      orange in the output lattice.
#    - Difference from the base task:
#      the source shape is no longer an interior blob, and the output palette is no
#      longer hard-coded. Both pieces must be read from different regions.
# b) Complexity calibration:
#    - The glyph family stays fixed to the four original contour forms, so shape decoding
#      is still learnable from a few examples.
#    - The palette legend remaps only two semantic roles and uses a repeated bracket
#      pattern, which keeps the color rule explicit instead of arbitrary.
#    - The body uses one legend color only, while the top strip handles the palette, so
#      the two control channels do not collapse into one ambiguous multi-color border.
# c) Implicit assumption audit against the original main():
#    - `assert g.shape == (6, 6)` no longer holds: the combo input is 7x6.
#    - `bg == Color.ORANGE` on the whole input is invalid because the top legend row
#      intentionally contains several colors.
#    - `len(colors_present(g, exclude={bg})) == 1` on the whole grid is invalid for the
#      same reason; the one-foreground assumption now applies only to the 6x6 body.
#    - `bbox(...)` on the full input is invalid because the glyph is discontiguous border
#      data rather than a single interior 3x3 object.
#    - `phase_colors = np.where(phase, bg, Color.BLACK)` is invalid because the output
#      colors must come from the legend, not from fixed orange/black.
#    - `out_shape` must be derived from the 6x6 body, not the full 7x6 input.
# d) Placement legality:
#    - Row 0 is reserved for the palette legend only.
#    - The 6x6 body's reserved glyph cells are the same safe positions as the D2 single
#      variant: top strip, left strip, right strip, bottom strip, and one center marker.
#    - The generator writes only into those reserved cells, so there is no overlap and no
#      out-of-bounds placement.
# e) Visual observability:
#    - Across examples, the top row's middle two cells visibly reappear as the two lattice
#      colors in the output, while the body border pattern reappears as the centered 6x6
#      maroon stamp. The two clues support two distinct inferences.
#    - Humans do not need numeric color ids: the bracket colors black/orange indicate the
#      semantic roles, and the body border geometry indicates the glyph.
# f) Visual cleanliness:
#    - The input has a strict separation of duties: palette on row 0, glyph on the body
#      border, empty interior in the body. No region mixes both encodings.
# g) Color-change inferability:
#    - Only two output roles change color, and both are shown explicitly in the top legend.
#    - The body background and glyph-legend color are sampled independently from the output
#      palette so that "same color means same role" shortcuts do not accidentally work.
# h) Variant metrics to guard against degeneration:
#    - The body interior should stay entirely background-colored.
#    - The top legend should use exactly the canonical frame `[0, L, 0, 7, F, 7]`.
#    - Over generated samples, both the reconstructed glyph identity and the legend-chosen
#      `(line_color, fill_color)` pair should vary; otherwise the combo would collapse back
#      toward a single-control variant.
# i) New main() assumptions and generator guarantees:
#    - Assumption: row 0 is the palette legend in the fixed bracketed format.
#      Guarantee: generator constructs it directly from sampled line/fill colors.
#    - Assumption: the 6x6 body has exactly one non-background color and that color appears
#      only on the reserved glyph-legend cells.
#      Guarantee: generator fills the body with one background color and stamps one legend
#      color only into those reserved positions.
#    - Assumption: the reconstructed motif is one of the four supported contour glyphs.
#      Guarantee: generator samples only from that fixed glyph set and oracle-validates.


CANON_LINE = Color.BLACK
CANON_FILL = Color.ORANGE
INPUT_BODY_H, INPUT_BODY_W = 6, 6
INPUT_H, INPUT_W = INPUT_BODY_H + 1, INPUT_BODY_W
OUT_H, OUT_W = 3 * INPUT_BODY_H - 2, 3 * INPUT_BODY_W - 2
MAX_RETRIES = 300
NON_SPECIAL_COLORS = [1, 2, 3, 4, 5, 6, 8]

TOP_SLICE = (0, slice(1, 4))
LEFT_SLICE = (slice(1, 4), 0)
RIGHT_SLICE = (slice(1, 4), 5)
BOTTOM_SLICE = (5, slice(1, 4))
CENTER_POS = (0, 5)

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the original period-3 phase mask."""
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


def _read_palette_legend(g: np.ndarray) -> tuple[int, int]:
    legend = g[0, :]
    assert legend.shape == (INPUT_W,)
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


def _glyph_legend_mask() -> np.ndarray:
    mask = np.zeros((INPUT_BODY_H, INPUT_BODY_W), dtype=bool)
    mask[TOP_SLICE] = True
    mask[LEFT_SLICE] = True
    mask[RIGHT_SLICE] = True
    mask[BOTTOM_SLICE] = True
    mask[CENTER_POS] = True
    return mask


def _read_border_motif(body: np.ndarray, bg: int) -> np.ndarray:
    fg = body != bg
    mask = _glyph_legend_mask()
    assert not bool(np.any(fg & ~mask))

    top = fg[TOP_SLICE]
    left = fg[LEFT_SLICE]
    right = fg[RIGHT_SLICE]
    bottom = fg[BOTTOM_SLICE]
    center = bool(fg[CENTER_POS])

    assert bool(top[0]) == bool(left[0])
    assert bool(top[2]) == bool(right[0])
    assert bool(bottom[0]) == bool(left[2])
    assert bool(bottom[2]) == bool(right[2])

    motif = np.array(
        [
            [top[0], top[1], top[2]],
            [left[1], center, right[1]],
            [bottom[0], bottom[1], bottom[2]],
        ],
        dtype=bool,
    )
    assert any(np.array_equal(motif, glyph) for glyph in GLYPHS.values())
    return motif


def _stamp_border_legend(body: np.ndarray, motif: np.ndarray, fg: int) -> None:
    body[TOP_SLICE][motif[0]] = fg
    body[LEFT_SLICE][motif[:, 0]] = fg
    body[RIGHT_SLICE][motif[:, 2]] = fg
    body[BOTTOM_SLICE][motif[2]] = fg
    if bool(motif[1, 1]):
        body[CENTER_POS] = fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a palette legend from the top strip and a contour glyph from the 6x6 body's border, then combine both to build the periodic lattice and centered maroon stamp.

    Input structure:
    - The grid is 7x6: row 0 is an independent color legend, and rows 1:7 form a 6x6 body.
    - The top legend row is `[black, line_color, black, orange, fill_color, orange]`, which states how the canonical lattice roles should be recolored.
    - The 6x6 body uses one background color plus one glyph-legend color; that foreground appears only on reserved border cells that encode one of the four contour glyphs (plus, H, X, or four corners).

    Rule:
    - Read `line_color` and `fill_color` from the top legend row.
    - Reconstruct the 3x3 contour motif from the body's border bars and decode it into the original period-3 phase tile.
    - Tile a 16x16 canvas with the phase using the legend-selected colors, then place a 2x2 maroon tiling of the reconstructed glyph in the centered 6x6 region.

    Key insight:
    - The input contains two independent control channels: the body border determines the geometry of the lattice phase, while the top legend independently determines the lattice palette. Neither cue is sufficient alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_H, INPUT_W)

    line_color, fill_color = _read_palette_legend(g)
    body = g[1:, :]
    body_bg = int(get_bg(body))
    assert len(colors_present(body, exclude={body_bg})) == 1

    motif = _read_border_motif(body, body_bg)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, fill_color, line_color).astype(int)
    out = tile_fill((OUT_H, OUT_W), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    line_color, fill_color, body_bg, glyph_color = map(
        int, rng.choice(NON_SPECIAL_COLORS, size=4, replace=False).tolist()
    )

    legend = np.array(
        [CANON_LINE, line_color, CANON_LINE, CANON_FILL, fill_color, CANON_FILL],
        dtype=int,
    )
    body = np.full((INPUT_BODY_H, INPUT_BODY_W), body_bg, dtype=int)
    _stamp_border_legend(body, motif, glyph_color)
    inp = np.vstack([legend[np.newaxis, :], body])

    assert not bool(np.any(body[1:5, 1:5] != body_bg))
    assert tuple(int(x) for x in inp[0, :]) == (
        CANON_LINE,
        line_color,
        CANON_LINE,
        CANON_FILL,
        fill_color,
        CANON_FILL,
    )

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
