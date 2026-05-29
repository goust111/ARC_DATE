# Variant of eee78d87
# Dimension: D5 — 输出网格尺寸与裁剪方式 (grid_shape)
# Change: 固定16x16方形输出（计算公式为3*6-2=16）... → 输出尺寸由字形像素数编码

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D5 variant
# ----------------------------------
# a) Concretization and behavior trace
# - Input stays 6x6, orange background, with one 3x3 contour glyph in one non-orange color.
# - The four legal glyphs remain:
#   plus(5 pixels), H(7 pixels), X(5 pixels), 4corners(4 pixels).
# - The phase decoding is unchanged: the 3x3 contour still selects one of the four period-3
#   orange/black tiles.
# - The changed step is canvas sizing. Original behavior used a fixed 16x16 canvas because it
#   derived shape only from the 6x6 input size. New behavior derives the output side from the
#   glyph's visible pixel count: side = 3 * foreground_pixel_count.
# - Example trace for plus:
#   original: 5-pixel plus -> phase tile -> always 16x16 lattice -> centered 6x6 maroon stamp.
#   variant:  5-pixel plus -> phase tile -> 15x15 lattice -> centered 6x6 maroon stamp.
#   The phase pattern and stamp are the same kind of objects; only the canvas extent changes.
# - Example trace for H:
#   original: 7-pixel H -> always 16x16.
#   variant:  7-pixel H -> 21x21.
#
# b) Complexity calibration
# - The rule is small but visible: the solver must infer a count-to-size relation, not just a
#   different constant.
# - Only three output sizes can occur: 12 (4corners), 15 (plus or X), 21 (H). This is enough
#   variety to make the D5 change obvious without creating too many branches.
# - plus and X intentionally share size 15; they remain distinguishable because their decoded
#   phase lattices differ, so output size alone is not sufficient to solve the task.
#
# c) Implicit-assumption review against the original main()
# - `assert g.shape == (6, 6)`: still valid; D5 only changes output shape.
# - `bg == ORANGE` and exactly one other color: still valid.
# - `motif.shape == (3, 3)`: still valid; the code still expects one tight 3x3 contour code.
# - `_decode_phase_tile(motif)`: still valid; the meaning of the contour glyph is unchanged.
# - `out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)`: no longer valid because it ignores
#   the new controlling signal (motif pixel count) and would collapse every case back to 16x16.
# - Centering logic remains valid if the canvas is at least 6x6. The smallest legal motif has
#   4 pixels, so the smallest side is 12, which safely contains the 6x6 center stamp.
#
# d) Placement legality
# - The generator still places exactly one 3x3 motif inside a 6x6 input with top-left rows and
#   columns in [0, 3], so it never goes out of bounds.
# - There are no multiple objects, so overlap constraints are unchanged.
# - Every generated motif is one of the solver's accepted contour forms, so bbox extraction and
#   phase decoding remain deterministic.
#
# e) Visual observability
# - From 2-4 examples, a human can notice that larger glyphs by pixel count create larger output
#   canvases, while the internal lattice family stays tied to glyph shape.
# - The rule uses a visible property (number of painted cells), not hidden ids or metadata.
#
# f) Visual cleanliness
# - The output still has only two layers: a repeated orange/black lattice and a centered maroon
#   stamp. No extra annotations are introduced.
# - Even the smallest 12x12 output leaves a 3-cell border around the 6x6 central stamp, so the
#   composition stays legible.
#
# g) Color inferability
# - Color logic is unchanged and remains simple: input uses orange + one arbitrary foreground
#   color; output uses only black, orange, and maroon.
# - The variant does not introduce any new color mapping burden.
#
# Post-change main() assumptions to audit
# - The glyph must be one of the four supported contour templates so its foreground count is one
#   of {4, 5, 7}. The generator enforces that by sampling from `GLYPHS`.
# - The output side is always `3 * count`, which is >= 12; this guarantees the centered 6x6
#   stamp placement cannot underflow or overflow.


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


def _output_side_from_motif(motif: np.ndarray) -> int:
    """Encode the output side length from the contour glyph's foreground pixel count."""
    count = int(np.count_nonzero(motif))
    assert count in {4, 5, 7}
    side = 3 * count
    assert side >= 6
    return side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a variable-size periodic lattice whose side length is set by the glyph's pixel count, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels occupy a tight 3x3 bounding box that contains one of the task's contour glyphs.
    - The glyph is one of four legal forms: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 contour glyph into the same period-3 orange/black phase tile as in the original task.
    - Count the glyph's foreground pixels and set the output size to `(3 * count, 3 * count)`.
    - Fill that canvas by tiling the phase tile, then place a centered 6x6 maroon stamp made from a 2x2 tiling of the original glyph.

    Key insight:
    - The input glyph carries two signals at once: its contour still selects the lattice phase pattern, and its visible pixel count now also controls the size of the entire output canvas.
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
    side = _output_side_from_motif(motif)
    out = tile_fill((side, side), phase_colors)

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
        expected_side = 3 * int(np.count_nonzero(motif))
        assert out.shape == (expected_side, expected_side)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D5 variant")
