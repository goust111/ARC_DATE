# Combo variant of eee78d87 (index 19)
# combo_spec: D5/1+D12/0
# Name: 输出尺寸由像素数编码+输入尺寸可变

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Preserved core object language:
#   the input still contains one tight 3x3 contour glyph on an orange square grid, and the
#   output still consists of a tiled orange/black phase lattice plus a centered 2x2 maroon stamp
#   of that same glyph.
# - D12 changes the input carrier: the orange grid side is no longer fixed at 6, and can vary
#   from 4 to 9. The 3x3 glyph may therefore appear in different amounts of surrounding orange
#   padding, but it still occupies a single 3x3 bbox.
# - D5 changes the output size controller: the lattice side no longer follows the input side.
#   Instead, it is encoded by the glyph's visible pixel count, using `side = 3 * count`.
# - Example trace:
#   input A = 4x4 with a plus glyph (5 pixels). Original task would be impossible because the
#   input size was fixed. A pure D12 variant would output 10x10 from `3N-2`. This combo instead
#   decodes the plus to its phase tile and outputs 15x15 because `3 * 5 = 15`.
#   input B = 9x9 with the same plus glyph. The combo output is still 15x15 even though the
#   input got larger, because the size controller is the motif count rather than N.
#   input C = 6x6 with an H glyph (7 pixels). The combo output becomes 21x21.
# - The key difference from the original is therefore twofold:
#   `N` can change across examples, but output size is intentionally decoupled from `N`.
#
# b) Complexity calibration
# - There are only three output-side values: 12 (4corners), 15 (plus or X), and 21 (H). This is
#   enough variability to show the count-to-size rule without making the size mapping noisy.
# - Input sizes range across six values (4..9), which is enough for the solver to see that input
#   size varies independently from output size.
# - plus and X share the same count and thus the same output side, but they remain separable
#   because their phase tiles differ. This keeps the fusion non-trivial.
#
# c) Implicit-assumption review against the original main()
# - `assert g.shape == (6, 6)` breaks under D12 and must be replaced by checks for a square grid
#   with side >= 4.
# - `out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)` breaks under D5 because output size
#   now depends on motif count, not on the carrier grid size.
# - `motif.shape == (3, 3)` remains valid because the generator still inserts exactly one legal
#   3x3 contour code and the solver still extracts a tight bbox.
# - `_decode_phase_tile(motif)` remains valid because the contour semantics are unchanged.
# - Center-stamp placement still works because the minimum legal side is `3 * 4 = 12`, which is
#   comfortably larger than the fixed 6x6 maroon stamp.
#
# d) Placement legality
# - The generator samples `N` first, then samples a top-left placement in `0..N-3`, so the 3x3
#   motif never crosses the border.
# - There is only one foreground object, so no overlap management is needed.
# - Foreground color is chosen from the same safe set excluding black, orange, and maroon.
#
# e) Visual observability
# - The task is learnable only if examples show both:
#   1. same glyph count appearing inside different input sizes but producing the same output side;
#   2. different glyph counts appearing inside the same input size but producing different sides.
# - To make that observable from a small set of examples, the generator cycles through a short
#   schedule of `(N, glyph)` cases instead of sampling both dimensions fully at random.
# - The rule depends only on visible geometry: input carrier size and number of painted cells.
#
# f) Visual cleanliness
# - The output still has exactly two visually clean layers: a full-canvas lattice and a centered
#   maroon stamp. No extra markers are added.
# - Even at the minimum 12x12 size there is a 3-cell border around the 6x6 center stamp, so the
#   lattice remains readable after stamping.
#
# g) Change inferability
# - No new color semantics are introduced. The input still uses orange plus one arbitrary marker
#   color, while the output still uses only black, orange, and maroon.
# - Because the output side values are driven by motif counts {4, 5, 7}, the count rule can be
#   inferred from repeated examples without relying on hidden metadata.
#
# Post-change solver assumptions to audit
# - The grid is square and has side at least 4.
# - Exactly one non-orange color is present.
# - The foreground bbox is exactly 3x3 and matches one of the four supported contour glyphs.
# - The motif foreground count is in {4, 5, 7}, which makes `side = 3 * count` one of {12, 15, 21}.
# The generator enforces all of these assumptions directly.


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
    """Map motif foreground count to the combo output side length."""
    count = int(np.count_nonzero(motif))
    assert count in {4, 5, 7}
    side = 3 * count
    assert side in {12, 15, 21}
    return side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a periodic lattice whose side length is encoded by the glyph's pixel count rather than by the surrounding input size.

    Input structure:
    - The grid is square, orange-background, and its side length can vary from example to example.
    - Exactly one non-orange color appears, forming a tight 3x3 bounding box that contains one contour glyph.
    - The glyph is one of the four legal contour forms: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 glyph as a code for the same 3x3 orange/black phase tile family as in the original task.
    - Count the glyph's foreground pixels and set the output canvas to `(3 * count, 3 * count)`, ignoring the input carrier size.
    - Tile the phase tile across that canvas, then place a centered 6x6 maroon stamp made by 2x2 tiling of the original glyph.

    Key insight:
    - The input now contains two independent geometric signals: the varying outer grid size is just context, while the glyph itself determines both the lattice phase and the final canvas size through its foreground pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h == w
    assert h >= 4

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
MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 9
MOTIF_SIZE = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

# This schedule deliberately exhibits the fused rule:
# - repeated counts across different N values (`plus` at 4, 7, and 9; `4corners` at 5 and 6)
# - repeated N values across different counts (N=6 with `H` and `4corners`; N=7 with `X` and `plus`)
CASE_SCHEDULE = (
    (4, "plus"),
    (9, "plus"),
    (6, "H"),
    (6, "4corners"),
    (7, "X"),
    (5, "4corners"),
    (8, "H"),
    (7, "plus"),
)
_CASE_INDEX = 0


def _next_case() -> tuple[int, str]:
    global _CASE_INDEX
    case = CASE_SCHEDULE[_CASE_INDEX % len(CASE_SCHEDULE)]
    _CASE_INDEX += 1
    return case


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n, motif_name = _next_case()
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    pos_max = n - MOTIF_SIZE + 1
    r0 = int(rng.integers(0, pos_max))
    c0 = int(rng.integers(0, pos_max))

    inp = np.full((n, n), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
        expected_side = 3 * int(np.count_nonzero(motif))
        assert out.shape == (expected_side, expected_side)
        assert out.shape != (3 * n - 2, 3 * n - 2)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 19")
