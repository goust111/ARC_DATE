# Variant of eee78d87
# Dimension: D1 — 字形到相位瓦片的解码规则 (match_cond)
# Change: 4种轮廓字形（plus/H/X/4corners）各对应一种固定的3x3相位瓦片，通过角点与边中心像素的布尔组合确定橙色... → 字形像素直接标记橙色位置（正负反转）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input stays the same: a 6x6 orange field with exactly one 3x3 non-orange contour
#      glyph. The glyph is one of plus / H / X / 4corners and may sit anywhere that
#      keeps its 3x3 bbox inside the board.
#    - Output also keeps the same macro layout: a 16x16 tiled lattice plus a centered
#      6x6 maroon stamp made by tiling the 3x3 glyph 2x2.
#    - The only changed stage is the glyph -> 3x3 phase tile decode. Originally the
#      solver read corners and edge-centers, then mapped each of the 4 allowed glyphs to
#      a hand-written orange/black phase mask. In this variant the glyph mask itself is
#      the phase mask: motif pixels become orange(7), non-motif pixels become black(0).
#    - Example trace with plus:
#        original glyph mask = [[0,1,0],[1,1,1],[0,1,0]]
#        old phase tile      = [[0,0,0],[0,1,1],[0,1,1]]
#        new phase tile      = [[0,1,0],[1,1,1],[0,1,0]]
#      So the repeated lattice changes from an offset corner block to a direct cross.
#      This is the intended positive/negative inversion of the old decode logic.
# b) Complexity calibration
#    - The variant is simpler algorithmically, but not visually degenerate: the 4 glyphs
#      still produce 4 distinct period-3 lattices, and all of them differ from the
#      original task's lattices. The repeated background region therefore still carries
#      enough evidence for humans to infer the rule from a few examples.
#    - We keep the glyph family fixed at 4 patterns so the task does not explode into an
#      unrestricted "copy any 3x3 bitmap" problem.
# c) Implicit-assumption audit
#    - `g.shape == (6, 6)`: unchanged and still guaranteed by the generator.
#    - Background is orange and there is exactly one non-orange color: unchanged.
#    - `bbox(fg)` must exist and `crop_bbox(...).shape == (3, 3)`: still required, so the
#      generator must stamp exactly one 3x3 glyph and nothing else.
#    - The old `_decode_phase_tile` depended on contour-specific corner/edge assertions.
#      Those assumptions are removed because they are no longer needed for decoding. To
#      avoid silently accepting arbitrary 3x3 noise, `main()` instead checks that the
#      cropped mask equals one of the 4 known glyph templates.
#    - Center stamping still assumes the lattice can be overwritten by maroon on motif
#      pixels only. That remains safe because the stamp writes only inside the centered
#      6x6 window and uses the same mask as before.
# d) Placement legality
#    - The generator chooses `(r0, c0)` from 0..3 so the 3x3 glyph never leaves the 6x6
#      input board.
#    - There is only one object, so overlap cannot happen.
#    - Because the generator uses the exact known glyph masks, the solver's bbox and
#      template assertions are always satisfied.
# e) Visual observability
#    - Humans can see the change directly: outside the maroon center, orange cells now
#      appear exactly where the input glyph had foreground pixels, repeated every 3 cells.
#      The old task required a less obvious remapping, so the variant's inversion is easy
#      to spot from 2-3 examples.
#    - No hidden ordering by color id is used; the input foreground color remains
#      irrelevant except for separating the glyph from the orange background.
# f) Visual cleanliness
#    - The layout still has two clean layers: an orange/black periodic background and a
#      centered maroon overlay. The overlay is confined to a 6x6 block, so most of the
#      lattice remains visible for rule induction.
# g) Color-change inferability
#    - Output colors stay limited to black(0), orange(7), and maroon(9). Orange marks the
#      repeated positive glyph mask, black marks the complement inside each 3x3 tile, and
#      maroon marks the centered 2x2 glyph stamp. This structured reuse keeps the rule
#      easy to infer.


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_NAMES = tuple(GLYPHS.keys())
KNOWN_GLYPHS = tuple(GLYPHS.values())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Use the glyph mask itself as the 3x3 orange phase tile."""
    assert motif.shape == (3, 3)
    return motif.astype(bool, copy=True)


def _is_known_glyph(motif: np.ndarray) -> bool:
    return any(np.array_equal(motif, glyph) for glyph in KNOWN_GLYPHS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the colored 3x3 contour glyph into a 16x16 periodic lattice that uses the glyph itself as the orange phase tile, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels occupy a tight 3x3 bounding box containing one contour glyph.
    - The glyph is one of the task's four allowed forms: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 glyph as a boolean mask and use that mask directly as the period-3 orange tile.
    - Repeat this orange/black tile across a 16x16 canvas, with non-glyph positions rendered in black.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The input object is still a code rather than a copied shape, but in this variant the code is literal: foreground pixels directly specify where orange appears in each repeated 3x3 phase tile.
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
    assert _is_known_glyph(motif)

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
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(GLYPH_NAMES))
    motif = GLYPHS[motif_name]
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
    raise RuntimeError("Failed to generate input for eee78d87")
