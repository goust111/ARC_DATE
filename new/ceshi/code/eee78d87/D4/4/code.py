# Variant of eee78d87
# Dimension: D4 — 栗色重着色变换 (transform)
# Change: 中心6x6区域中字形像素位置统一重着色为栗色(9)，非字形位置保持格栅原样
#         → 中心6x6区域中字形像素位置改为复制输入字形的前景色，非字形位置仍保持格栅原样
"""
Design analysis:
  variant_core:
    - Input geometry is unchanged: a 6x6 orange field contains exactly one 3x3 contour glyph
      in a single non-orange foreground color.
    - Output geometry is unchanged: decode the glyph into a period-3 orange/black tile,
      repeat it across a 16x16 canvas, and place a 2x2 repetition of the glyph in the
      centered 6x6 window.
    - The only rule change is the stamp color. The original task overwrote all glyph pixels
      in the center window with fixed maroon(9). This variant transfers the visible input
      glyph color directly to the center stamp.

  behavior_trace:
    - Example: if the input glyph is blue plus-shaped, the lattice is decoded exactly as
      before from the plus contour.
    - Original behavior at the last step: center glyph pixels become maroon regardless of
      blue input.
    - Variant behavior at the last step: center glyph pixels stay blue, because the stamp
      color is copied from the only non-orange input color.
    - Difference: shape controls the lattice; color now controls the center recolor target.

  complexity_calibration:
    - This keeps the task close to the original: no new objects, no new placement logic, no
      extra branches in phase decoding.
    - The color-transfer rule is still visually meaningful because every sample shows the
      same 6x6 center structure but in a different color tied to the input glyph.
    - Identity transfer is intentionally simple: with only 2-3 training pairs, a human can
      infer that the center stamp color matches the input glyph color.

  implicit_assumptions_review:
    - `bg == ORANGE` still holds and is enforced.
    - `len(colors_present(exclude={bg})) == 1` still holds; this single color is now also
      the transferred stamp color.
    - `motif = crop_bbox(g != bg, b)` remains valid because lattice decoding depends only on
      the glyph mask, not on its color value.
    - The old write `... = Color.MAROON` depended on a constant target color; that exact
      assumption is what the variant changes, so it must become a dynamic transferred color.
    - If the transferred color were black or orange, the center stamp could visually merge
      with the lattice/background; the generator therefore excludes those colors.

  placement_legality:
    - The 3x3 motif is still placed with top-left row/col in 0..3, so it always fits inside
      the 6x6 input.
    - The output center stamp still occupies the original centered 6x6 window, so no new
      overlap or boundary case is introduced.

  visual_observability:
    - The changed behavior is easy to spot: blue input gives a blue center stamp, green
      input gives a green center stamp, etc.
    - The rule uses direct visible color copying, not hidden arithmetic on color ids.
    - The lattice region and the center stamp remain visually separated because the lattice
      uses only black/orange while the transferred stamp uses one other color.

  generator_guarantees:
    - Foreground colors are sampled from {1,2,3,4,5,6,8}, excluding black, orange, and
      maroon.
    - Excluding maroon ensures generated samples do not collapse back to the original fixed
      maroon behavior, so the D4 change stays observable.
"""

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


def _stamp_color_from_fg(fg_color: int) -> int:
    """Transfer the input glyph's visible foreground color to the center stamp."""
    fg_color = int(fg_color)
    assert fg_color not in (Color.BLACK, Color.ORANGE)
    return fg_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped at the center using the glyph's own color.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange foreground color.
    - The non-orange pixels form a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the period-3 orange phase tile.
    - Fill a black 16x16 canvas by repeating that orange phase tile.
    - Tile the original 3x3 glyph 2x2 into the centered 6x6 region and recolor its true pixels with the input glyph's foreground color.

    Key insight:
    - The glyph's shape determines the orange/black lattice, while the glyph's color is transferred separately to the centered stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    stamp_color = _stamp_color_from_fg(fg_colors[0])

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    stamp = out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]]
    stamp[center] = stamp_color
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
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
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
    raise RuntimeError("Failed to generate input for eee78d87 D4 variant")
