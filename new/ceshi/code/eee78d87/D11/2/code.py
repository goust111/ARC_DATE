# Variant of eee78d87
# Dimension: D11 — 字形到印章的平铺倍数 (transform)
# Change: 字形2x2简单平铺（np.tile(motif,(2,2))）生成6x6印章块... → 像素放大而非平铺

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D11 variant
# spec_variant:
#   preserved:
#   - input remains a 6x6 orange grid with one 3x3 contour glyph in a single non-orange color
#   - the 3x3 glyph still decodes the same period-3 orange/black lattice on a 16x16 output
#   - glyph vocabulary stays {plus, H, X, 4corners}; no new object types are introduced
#   changed_core:
#   - only the center stamp changes: instead of repeating the 3x3 glyph as a 2x2 array, each glyph pixel is magnified
#     into a 2x2 maroon block, yielding a 6x6 coarse stamp
#   concrete_trace:
#   - original center for plus used np.tile on 010/111/010, producing rows 010010 / 111111 / 010010 repeated vertically
#   - new center uses pixel enlargement, producing 001100 / 001100 / 111111 / 111111 / 001100 / 001100
#   - both use the same 6x6 footprint, so the only visible difference is "repetition" versus "blocky scaling"
#   complexity_calibration:
#   - keeping the scale fixed at 2 avoids adding a second hidden control variable while still making the visual rule obvious
#   - all four glyphs remain distinguishable after enlargement, and none of them collapse to the old 2x2 tiling pattern
#   implicit_assumption_audit:
#   - g.shape == (6, 6), orange background, one foreground color, 3x3 tight bbox: still required and still guaranteed
#   - _decode_phase_tile depends only on the 3x3 boolean glyph, so it is unchanged
#   - the old line center = np.tile(motif, (2, 2)) depended on "stamp = repeated array"; this is the only invalidated assumption
#   - center placement by geometric centering remains valid because the enlarged stamp is also 6x6
#   placement_legality:
#   - generator still places a single 3x3 motif fully inside the 6x6 input, with top-left in 0..3
#   - enlarged stamp is drawn only in the centered 6x6 window of the 16x16 output, so it cannot exceed bounds
#   visual_observability:
#   - the difference is human-visible in 2-3 examples because repeated seams disappear and contiguous 2x2 maroon blocks appear
#   visual_cleanliness:
#   - the stamp still occupies one compact centered area over the same orange/black lattice, so no extra overlaps are introduced
#   color_inference:
#   - colors remain black/orange/maroon in the output; no new color mapping is introduced, so the only new inference is geometric
#   generator_validation:
#   - oracle validation still works by calling main(inp)
#   - anti-degeneration check should compare enlarged center versus legacy tiled center and confirm they differ for generated samples
#   new_main_assumptions:
#   - the new helper assumes a fixed enlargement factor of 2
#   - generate_input guarantees this is always meaningful because the motif is always exactly 3x3, so the stamp is always 6x6

STAMP_SCALE = 2


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


def _pixel_enlarge(mask: np.ndarray, scale: int) -> np.ndarray:
    """Expand each boolean pixel into a scale x scale solid block."""
    assert mask.ndim == 2
    assert scale >= 1
    block = np.ones((scale, scale), dtype=int)
    return np.kron(mask.astype(int), block).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a periodic orange lattice and stamp a magnified maroon version at the center.

    Input structure:
    - The input is a 6x6 orange field with exactly one non-orange color.
    - Those non-orange cells form one tight 3x3 contour glyph.
    - The glyph is one of four code shapes: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 contour glyph as the same period-3 orange/black phase tile used in the base task.
    - Tile that phase tile across a 16x16 black canvas.
    - Recolor the glyph to maroon and enlarge each of its pixels into a 2x2 solid block, producing a centered 6x6 stamp.

    Key insight:
    - The input glyph still plays two roles: it encodes the lattice phase, and it also provides the central emblem, but the emblem is now obtained by pixel scaling rather than by repeating the glyph array.
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
    out = tile_fill(out_shape, phase_colors)

    center = _pixel_enlarge(motif, STAMP_SCALE)
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
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
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
