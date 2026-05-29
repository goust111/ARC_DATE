# Variant of eee78d87
# Dimension: D1 — 字形到相位瓦片的解码规则 (match_cond)
# Change: 4种轮廓字形（plus/H/X/4corners）各对应一种固定的3x3相位瓦片，通过角点与边中心像素的布尔组合确定橙色... → 字形位置编码相位偏移

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D1 variant
# a) Concrete behavior and trace:
#    - Input layout stays the same: a 6x6 orange canvas with exactly one 3x3 contour glyph
#      in one non-orange color, placed anywhere that fits.
#    - Output still has two layers: a 16x16 black/orange periodic lattice and a centered 6x6
#      maroon stamp made by tiling the 3x3 glyph 2x2.
#    - The changed step is the lattice phase. In the original task, a glyph directly determined
#      a fixed 3x3 phase tile and the tiling always started at offset (0,0). Here the glyph
#      still determines the same base 3x3 tile, but the glyph bbox top-left (r0,c0) shifts the
#      whole tiling by (r0 mod 3, c0 mod 3).
#    - Example trace: for a plus glyph, the base tile is still
#          000
#          011
#          011
#      If the plus is at input position (0,0), the lattice matches the original alignment. If
#      the same plus is at (1,2), the exact same tile is repeated with offset (1,2), so every
#      orange cell in the 16x16 background moves one row down and two columns right modulo 3.
# b) Complexity calibration:
#    - The rule now depends on 4 glyph classes and 9 phase-offset classes, which is richer than
#      the original but still small enough to infer from a few examples.
#    - The modulo-3 offset is visually strong because it shifts the entire 16x16 lattice while
#      leaving the centered maroon stamp unchanged.
# c) Implicit assumptions to audit:
#    - `bbox(fg)` must still isolate exactly one object; the generator keeps one glyph only.
#    - `crop_bbox(fg, b)` must still be 3x3; the generator only places the known 3x3 glyphs.
#    - `_decode_base_phase_tile` still assumes one of the four original contour families; that
#      remains true because the variant changes only the phase origin, not glyph semantics.
#    - `tile_fill(..., offset=...)` now matters. Using the bbox top-left directly is safe because
#      the glyph always fits inside the 6x6 grid and modulo-3 collapses positions to valid phases.
# d) Placement legality:
#    - The glyph bbox top-left must satisfy 0 <= r0,c0 <= 3, so the 3x3 motif never leaves the
#      6x6 input. There are no other objects, so overlap is impossible.
#    - The centered 6x6 maroon stamp in output is unchanged and cannot conflict with generation.
# e) Visual observability:
#    - The same glyph can now yield different outputs solely because of placement, which is
#      directly visible from the shifted orange/black background. This is inferable from 2-3
#      examples as long as multiple modulo classes appear.
# f) Visual cleanliness:
#    - The output still has only two clean layers: periodic background and centered maroon stamp.
#      No extra markers are introduced, so the scene remains legible.
# g) Color inferability:
#    - Colors are unchanged from the original task: orange/black for the lattice and maroon for
#      the center stamp. The new rule does not depend on hidden color ids, only on visible glyph
#      placement.


def _decode_base_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the original period-3 orange phase mask."""
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
    Turn the colored 3x3 contour glyph into a 16x16 periodic lattice whose phase is shifted by the glyph placement, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels form a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's four contour forms, and its bbox top-left position is meaningful.

    Rule:
    - Decode the 3x3 glyph into the same base period-3 orange phase tile as in the original task.
    - Compute a phase offset from the glyph bbox top-left: `(r0 % 3, c0 % 3)`.
    - Fill a black 16x16 canvas by repeating the decoded tile with that offset.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The contour still selects which 3x3 lattice tile is repeated, but the glyph's location now controls where the repeating orange pattern starts.
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

    phase = _decode_base_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    phase_offset = (int(b[0]) % 3, int(b[1]) % 3)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors, offset=phase_offset)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

POSITIONS_BY_PHASE = {
    0: (0, 3),
    1: (1,),
    2: (2,),
}


def _sample_position(rng: np.random.Generator) -> tuple[int, int]:
    """Sample actual 6x6 positions while keeping modulo-3 phase classes balanced."""
    r_phase = int(rng.integers(0, 3))
    c_phase = int(rng.integers(0, 3))
    r0 = int(rng.choice(POSITIONS_BY_PHASE[r_phase]))
    c0 = int(rng.choice(POSITIONS_BY_PHASE[c_phase]))
    return r0, c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    r0, c0 = _sample_position(rng)

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
    raise RuntimeError("Failed to generate input for eee78d87 D1/3 variant")
