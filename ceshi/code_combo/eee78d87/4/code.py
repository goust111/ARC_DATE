# Combo variant of eee78d87 (index 4)
# combo_spec: D3/1+D9/0
# Name: 多印章分布+位置映射

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, paste, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   preserved:
#     - input stays a 6x6 orange grid with exactly one non-orange contour glyph
#     - the glyph family and phase decoding logic are unchanged
#     - output stays a 16x16 orange/black lattice derived from the decoded 3x3 phase tile
#   fused_changes:
#     - D3/1 contributes a repeated 2x2 stamp arrangement instead of one centered stamp
#     - D9/0 contributes position control from the input glyph bbox top-left
#     - the fusion is not "move each stamp independently"; instead the four-stamp arrangement
#       is treated as one regular group whose anchor is shifted by the mapped input position
# concretization:
#   - a 6x6 input glyph bbox can start at row/col 0..3
#   - a maroon stamp is always the 2x2 tiling of the 3x3 glyph, so each stamp is 6x6
#   - the D3 layout uses two stamp slots per axis with fixed spacing 8, i.e. relative offsets [0, 8]
#   - on a 16-cell axis, that 2-stamp group spans 14 cells, so its left/top anchor can only be 0, 1, or 2
#   - D9 therefore scales each input coordinate 0..3 onto anchor range 0..2 by round(src * 2 / 3),
#     yielding visible anchor levels [0, 1, 1, 2]
# behavior_trace:
#   example:
#     - suppose the glyph is plus-shaped and its bbox top-left is at input (3, 0)
#   original_task:
#     - plus decodes the same orange phase tile as always
#     - output gets one centered 6x6 maroon stamp at (5, 5)
#   D3-only:
#     - output gets four stamps at fixed origins (1,1), (1,9), (9,1), (9,9)
#   fused_combo:
#     - row 3 maps to anchor 2 and col 0 maps to anchor 0
#     - row origins become [2, 10], col origins become [0, 8]
#     - four stamps appear at (2,0), (2,8), (10,0), (10,8)
#   visible_difference:
#     - compared with D3-only, the 2x2 stamp lattice is still evenly spaced, but the whole group shifts
#     - compared with D9-only, there is no single privileged stamp; the same mapped offset controls all four
# complexity_calibration:
#   - there are still only four glyph shapes and one foreground color, so shape decoding remains readable
#   - each axis now has three output anchor levels, producing nine distinct group placements overall
#   - the mapped levels are coarse and repeated enough to infer from a few examples without hidden metadata
# implicit_assumption_review:
#   - get_bg/colors_present still assume orange background and one foreground color; generator preserves that
#   - bbox/crop_bbox still require one tight 3x3 glyph bbox; generator places exactly one glyph
#   - _decode_phase_tile still assumes the original four contour codes; unchanged
#   - the old "single centered stamp" assumption is removed
#   - the D3 helper's fixed spacing assumption is retained, but only as intra-group spacing, not absolute position
#   - new assumption: the whole repeated-stamp group must fit after shifting; helper code computes legal anchor range
# placement_legality:
#   - for axis_len=16 and stamp_len=6, relative offsets [0, 8] imply total group span 14
#   - anchor range 0..2 guarantees origins [anchor, anchor+8] stay in-bounds and non-overlapping
#   - rows/cols are mapped independently, so every sampled input position yields one legal 2x2 placement grid
# visual_observability:
#   - maroon stamps are large and repeated, making the regular spacing easy to see
#   - the group shift is also easy to see because the outer border changes from 0/1/2 cells on opposite sides
# visual_cleanliness:
#   - stamps only overwrite the orange/black lattice, never one another
#   - the 2-cell gap between same-axis stamps is preserved under every legal anchor
# inference_safety:
#   - no color identity or numeric palette ordering is used as a hidden control variable
#   - only shape and visible input position matter, both directly observable in the 6x6 input
# generator_alignment:
#   - random glyph positions still cover all input origins 0..3
#   - because scaling 0..3 -> 0..2 is many-to-one, some centered outputs remain, but edge-shifted cases
#     appear frequently enough to prevent collapse back to the D3-only pattern


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


def _scale_origin(src: int, src_limit: int, dst_limit: int) -> int:
    """Map a discrete source origin into a discrete destination origin range by proportional scaling."""
    src = int(src)
    src_limit = int(src_limit)
    dst_limit = int(dst_limit)
    assert 0 <= src <= src_limit
    assert dst_limit >= 0
    if src_limit == 0:
        return 0
    return int(round(src * dst_limit / src_limit))


def _regular_slot_origins(axis_len: int, stamp_len: int) -> list[int]:
    """Return the base 2-slot regular lattice used by the repeated-stamp variant."""
    axis_len = int(axis_len)
    stamp_len = int(stamp_len)
    outer_margin = 1
    gap = axis_len - 2 * outer_margin - 2 * stamp_len
    assert gap >= 0
    step = stamp_len + gap
    origins = list(range(outer_margin, axis_len - stamp_len, step))
    assert len(origins) == 2
    return origins


def _mapped_group_origins(src_origin: int, src_limit: int, axis_len: int, stamp_len: int) -> list[int]:
    """
    Shift the whole repeated-stamp group by mapping the input glyph position onto the legal group-anchor range.

    For the task dimensions this yields two evenly spaced origins per axis:
    base [1, 9] -> offsets [0, 8], anchor in {0,1,2} -> mapped origins [0,8], [1,9], or [2,10].
    """
    base = _regular_slot_origins(axis_len, stamp_len)
    offsets = [origin - base[0] for origin in base]
    group_span = offsets[-1] + stamp_len
    anchor_limit = int(axis_len) - group_span
    assert anchor_limit >= 0

    anchor = _scale_origin(src_origin, src_limit, anchor_limit)
    origins = [anchor + offset for offset in offsets]

    assert len(origins) == 2
    assert all(0 <= origin and origin + stamp_len <= axis_len for origin in origins)
    assert origins[1] - origins[0] == offsets[1] - offsets[0]
    return origins


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a 16x16 periodic lattice and stamp a shifted 2x2 maroon stamp array.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one contour glyph with a tight 3x3 bounding box.
    - The glyph is one of the four original contour codes, and its bbox top-left can appear anywhere from row/col 0..3.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the base task.
    - Fill a black 16x16 canvas by repeating that orange/black phase tile.
    - Tile the original 3x3 glyph 2x2 to form a 6x6 maroon stamp.
    - Place four copies of that stamp in a regular 2x2 arrangement with fixed spacing, but map the input bbox row and column onto the legal anchor range of that whole arrangement so the four-stamp group shifts together.

    Key insight:
    - The glyph carries two independent visible signals: its shape still determines the orange phase lattice, while its input position now controls the global offset of an evenly spaced four-stamp maroon array.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    stamp_mask = np.tile(motif, (2, 2))
    stamp = np.where(stamp_mask, Color.MAROON, Color.BLACK).astype(int)
    in_r_limit = g.shape[0] - motif.shape[0]
    in_c_limit = g.shape[1] - motif.shape[1]
    row_origins = _mapped_group_origins(b[0], in_r_limit, out.shape[0], stamp.shape[0])
    col_origins = _mapped_group_origins(b[1], in_c_limit, out.shape[1], stamp.shape[1])

    for r0 in row_origins:
        for c0 in col_origins:
            paste(out, stamp, r0, c0, transparent=Color.BLACK)

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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant D3/1+D9/0")
