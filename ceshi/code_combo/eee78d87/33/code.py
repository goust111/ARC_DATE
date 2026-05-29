# Combo variant of eee78d87 (index 33)
# combo_spec: D3/1+D4/3+D11/0
# Name: 多印章+旋转拼接+大平铺

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, paste, tile_fill


# Design analysis for the fused variant
# -------------------------------------
# fused_spec:
#   preserved:
#     - input stays a 6x6 orange grid containing exactly one non-orange 3x3 contour glyph
#     - the glyph vocabulary remains the original four codes: plus / H / X / 4corners
#     - the glyph still decodes the same 3x3 orange/black phase tile used to fill the background lattice
#   fused_changes:
#     - D11/0 enlarges the stamp from 6x6 to 9x9 by using a 3x3 arrangement of 3x3 motif copies
#     - D4/3 replaces plain repetition with a rotation mosaic: each of the 9 sub-blocks is a rotated copy
#       of the motif, using clockwise quarter-turn counts
#           0 1 2
#           3 0 1
#           2 3 0
#       where 0=original, 1=90 deg cw, 2=180 deg, 3=270 deg cw
#     - D3/1 distributes that larger 9x9 stamp to multiple evenly spaced output slots
#   physical_feasibility:
#     - a 16x16 canvas cannot host four non-overlapping 9x9 stamps, so keeping the old output size would
#       force heavy overlap and visually collapse the D3 change
#     - the smallest clean square layout for a 2x2 lattice of 9x9 stamps with a 1-cell outer border and
#       a 1-cell gutter is 21x21 = 1 + 9 + 1 + 9 + 1
#     - the fused output is therefore a 21x21 orange/black lattice with four 9x9 maroon stamps at
#       origins (1,1), (1,11), (11,1), (11,11)
# behavior_trace:
#   example_glyph:
#     - use the H glyph:
#         101
#         111
#         101
#   original_flow:
#     - decode H into its period-3 phase tile
#     - tile that phase across a 16x16 black canvas
#     - place one centered 6x6 stamp made by simple 2x2 repetition of H
#   fused_flow:
#     - decode H into exactly the same phase tile as before
#     - tile the phase across a 21x21 black canvas
#     - build one 9x9 stamp from 9 local 3x3 copies of H, rotated according to the turn map above
#     - place that same 9x9 stamp in a regular 2x2 lattice of output slots
#   visible_difference:
#     - the output is larger
#     - there are four stamps instead of one
#     - each stamp has internal orientation structure instead of simple repetition
# complexity_calibration:
#   - the rule stays discrete and low-branching: one fixed output size, one fixed 2x2 slot lattice,
#     one fixed 3x3 turn map
#   - the fusion is still visually strong because all three changed dimensions alter obvious geometry:
#     count of stamps, stamp size, and internal orientation
#   - the turn map is intentionally regular rather than data-dependent, so the solver need not infer any
#     hidden per-color or per-position lookup
# implicit_assumption_review:
#   - `g.shape == (6, 6)`, orange background, exactly one non-orange color, and a tight 3x3 bbox are still
#     required; the generator preserves all of them
#   - `_decode_phase_tile` depends only on the boolean 3x3 glyph mask, so it remains unchanged
#   - the old `out_shape = (3*H-2, 3*W-2)` assumption is no longer valid, because that 16x16 canvas is too
#     small for D3/1 + D11/0 together
#   - the old centered single-stamp placement assumption is removed; placement now depends on a fixed
#     two-slot lattice per axis
#   - the new rotation assembly assumes the motif is exactly 3x3; generator guarantees this via the same
#     input format as the source task
# placement_legality:
#   - each stamp is 9x9
#   - row/col origins [1, 11] yield occupied intervals [1,9] and [11,19]
#   - stamps stay in bounds, never overlap, and leave row/col 0 and 20 as a clean border
# visual_observability:
#   - multiple 9x9 maroon regions are obvious in every output
#   - the enlarged stamp size is obvious because each region is much larger than the original 6x6 center block
#   - the rotation mosaic is especially visible on H samples; the generator biases toward H so the D4 rule
#     is frequently represented rather than disappearing into rotationally symmetric glyphs
# visual_cleanliness:
#   - maroon overlays only the orange/black lattice
#   - the 1-cell gutters cleanly separate the four stamp regions
#   - within each stamp, all 9 local 3x3 copies align to the same block grid, so the pattern remains readable
# color_inference:
#   - output colors stay simple and structural: black/orange lattice plus maroon stamps only
#   - no color-id arithmetic or hidden palette ordering is introduced
# generator_validation:
#   - oracle validation still calls `main(inp)` directly
#   - anti-degeneration should measure both the count of stamp regions and whether any rotated 3x3 block
#     differs from the unrotated motif; H examples score positively on the latter
# new_main_assumptions:
#   - output size is fixed at 21x21
#   - stamp slots are fixed at a 2x2 lattice
#   - the 3x3 turn map is fixed and shared by all samples

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

STAMP_TILE_SIDE = 3
STAMP_SIZE = MOTIF_SIZE * STAMP_TILE_SIDE
OUTER_MARGIN = 1
STAMP_GUTTER = 1
OUTPUT_SIZE = OUTER_MARGIN + STAMP_SIZE + STAMP_GUTTER + STAMP_SIZE + OUTER_MARGIN

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

# Bias toward H so the rotation mosaic is often visibly different from plain repetition.
GLYPH_NAMES = ["plus", "H", "X", "4corners"]
GLYPH_WEIGHTS = np.array([1.0, 13.0, 1.0, 1.0], dtype=float)
ROTATION_TURNS_CW = np.array(
    [
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0],
    ],
    dtype=int,
)


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


def _build_rotary_stamp_mask(motif: np.ndarray) -> np.ndarray:
    """Assemble a 9x9 stamp from a 3x3 grid of rotated 3x3 motif copies."""
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)
    stamp = np.zeros((STAMP_SIZE, STAMP_SIZE), dtype=bool)
    for block_r in range(STAMP_TILE_SIDE):
        for block_c in range(STAMP_TILE_SIDE):
            turns_cw = int(ROTATION_TURNS_CW[block_r, block_c])
            block = np.rot90(motif, -turns_cw)
            r0 = block_r * MOTIF_SIZE
            c0 = block_c * MOTIF_SIZE
            stamp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE] = block
    return stamp


def _stamp_origins(axis_len: int, stamp_len: int) -> list[int]:
    """Return the two regular slot origins for the 2x2 stamp lattice."""
    axis_len = int(axis_len)
    stamp_len = int(stamp_len)
    gap = axis_len - 2 * OUTER_MARGIN - 2 * stamp_len
    assert gap >= 0
    origins = [OUTER_MARGIN, OUTER_MARGIN + stamp_len + gap]
    assert len(origins) == 2
    assert origins[1] + stamp_len == axis_len - OUTER_MARGIN
    return origins


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a larger periodic lattice and stamp a rotated 9x9 mosaic at multiple output positions.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels form one tight 3x3 contour glyph bbox.
    - The glyph is one of the original period-3 contour codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same orange/black phase tile as in the base task.
    - Fill a black 21x21 canvas by repeating that phase tile.
    - Build a 9x9 maroon stamp from a 3x3 arrangement of the glyph, rotating each local 3x3 copy by the fixed turn map.
    - Place that same 9x9 stamp at every slot of a regular 2x2 output lattice.

    Key insight:
    - The input glyph still controls the orange lattice phase, but its maroon imprint is no longer one centered plain repetition:
      it is enlarged to a 9x9 rotation mosaic and then distributed to multiple evenly spaced stamp zones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((OUTPUT_SIZE, OUTPUT_SIZE), phase_colors)

    stamp_mask = _build_rotary_stamp_mask(motif)
    stamp = np.where(stamp_mask, Color.MAROON, Color.BLACK).astype(int)
    row_origins = _stamp_origins(out.shape[0], stamp.shape[0])
    col_origins = _stamp_origins(out.shape[1], stamp.shape[1])
    for r0 in row_origins:
        for c0 in col_origins:
            paste(out, stamp, r0, c0, transparent=Color.BLACK)

    return out


def _choose_motif(rng: np.random.Generator) -> np.ndarray:
    """Sample a glyph with a bias toward the non-90-symmetric H pattern."""
    idx = int(rng.choice(len(GLYPH_NAMES), p=GLYPH_WEIGHTS / GLYPH_WEIGHTS.sum()))
    return GLYPHS[GLYPH_NAMES[idx]]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = _choose_motif(rng)
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 33")
