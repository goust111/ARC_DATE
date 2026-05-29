# Variant of eee78d87
# Dimension: D9 — 中心印章的放置位置 (dst)
# Change: 印章固定放置在16x16画布正中心(5,5)-(10,10)... → 印章位置由字形在输入中的位置映射

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

"""
design_analysis:
  concretization:
    input:
      - 6x6 orange background.
      - Exactly one non-orange contour glyph with a tight 3x3 bbox.
      - The glyph bbox top-left is the visible control signal for D9 and can be at any row/col in 0..3.
    output:
      - Keep the original 16x16 orange/black period-3 lattice.
      - Keep the original 2x2 maroon stamp of the 3x3 glyph, so the stamp is still 6x6.
      - Change only the stamp destination: its 6x6 top-left is no longer fixed at (5,5); instead it is scaled from the input bbox top-left.
  behavior_trace:
    example:
      - If the glyph bbox is at input top-left (3,0), the base task would still stamp at output (5,5).
      - In this variant, the legal input origin range is 0..3 while the legal output stamp-origin range is 0..10.
      - So row and col are mapped independently by proportional scaling: round(src * 10 / 3).
      - The same example therefore moves the stamp to output (10,0), i.e. bottom-left.
  complexity_calibration:
    - The lattice-decoding logic stays unchanged, so the task still has the original shape-decoding core.
    - D9 adds a second visible cue with four row levels and four column levels, yielding 16 possible stamp placements.
    - The mapped output origins are {0,3,7,10}; these are far apart enough to be obvious in 16x16 outputs.
  implicit_assumption_review:
    - Background=orange and one foreground color are still valid assumptions.
    - Tight 3x3 bbox is still required because the phase decoder only accepts 3x3 motifs.
    - The old centering code depended on the stamp origin being constant; that is the only obsolete assumption.
    - Overwriting lattice cells with maroon is still safe because the original task already allowed the stamp to cover the lattice.
  placement_legality:
    - Input generation keeps the motif within 6x6 by choosing top-left in 0..3.
    - Output placement is always legal because scaling 0..3 onto 0..10 keeps the 6x6 stamp inside the 16x16 canvas.
    - There are no extra objects, so no new non-overlap constraints are needed.
  observability_and_tidiness:
    - The only changed visual variable is stamp position; colors and glyph families remain unchanged, making the rule easier to isolate.
    - A left/right or top/bottom shift is immediately visible because the maroon stamp is large and contrasts with the orange/black lattice.
  new_solver_assumptions:
    - The solver now assumes the bbox top-left, not the glyph color, determines the stamp destination.
    - The generator guarantees this remains readable by sampling all motif positions rather than collapsing to one fixed area.
"""


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
    """Map a discrete origin from one legal range to another by proportional scaling."""
    src = int(src)
    src_limit = int(src_limit)
    dst_limit = int(dst_limit)
    assert 0 <= src <= src_limit
    if src_limit == 0:
        return 0
    return int(round(src * dst_limit / src_limit))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a 16x16 periodic lattice and place the maroon 2x2 stamp at a location derived from the glyph's input position.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one contour glyph with a tight 3x3 bounding box.
    - The glyph is one of the four original contour codes, and its bbox top-left can appear anywhere from row/col 0..3.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the base task.
    - Fill a black 16x16 canvas by repeating that orange/black phase tile.
    - Tile the original 3x3 glyph 2x2, recolor its True cells to maroon, and treat that as a 6x6 stamp.
    - Map the glyph bbox top-left from the input placement range onto the output stamp-placement range, independently by row and column, then place the stamp there.

    Key insight:
    - The glyph's shape still chooses the lattice phase, but the glyph's position in the 6x6 input now also encodes where the maroon stamp belongs on the 16x16 output.
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

    stamp = np.tile(motif, (2, 2))
    in_r_limit = g.shape[0] - motif.shape[0]
    in_c_limit = g.shape[1] - motif.shape[1]
    out_r_limit = out.shape[0] - stamp.shape[0]
    out_c_limit = out.shape[1] - stamp.shape[1]
    r0 = _scale_origin(b[0], in_r_limit, out_r_limit)
    c0 = _scale_origin(b[1], in_c_limit, out_c_limit)
    out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]][stamp] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6                        # input grid dimensions
MOTIF_SIZE = 3                               # glyph bounding box side length
MOTIF_POS_MAX = 4                            # exclusive upper bound for top-left position (0..3)
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]           # valid foreground colors (exclude 0, 7, 9)
MAX_RETRIES = 300                            # oracle-validation retry budget

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
    raise RuntimeError("Failed to generate input for eee78d87 D9 variant")
