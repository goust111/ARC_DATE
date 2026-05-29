# Combo variant of eee78d87 (index 14)
# combo_spec: D1/3+D9/0
# Name: 字形位置编码相位偏移+印章位置映射

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the fused D1/3 + D9/0 variant
# a) Concrete behavior and trace:
#    - Input remains a 6x6 orange field with exactly one 3x3 contour glyph in one non-orange color.
#    - Output still has a 16x16 orange/black lattice plus a maroon 2x2 stamp of the glyph.
#    - The shared control signal is the glyph bbox top-left (r0,c0).
#    - Example: a plus glyph at input origin (3,1) still decodes to the original plus base tile.
#      Original task: lattice starts at phase (0,0), stamp is fixed at center (5,5).
#      Fused task: lattice offset becomes (3 mod 3, 1 mod 3) = (0,1), while the 6x6 stamp origin
#      becomes (round(3 * 10 / 3), round(1 * 10 / 3)) = (10,3). The same visible position cue
#      therefore drives two different downstream effects.
# b) Complexity calibration:
#    - Glyph shape still contributes the original 4-way phase-tile identity.
#    - Placement now adds a 3x3 modulo phase class and a 4x4 scaled stamp-origin class.
#    - This is richer than either single variant but still human-inferable because both effects are
#      coarse and highly visible: the whole lattice shifts, and the large maroon stamp moves.
# c) Implicit assumptions to audit:
#    - `bbox(fg)` must isolate one object; generator keeps exactly one glyph.
#    - `crop_bbox(fg, b)` must be 3x3; generator only stamps the known contour glyphs.
#    - `_decode_base_phase_tile` still assumes one of the four original contour families; fusion
#      changes only how position is consumed, not the glyph vocabulary.
#    - `tile_fill(..., offset=...)` now depends on bbox top-left modulo 3.
#    - Stamp placement no longer assumes a constant center; it assumes the bbox origin is within
#      the legal 0..3 range so scaling into the 0..10 output-origin range stays in bounds.
# d) Placement legality:
#    - Input glyph top-left is always sampled in 0..3 for both axes, so the 3x3 bbox fits in 6x6.
#    - A 6x6 stamp inside a 16x16 output has legal top-left range 0..10. Scaling 0..3 -> 0..10
#      by rounding produces exactly {0,3,7,10}, so the stamp never leaves the canvas.
#    - Overlap between stamp and lattice is intentional; the stamp simply overwrites lattice cells.
# e) Visual observability:
#    - Two samples with the same glyph but different positions visibly differ in two ways at once.
#    - Positions 0 and 3 share modulo class 0, so they keep the same lattice phase but map the
#      stamp to opposite sides of the output. Positions 1 and 2 change both phase and stamp.
#    - This makes it possible to infer that the same input location feeds two separate rules.
# f) Visual cleanliness:
#    - The scene still contains only two large layers: a periodic orange/black background and one
#      solid maroon stamp. No extra markers or hidden encodings are introduced.
#    - The scaled stamp destinations {0,3,7,10} are widely separated, so stamp movement is easy to read.
# g) Change inferability:
#    - No new color mapping is introduced. Orange/black still encode the lattice and maroon still
#      encodes the stamp, so the only new inference burden is geometric: one position controls two outputs.


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


def _scale_origin(src: int, src_limit: int, dst_limit: int) -> int:
    """Map a discrete legal origin from one range to another by proportional scaling."""
    src = int(src)
    src_limit = int(src_limit)
    dst_limit = int(dst_limit)
    assert 0 <= src <= src_limit
    if src_limit == 0:
        return 0
    return int(round(src * dst_limit / src_limit))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice whose phase and maroon stamp position both depend on the motif's input placement.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one contour glyph with a tight 3x3 bounding box.
    - The glyph is one of the four original contour codes, and its bbox top-left position is meaningful.

    Rule:
    - Decode the 3x3 glyph into the same base period-3 orange phase tile as in the original task.
    - Read the glyph bbox top-left `(r0, c0)` and use `(r0 % 3, c0 % 3)` as the lattice phase offset.
    - Fill a black 16x16 canvas by repeating the decoded tile with that offset.
    - Tile the original 3x3 glyph 2x2, recolor its True cells to maroon, scale `(r0, c0)` from the
      input origin range onto the legal 6x6-stamp origin range in the 16x16 output, and place the stamp there.

    Key insight:
    - The same visible glyph location is consumed twice: modulo 3 decides where the orange lattice starts,
      while proportional scaling decides where the maroon stamp lands.
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
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    phase_offset = (int(b[0]) % 3, int(b[1]) % 3)
    out = tile_fill(out_shape, phase_colors, offset=phase_offset)

    stamp = np.tile(motif, (2, 2))
    in_r_limit = g.shape[0] - motif.shape[0]
    in_c_limit = g.shape[1] - motif.shape[1]
    out_r_limit = out.shape[0] - stamp.shape[0]
    out_c_limit = out.shape[1] - stamp.shape[1]
    stamp_r0 = _scale_origin(int(b[0]), in_r_limit, out_r_limit)
    stamp_c0 = _scale_origin(int(b[1]), in_c_limit, out_c_limit)
    out[stamp_r0 : stamp_r0 + stamp.shape[0], stamp_c0 : stamp_c0 + stamp.shape[1]][stamp] = Color.MAROON
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


def _sample_origin(rng: np.random.Generator) -> tuple[int, int]:
    """
    Sample legal input origins while keeping modulo-3 phase classes active.

    The phase offset only depends on modulo 3, but the stamp mapping distinguishes 0 from 3.
    Sampling phase first keeps nonzero offsets common, and sampling within the phase-0 bucket
    preserves the D9 contrast between same-phase origins 0 and 3.
    """
    r_phase = int(rng.integers(0, 3))
    c_phase = int(rng.integers(0, 3))
    r0 = int(rng.choice(POSITIONS_BY_PHASE[r_phase]))
    c0 = int(rng.choice(POSITIONS_BY_PHASE[c_phase]))
    return r0, c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    r0, c0 = _sample_origin(rng)

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
    raise RuntimeError("Failed to generate input for eee78d87 combo D1/3+D9/0")
