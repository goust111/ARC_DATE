# Combo variant of eee78d87 (index 39)
# combo_spec: D2/0+D6/2
# Name: 大尺寸字形+周期4瓦片

from __future__ import annotations

# Design analysis
# ---------------
# a) Concrete behavior:
#    - Original task: a 6x6 orange input contains one 3x3 contour glyph.  That
#      glyph is decoded into a 3x3 orange/black phase tile, repeated on a 16x16
#      canvas, and the same 3x3 glyph is tiled 2x2 as a centered 6x6 maroon
#      stamp.
#    - Fused task: keep the same one-object scene and the same four contour
#      families, but enlarge the visible codeword itself to 4x4 by stretching the
#      coarse 3x3 family mask with row/col bands [1,2,1].  The decoded phase tile
#      is enlarged by the same banding, so the lattice period becomes 4 in lockstep
#      with the glyph size.
#    - Small trace for the plus family:
#        original motif: .#./###/.#.
#        original phase: ... /.OO /.OO
#        fused motif:    .##./####/####/.##.
#        fused phase:    .... /.OOO /.OOO /.OOO
#      The difference is visible in both the larger 4x4 input codeword and the
#      denser period-4 lattice behind the centered maroon stamp.
#
# b) Complexity calibration:
#    - I keep exactly four glyph families rather than inventing many unrelated 4x4
#      symbols.  That gives repeated support across examples and preserves the
#      original family semantics.
#    - The size freedom from standalone D2/0 is intentionally collapsed to one
#      value, 4x4, because this combo is specifically "large glyph + period-4
#      tile".  Allowing 5x5 here would break the intended synchronization between
#      glyph scale and lattice period.
#    - The changed degrees of freedom are therefore limited and legible: family
#      selects the phase pattern, while the fused size/period switch is always 4.
#
# c) Implicit assumption audit:
#    - `g.shape == (6, 6)` still holds, because a single 4x4 glyph still fits in
#      the original 6x6 input with visible position variation.
#    - `motif.shape == (3, 3)` is no longer valid; it must become `(4, 4)`.
#    - The original `_decode_phase_tile()` read exact 3x3 pixels.  That logic would
#      be brittle on a 4x4 surface, so the fused solver first identifies which
#      canonical 4x4 family is present, then decodes the corresponding coarse 3x3
#      family and expands the phase to 4x4.
#    - `out_shape = (3 * g.shape[0] - 2, ...)` is no longer justified, because the
#      combo changes tile period without changing input canvas size.  The output is
#      fixed explicitly at 16x16, matching the original task's canvas extent.
#    - `center = np.tile(motif, (2, 2))` still holds, but now the centered maroon
#      stamp is 8x8 instead of 6x6.
#
# d) Placement legality:
#    - A 4x4 motif inside a 6x6 grid fits only at top-left rows/cols 0..2.
#    - The generator samples exactly one motif, so there is no overlap ambiguity.
#    - Every canonical 4x4 glyph touches all four sides of its bbox, so `bbox()`
#      recovers the intended 4x4 crop exactly.
#
# e) Visual observability:
#    - A human can see that the input glyph is larger than in the base task, and
#      the output background repeats every 4 cells rather than every 3.
#    - The four contour families remain visually distinct after expansion, so a few
#      examples are enough to align family -> phase-pattern correspondences.
#    - No hidden use of numeric color ids is involved; the rule is entirely carried
#      by visible contour geometry, repetition period, and centered restamping.
#
# f) Visual cleanliness:
#    - The output still has only two clearly separated layers: a black/orange
#      lattice and one centered maroon stamp.
#    - On a 16x16 canvas, the 8x8 stamp leaves a 4-cell border on each side, so the
#      period-4 lattice remains readable around the center.
#
# g) Change inferability:
#    - The foreground input color stays arbitrary-but-singleton, so color does not
#      add an extra inference burden beyond marking the glyph.
#    - Because glyph size and lattice period always change together, examples show a
#      simple structural rule: "read the enlarged 4x4 contour code, then repeat its
#      matching enlarged phase tile."
#
# Post-change solver assumptions checked against the generator:
#    - The foreground mask must match exactly one of the four canonical 4x4 glyphs.
#    - The foreground bbox must be exactly 4x4.
#    - There must be exactly one non-orange color in the input.
#    - The generator enforces all of these by sampling from the same glyph table and
#      oracle-checking every constructed input with `main()`.

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 4
OUTPUT_H, OUTPUT_W = 16, 16
MOTIF_POS_MAX = GRID_H - MOTIF_SIZE + 1
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
_BANDS = (1, 2, 1)

COARSE_GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _expand_121(mask: np.ndarray) -> np.ndarray:
    """Expand a 3x3 boolean mask to 4x4 using row/col band widths [1,2,1]."""
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (3, 3)

    out = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    r0 = 0
    for r, rh in enumerate(_BANDS):
        c0 = 0
        for c, cw in enumerate(_BANDS):
            if bool(mask[r, c]):
                out[r0 : r0 + rh, c0 : c0 + cw] = True
            c0 += cw
        r0 += rh
    return out


GLYPHS = {name: _expand_121(mask) for name, mask in COARSE_GLYPHS.items()}


def _decode_phase_tile_3(motif: np.ndarray) -> np.ndarray:
    """Decode the original 3x3 contour glyph into the original period-3 phase mask."""
    motif = np.asarray(motif, dtype=bool)
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


def _identify_glyph(motif: np.ndarray) -> str:
    motif = np.asarray(motif, dtype=bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    raise AssertionError("Unrecognized 4x4 contour glyph")


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 4x4 contour glyph into the matching period-4 phase mask."""
    glyph_name = _identify_glyph(motif)
    coarse_phase = _decode_phase_tile_3(COARSE_GLYPHS[glyph_name])
    phase = _expand_121(coarse_phase)
    assert phase.shape == (MOTIF_SIZE, MOTIF_SIZE)
    return phase


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the enlarged 4x4 contour glyph into a period-4 orange/black lattice and stamp a 2x2 maroon copy of that glyph at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels have a tight 4x4 bbox containing one enlarged contour glyph.
    - The glyph is one of four canonical 4x4 forms obtained by stretching the original contour families with [1,2,1] row/col bands.

    Rule:
    - Identify which 4x4 contour family the motif belongs to.
    - Decode that family's original 3x3 phase meaning, then enlarge the phase tile with the same [1,2,1] banding so glyph scale and lattice period are both 4.
    - Fill a black 16x16 canvas by repeating that period-4 tile, then tile the original 4x4 glyph 2x2 and recolor its true pixels to maroon in the centered 8x8 region.

    Key insight:
    - The colored object is still a compact codeword rather than a picture to preserve in place; the fusion ties two dimensions together, so the same enlargement that makes the glyph 4x4 also makes the decoded lattice repeat every 4 cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((OUTPUT_H, OUTPUT_W), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (OUTPUT_H - center.shape[0]) // 2
    c0 = (OUTPUT_W - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
