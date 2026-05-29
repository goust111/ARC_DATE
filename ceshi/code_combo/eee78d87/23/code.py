# Combo variant of eee78d87 (index 23)
# combo_spec: D9/2+D11/0
# Name: 印章对齐瓦片边界+3x3平铺大印章

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Input stays structurally simple: one 3x3 contour glyph inside a 6x6 orange
#      field. The glyph family is still {plus, H, X, 4corners}.
#    - Original task:
#      decode glyph -> build 16x16 orange/black period-3 lattice -> tile glyph 2x2
#      into a 6x6 maroon stamp -> place that stamp at the fixed centered box
#      rows/cols 5..10.
#    - Fused task:
#      decode glyph exactly as before -> build the same 16x16 lattice -> tile glyph
#      3x3 into a 9x9 maroon stamp -> aim for the geometric center first, then snap
#      the stamp's top-left to the nearest legal 3x3 tile boundary.
#    - Small trace:
#      suppose the glyph is plus. The lattice phase is still the plus-specific one.
#      A pure "make the stamp bigger" rule would center a 9x9 block around the
#      16x16 canvas, giving the rounded top-left (4,4). The D9/2 alignment step then
#      snaps that to the nearest period-3 boundary, namely (3,3). The final maroon
#      region is therefore rows/cols 3..11 and covers exactly 3x3 full lattice tiles.
#
# b) Complexity calibration
#    - The combo changes only the stamp geometry, not the glyph vocabulary or the
#      lattice decoding. This keeps the task readable while making the maroon region
#      visibly different from the source task.
#    - The two fused changes interact tightly instead of independently:
#      * D11/0 enlarges the stamp from 6x6 to 9x9.
#      * D9/2 forces the enlarged stamp to land on 3x3 lattice boundaries.
#      Because 9 is exactly 3 periods, the aligned stamp cleanly occupies a 3x3
#      block of lattice cells, which is an obvious visual cue.
#    - The design avoids extra hidden control variables such as color-id-based
#      branching or input-position mapping, so the rule remains inferable from 2-4
#      examples.
#
# c) Hidden assumptions audit
#    - `bg == ORANGE`, "exactly one non-orange color", and "tight 3x3 bbox" all
#      remain valid assumptions because the glyph decoder is unchanged.
#    - `_decode_phase_tile(...)` still assumes the source motif is one of the four
#      contour families; the generator continues to enforce that.
#    - The original `center = np.tile(motif, (2, 2))` is no longer valid: the fused
#      stamp must be `np.tile(motif, (3, 3))`.
#    - The original fixed-center placement `(out - center)//2` is also obsolete.
#      The new placement assumes there exists at least one legal period-3-aligned
#      origin for the larger stamp; on a 16x16 canvas with a 9x9 stamp, the legal
#      aligned top-left coordinates are {0,3,6}, so this assumption holds.
#
# d) Placement legality
#    - The 9x9 stamp always fits inside the 16x16 output because its top-left is
#      chosen from legal aligned origins only.
#    - There are no multiple moving objects, so there is no overlap ambiguity beyond
#      the intended maroon-over-lattice overwrite already present in the base task.
#    - Input generation keeps the motif bbox inside the 6x6 source grid by sampling
#      top-left row/col in 0..3, exactly as in the source task.
#
# e) Visual observability
#    - The maroon area becomes both larger and shifted relative to naive centering.
#      Humans can see that the stamp now spans a 9x9 square and that its edges sit
#      flush with the 3x3 lattice cell boundaries.
#    - The lattice phase still varies with glyph shape, so the training set can show
#      repeated evidence for the unchanged decode rule while the stamp geometry stays
#      consistently altered across examples.
#
# f) Visual cleanliness
#    - The output still has only three colors: black, orange, and maroon.
#    - The aligned 9x9 stamp forms one contiguous, regular overlay region. Because
#      it matches tile boundaries, the image looks cleaner than an off-grid overlay
#      would, and the lattice remains readable around it.
#
# g) Change inferability and degeneration check
#    - No invisible color ordering or numeric tie-break is used. Alignment depends
#      only on visible geometry: a period-3 lattice and a 9x9 stamp.
#    - A potential degeneration is that a size-only D11/0 implementation might also
#      appear to place the 9x9 stamp "near the center". To keep D9/2 active, this
#      solver defines the pre-alignment centered target by rounded geometric centering
#      (`round((16-9)/2) = 4`), then snaps to the nearest boundary `(3,3)`. Thus the
#      combo output differs from the no-alignment large-stamp baseline.
#
# main()'s new assumptions and generator guarantees
# - Assumption 1: the foreground bbox crops to exactly 3x3. Generator constructs the
#   input from a catalog of 3x3 glyphs and places them without clipping.
# - Assumption 2: the cropped 3x3 mask is one of the four decoder-recognized contour
#   codes. Generator samples only from that catalog.
# - Assumption 3: the output stamp size is a multiple of the lattice period.
#   Generator and solver both fix period=3 and tiling factor=3, so stamp size is 9.
# - Assumption 4: a nearest legal aligned origin exists. On the fixed 16x16 canvas,
#   legal aligned origins are deterministic and non-empty, so generation cannot break
#   this precondition.
# ---------------------------------------------------------------------------


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
OUT_SHAPE = (16, 16)
LATTICE_PERIOD = 3
STAMP_TILE_REPS = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
GLYPH_ORDER = ["plus", "H", "X", "4corners"]
_GLYPH_COUNTER = 0

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


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


def _nearest_aligned_origin(length: int, stamp_length: int, period: int) -> int:
    """Round the centered target, then snap it to the nearest legal period boundary."""
    length = int(length)
    stamp_length = int(stamp_length)
    period = int(period)
    assert 0 < stamp_length <= length
    assert period > 0

    max_start = length - stamp_length
    centered = int(round((length - stamp_length) / 2.0))
    legal = [start for start in range(0, max_start + 1) if start % period == 0]
    assert legal
    return min(legal, key=lambda start: (abs(start - centered), start))


def _stamp_origin(out_shape: tuple[int, int], stamp_shape: tuple[int, int], period: int) -> tuple[int, int]:
    """Choose the nearest legal tile-aligned placement for the stamp."""
    sr, sc = map(int, stamp_shape)
    assert sr % period == 0 and sc % period == 0
    r0 = _nearest_aligned_origin(int(out_shape[0]), sr, period)
    c0 = _nearest_aligned_origin(int(out_shape[1]), sc, period)
    return r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a periodic 16x16 lattice, then overlay a larger maroon stamp snapped to the lattice grid.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one other color.
    - The non-orange pixels form one contour glyph inside a tight 3x3 bounding box.
    - The glyph family is unchanged from the source task: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange/black phase tile as in the base task.
    - Repeat that phase tile to fill a 16x16 output canvas.
    - Tile the original glyph 3x3 to build a 9x9 stamp, recolor its True pixels to maroon, and place it at the nearest legal 3x3 tile boundary rather than at the old fixed center box.

    Key insight:
    - The contour still determines the lattice phase, but the stamp geometry is now governed by two coupled constraints: it is larger because of 3x3 tiling, and it must sit on the lattice's own 3x3 boundary system, so the maroon region lands as a clean 3-by-3 block of tiles.
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
    out = tile_fill(OUT_SHAPE, phase_colors)

    stamp = np.tile(motif, (STAMP_TILE_REPS, STAMP_TILE_REPS))
    r0, c0 = _stamp_origin(out.shape, stamp.shape, LATTICE_PERIOD)
    out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]][stamp] = Color.MAROON
    return out


def _choose_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_COUNTER
    offset = int(rng.integers(0, len(GLYPH_ORDER))) if _GLYPH_COUNTER == 0 else 0
    name = GLYPH_ORDER[(_GLYPH_COUNTER + offset) % len(GLYPH_ORDER)]
    _GLYPH_COUNTER += 1
    return name


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = _choose_glyph_name(rng)
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
    raise RuntimeError("Failed to generate input for eee78d87 combo D9/2+D11/0")
