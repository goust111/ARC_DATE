# Combo variant of eee78d87 (index 18)
# combo_spec: D3/3+D13/2
# Name: 铺贴后擦除+中心空洞

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in THIS_FILE.parents if (parent / "solvers" / "arc_utils.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Could not locate solvers/arc_utils.py from combo variant path")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for combo D3/3 + D13/2
# ---------------------------------------
# a) Concrete behavior and trace
#    - Input remains the original 6x6 orange field with exactly one tight 3x3 contour glyph.
#    - Original task:
#        glyph -> decode 3x3 phase tile -> tile full 16x16 lattice -> stamp 2x2 maroon glyph at center.
#    - This combo:
#        glyph -> decode 3x3 phase tile -> tile full 16x16 lattice
#              -> classify the glyph's symmetry family
#              -> keep only a large cross window (plus/H) or a large diamond window (X/4corners)
#              -> force the centered 6x6 block to pure black
#              -> stamp one maroon 3x3 glyph in each corner.
#    - Example trace:
#        plus:
#          source output = full periodic lattice + centered maroon 2x2 plus
#          combo output  = cross-shaped lattice silhouette with a 6x6 black hole in the middle
#                          + four corner maroon plus stamps
#        X:
#          source output = full periodic lattice + centered maroon 2x2 X
#          combo output  = diamond-shaped lattice silhouette with the same 6x6 black hole
#                          + four corner maroon X stamps
#    - The fusion is not additive-only: the erase stage changes which lattice survives,
#      and the old center stamp is removed entirely so the center can stay empty.
#
# b) Complexity calibration
#    - The phase decoding still depends on the exact glyph, so all four glyph types matter.
#    - The erase stage uses only two large window families: cross or diamond.
#      That keeps the branch count small enough for ARC while still being visually obvious.
#    - The orthogonal family (plus/H) and diagonal-corner family (X/4corners) each contain
#      two glyphs, so the branch split is balanced and repeated across samples.
#
# c) Hidden assumptions from the original main()
#    - The original shape assertions still hold: 6x6 input, orange background, one non-orange
#      color, and a tight 3x3 motif bbox.
#    - The original centered 2x2 maroon stamp assumption is invalid now and must be removed.
#    - A new assumption appears: the glyph must be one of the four known contour codes so that
#      both phase decoding and symmetry-family classification are valid. The generator preserves this.
#    - Ordering matters more than before:
#        1. tile lattice first
#        2. erase by window
#        3. carve the center hole
#        4. stamp corners last
#      If corner stamping happened earlier, the erase pass would delete the maroon markers.
#
# d) Placement legality
#    - Output size is still fixed at 16x16, so the center hole is rows 5:11 and cols 5:11.
#    - Corner anchors (0,0), (0,13), (13,0), (13,13) fit four non-overlapping 3x3 stamps.
#    - The 6x6 center hole is disjoint from all corner stamps.
#    - Input placement is unchanged from the source task: the 3x3 motif can start at any
#      top-left in rows/cols 0..3 without leaving the 6x6 canvas.
#
# e) Visual observability
#    - A human can infer the new rule from a few examples because two stable cues repeat:
#        1. the center is always a black 6x6 void,
#        2. the surviving lattice silhouette flips between a cross and a diamond depending
#           on whether the glyph has orthogonal middle-row arms or diagonal/corner emphasis.
#    - The branch is driven by visible geometry, not hidden metadata such as color ids.
#
# f) Visual cleanliness
#    - The lattice layer is orange on black only inside one large aperture.
#    - The center hole is a single clean black block, not a noisy partial overwrite.
#    - Maroon appears only in the four corners, far away from the erased center and from the
#      main silhouette boundary, so the three roles are visually separated.
#
# g) Change inferability
#    - No extra color mapping is introduced: output colors stay {black, orange, maroon}.
#    - The meaning of maroon is still "decoded glyph marker", but its placement changes from
#      center to corners, which is a strong positional cue rather than an arbitrary recoloring rule.
#
# h) New main() assumptions after the fusion
#    - _window_kind() assumes the four source glyphs split cleanly into two symmetry families.
#      It reads motif[1,0] (equivalently motif[1,2]) to distinguish orthogonal vs diagonal families.
#      The generator only emits plus/H/X/4corners, so this assumption is guaranteed.
#    - The corner-stamp routine assumes a 3x3 motif; main() asserts that before any stamping.
#    - The erase window is defined for the fixed 16x16 output size used by the source spec;
#      generate_input() leaves the source input size unchanged so output size remains 16x16.


OUT_SHAPE = (16, 16)
CENTER_HOLE = (slice(5, 11), slice(5, 11))
CORNER_ANCHORS = ((0, 0), (0, 13), (13, 0), (13, 13))


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


def _window_kind(motif: np.ndarray) -> str:
    """Use orthogonal middle-row arms for cross windows; diagonal/corner-only forms get diamonds."""
    assert motif.shape == (3, 3)
    return "cross" if bool(motif[1, 0]) else "diamond"


def _window_mask(shape: tuple[int, int], kind: str) -> np.ndarray:
    """Return the keep-mask used by the erase stage."""
    h, w = shape
    rr, cc = np.indices((h, w))
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0

    if kind == "cross":
        return (np.abs(rr - center_r) <= 2.5) | (np.abs(cc - center_c) <= 2.5)
    if kind == "diamond":
        return (np.abs(rr - center_r) + np.abs(cc - center_c)) <= 8.5
    raise ValueError(f"unknown window kind: {kind}")


def _stamp_corner_glyphs(out: np.ndarray, motif: np.ndarray) -> None:
    for r0, c0 in CORNER_ANCHORS:
        out[r0 : r0 + 3, c0 : c0 + 3][motif] = Color.MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a tiled lattice, erase that lattice down to a symmetry-selected aperture, and leave the center as a black hole with maroon corner stamps.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one tight 3x3 contour glyph.
    - The glyph is one of the four valid code shapes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the source task's period-3 orange phase tile.
    - Repeat that phase tile across a 16x16 black canvas.
    - Use the glyph's visible symmetry family to choose an erase window: plus/H keep a cross, X/four-corners keep a diamond.
    - Black out everything outside that window, then black out the centered 6x6 block as a permanent hole.
    - Stamp one maroon 3x3 copy of the glyph into each of the four output corners.

    Key insight:
    - The same small contour now controls two different spatial decisions: its exact shape still decodes the lattice phase, while its orthogonal-vs-diagonal arm structure decides which large aperture survives around the central void.
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

    tiled = tile_fill(OUT_SHAPE, phase_colors)
    keep = _window_mask(OUT_SHAPE, _window_kind(motif))
    out = np.where(keep, tiled, Color.BLACK).astype(int)

    out[CENTER_HOLE] = Color.BLACK
    _stamp_corner_glyphs(out, motif)
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
WINDOW_FAMILIES = {
    "cross": ("plus", "H"),
    "diamond": ("X", "4corners"),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    family = rng.choice(tuple(WINDOW_FAMILIES.keys()))
    glyph_name = rng.choice(WINDOW_FAMILIES[family])
    motif = GLYPHS[str(glyph_name)]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    keep = _window_mask(OUT_SHAPE, family)
    hole_mask = np.zeros(OUT_SHAPE, dtype=bool)
    hole_mask[CENTER_HOLE] = True

    if not np.all(out[CENTER_HOLE] == Color.BLACK):
        return None

    for rr, cc in CORNER_ANCHORS:
        patch = out[rr : rr + 3, cc : cc + 3]
        if not np.array_equal(patch == Color.MAROON, motif):
            return None

    # Outside the kept aperture, only the four corner maroon stamps are allowed to survive.
    allowed_outside = np.zeros(OUT_SHAPE, dtype=bool)
    for rr, cc in CORNER_ANCHORS:
        allowed_outside[rr : rr + 3, cc : cc + 3] |= motif
    illegal = (~keep) & (~allowed_outside) & (out != Color.BLACK)
    if np.any(illegal):
        return None

    # The combo should not collapse back to the D3-only variant: some non-hole cells
    # outside the chosen aperture must have been erased from the full tiled lattice.
    full_tiled = tile_fill(OUT_SHAPE, np.where(_decode_phase_tile(motif), Color.ORANGE, Color.BLACK).astype(int))
    erased = (~keep) & (~hole_mask) & (full_tiled == Color.ORANGE) & (out == Color.BLACK)
    if not np.any(erased):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 18")
