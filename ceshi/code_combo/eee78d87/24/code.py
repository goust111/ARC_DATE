# Combo variant of eee78d87 (index 24)
# combo_spec: D2/0+D13/1
# Name: 分形迭代放大+大尺寸字形
"""
Design analysis:
  concrete_spec:
    input_grid: fixed 10x10 orange board so both 4x4 and 5x5 enlarged glyphs fit with slack
    source_object:
      count: exactly 1
      bbox: exact square of side 4 or 5
      color: exactly 1 non-orange color from {1,2,3,4,5,6,8}
      families:
        - plus: expand the coarse 3x3 plus with row/col bands [1, size-2, 1]
        - H: expand the coarse 3x3 H with the same bands
        - X: expand the coarse 3x3 X with the same bands
        - 4corners: expand the coarse 3x3 four-corners with the same bands
    output_process:
      step1: identify the enlarged glyph family and decode the matching enlarged period phase tile
      step2: tile that phase over a 10x10 coarse raster using orange for True and black for False
      step3: overwrite the centered coarse 2x2 block with maroon
      step4: expand every coarse pixel into a size x size child tile shaped like the same enlarged glyph
    output_grid:
      size4: 40x40 final grid
      size5: 50x50 final grid
      colors: black, orange, maroon only

  behavior_trace:
    original_path:
      input_bbox: always 3x3
      decoded_period: always 3
      final_write: direct 16x16 lattice plus flat 6x6 maroon overwrite
    fused_path:
      input_bbox: 4x4 or 5x5
      decoded_period: 4 or 5 after enlarged-glyph identification
      final_write: first build a 10x10 macro lattice, then recursively expand each macro pixel by the same enlarged glyph
    concrete_example:
      size4_plus:
        motif: .##. / #### / #### / .##.
        phase: .... / .### / .### / .###
        coarse_center: rows 4..5 and cols 4..5 become maroon macro cells
        final_effect: every orange or maroon macro cell becomes a 4x4 plus-shaped child stamp, giving a 40x40 perforated lattice
      difference:
        - D2 changes which glyph sizes are legal and which phase period is used
        - D13 changes the write target from a direct final canvas to a macro raster followed by one recursive expansion
        - Together, size controls both the macro period and the child-tile magnification, so the two changes are coupled rather than independent

  complexity_calibration:
    - There are 4 families x 2 sizes = 8 discrete visual codewords, enough variation to avoid trivial memorization.
    - Size is visible twice in each example: in the macro lattice period and in the child stamp dimensions.
    - Family is visible twice as well: it selects the macro phase layout and also the holes inside every non-black child tile.
    - The design stays learnable because only one recursion level, one centered maroon macro block, and three output colors are used.

  implicit_assumption_audit:
    - `g.shape == (6, 6)` from the base task no longer holds; the fused task needs 10x10 to fit 5x5 glyphs comfortably.
    - `motif.shape == (3, 3)` no longer holds; solver must accept only 4x4 or 5x5 prototypes.
    - Base `_decode_phase_tile()` relied on literal 3x3 pixel tests; here decoding must go through family identification plus enlarged phase lookup.
    - Direct final tiling (`out_shape = 3*g.shape - 2`) is no longer the right geometry; the final canvas now comes from coarse-grid size times glyph size.
    - The maroon mark cannot be painted after expansion as one flat rectangle, because that would erase the fractal holes. It must replace a centered 2x2 block in the coarse raster before recursive expansion.
    - `_expand_by_glyph()` assumes coarse pixels are only black/orange/maroon; `_build_coarse_raster()` enforces exactly that palette.

  placement_legality:
    - Generator samples a glyph size first, then chooses a top-left uniformly from all placements fully inside 10x10.
    - Only one foreground object exists, so there is no overlap ambiguity.
    - Every prototype touches all four sides of its bbox, so `bbox()` recovers the intended 4x4 or 5x5 square exactly.

  visual_observability:
    - A human can spot the fusion from 2-3 examples because outputs jump to 40x40 or 50x50 and all colored regions are perforated by repeated large glyph silhouettes.
    - Size-4 and size-5 examples differ in both final resolution and local stroke thickness.
    - Different families remain visually distinct because the same hole pattern repeats in every non-black child tile.

  visual_cleanliness:
    - Macro structure and micro structure are separated cleanly: macro cells decide where colored tiles appear, while the glyph mask decides their internal holes.
    - The centered maroon feature is still a single contiguous macro region, but it inherits the same glyph perforation as the orange lattice, so the scene stays readable instead of becoming a flat overwrite.

  color_inference:
    - Input color remains arbitrary and carries no extra meaning beyond locating the single glyph.
    - Output colors stay structurally interpretable: orange for phase-on macro cells, maroon for the centered macro stamp, black for empty macro cells and all holes inside child glyph tiles.

  generator_validation:
    - Every sampled input is oracle-checked with `main()`.
    - The generator explicitly varies both controlling dimensions: glyph family and glyph size.
    - The intended fusion metric is final output side length, which must be either 40 or 50 and therefore cannot collapse back to the original 16 or the single-variant 28.

  main_assumptions_after_change:
    - The foreground bbox must match one exact enlarged prototype from the shared lookup table.
    - The bbox must be square with side 4 or 5.
    - The centered coarse 2x2 block is well-defined because the coarse board size is fixed at 10x10.
    - `generate_input()` guarantees all of these assumptions by constructing inputs from the same prototype family/size table used by the solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 10, 10
GLYPH_SIZES = (4, 5)
CENTER_MACRO_SIZE = 2
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

COARSE_GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

COARSE_PHASES = {
    "plus": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    "H": np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "X": np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "4corners": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
}


def _band_sizes(size: int) -> tuple[int, int, int]:
    size = int(size)
    assert size in GLYPH_SIZES
    return 1, size - 2, 1


def _expand_coarse(pattern: np.ndarray, size: int) -> np.ndarray:
    bands = np.array(_band_sizes(size), dtype=int)
    expanded = np.repeat(np.repeat(np.asarray(pattern, dtype=bool), bands, axis=0), bands, axis=1)
    assert expanded.shape == (size, size)
    return expanded


EXPANDED_GLYPHS = {
    (name, size): _expand_coarse(pattern, size)
    for name, pattern in COARSE_GLYPHS.items()
    for size in GLYPH_SIZES
}

EXPANDED_PHASES = {
    (name, size): _expand_coarse(pattern, size)
    for name, pattern in COARSE_PHASES.items()
    for size in GLYPH_SIZES
}


def _identify_glyph(motif: np.ndarray) -> tuple[str, int]:
    motif = np.asarray(motif, dtype=bool)
    assert motif.ndim == 2
    assert motif.shape[0] == motif.shape[1]
    size = int(motif.shape[0])
    assert size in GLYPH_SIZES

    for (name, glyph_size), glyph in EXPANDED_GLYPHS.items():
        if glyph_size == size and np.array_equal(motif, glyph):
            return name, size
    raise AssertionError("Unrecognized enlarged glyph")


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode an enlarged contour glyph into the matching enlarged phase tile."""
    glyph_name, size = _identify_glyph(motif)
    return EXPANDED_PHASES[(glyph_name, size)]


def _build_coarse_raster(motif: np.ndarray) -> np.ndarray:
    """Create the 10x10 macro raster before the recursive expansion step."""
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    coarse = tile_fill((GRID_H, GRID_W), phase_colors)

    r0 = (GRID_H - CENTER_MACRO_SIZE) // 2
    c0 = (GRID_W - CENTER_MACRO_SIZE) // 2
    coarse[r0 : r0 + CENTER_MACRO_SIZE, c0 : c0 + CENTER_MACRO_SIZE] = Color.MAROON
    return coarse


def _expand_by_glyph(coarse: np.ndarray, motif: np.ndarray) -> np.ndarray:
    """Expand each macro pixel into a child tile shaped like the same enlarged glyph."""
    motif = np.asarray(motif, dtype=bool)
    size = int(motif.shape[0])
    assert coarse.shape == (GRID_H, GRID_W)
    assert motif.shape == (size, size)
    assert size in GLYPH_SIZES

    out = np.full((coarse.shape[0] * size, coarse.shape[1] * size), Color.BLACK, dtype=int)
    child_tiles = {
        Color.BLACK: np.full((size, size), Color.BLACK, dtype=int),
        Color.ORANGE: np.where(motif, Color.ORANGE, Color.BLACK).astype(int),
        Color.MAROON: np.where(motif, Color.MAROON, Color.BLACK).astype(int),
    }

    for r in range(coarse.shape[0]):
        for c in range(coarse.shape[1]):
            color = int(coarse[r, c])
            assert color in child_tiles
            rr = r * size
            cc = c * size
            out[rr : rr + size, cc : cc + size] = child_tiles[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 4x4 or 5x5 enlarged contour glyph into a coarse lattice, then recursively expand that lattice with the same enlarged glyph as the child tile.

    Input structure:
    - The grid is 10x10 with orange background and exactly one non-orange color.
    - The colored pixels form one tight square bbox of side 4 or 5.
    - That bbox matches one of four enlarged contour families: plus, H, X, or four corners.

    Rule:
    - Identify the enlarged glyph family and decode its matching period-4 or period-5 orange/black phase tile.
    - Tile that phase over a 10x10 coarse raster and overwrite the centered coarse 2x2 block with maroon.
    - Expand each coarse pixel into a size-matched child tile shaped like the same enlarged glyph: orange cells become orange glyph tiles, maroon cells become maroon glyph tiles, and black cells stay all black.

    Key insight:
    - The enlarged input glyph controls both scales at once: its family determines the macro phase arrangement, while its size determines both the macro period and the magnification factor of every recursive child tile.
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
    _glyph_name, motif_size = _identify_glyph(motif)

    coarse = _build_coarse_raster(motif)
    out = _expand_by_glyph(coarse, motif)
    assert out.shape == (GRID_H * motif_size, GRID_W * motif_size)
    return out


GLYPHS = EXPANDED_GLYPHS


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(COARSE_GLYPHS.keys())))
    motif_size = int(rng.choice(GLYPH_SIZES))
    motif = GLYPHS[(glyph_name, motif_size)]
    fg = int(rng.choice(FG_COLORS))

    r0 = int(rng.integers(0, GRID_H - motif_size + 1))
    c0 = int(rng.integers(0, GRID_W - motif_size + 1))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + motif_size, c0 : c0 + motif_size][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    assert out.shape == (GRID_H * motif_size, GRID_W * motif_size)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
