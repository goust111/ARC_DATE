# Combo variant of eee78d87 (index 41)
# combo_spec: D12/2+D16/2
# Name: 多字形指令序列+大输入多字形区域

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, colors_present, tile_fill


# Design analysis for the combo variant
# a) Concretization + behavior trace
#    - The base task had one 6x6 orange field with one 3x3 contour glyph; that single glyph
#      simultaneously determined the lattice phase and the centered stamp motif.
#    - Here the input is a 12x12 orange field split into a 4x4 array of 3x3 macro-cells.
#      Exactly three macro-cells are occupied, each by one monochrome contour glyph.
#    - Read the occupied macro-cells in row-major order. They form a three-step instruction
#      sequence rather than three independent local panels:
#        step 1: glyph shape -> decode the base 3x3 phase tile.
#        step 2: glyph shape -> choose the 3x3 stamp motif; glyph color -> choose lattice color.
#        step 3: glyph shape -> choose a quarter-turn rotation applied to both results above.
#    - Example trace:
#      if the occupied cells in row-major order are X(red), H(blue), 4corners(green), then
#      the solver decodes X into the original phase tile, uses blue as the repeated lattice
#      color, uses H as the centered 2x2 stamp shape, then rotates both by 270 degrees
#      because the third glyph is 4corners. The final output is one global 16x16 canvas.
#    - Difference from the base task:
#      original: one glyph -> one direct decode.
#      combo: three separated glyph regions -> ordered instruction fusion -> one decode.
# b) Complexity calibration
#    - The glyph vocabulary stays the original four shapes so the task does not explode into
#      arbitrary symbol tables.
#    - Only one global canvas is produced, which forces genuine instruction composition and
#      avoids degenerating into the D12 single-variant behavior of per-region independent
#      panels.
#    - The color rule is kept visually grounded: the second instruction's visible input color
#      becomes the visible lattice color, so at least one control channel is inferable from a
#      single example instead of requiring hidden memorization.
# c) Implicit-assumption audit of the original main()
#    - `assert g.shape == (6, 6)`: invalid now; replaced by an exact 12x12 macro-grid.
#    - `len(colors_present(...)) == 1`: invalid now; three foreground colors are allowed.
#    - Global bbox extraction of all foreground pixels would merge multiple instructions and
#      lose their order; replaced by fixed 3x3 macro-cell parsing.
#    - `motif` serving both as lattice decoder and stamp source is no longer valid; those are
#      split across instruction 1 and instruction 2.
#    - Hard-coding orange as the lattice foreground is no longer valid; instruction 2 selects
#      the lattice color.
#    - No-rotation output is no longer guaranteed; instruction 3 can rotate both components.
# d) Placement legality
#    - The generator only places glyphs on macro-cell boundaries, so they never overlap and
#      always fit exactly inside one 3x3 region.
#    - Exactly three cells are occupied, and the chosen cells span at least two macro rows and
#      two macro columns so the "large input with multiple regions" behavior is visually real.
#    - All three glyph colors are distinct and avoid black, orange, and maroon.
# e) Visual observability
#    - A human can spot three separated 3x3 regions in a much larger orange grid, infer that
#      reading order matters, and see that only one final output is produced.
#    - The second instruction is especially observable because its input color reappears as the
#      repeated lattice color, while its shape reappears as the maroon center stamp.
#    - The third instruction affects the orientation of both the lattice phase and the stamp,
#      so it has a global effect rather than a local decoration.
# f) Visual cleanliness
#    - Input regions never touch and output layers are limited to a periodic black/color field
#      plus one centered maroon stamp, matching the base task's clean composition.
#    - There is no multi-panel output, so no region competes for space with another region in
#      the output.
# g) Change inferability
#    - Color usage is constrained to one meaningful transferred color plus maroon on black, so
#      the color logic stays legible.
#    - The only arbitrary lookup retained is third-glyph-shape -> quarter-turn count, but it is
#      applied to the whole output and therefore produces a strong, repeated visual signature.
#    - New solver assumptions introduced by this combo are:
#      row-major occupied-cell order defines instruction order; exactly three occupied cells
#      exist; every occupied cell is a valid contour glyph; the second glyph's color is a valid
#      lattice color; the third glyph's shape is one of the four rotation codes. The generator
#      explicitly constructs inputs to satisfy each of those assumptions.


MACRO_SIZE = 3
GRID_MACROS = 4
GRID_H = GRID_W = MACRO_SIZE * GRID_MACROS
OUT_SHAPE = (16, 16)
STAMP_SIZE = 6
OCCUPIED_COUNT = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
PATTERN_TO_NAME = {tuple(map(int, motif.flatten())): name for name, motif in GLYPHS.items()}
ROTATION_BY_NAME = {
    "plus": 0,
    "H": 1,
    "X": 2,
    "4corners": 3,
}


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 phase mask used by the base task."""
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


def _motif_name(motif: np.ndarray) -> str:
    key = tuple(map(int, motif.astype(int).flatten()))
    assert key in PATTERN_TO_NAME
    return PATTERN_TO_NAME[key]


def _extract_instruction_sequence(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(np.bincount(g.reshape(-1)).argmax())
    assert bg == Color.ORANGE

    seen_colors: set[int] = set()
    instructions: list[tuple[int, int, np.ndarray, int]] = []
    for mr in range(GRID_MACROS):
        for mc in range(GRID_MACROS):
            r0 = mr * MACRO_SIZE
            c0 = mc * MACRO_SIZE
            cell = g[r0 : r0 + MACRO_SIZE, c0 : c0 + MACRO_SIZE]
            cols = colors_present(cell, exclude={bg})
            assert len(cols) <= 1
            if not cols:
                continue

            fg = int(cols[0])
            assert fg not in {Color.BLACK, Color.ORANGE, Color.MAROON}
            assert fg not in seen_colors
            seen_colors.add(fg)

            motif = (cell != bg)
            _motif_name(motif)
            instructions.append((mr, mc, motif, fg))

    assert len(instructions) == OCCUPIED_COUNT
    rows = {mr for mr, _mc, _motif, _fg in instructions}
    cols = {mc for _mr, mc, _motif, _fg in instructions}
    assert len(rows) >= 2
    assert len(cols) >= 2
    return instructions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fuse three separated contour glyph instructions from a larger orange macro-grid into one transformed lattice-and-stamp output.

    Input structure:
    - The grid is 12x12 with orange background, viewed as a 4x4 arrangement of 3x3 macro-cells.
    - Exactly three macro-cells are occupied, and each occupied cell contains one monochrome 3x3 contour glyph in a unique non-black, non-orange, non-maroon color.
    - Reading occupied cells in row-major order gives instruction 1, instruction 2, and instruction 3.

    Rule:
    - Instruction 1 uses its glyph shape to decode the same base 3x3 phase tile as in the original task.
    - Instruction 2 uses its glyph shape as the centered 2x2 stamp motif and uses its input color as the lattice foreground color.
    - Instruction 3 uses its glyph shape as a rotation code; rotate both the decoded phase tile and the stamp motif by 0, 90, 180, or 270 degrees, then tile the 16x16 canvas and stamp the rotated 6x6 motif in maroon at the center.

    Key insight:
    - The larger input is not a set of independent regions. It is a three-step program: one glyph chooses the tiling pattern, one glyph chooses both the repeated color and the stamped shape, and the last glyph globally reorients the whole construction.
    """
    instructions = _extract_instruction_sequence(grid)

    phase_motif = instructions[0][2]
    stamp_motif = instructions[1][2]
    lattice_color = int(instructions[1][3])
    rotation_name = _motif_name(instructions[2][2])
    rotation_k = int(ROTATION_BY_NAME[rotation_name])

    phase = np.rot90(_decode_phase_tile(phase_motif), k=rotation_k)
    stamp_motif = np.rot90(stamp_motif, k=rotation_k)

    phase_colors = np.where(phase, lattice_color, Color.BLACK).astype(int)
    out = tile_fill(OUT_SHAPE, phase_colors)

    center = np.tile(stamp_motif, (2, 2))
    r0 = (out.shape[0] - STAMP_SIZE) // 2
    c0 = (out.shape[1] - STAMP_SIZE) // 2
    out[r0 : r0 + STAMP_SIZE, c0 : c0 + STAMP_SIZE][center] = Color.MAROON
    return out


def _choose_macro_cells(rng: np.random.Generator) -> list[tuple[int, int]]:
    macro_ids = np.arange(GRID_MACROS * GRID_MACROS)
    for _ in range(100):
        chosen = [int(x) for x in rng.choice(macro_ids, size=OCCUPIED_COUNT, replace=False)]
        coords = [divmod(idx, GRID_MACROS) for idx in sorted(chosen)]
        rows = {mr for mr, _mc in coords}
        cols = {mc for _mr, mc in coords}
        if len(rows) >= 2 and len(cols) >= 2:
            return [(int(mr), int(mc)) for mr, mc in coords]
    raise RuntimeError("Failed to sample sufficiently spread macro-cells")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)

    cells = _choose_macro_cells(rng)
    colors = [int(x) for x in rng.choice(FG_COLORS, size=OCCUPIED_COUNT, replace=False)]
    glyph_names = list(GLYPHS.keys())

    # Step 1, 2, and 3 are assigned by row-major occupied-cell order.
    motifs = [GLYPHS[str(rng.choice(glyph_names))] for _ in range(OCCUPIED_COUNT)]

    for (mr, mc), motif, color in zip(cells, motifs, colors, strict=True):
        r0 = mr * MACRO_SIZE
        c0 = mc * MACRO_SIZE
        inp[r0 : r0 + MACRO_SIZE, c0 : c0 + MACRO_SIZE][motif] = color

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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 41")
