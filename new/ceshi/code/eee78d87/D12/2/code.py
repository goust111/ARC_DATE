# Variant of eee78d87
# Dimension: D12 — 输入网格尺寸 (meta)
# Change: 固定6x6输入网格... → 更大输入网格含多个字形区域

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, colors_present, tile_fill


# Design analysis for the D12 variant
# a) Concretization + behavior trace
#    - Input is no longer a single 6x6 field. It is a 9x9 orange grid split into a 3x3
#      array of 3x3 macro-cells.
#    - Each occupied macro-cell contains exactly one monochrome 3x3 contour glyph, using
#      one unique non-orange color. Empty macro-cells stay fully orange.
#    - Output is a 26x26 black canvas split into a matching 3x3 array of 8x8 panels,
#      separated by 1-cell black gutters. Each occupied input macro-cell controls only its
#      corresponding output panel: decode that glyph into the original period-3 orange/black
#      phase tile, tile the 8x8 panel with that phase, then stamp the glyph tiled 2x2 as a
#      centered 6x6 maroon mark. Empty input cells map to all-black panels.
#    - Trace difference:
#      original: one 3x3 glyph -> one global 16x16 lattice with one central stamp.
#      variant: three or more macro-cells may exist, and each occupied macro-cell performs
#      the same decode locally inside its own panel. The solver must separate regions first,
#      then decode each independently.
# b) Complexity calibration
#    - The core change is region multiplicity, not new glyph logic. Keeping the glyph set
#      fixed to the original four contour codes avoids rule explosion.
#    - Using a rigid 3x3 macro-grid prevents ambiguous grouping while still making the
#      multi-object behavior visually obvious.
# c) Implicit-assumption audit of the original main()
#    - `assert g.shape == (6, 6)`: no longer valid; replaced by an exact 9x9 macro-grid.
#    - `len(colors_present(...)) == 1`: no longer valid; now multiple distinct foreground
#      colors are required so glyph regions are visually separable.
#    - Global bbox extraction of all foreground pixels would wrongly merge multiple glyphs;
#      replaced by fixed 3x3 macro-cell inspection.
#    - One global tiled canvas plus one global center stamp would destroy per-region
#      independence; replaced by per-panel synthesis.
# d) Placement legality
#    - Generator places glyphs only on macro-cell boundaries, so 3x3 glyphs never overlap
#      and always fit. At least two macro-cells are occupied to ensure the variant is active.
#    - Each occupied macro-cell uses a unique foreground color, but output ignores that color
#      just like the original task ignored the specific input color.
# e) Visual observability
#    - Across 2-4 examples, a human can see that each colored 3x3 block influences the panel
#      in the same relative row/column of the output mosaic. Different glyph types yield
#      different orange/black phase panels, making the local decode rule observable.
# f) Visual cleanliness
#    - Panels are spatially disjoint, separated by black gutters, and maroon only appears
#      inside occupied panels. This avoids overlapping explanatory layers.
# g) Color inferability
#    - Input colors only separate glyph regions; there is no hidden color-dependent mapping.
#      All occupied colors behave identically, which is easy to infer from few examples.


MACRO_SIZE = 3
GRID_MACROS = 3
GRID_H = GRID_W = MACRO_SIZE * GRID_MACROS
PANEL_SIZE = 8
PANEL_GAP = 1
OUT_SIZE = GRID_MACROS * PANEL_SIZE + (GRID_MACROS - 1) * PANEL_GAP
STAMP_SIZE = 6
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_PATTERNS = {tuple(map(int, motif.flatten())) for motif in GLYPHS.values()}


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


def _extract_motifs(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(np.bincount(g.reshape(-1)).argmax())
    assert bg == Color.ORANGE

    seen_colors: set[int] = set()
    motifs: list[tuple[int, int, np.ndarray]] = []
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
            assert fg not in seen_colors
            seen_colors.add(fg)

            motif = cell != bg
            assert tuple(map(int, motif.flatten())) in GLYPH_PATTERNS
            motifs.append((mr, mc, motif))

    assert 2 <= len(motifs) <= 4
    return motifs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode several colored 3x3 contour glyphs from a larger orange macro-grid into separate local phase panels.

    Input structure:
    - The grid is 9x9 with orange background, viewed as a 3x3 arrangement of 3x3 macro-cells.
    - Between two and four macro-cells are occupied; each occupied cell contains exactly one 3x3 contour glyph in its own non-orange color.
    - Every glyph is one of the original four contour codes: plus, H, X, or four corners.

    Rule:
    - For each occupied macro-cell, decode its 3x3 glyph into the same period-3 orange phase tile as in the base task.
    - In the output mosaic, fill the corresponding 8x8 panel with that orange/black tile and stamp a 2x2 tiling of the glyph in maroon at the panel center.
    - Leave panels whose input macro-cells are empty entirely black; panels are separated by 1-cell black gutters.

    Key insight:
    - The task is no longer about one global lattice. Each colored 3x3 region is an independent code, and the output is a position-preserving mosaic where every occupied region generates its own local lattice panel.
    """
    motifs = _extract_motifs(grid)
    out = np.full((OUT_SIZE, OUT_SIZE), Color.BLACK, dtype=int)

    stamp_offset = (PANEL_SIZE - STAMP_SIZE) // 2
    for mr, mc, motif in motifs:
        phase = _decode_phase_tile(motif)
        phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
        panel = tile_fill((PANEL_SIZE, PANEL_SIZE), phase_colors)

        pr = mr * (PANEL_SIZE + PANEL_GAP)
        pc = mc * (PANEL_SIZE + PANEL_GAP)
        out[pr : pr + PANEL_SIZE, pc : pc + PANEL_SIZE] = panel

        center = np.tile(motif, (2, 2))
        stamp = out[
            pr + stamp_offset : pr + stamp_offset + STAMP_SIZE,
            pc + stamp_offset : pc + stamp_offset + STAMP_SIZE,
        ]
        stamp[center] = Color.MAROON

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)

    macro_ids = list(range(GRID_MACROS * GRID_MACROS))
    occupied_count = int(rng.integers(2, 5))
    chosen_cells = rng.choice(macro_ids, size=occupied_count, replace=False)
    chosen_colors = rng.choice(FG_COLORS, size=occupied_count, replace=False)
    glyph_names = list(GLYPHS.keys())

    for idx, macro_id in enumerate(chosen_cells):
        mr, mc = divmod(int(macro_id), GRID_MACROS)
        r0 = mr * MACRO_SIZE
        c0 = mc * MACRO_SIZE
        motif = GLYPHS[str(rng.choice(glyph_names))]
        color = int(chosen_colors[idx])
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
    raise RuntimeError("Failed to generate input for eee78d87 D12 variant")
