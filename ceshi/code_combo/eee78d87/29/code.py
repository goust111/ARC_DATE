# Combo variant of eee78d87 (index 29)
# combo_spec: D12/2+D14/2
# Name: 大输入多字形+多色多层

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the D12/2 + D14/2 combo
# a) Concretization + behavior trace
#    - The base task had one 6x6 orange board, one monochrome 3x3 contour glyph, one
#      orange/black lattice, and one central maroon 6x6 stamp.
#    - Here the input becomes a larger 9x9 or 12x12 orange board aligned to 3x3
#      macro-cells. Two or three different macro-cells are occupied, and each occupied
#      cell contains one full 3x3 contour glyph in its own distinct color.
#    - Example trace:
#      original: one teal X -> decode one phase tile -> tile one orange layer globally.
#      combo: blue plus, red X, grey H in three macro-cells -> decode three phase tiles
#      independently -> tile three full-canvas colored layers -> paint them in visible
#      row-major macro-cell order so later cells overwrite earlier ones -> stamp the
#      union of all 2x2-tiled motifs as one maroon 6x6 mark at the canvas center.
#    - The essential difference from the D12-only panel solution is that macro-cells do
#      not map to separate output panels. They only supply multiple glyph sources that
#      all influence the same large periodic canvas, which is the real fusion with D14.
# b) Complexity calibration
#    - Keeping the original four glyph codes preserves the base decoding vocabulary.
#    - Restricting the input to 2-3 occupied macro-cells prevents the overlay from
#      becoming unreadably dense while still making the "multiple sources" change clear.
#    - Using only 9x9 or 12x12 preserves obvious 3x3 macro alignment and creates a
#      visible size change relative to the fixed 6x6 original.
# c) Implicit-assumption audit of the original main()
#    - `assert g.shape == (6, 6)` is invalid; the combo accepts 9x9 and 12x12 only.
#    - `len(colors_present(...)) == 1` is invalid; the combo requires 2-3 distinct
#      foreground colors because each color is a separate layer source.
#    - One global bbox over all foreground pixels is invalid because it would merge
#      multiple glyphs. Extraction must happen per 3x3 macro-cell.
#    - `phase_colors = np.where(phase, bg, black)` is invalid because lattice color is
#      no longer always orange. Each layer keeps its own input color in the output.
#    - One decoded motif and one decoded lattice are invalid assumptions. The combo needs
#      a loop over ordered layers plus explicit overwrite composition.
# d) Placement legality
#    - The generator only places glyphs on macro-cell boundaries, so every glyph has an
#      exact 3x3 box, never overlaps another glyph, and always fits the larger board.
#    - Each occupied macro-cell gets a unique non-orange color, making color ownership
#      of each glyph unambiguous.
# e) Visual observability
#    - Priority depends only on visible row-major macro-cell order, never on hidden
#      numeric color ids.
#    - The generator rejects samples whose decoded layers never overlap, because then the
#      overlay rule would exist in code but not be inferable from the image pair.
#    - It also rejects samples where any input color disappears entirely from the output,
#      so each layer remains visibly tied to its source glyph.
# f) Visual cleanliness
#    - All lattice layers share the same 3-period scaffold, which keeps the composite
#      organized compared with unrelated free-form overlays.
#    - The maroon stamp remains a single centered 6x6 region, so the rest of the canvas
#      is still available to show the layer-composition rule.
# g) Change inferability
#    - Color has a structured role: same input color -> same output layer color, and the
#      layer's visibility depends on its decoded phase plus row-major priority. Because
#      colors are reused directly rather than remapped arbitrarily, a human can infer the
#      semantics from a few examples.


MACRO_SIZE = 3
VALID_GRID_MACROS = (3, 4)  # 9x9 or 12x12 input
VALID_INPUT_SIZES = {macro_count * MACRO_SIZE for macro_count in VALID_GRID_MACROS}
STAMP_SIZE = 6
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MIN_LAYERS = 2
MAX_LAYERS = 3
MAX_RETRIES = 400

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_PATTERNS = {tuple(map(int, motif.reshape(-1))) for motif in GLYPHS.values()}


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the original period-3 phase mask."""
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


PHASES = {name: _decode_phase_tile(motif) for name, motif in GLYPHS.items()}


def _sequence_has_visible_overlay(glyph_names: tuple[str, ...]) -> bool:
    owner = np.full((3, 3), -1, dtype=int)
    overlap_seen = False
    for idx, name in enumerate(glyph_names):
        phase = PHASES[name]
        overlap_seen |= bool(np.any(phase & (owner != -1)))
        owner[phase] = idx
    return overlap_seen and all(bool(np.any(owner == idx)) for idx in range(len(glyph_names)))


VALID_GLYPH_SEQUENCES = {
    count: [
        seq
        for seq in product(GLYPHS.keys(), repeat=count)
        if _sequence_has_visible_overlay(seq)
    ]
    for count in range(MIN_LAYERS, MAX_LAYERS + 1)
}


def _output_shape_for_input(shape: tuple[int, int]) -> tuple[int, int]:
    return 3 * int(shape[0]) - 2, 3 * int(shape[1]) - 2


def _extract_layers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == g.shape[1]
    assert g.shape[0] in VALID_INPUT_SIZES

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    macro_count = g.shape[0] // MACRO_SIZE
    seen_colors: set[int] = set()
    layers: list[dict[str, object]] = []

    for mr in range(macro_count):
        for mc in range(macro_count):
            r0 = mr * MACRO_SIZE
            c0 = mc * MACRO_SIZE
            cell = g[r0 : r0 + MACRO_SIZE, c0 : c0 + MACRO_SIZE]
            cols = colors_present(cell, exclude={bg})
            assert len(cols) <= 1
            if not cols:
                continue

            color = int(cols[0])
            assert color in FG_COLORS
            assert color not in seen_colors
            seen_colors.add(color)

            motif = cell == color
            assert tuple(map(int, motif.reshape(-1))) in GLYPH_PATTERNS
            layers.append(
                {
                    "color": color,
                    "macro": (mr, mc),
                    "motif": motif,
                    "phase": _decode_phase_tile(motif),
                }
            )

    assert MIN_LAYERS <= len(layers) <= MAX_LAYERS
    return layers


def _compose_lattice(layers: list[dict[str, object]], out_shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    out = np.full(out_shape, Color.BLACK, dtype=int)
    overlap_cells = 0

    for layer in layers:
        phase = np.asarray(layer["phase"], dtype=bool)
        color = int(layer["color"])
        phase_colors = np.where(phase, color, Color.BLACK).astype(int)
        tiled = tile_fill(out_shape, phase_colors)
        mask = tiled != Color.BLACK
        overlap_cells += int(np.count_nonzero(mask & (out != Color.BLACK)))
        out[mask] = tiled[mask]

    return out, overlap_cells


def _apply_center_stamp(out: np.ndarray, layers: list[dict[str, object]]) -> np.ndarray:
    stamped = out.copy()
    center_mask = np.zeros((STAMP_SIZE, STAMP_SIZE), dtype=bool)
    for layer in layers:
        center_mask |= np.tile(np.asarray(layer["motif"], dtype=bool), (2, 2))

    r0 = (stamped.shape[0] - STAMP_SIZE) // 2
    c0 = (stamped.shape[1] - STAMP_SIZE) // 2
    stamped[r0 : r0 + STAMP_SIZE, c0 : c0 + STAMP_SIZE][center_mask] = Color.MAROON
    return stamped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode several colored 3x3 contour glyphs from a larger orange macro-grid into one stacked periodic lattice, then mark their combined code at the center.

    Input structure:
    - The grid is either 9x9 or 12x12 with orange background, interpreted as a 3x3 or 4x4 array of 3x3 macro-cells.
    - Exactly two or three macro-cells are occupied; each occupied cell contains one valid contour glyph in its own non-orange color.
    - The occupied cells are read in row-major order, and that visible order defines lattice draw priority from bottom layer to top layer.

    Rule:
    - Decode each occupied 3x3 glyph independently into the original task's period-3 phase tile.
    - Repeat every decoded phase across the full output canvas of size (3H-2)x(3W-2), using that glyph's own input color as the lattice foreground.
    - Composite the full-canvas layers in row-major macro-cell order so later glyph regions overwrite earlier ones, then place the 2x2-tiled union of all glyphs as a maroon 6x6 stamp at the output center.

    Key insight:
    - The bigger input is not a panel layout. It is a way to host multiple color-labeled glyph codes, and each code generates a global periodic layer whose final visibility depends on both phase overlap and position-based priority.
    """
    layers = _extract_layers(grid)
    lattice, _overlap_cells = _compose_lattice(layers, _output_shape_for_input(np.asarray(grid).shape))
    return _apply_center_stamp(lattice, layers)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    macro_count = int(rng.choice(VALID_GRID_MACROS))
    grid_shape = (macro_count * MACRO_SIZE, macro_count * MACRO_SIZE)
    layer_count = int(rng.integers(MIN_LAYERS, MAX_LAYERS + 1))

    cell_ids = sorted(map(int, rng.choice(macro_count * macro_count, size=layer_count, replace=False)))
    colors = list(map(int, rng.choice(FG_COLORS, size=layer_count, replace=False)))
    glyph_names = VALID_GLYPH_SEQUENCES[layer_count][int(rng.integers(len(VALID_GLYPH_SEQUENCES[layer_count])))]

    inp = np.full(grid_shape, Color.ORANGE, dtype=int)
    for cell_id, color, glyph_name in zip(cell_ids, colors, glyph_names):
        mr, mc = divmod(cell_id, macro_count)
        r0 = mr * MACRO_SIZE
        c0 = mc * MACRO_SIZE
        inp[r0 : r0 + MACRO_SIZE, c0 : c0 + MACRO_SIZE][GLYPHS[glyph_name]] = color

    try:
        layers = _extract_layers(inp)
        lattice, overlap_cells = _compose_lattice(layers, _output_shape_for_input(inp.shape))
        out = _apply_center_stamp(lattice, layers)
    except Exception:
        return None

    visible_colors = set(map(int, np.unique(out))) - {Color.BLACK, Color.MAROON}
    input_colors = {int(layer["color"]) for layer in layers}
    if overlap_cells == 0:
        return None
    if visible_colors != input_colors:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 29")
