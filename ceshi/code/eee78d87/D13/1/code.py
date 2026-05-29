# Variant of eee78d87
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 单一流程：提取字形 -> 解码相位瓦片 -> 铺满画布 -> 放置中心印章，无迭代... → 分形迭代放大

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis for the D13 fractal-scaling variant
# ---------------------------------------------------
# concrete_behavior:
#   input:
#   - Keep the original 6x6 orange board with exactly one non-orange 3x3 contour glyph.
#   - The glyph still determines the same decoded 3x3 orange/black phase tile as in the base task.
#   output:
#   - Step 1 builds a coarse 6x6 raster: repeat the decoded phase tile over the coarse board.
#   - Step 2 overwrites the centered coarse 2x2 block with maroon cells.
#   - Step 3 performs one recursive expansion: each coarse pixel becomes a 3x3 child tile
#     shaped like the original glyph mask. Black coarse pixels expand to all-black tiles,
#     orange coarse pixels expand to orange glyph tiles, and maroon coarse pixels expand to
#     maroon glyph tiles. The final canvas is therefore 18x18.
#   behavior_trace:
#   - Base task on a plus glyph: decode plus -> 3x3 phase tile -> tile directly to a 16x16
#     orange/black lattice -> stamp a flat 6x6 maroon 2x2 repetition at the center.
#   - Variant on the same plus glyph: decode plus -> tile phase on a coarse 6x6 raster ->
#     place a coarse 2x2 maroon block -> recursively replace every non-black coarse cell by a
#     plus-shaped 3x3 child tile. The center is no longer a flat patch; it is a 2x2 array of
#     plus stamps, and the orange lattice is likewise made of repeated plus stamps.
#   - Difference: the output now exposes an intermediate macro pattern and a micro pattern,
#     so the solver must recover a two-level construction rather than a single direct fill.
# complexity_calibration:
#   - Only four glyph classes remain, matching the original task's discrete branch structure.
#   - The recursive layer is visually obvious because the output size changes from 16x16 to
#     18x18 and every non-black area is internally perforated by the glyph silhouette.
#   - The freedom stays controlled: no extra colors, no rotations, no multiple recursive
#     depths, and a fixed one-step expansion factor of 3.
# implicit_assumption_review:
#   old_assumptions_changed:
#   - The base solver assumed the decoded phase tile was written straight to the final canvas.
#     That is no longer true; the phase tiling now populates an intermediate coarse raster.
#   - The base solver assumed the center stamp was a direct 6x6 overwrite. That would destroy
#     the new fractal structure, so the overwrite must happen before expansion at coarse scale.
#   - The base solver's final shape formula (3*h-2) relied on direct tiling. The variant needs
#     a clean macro grid that can be expanded by 3, so the final size becomes 3*h by 3*w.
#   new_assumptions:
#   - The foreground glyph still has a tight 3x3 bbox and matches one of the four legal
#     contour forms validated by _decode_phase_tile.
#   - The expanded output assumes coarse pixels use only black/orange/maroon, which main()
#     guarantees by construction.
# placement_legality:
#   - Input placement remains the original safe regime: one 3x3 motif with top-left in 0..3,
#     so it never clips the 6x6 board.
#   - The centered coarse 2x2 maroon block occupies rows 2..3 and cols 2..3 of the 6x6 macro
#     raster, yielding a centered 6x6 maroon-bbox in the 18x18 final output.
# visual_observability:
#   - Across examples, different glyphs create distinct micro-tiles, and the same glyph is
#     reused everywhere in the output, making the recursive rule inferable by eye.
#   - The change does not depend on color ids or hidden ordering; only visible shape and color
#     placement matter.
# visualization_cleanliness:
#   - Layers do not overlap ambiguously: the macro lattice decides where colored child tiles go,
#     while the glyph mask decides each tile's internal holes. All holes are black, so the
#     composition stays legible.
# color_inference:
#   - Colors remain structurally simple: orange marks phase-lattice cells, maroon marks the
#     centered stamp cells, and black is the shared empty background inside and between tiles.


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


def _build_coarse_raster(motif: np.ndarray) -> np.ndarray:
    """Create the intermediate 6x6 raster before recursive expansion."""
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    coarse = tile_fill((GRID_H, GRID_W), phase_colors)

    r0 = (coarse.shape[0] - CENTER_MACRO_SIZE) // 2
    c0 = (coarse.shape[1] - CENTER_MACRO_SIZE) // 2
    coarse[r0 : r0 + CENTER_MACRO_SIZE, c0 : c0 + CENTER_MACRO_SIZE] = Color.MAROON
    return coarse


def _expand_by_glyph(coarse: np.ndarray, motif: np.ndarray) -> np.ndarray:
    """Expand each coarse pixel into a 3x3 tile shaped like the input glyph."""
    assert coarse.shape == (GRID_H, GRID_W)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    out = np.full((coarse.shape[0] * MOTIF_SIZE, coarse.shape[1] * MOTIF_SIZE), Color.BLACK, dtype=int)
    child_tiles = {
        Color.BLACK: np.full((MOTIF_SIZE, MOTIF_SIZE), Color.BLACK, dtype=int),
        Color.ORANGE: np.where(motif, Color.ORANGE, Color.BLACK).astype(int),
        Color.MAROON: np.where(motif, Color.MAROON, Color.BLACK).astype(int),
    }

    for r in range(coarse.shape[0]):
        for c in range(coarse.shape[1]):
            color = int(coarse[r, c])
            assert color in child_tiles
            rr = r * MOTIF_SIZE
            cc = c * MOTIF_SIZE
            out[rr : rr + MOTIF_SIZE, cc : cc + MOTIF_SIZE] = child_tiles[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored contour glyph into a coarse phase lattice, then recursively expand that lattice with the same glyph as the child tile.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange color.
    - The non-orange pixels form one tight 3x3 contour glyph somewhere inside the board.
    - The glyph is one of the four legal contour codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 contour glyph into the original period-3 orange/black phase tile.
    - Build a coarse 6x6 lattice by tiling that phase tile and overwrite the centered coarse 2x2 block with maroon.
    - Expand each coarse pixel into a 3x3 child tile shaped like the input glyph: orange cells become orange glyph tiles, maroon cells become maroon glyph tiles, and black cells stay black.

    Key insight:
    - The input glyph now serves two roles at different scales: it still decodes the macro lattice, and it is also the micro stamp used to recursively enlarge every colored macro cell.
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

    coarse = _build_coarse_raster(motif)
    return _expand_by_glyph(coarse, motif)


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CENTER_MACRO_SIZE = 2
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    assert out.shape == (GRID_H * MOTIF_SIZE, GRID_W * MOTIF_SIZE)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D13 variant")
