# Variant of eee78d87
# Dimension: D14 — 前景色的选择范围与语义 (meta)
# Change: 字形前景色从{1,2,3,4,5,6,8}中任选，排除黑色(0)、橙色(7)、栗色(9)，颜色仅起标识作用无额外语义... → 多个字形使用不同前景色形成多层

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D14 variant
# a) Concrete behavior:
#    The input keeps the original 6x6 orange board and 3x3 contour-glyph vocabulary,
#    but now there are exactly two foreground colors. Each color owns one complete 3x3
#    glyph bbox, so decoding must happen per color instead of on the union mask.
#    Example trace:
#    - Original: one blue X -> decode one phase tile -> repeat one orange lattice ->
#      stamp one maroon 2x2 motif at the center.
#    - Variant: one blue 4corners and one red H -> decode both independently ->
#      repeat a blue lattice and a red lattice -> paint the earlier bbox layer first
#      and the later bbox layer second -> stamp the union of the two 2x2 motifs in maroon.
#    The changed step is therefore the "single foreground extraction" stage: it becomes
#    a two-source extraction, followed by explicit layer composition.
# b) Complexity calibration:
#    The rule must stay human-visible in 2-4 examples. If the sampled pair never overlaps
#    in phase space, the priority rule would exist in code but be invisible. The generator
#    therefore chooses ordered glyph pairs whose decoded phase masks overlap and also leave
#    at least one uncovered cell from the lower-priority layer, ensuring both colors remain
#    visible in the output.
# c) Implicit assumptions in the original main():
#    - `len(colors_present(...)) == 1` no longer holds and must become `== 2`.
#    - `fg = g != bg` merged every non-orange pixel into one motif; that is invalid now
#      because disconnected colors can describe two unrelated glyphs.
#    - `crop_bbox(fg, b)` assumed one global bbox; the variant needs one bbox per color.
#    - `phase_colors = np.where(phase, bg, black)` assumed a fixed orange lattice color.
#      The variant must preserve each glyph's own color in its decoded layer.
#    - The center stamp was a single motif. Now both motifs contribute, but the original
#      maroon "stamp" idea is preserved by taking the union after 2x2 tiling.
# d) Placement legality:
#    The generator only uses pairs of 3x3 bboxes that fit in the 6x6 input without
#    overlap. Each bbox contains only orange and its own foreground color, so solver-side
#    color-specific cropping is well-defined.
# e) Visual observability:
#    Layer priority is based on visible row-major bbox order: the glyph that appears later
#    in reading order is drawn on top. This avoids any hidden dependence on numeric color id.
# f) Visual cleanliness:
#    There are only two lattice colors plus black and the single maroon center stamp. The
#    composition remains easy to parse because the two layers share one periodic scaffold
#    instead of creating many independent regions.
# g) Color inference:
#    Input colors now matter only as persistent layer labels. The output reuses those same
#    two colors directly in the lattice, so the solver can infer "same color => same layer"
#    without learning an arbitrary remapping table.


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 phase mask."""
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


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

PHASES = {name: _decode_phase_tile(motif) for name, motif in GLYPHS.items()}


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_from_topleft(r0: int, c0: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + 2, c0 + 2


def _extract_layers(g: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2

    layers: list[dict[str, object]] = []
    for color in fg_colors:
        mask = g == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)

        cropped = crop_bbox(g, b)
        assert np.isin(cropped, [bg, color]).all()
        motif = cropped == color
        assert any(np.array_equal(motif, glyph) for glyph in GLYPHS.values())

        layers.append(
            {
                "color": int(color),
                "bbox": b,
                "motif": motif,
                "phase": _decode_phase_tile(motif),
            }
        )

    assert not _boxes_overlap(layers[0]["bbox"], layers[1]["bbox"])
    layers.sort(key=lambda layer: (layer["bbox"][0], layer["bbox"][1]))
    return bg, layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two colored contour glyphs into stacked periodic lattices, then seal their combined motif as a centered maroon stamp.

    Input structure:
    - The grid is 6x6 with orange background and exactly two non-orange colors.
    - Each foreground color occupies its own tight 3x3 bbox and forms one valid contour glyph.
    - The two 3x3 glyph bboxes do not overlap; their row-major order gives the draw priority.

    Rule:
    - Decode each 3x3 glyph independently into the task's period-3 phase tile.
    - Repeat each decoded tile across a black 16x16 canvas using that glyph's own input color.
    - Composite the two repeated layers so the later bbox in row-major order overwrites the earlier one, then stamp the 2x2-tiled union of both glyphs in maroon inside the centered 6x6 window.

    Key insight:
    - Foreground colors are no longer disposable labels for the same object type; each color identifies a separate lattice layer, so solving the task requires per-color decoding and visible layer composition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    _bg, layers = _extract_layers(g)

    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = np.full(out_shape, Color.BLACK, dtype=int)

    for layer in layers:
        phase_colors = np.where(layer["phase"], layer["color"], Color.BLACK).astype(int)
        tiled = tile_fill(out_shape, phase_colors)
        mask = tiled != Color.BLACK
        out[mask] = tiled[mask]

    center_mask = np.zeros((6, 6), dtype=bool)
    for layer in layers:
        center_mask |= np.tile(layer["motif"], (2, 2))

    r0 = (out.shape[0] - center_mask.shape[0]) // 2
    c0 = (out.shape[1] - center_mask.shape[1]) // 2
    out[r0 : r0 + center_mask.shape[0], c0 : c0 + center_mask.shape[1]][center_mask] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

POSITIONS = [(r, c) for r in range(MOTIF_POS_MAX) for c in range(MOTIF_POS_MAX)]
VALID_POSITION_PAIRS = [
    (p1, p2)
    for i, p1 in enumerate(POSITIONS)
    for p2 in POSITIONS[i + 1 :]
    if not _boxes_overlap(_bbox_from_topleft(*p1), _bbox_from_topleft(*p2))
]

# Use only ordered pairs where:
# 1) the two decoded phases overlap, so priority matters;
# 2) the lower-priority layer keeps at least one unique visible cell after overlay,
#    so both input colors appear in the output.
VISIBLE_ORDERED_GLYPH_PAIRS = [
    (low_name, high_name)
    for low_name, low_phase in PHASES.items()
    for high_name, high_phase in PHASES.items()
    if bool((low_phase & high_phase).any()) and bool((low_phase & ~high_phase).any())
]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    (low_r0, low_c0), (high_r0, high_c0) = VALID_POSITION_PAIRS[int(rng.integers(len(VALID_POSITION_PAIRS)))]
    low_name, high_name = VISIBLE_ORDERED_GLYPH_PAIRS[int(rng.integers(len(VISIBLE_ORDERED_GLYPH_PAIRS)))]
    low_color, high_color = map(int, rng.choice(FG_COLORS, size=2, replace=False))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[low_r0 : low_r0 + MOTIF_SIZE, low_c0 : low_c0 + MOTIF_SIZE][GLYPHS[low_name]] = low_color
    inp[high_r0 : high_r0 + MOTIF_SIZE, high_c0 : high_c0 + MOTIF_SIZE][GLYPHS[high_name]] = high_color

    try:
        out = main(inp)
    except Exception:
        return None

    out_colors = set(map(int, np.unique(out)))
    if not {low_color, high_color, Color.MAROON}.issubset(out_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D14 variant")
