# Variant of d59b0160
# Dimension: D3 — 图例区域的结构与位置 (symbol_instructions)
# Change: 图例位于网格左上角，由颜色3（绿色）构成L形边框围成的3x3区域，图例内非背景像素的颜色构成目标颜色集... → 图例由分散的单像素标记组成
#
# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#     - 16x16 grid, background color 7.
#     - Non-legend objects are 4-connected rectangular components filled with color 0 and sprinkled palette colors.
#     - Delete exactly those components whose color set contains every target color.
#   D3_change:
#     - Remove the top-left legend box entirely.
#     - Place four single green control pixels on the outer border: one on each edge.
#     - Each green border pixel has exactly one inward 4-neighbor that is a colored swatch.
#     - The union of those four swatch colors is the target set; there are still exactly 3 distinct targets, so one target repeats once.
#   concrete_behavior_trace:
#     - Old task: solver scanned the top-left corner for an L-shaped border, then read target colors from one compact 3x3 interior.
#     - New task: solver must scan the whole border for green singletons, then read one adjacent swatch per marker.
#     - Example: if edge pairs are top=(3->6), bottom=(3->1), left=(3->4), right=(3->6), then the legend set is {1,4,6}.
#       A component containing {0,1,4,6,9} is deleted; a component containing {0,1,6,8} stays because color 4 is missing.
#   complexity_calibration:
#     - Using four markers instead of three gives repetition support and makes the edge-search pattern visible in 2-3 examples.
#     - Restricting markers to one per edge keeps the control grammar simple and avoids combinatorial ambiguity.
#     - Palette still excludes color 3, so green remains a unique legend cue instead of blending with object colors.
#   implicit_assumption_audit_for_main:
#     - `_find_legend` from the original code assumed a compact top-left square and would now always fail.
#     - The original component pass over `g != bg` is still valid, but now it also includes tiny control-marker components.
#       That is harmless because they never contain all 3 targets.
#     - The new marker reader assumes every green border pixel has exactly one non-bg, non-green adjacent swatch.
#       The generator guarantees this by isolating markers and keeping components padded away from them.
#   placement_legality:
#     - Edge markers never use corners; each pair occupies two cells fully inside bounds.
#     - Components are placed with a 1-cell padding from all existing non-bg cells, so they cannot touch legend pairs.
#     - Validation remains output-first: build `out`, synthesize `inp`, and require `main(inp) == out`.
#   visual_observability:
#     - The old concentrated legend disappears completely; instead, green clues are visibly scattered across all four edges.
#     - Because one target repeats across four markers, the solver must gather a set rather than read one local motif.
#   cleanliness:
#     - Reserving the actual border for controls keeps the legend visually separate from interior components.
#     - Using exactly one inward swatch per marker avoids clutter and makes each clue legible.

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_colors_from_edge_markers(g: np.ndarray, bg: int) -> set[int]:
    """
    Read the distributed legend from green edge markers.

    Every green border pixel contributes the color of its unique adjacent
    non-bg, non-green 4-neighbor. The target set is the union of those colors.
    """
    h, w = g.shape
    marker_pts = [
        (int(r), int(c))
        for r, c in np.argwhere(g == BORDER)
        if int(r) in (0, h - 1) or int(c) in (0, w - 1)
    ]
    assert marker_pts, "no edge markers found"

    legend_colors: set[int] = set()
    for r, c in marker_pts:
        adj = set()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                color = int(g[nr, nc])
                if color not in (bg, BORDER):
                    adj.add(color)
        assert len(adj) == 1, "edge marker must touch exactly one swatch color"
        legend_colors |= adj
    return legend_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose colors match a distributed edge legend.

    Input structure:
    - Background (bg) fills most of the grid.
    - On the outer border, several single green control pixels are scattered; each one has a single colored swatch immediately inward from the edge.
    - Elsewhere: multicolored 4-connected objects on the bg.

    Rule:
    - Collect the target color set by scanning all green edge markers and reading the adjacent swatch color next to each marker.
    - For every non-bg connected component, check whether its color set contains **all** target colors.
    - Erase (fill with bg) every component that does; leave all others unchanged.
    - Output size stays the same.

    Key insight: The legend is no longer a boxed local region; it is encoded as distributed edge clues, so the solver must aggregate colors globally before filtering components.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_colors = _legend_colors_from_edge_markers(g, bg)
    assert legend_colors, "legend contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (D3 variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [7, 2, 1]
TALL_H = (6, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (4, 8)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [8, 4, 1]
NUM_KEPT = 3

MAX_RETRIES = 1200
EDGE_ORDER = ("top", "bottom", "left", "right")


def _edge_pair(edge: str, pos: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if edge == "top":
        return (0, pos), (1, pos)
    if edge == "bottom":
        return (GRID_H - 1, pos), (GRID_H - 2, pos)
    if edge == "left":
        return (pos, 0), (pos, 1)
    return (pos, GRID_W - 1), (pos, GRID_W - 2)


def _place_distributed_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Place four green edge markers, one per edge, with inward swatches."""
    repeated = targets + [random.choice(targets)]
    random.shuffle(repeated)

    positions = {
        "top": random.randint(5, GRID_W - 3),
        "bottom": random.randint(2, GRID_W - 3),
        "left": random.randint(5, GRID_H - 3),
        "right": random.randint(2, GRID_H - 3),
    }

    for edge, color in zip(EDGE_ORDER, repeated):
        marker, swatch = _edge_pair(edge, positions[edge])
        for g in (inp, out):
            g[marker] = BORDER
            g[swatch] = color


def _component_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_distributed_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not deleted:
            paste(out, patch, r, c)

    if np.any(inp[:4, :4] != BG):
        return None

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
