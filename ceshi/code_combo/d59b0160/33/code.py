# Combo variant of d59b0160 (index 33)
# combo_spec: D9/3+D15/1
# Name: 图例参与匹配+单一图例颜色
#
# Design analysis
# ===============
# fusion_rule:
#   - Baseline d59b0160 reads a color set from the top-left legend, then deletes every external
#     4-connected multicolor component whose color-set contains all legend colors.
#   - D15/1 collapses the legend palette to a single target color T, so the matching test becomes
#     "does this candidate contain T?" rather than "does it contain all three targets?".
#   - D9/3 says the legend panel itself also participates in matching. Because the whole 4x4 legend
#     square visibly contains T, that square is always deleted as one candidate region even though
#     its colored pixels are not a single 4-connected component in the raw bitmap.
#   - The fused rule is therefore:
#       1. Read the lone target color T from the legend interior.
#       2. Treat the full legend square as one extra candidate region and erase it.
#       3. Separately scan the non-legend area for ordinary 4-connected foreground components.
#       4. Erase every external component whose color-set contains T; keep the others.
#
# behavior_trace:
#   - Example: the legend interior contains only color 6.
#   - Original task with a 3-color legend would preserve the legend and would erase only components
#     containing all three target colors.
#   - Single-color-only behavior without D9/3 would preserve the legend but erase every external
#     component containing color 6.
#   - Fused behavior does the same external erasures and additionally turns the entire 4x4 legend
#     square into background. This top-left disappearance is the crucial visible difference.
#
# complexity_calibration:
#   - If the generator produced no external component carrying T, the task would degenerate into
#     "just delete the legend". To avoid that, every sample contains 1-3 deleted external components.
#   - If almost every component carried T, the board would become too empty and the rule would look
#     noisy rather than structured. We keep exactly 3 external survivors so both classes are visible.
#   - The legend still uses the original L-shaped border and sparse interior, so the task stays
#     recognizable as d59b0160 rather than becoming a different visual family.
#
# implicit_assumption_review:
#   - `_find_legend(...)` still assumes a top-left square whose bottom row and right column share one
#     border color. The generator keeps that geometry unchanged.
#   - The old solver's `components_mask(g != bg, connectivity=4)` is no longer sufficient by itself:
#     it would split the legend into several tiny pieces, but D9/3 requires treating the full legend
#     square as one matchable region. The new solver handles the legend explicitly, then analyzes only
#     the non-legend area with connected components.
#   - The old "legend has multiple colors" assumption is broken. The new solver asserts that exactly
#     one target color is present, and the generator constructs exactly one interior swatch.
#
# placement_legality:
#   - The legend occupies the top-left 4x4 block.
#   - External objects are still placed as full rectangular patches with `random_free_location(...)`
#     and `padding=1`, so they neither overlap each other nor touch the legend by 8-neighborhood.
#   - Component sizes remain within the original ranges and fit inside the 16x16 board with retries.
#
# visual_observability_and_cleanliness:
#   - The legend's disappearance is directly visible in the output and does not rely on color IDs or
#     other hidden metadata.
#   - External deleted objects all share one visible trait: they contain the lone legend color.
#     External kept objects all lack that color. This gives repeated support for the single-color rule.
#   - The grid remains visually tidy: one legend panel plus separated rectangular objects, with no
#     layered annotations or ambiguous overlaps.
#
# generator_guarantees_for_solver_assumptions:
#   - Exactly one legend color is present.
#   - At least one and at most three external components contain that target color.
#   - Exactly three external components omit the target color and survive.
#   - Every generated sample differs from the "single-color but legend excluded" baseline because the
#     fused solver always erases the whole legend square.

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend square whose bottom row and right column share one border color.
    Return (size, border_color, interior_target_colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_color)
            return int(sz), int(border_color), colors
    raise AssertionError("no legend border found in grid")


def _single_color_external_only(grid: np.ndarray) -> np.ndarray:
    """Single-color legend baseline that keeps the legend panel intact."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert len(legend_colors) == 1, "expected a single legend color"

    out = g.copy()
    external_mask = (g != bg).copy()
    external_mask[:legend_size, :legend_size] = False
    for pts in components_mask(external_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every candidate region that contains the lone legend color, including the legend itself.

    Input structure:
    - One background color fills most of the board and is recoverable from the outer frame.
    - The top-left corner holds the usual small legend square with a shared border color on its
      bottom and right edges.
    - The legend interior now shows exactly one non-background, non-border color.
    - Outside the legend sit several separated 4-connected multicolor components on the background.

    Rule:
    - Read the single target color from the legend interior.
    - Treat the entire legend square as one candidate region and erase it, because that region
      necessarily contains the target color.
    - Find all ordinary 4-connected foreground components outside the legend square.
    - Erase each external component whose color-set contains the target color; leave every component
      that lacks the target color unchanged.

    Key insight:
    - With only one legend color, the old "contains the full palette" test collapses to a simple
      presence test, and D9/3 makes the legend panel itself part of the objects being filtered.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert len(legend_colors) == 1, "legend must contain exactly one target color"

    out = g.copy()
    out[:legend_size, :legend_size] = bg

    external_mask = (g != bg).copy()
    external_mask[:legend_size, :legend_size] = False

    for pts in components_mask(external_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 1

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3
MAX_RETRIES = 500


def _place_legend(grid: np.ndarray, target: int) -> None:
    """Draw the 4x4 legend with exactly one interior target swatch."""
    s = LEGEND_SIZE
    grid[:s, :s] = BG
    grid[s - 1, :s] = BORDER
    grid[:s, s - 1] = BORDER

    r, c = random.choice([(rr, cc) for rr in range(s - 1) for cc in range(s - 1)])
    grid[r, c] = int(target)


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions using the original medium / tall / wide modes."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, target: int, deleted: bool) -> np.ndarray:
    """Build one rectangular multicolor component on a color-0 base."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, min(area, math.floor(area * DENSITY_HI)))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v != target]

    if deleted:
        colors = [target] + [random.choice(PALETTE) for _ in range(n_colored - 1)]
    else:
        colors = [random.choice(non_targets) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _count_deleted_external_components(grid: np.ndarray) -> int:
    """Count how many external components satisfy the fused deletion test."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)

    count = 0
    external_mask = (g != bg).copy()
    external_mask[:legend_size, :legend_size] = False
    for pts in components_mask(external_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid fused-variant input."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    occupancy = grid.copy()

    target = random.choice(PALETTE)
    _place_legend(grid, target)
    occupancy[:LEGEND_SIZE, :LEGEND_SIZE] = BORDER

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, target, deleted)
        try:
            r, c = random_free_location(
                occupancy,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, r, c)
        paste(grid, patch, r, c)

    try:
        legend_size, border_color, legend_colors = _find_legend(grid, BG)
    except AssertionError:
        return None
    if legend_size != LEGEND_SIZE or border_color != BORDER or legend_colors != {target}:
        return None

    try:
        fused_out = main(grid)
        baseline_out = _single_color_external_only(grid)
    except AssertionError:
        return None
    if np.array_equal(fused_out, baseline_out):
        return None

    deleted_external = _count_deleted_external_components(grid)
    if not (1 <= deleted_external <= component_count - NUM_KEPT):
        return None

    if not np.all(fused_out[:LEGEND_SIZE, :LEGEND_SIZE] == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
