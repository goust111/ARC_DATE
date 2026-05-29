# Combo variant of d59b0160 (index 8)
# combo_spec: D4/1+D12/0
# Name: 同色连通分量+颜色0可作图例颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the fused variant:
# - Original behavior grouped every non-bg pixel into multicolor 4-connected objects and
#   removed an object only when its color-set contained the full legend palette.
# - D4/1 changes the object model itself: components are now monochromatic 4-connected
#   regions, so a component no longer has an internal palette to compare against.
# - D12/0 lets color 0 appear in the legend; because bg is still 7, black regions are
#   visible objects and must be matched like any other target color.
# - Therefore the matching rule becomes per-component color equality:
#   every external monochrome component whose single color is listed in the legend is
#   erased. This is the only interpretation that makes the two changes work together.
# - A concrete example is legend {0, 4, 8}: under the original rule a black rectangle,
#   a yellow rectangle, and an azure rectangle would all survive because none contains
#   all three colors together; under the fused rule all three are deleted individually.
# - The solver must now exclude the legend square explicitly. Otherwise the legend's own
#   target swatches would also be monochrome components and would be deleted incorrectly.
# - The generator switches from sparse multicolor rectangles to separated solid-color
#   rectangles. That keeps the single-color component structure visually clean and makes
#   "all components of these legend colors disappear" inferable from a few examples.
# - To keep D12/0 observable, every generated grid contains at least one external black
#   component; when 0 is listed in the legend it disappears, otherwise it stays.


def _bg_boundary(g: np.ndarray) -> int:
    """Background is the most common color along the outer frame."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend whose bottom row and right column share one border color.

    The returned color set comes from the legend interior and excludes bg and border.
    Color 0 is allowed to remain in that set.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_col = int(g[sz - 1, sz - 1])
        if border_col == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_col) and np.all(g[0:sz, sz - 1] == border_col):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_col)
            return int(sz), int(border_col), colors
    raise AssertionError("no legend border found in grid")


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original d59b0160 behavior, kept only to reject degenerate generated samples."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete monochrome connected components whose single color appears in the legend.

    Input structure:
    - The background color is defined by the outer frame and still fills most cells.
    - A top-left legend square is bordered on its bottom and right edges by one color.
    - The legend interior lists target colors, and color 0 may now be one of them.
    - Outside the legend, objects are separated monochrome 4-connected components.

    Rule:
    - Read the target color set from the legend interior, keeping color 0 if present.
    - Ignore the legend area itself; it is only a specification panel.
    - For each target color, erase every external 4-connected component made of exactly
      that color by filling it with the background.
    - Leave all components of non-legend colors unchanged.

    Key insight:
    - The fusion moves the test from "does a multicolor object contain the whole legend
      palette?" to "is this monochrome component's color listed in the legend?".
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend_size, _border_col, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    external_mask = np.ones_like(g, dtype=bool)
    external_mask[:legend_size, :legend_size] = False

    for color in sorted(legend_colors):
        comps = components_mask((g == color) & external_mask, connectivity=4)
        for pts in comps:
            out[pts[:, 0], pts[:, 1]] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
TARGET_COUNT = 3

ALL_OBJECT_COLORS = [0, 1, 2, 4, 5, 6, 8, 9]
NONZERO_OBJECT_COLORS = [1, 2, 4, 5, 6, 8, 9]

SHAPE_WEIGHTS = [4, 3, 3]  # compact, tall, wide
COMPACT_H = (2, 4)
COMPACT_W = (2, 4)
TALL_H = (4, 7)
TALL_W = (2, 3)
WIDE_H = (2, 3)
WIDE_W = (4, 7)

KEEP_COUNT = 3
MAX_RETRIES = 500


def _place_legend(grid: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend in the top-left corner."""
    s = LEGEND_SIZE
    grid[:s, :s] = BG
    grid[s - 1, :s] = BORDER
    grid[:s, s - 1] = BORDER

    positions = random.sample([(r, c) for r in range(s - 1) for c in range(s - 1)], len(targets))
    for color, (r, c) in zip(targets, positions):
        grid[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample a small rectangle size that leaves room for several separated components."""
    mode = random.choices(["compact", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*COMPACT_H), random.randint(*COMPACT_W)


def _sample_targets() -> list[int]:
    """Choose three legend colors; color 0 is included often enough to stay observable."""
    if random.random() < 0.65:
        targets = [0] + random.sample(NONZERO_OBJECT_COLORS, TARGET_COUNT - 1)
    else:
        targets = random.sample(NONZERO_OBJECT_COLORS, TARGET_COUNT)
    random.shuffle(targets)
    return targets


def _component_plan(targets: list[int]) -> list[int]:
    """
    Produce a color list for external monochrome components.

    Every sample contains:
    - at least one component for each target color,
    - one duplicated target color to make deletion-by-color easy to see,
    - exactly three kept components,
    - at least one external black component, deleted iff 0 is a target.
    """
    deleted_colors = list(targets) + [random.choice(targets)]
    if 0 in targets and deleted_colors.count(0) == 1:
        deleted_colors.append(0)

    non_targets = [c for c in ALL_OBJECT_COLORS if c not in targets]
    kept_colors: list[int] = []
    if 0 not in targets:
        kept_colors.append(0)

    while len(kept_colors) < KEEP_COUNT:
        if kept_colors and random.random() < 0.35:
            kept_colors.append(random.choice(kept_colors))
            continue
        choices = [c for c in non_targets if c != 0 or 0 not in kept_colors]
        kept_colors.append(random.choice(choices))

    colors = deleted_colors + kept_colors
    random.shuffle(colors)
    return colors


def _place_components(grid: np.ndarray, colors: list[int]) -> bool:
    """Place each monochrome rectangle with a guaranteed bg gap around it."""
    for color in colors:
        h, w = _component_dims()
        patch = np.full((h, w), color, dtype=int)
        try:
            r, c = random_free_location(
                grid,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(grid, patch, r, c)
    return True


def _has_black_component_outside_legend(grid: np.ndarray) -> bool:
    """Generator-side sanity check used for D12/0 observability."""
    mask = np.zeros_like(grid, dtype=bool)
    mask[LEGEND_SIZE:, :] = grid[LEGEND_SIZE:, :] == 0
    mask[:LEGEND_SIZE, LEGEND_SIZE:] = grid[:LEGEND_SIZE, LEGEND_SIZE:] == 0
    return bool(mask.any())


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input grid and reject degenerate cases."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)

    targets = _sample_targets()
    _place_legend(grid, targets)

    if not _place_components(grid, _component_plan(targets)):
        return None
    if not _has_black_component_outside_legend(grid):
        return None

    fused_out = main(grid)
    legacy_out = _legacy_main(grid)
    if np.array_equal(fused_out, legacy_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
