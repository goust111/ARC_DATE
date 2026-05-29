# Variant of d59b0160
# Dimension: D5 — 图例颜色集合的提取方式 (legend_lookup)
# Change: 图例区域内所有非背景(7)、非边框(3)颜色的集合作为目标颜色集... → 图例颜色的出现次数决定匹配阈值

import math
import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis:
# - Concrete appearance: the top-left 4x4 legend still uses border color 3 on its bottom row
#   and right column, but the 3x3 interior can repeat colors. Repetition is meaningful:
#   if the legend shows color 5 twice and color 8 once, then a component must contain
#   at least two 5-pixels and at least one 8-pixel to be deleted.
# - Behavior trace:
#   original rule -> legend {5,8,9}, component colors {5,8,9} => delete.
#   new rule      -> legend counts {5:2,8:1,9:1}, component counts {5:1,8:3,9:1} => keep.
#   The decisive change is at the matching step: set inclusion becomes per-color counting.
# - Complexity calibration: keep exactly 3 target colors as in the base task, but sample
#   count patterns from a small family such as [2,1,1], [2,2,1], [3,1,1]. This keeps the
#   visual language simple while making repeated legend cells visibly relevant.
# - Hidden-assumption audit for main():
#   _find_legend in the original solver assumed unique colors and returned a set; that would
#   silently discard multiplicity, which is no longer harmless.
#   The component loop originally built only a color set; under the variant that would wrongly
#   delete "under-quota" components, so it must build exact pixel counts.
# - Placement legality: the legend must not delete itself. To guarantee this, one occurrence of
#   a repeated target color is placed at (0,0) and its neighbors (0,1) and (1,0) stay bg, so
#   the border-connected legend component is always short of that color's full quota.
# - Visual observability: every generated grid has at least one repeated legend color and at
#   least one kept component that contains all target color kinds but misses a repeated quota
#   by exactly one pixel. Comparing input/output pairs makes the counting rule observable.
# - Visual cleanliness: components remain separated by a bg moat because the placement logic
#   reserves the whole rectangular bbox, not only the sprinkled colored pixels.
# - New solver assumptions: the solver now assumes the legend can be summarized by counts from
#   its interior and that components are still 4-connected regions of non-bg cells. The
#   generator preserves both by reusing the original rectangle-based component construction.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, dict[int, int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend requirement map is the color histogram of the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1].ravel()
            counts = Counter(int(v) for v in inside if int(v) not in (bg, x))
            return int(sz), int(x), dict(counts)
    assert False, "no legend border found in grid"


def _matches_requirements(component_counts: Counter, legend_counts: dict[int, int]) -> bool:
    return all(component_counts.get(color, 0) >= need for color, need in legend_counts.items())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose color tallies satisfy a legend quota.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior lists target colors, and some colors may appear multiple times.
    - Elsewhere: multicolored 4-connected objects sit on the bg.

    Rule:
    - Read the per-color pixel requirements from the legend interior by counting how many times each non-bg, non-border color appears there.
    - For each connected component, count how many pixels it has of each color.
    - Erase (fill with bg) every component whose counts meet or exceed the legend counts for every target color; leave all other components unchanged.
    - Output grid keeps the same shape as the input.

    Key insight: The legend is not just a palette anymore; it is a quota table, so matching depends on repeated swatches and exact color multiplicities inside each component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_counts = _find_legend(g, bg)
    assert legend_counts, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        comp_counts = Counter(int(g[r, c]) for r, c in pts)
        if _matches_requirements(comp_counts, legend_counts):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (o5x-d5-counting) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
COUNT_PATTERNS = [(2, 1, 1), (2, 2, 1), (3, 1, 1)]

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 600


def _sample_requirements() -> dict[int, int]:
    targets = random.sample(PALETTE, NUM_TARGETS)
    pattern = list(random.choice(COUNT_PATTERNS))
    random.shuffle(pattern)
    return {color: count for color, count in zip(targets, pattern)}


def _place_legend(inp: np.ndarray, out: np.ndarray, requirements: dict[int, int]) -> None:
    """Draw the 4x4 legend box and place repeated target colors according to the quota map."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    repeated_colors = [color for color, count in requirements.items() if count > 1]
    anchor_color = random.choice(repeated_colors)

    # Keep (0,0) isolated by leaving (0,1) and (1,0) as BG.
    positions = [(0, 0)]
    positions.extend(random.sample([(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], sum(requirements.values()) - 1))

    cells = [anchor_color]
    remaining = requirements.copy()
    remaining[anchor_color] -= 1
    for color, count in remaining.items():
        cells.extend([color] * count)
    tail = cells[1:]
    random.shuffle(tail)
    cells = [cells[0]] + tail

    for color, (r, c) in zip(cells, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims(min_needed: int) -> tuple[int, int]:
    """Sample component dimensions whose colored-capacity can accommodate min_needed pixels."""
    min_area = max(1, math.ceil(min_needed / DENSITY_HI))
    for _ in range(100):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if h * w >= min_area:
            return h, w
    return max(3, min_area), max(2, math.ceil(min_area / max(3, min_area)))


def _sample_colored_count(area: int, min_needed: int) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO), min_needed)
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _deleted_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    area = h * w
    n_colored = _sample_colored_count(area, sum(requirements.values()))
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = []
    for color, need in requirements.items():
        colors.extend([color] * need)
    colors.extend(random.choice(PALETTE) for _ in range(n_colored - len(colors)))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _under_quota_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    area = h * w
    shortage_color = random.choice([color for color, need in requirements.items() if need > 1])
    min_needed = sum(requirements.values()) - 1
    n_colored = _sample_colored_count(area, min_needed)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    colors = []
    for color, need in requirements.items():
        take = need - 1 if color == shortage_color else need
        colors.extend([color] * take)

    extras_pool = [color for color in PALETTE if color != shortage_color]
    colors.extend(random.choice(extras_pool) for _ in range(n_colored - len(colors)))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _missing_kind_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    area = h * w
    n_colored = _sample_colored_count(area, 1)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    omitted = random.choice(list(requirements))
    remaining_targets = [color for color in requirements if color != omitted]
    used_target_count = random.randint(0, len(remaining_targets))
    included_targets = random.sample(remaining_targets, used_target_count)
    allowed = [color for color in PALETTE if color != omitted] + included_targets
    colors = [random.choice(allowed) for _ in range(n_colored)]

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_patch(h: int, w: int, requirements: dict[int, int], role: str) -> np.ndarray:
    if role == "deleted":
        return _deleted_patch(h, w, requirements)
    if role == "under_quota":
        return _under_quota_patch(h, w, requirements)
    return _missing_kind_patch(h, w, requirements)


def _min_needed_for_role(requirements: dict[int, int], role: str) -> int:
    if role == "deleted":
        return sum(requirements.values())
    if role == "under_quota":
        return sum(requirements.values()) - 1
    return 1


def _component_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    kept_roles = ["under_quota", "missing_kind", random.choice(["missing_kind", "under_quota"])]
    roles = ["deleted"] * deleted_count + kept_roles
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input; return it or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    requirements = _sample_requirements()
    _place_legend(inp, out, requirements)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _component_roles(component_count)

    for role in roles:
        h, w = _component_dims(_min_needed_for_role(requirements, role))
        patch = _component_patch(h, w, requirements, role)
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
        if role != "deleted":
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
