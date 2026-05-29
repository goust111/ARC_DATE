# Variant of d59b0160
# Dimension: D1 — 颜色匹配条件：全部包含 vs 其他逻辑 (match_cond)
# Change: 连通分量必须包含图例中的所有颜色(ALL)才被删除，缺少任何一种图例颜色则保留... → 包含图例颜色数量超过阈值则删除

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    _candidate = _parent / "solvers" / "arc_utils.py"
    if _candidate.exists():
        sys.path.insert(0, str(_candidate.parent))
        break

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D1 variant
# a) Concreteization and behavior trace
# - The legend stays in the top-left corner and still lists target colors, but it now shows 4 colors
#   instead of 3. This is necessary because with only 3 legend colors, the proposed threshold
#   ceil(3/2)=2 and the deletion test "matched count > 2" collapses back to the original ALL rule.
# - With 4 legend colors, the threshold is ceil(4/2)=2, so any component containing 3 or 4 legend
#   colors is erased, while components with 0, 1, or 2 legend colors remain.
# - Example trace:
#   legend = {1,2,4,5}; component colors = {0,1,2,4,9}
#   original rule: keep it, because 5 is missing
#   new rule: matched legend colors = 3, threshold = 2, so delete it
#   This is the exact behavioral gap the generator must expose.
# b) Complexity calibration
# - The target-count is fixed at 4 in every sample. That keeps the threshold stable and visible.
# - The generator always includes:
#   1. at least one deleted component with exactly 3 legend colors
#   2. at least one kept component with exactly 2 legend colors
#   This makes the majority-style counting rule observable from a few examples.
# c) Hidden-assumption audit against the original main()
# - _bg_boundary still works because the background remains dominant on the outer frame.
# - _find_legend still works because the legend border geometry is unchanged.
# - components_mask(g != bg) also labels non-background fragments inside the legend. Under the new
#   rule that becomes dangerous: if one legend fragment carried 3 target colors, the solver would
#   erase part of the legend itself. The generator therefore uses a safe 4-color layout where every
#   legend-side connected fragment contains at most 2 target colors.
# - The only solver-side semantic change is replacing subset inclusion with an intersection-count
#   comparison against a derived threshold.
# d) Placement legality
# - Rectangular components are placed with one-cell padding from each other and from the legend, so
#   they never merge into accidental larger components.
# - Patches remain full rectangles of non-background cells (base color 0), so the solver sees each
#   placed object as exactly one 4-connected component.
# e) Visual observability
# - Because deleted objects can now miss one legend color, the input/output pairs visibly differ from
#   the original task: some 3-of-4 components disappear even though they do not contain the full set.
# - The paired kept example with exactly 2 legend colors shows the cutoff.
# f) Visual cleanliness
# - The legend is still small and isolated in one corner; the other components remain separated with
#   background gaps, so the scene stays readable.
# g) Color-rule inferability
# - No hidden ordering, no color-id arithmetic, and no color remapping are introduced. Only the
#   count of legend colors present in each component matters, which is directly visible.
# 9) Assumptions newly introduced by the updated main()
# - The threshold is computed from the number of distinct legend colors, so the legend must contain
#   enough colors for the rule to differ from ALL; the generator guarantees exactly 4.
# - Legend fragments must not exceed the threshold; _place_legend guarantees that.
# - Every non-legend object must stay as a single isolated component; the generator guarantees that
#   via solid rectangular patches and placement padding.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    assert False, "no legend border found in grid"


def _delete_threshold(n_legend_colors: int) -> int:
    """Threshold used by the variant: delete only if matched legend colors exceed ceil(n/2)."""
    return math.ceil(int(n_legend_colors) / 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove connected components whose overlap with the legend palette is large enough.

    Input structure:
    - The background color dominates the frame.
    - A small square legend sits in the top-left corner, marked by a single-color bottom edge and
      right edge; its interior lists the target colors.
    - Elsewhere, rectangular 4-connected objects contain color 0 plus scattered colored cells.

    Rule:
    - Read the distinct target colors from inside the legend border.
    - Let the deletion threshold be ceil(number_of_target_colors / 2).
    - For each 4-connected non-background component, count how many legend colors appear in it.
    - Erase the component iff its matched-color count is strictly greater than that threshold.

    Key insight: The legend is no longer an all-or-nothing filter; it defines a color quorum, and
    components disappear once they contain more than half of the legend colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    threshold = _delete_threshold(len(legend_colors))
    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        matched = len(cols & legend_colors)
        if matched > threshold:
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (D1 variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 4
DELETE_THRESHOLD = _delete_threshold(NUM_TARGETS)

SHAPE_WEIGHTS = [5, 3, 2]
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

MAX_RETRIES = 400

SAFE_LEGEND_LAYOUTS = [
    [(0, 0), (1, 1), (0, 2), (2, 0)],
    [(0, 0), (1, 1), (0, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 0), (2, 2)],
]


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """
    Draw the 4x4 legend box and place 4 target colors in a layout that never forms a 3-target
    connected fragment inside the legend itself.
    """
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    positions = random.choice(SAFE_LEGEND_LAYOUTS)
    colors = targets[:]
    random.shuffle(colors)
    for color, (r, c) in zip(colors, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _role_match_count(role: str) -> int:
    if role == "deleted_partial":
        return DELETE_THRESHOLD + 1
    if role == "deleted_full":
        return NUM_TARGETS
    if role == "kept_near":
        return DELETE_THRESHOLD
    if role == "kept_low":
        return random.randint(0, DELETE_THRESHOLD - 1)
    raise ValueError(f"unknown role: {role}")


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """Build an h x w solid component with an exact number of distinct legend colors."""
    area = h * w
    match_count = _role_match_count(role)
    lo = max(1, math.ceil(area * DENSITY_LO), match_count)
    hi = max(lo, min(area, math.floor(area * DENSITY_HI)))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    chosen_targets = random.sample(targets, match_count)
    non_targets = [v for v in PALETTE if v not in targets]

    if chosen_targets:
        allowed = non_targets + chosen_targets
        colors = chosen_targets + [random.choice(allowed) for _ in range(n_colored - match_count)]
    else:
        colors = [random.choice(non_targets) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _build_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    delete_roles = ["deleted_partial"]
    for _ in range(deleted_count - 1):
        delete_roles.append(random.choice(["deleted_partial", "deleted_full"]))

    keep_roles = ["kept_near"]
    for _ in range(NUM_KEPT - 1):
        keep_roles.append(random.choice(["kept_near", "kept_low"]))

    roles = delete_roles + keep_roles
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output exhibits the D1 threshold rule."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _build_roles(component_count)

    for role in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, role)
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
        if not role.startswith("deleted_"):
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
