# Combo variant of d59b0160 (index 1)
# combo_spec: D1/0+D2/1
# Name: ANY匹配+反转保留删除

# Design analysis for the combo variant
# - Concrete behavior:
#   The grid keeps the original visual grammar: a 4x4 legend sits in the top-left
#   corner, with border color 3 on the bottom row and right column, and exactly
#   three target colors shown inside the 3x3 interior. Elsewhere, separated
#   rectangle-like 4-connected components are built from base color 0 with extra
#   sprinkled palette cells. The fused rule keeps non-legend components that
#   contain at least one legend color and erases components that contain none.
# - Behavioral trace on a concrete example:
#   Suppose the legend colors are {1,4,8}. Component A has colors {0,1},
#   component B has {0,5,6}, and component C has {0,1,4,8}. In the base task,
#   only C matched ALL and was deleted; A and B survived. After the D1 change,
#   A and C both satisfy ANY. After the D2 inversion is applied to that relaxed
#   predicate, A and C are kept while B is deleted. The crucial witness is A:
#   it survives here even though it would be deleted under "ANY+delete" and
#   erased under "ALL+keep".
# - Complexity calibration:
#   To keep the fused rule observable from a few examples, generation always
#   includes at least one "partial_keep" component that uses only 1 or 2 target
#   colors, and at least one "miss_delete" component that uses zero target
#   colors. This prevents collapse back to either the base rule or the D2-only
#   inversion. Remaining kept components may be either partial-hit or full-hit,
#   giving modest variety without introducing too many categories.
# - Hidden assumption audit for main():
#   1. `_bg_boundary` still assumes the outer frame is dominated by the true
#      background. The generator preserves that by filling the board with 7 and
#      placing sparse objects inside it.
#   2. `_find_legend` assumes the smallest qualifying top-left square is the
#      legend. The generator keeps the original 4x4 geometry, so that remains
#      valid.
#   3. Component scanning can no longer safely include the legend region as a
#      normal object set, because the fused task conceptually filters only the
#      non-legend objects. The solver therefore removes the legend square from
#      the component mask before labeling.
#   4. The ANY predicate assumes it is easy to tell whether a component shares
#      at least one legend color. The generator supports this by making deleted
#      components use no target colors at all, and by forcing at least one kept
#      component to use only a strict subset of targets.
# - Placement legality:
#   Non-legend rectangles are still placed with
#   `random_free_location(..., padding=1, padding_connectivity=8)`, using a full
#   rectangular occupancy mask. That guarantees no overlap, no touching, no
#   out-of-bounds placement, and at least one background cell between every
#   object and the legend.
# - Visual observability and neatness:
#   The input remains a clean "legend + separated rectangles" scene. The output
#   is a positive filter, but not the ALL-filter from D2/1: a human can see that
#   some surviving objects share only part of the legend palette, while objects
#   with no shared legend color vanish. No hidden ordering, color arithmetic, or
#   overlapping markers are introduced.
# - New solver assumptions and generator support:
#   1. All task-relevant objects outside the legend are 4-connected components on
#      background 7. The generator constructs them that way.
#   2. At least one partial-hit survivor exists. `_component_roles()` guarantees
#      that.
#   3. At least one zero-hit deleted object exists. `_component_roles()` also
#      guarantees that, ensuring outputs differ from the base task on every
#      generated sample.

from pathlib import Path
import sys

import math
import random

import numpy as np

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    solvers_dir = parent / "solvers"
    if solvers_dir.is_dir():
        sys.path.insert(0, str(solvers_dir))
        break

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom edge and right edge share one
    non-background color, then read the interior legend colors from that square.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only connected components that overlap the legend palette at all.

    Input structure:
    - A dominant background fills the board and separates the objects.
    - The top-left corner contains a small square legend whose bottom row and
      right column share one border color; the interior shows the target colors.
    - Outside that legend, the remaining non-background cells form separate
      4-connected multicolor components.

    Rule:
    - Read the target color set from inside the legend frame.
    - Ignore the legend itself and inspect each other connected component.
    - Preserve a component if it contains **any** legend color.
    - Erase components that contain none of the legend colors.

    Key insight: the legend is a positive overlap filter, not a full-palette
    requirement. A single shared legend color is enough to keep an object, while
    objects with zero overlap disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    mask = (g != bg).copy()
    mask[:sz, :sz] = False
    comps = components_mask(mask, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not (cols & legend_colors):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (o5x) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of colored, non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & match/non-match split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_DELETED = 3

# -- Generation retry budget --
MAX_RETRIES = 1200


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component (h, w) according to the original shape weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _sample_colored_count(area: int, min_required: int) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO), min_required)
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build a rectangular component patch.

    Roles:
    - miss_delete: contains no target colors and will be erased
    - partial_keep: contains 1 or 2 target colors and must survive under ANY
    - full_keep: contains all target colors and also survives
    """
    area = h * w
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "miss_delete":
        n_colored = _sample_colored_count(area, 1)
        forced = []
        allowed = non_targets
    elif role == "partial_keep":
        hit_count = random.randint(1, NUM_TARGETS - 1)
        chosen_targets = random.sample(targets, hit_count)
        n_colored = _sample_colored_count(area, hit_count)
        forced = list(chosen_targets)
        allowed = non_targets + chosen_targets
    elif role == "full_keep":
        n_colored = _sample_colored_count(area, NUM_TARGETS)
        forced = list(targets)
        allowed = PALETTE
    else:
        raise ValueError(f"unknown role: {role}")

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = forced + [random.choice(allowed) for _ in range(n_colored - len(forced))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_roles(component_count: int) -> list[str]:
    """
    Guarantee at least one partial-hit survivor and exactly three zero-hit deletions.
    """
    keep_count = component_count - NUM_DELETED
    roles = ["partial_keep"] + ["miss_delete"] * NUM_DELETED
    for _ in range(keep_count - 1):
        roles.append(random.choices(["partial_keep", "full_keep"], weights=[3, 1])[0])
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output keeps all ANY-matching components."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _component_roles(component_count)

    for role in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, role)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if role != "miss_delete":
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
