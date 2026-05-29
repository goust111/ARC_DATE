# Variant of d59b0160
# Dimension: D2 — 核心变换操作：删除 vs 其他处理 (transform)
# Change: 匹配的连通分量被删除（所有像素替换为背景色7）... → 保留匹配分量、删除不匹配分量（反转保留/删除）

# Design analysis for the D2 inversion variant
# - Concrete behavior: the 4x4 top-left legend is unchanged. It still shows three target colors
#   inside a green bottom/right border. Outside the legend, each non-background object is one
#   4-connected rectangle-like component made from color 0 with extra colored pixels sprinkled in.
#   The output now keeps only the components whose color set covers all three legend colors; every
#   other non-legend component is erased to background 7.
# - Behavioral trace: in the original task, a component containing all legend colors was the one
#   removed. Here the exact same predicate is reused, but the action is flipped. Example:
#   legend={1,4,8}; component A has {0,1,4,8,5}; component B has {0,1,4}. Original output erased A
#   and kept B. This variant keeps A and erases B, so the surviving objects are precisely the full
#   palette matches.
# - Complexity calibration: to keep the inversion visually obvious, generation always includes both
#   categories: matching components and non-matching components. The component count remains 4-6 and
#   exactly three non-matching components are inserted, leaving 1-3 survivors in the output. That is
#   enough variation to avoid a trivial single fixed layout while still making the positive filter easy
#   to infer from a few examples.
# - Hidden assumptions review: the original solver could iterate over every non-bg component because
#   only matching components were erased, and the legend fragments never matched all three colors.
#   After inversion, blindly erasing all non-matching components would also erase the legend border and
#   swatches. The solver therefore must explicitly skip components that lie inside the legend square.
#   The rest of the legend parsing logic remains valid because the legend geometry does not change.
# - Placement legality: the generator still uses rectangular occupancy patches and
#   random_free_location(..., padding=1, padding_connectivity=8), so components do not overlap, do not
#   touch each other, and stay away from the legend by at least one background cell. This guarantees
#   clean 4-connected objects for the solver.
# - Visual observability: the output differs from the original in a strongly visible way because the
#   matched components remain while the distractors disappear. We avoid color-id-based ordering or any
#   invisible metadata; the decision depends only on visibly present legend colors inside each object.
# - Neatness: the legend occupies a dedicated corner and the scattered components are separated by
#   background gaps, so no extra marks or overlaps obscure which regions are being filtered.
# - New solver assumptions and generator support:
#   1. Legend components are confined to the top-left legend square. The generator enforces that by
#      reserving the legend region before placing any other object.
#   2. At least one matching component exists and at least one non-matching component exists. The role
#      split in the generator guarantees both.
#   3. A matching component must contain all three target colors. _component_patch(..., matches=True)
#      explicitly injects all target colors once before filling the remaining colored cells.

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


def _is_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend cells occupy the top-left sz x sz square and must survive unchanged."""
    return bool(np.all(pts[:, 0] < sz) and np.all(pts[:, 1] < sz))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only connected components that realize the full legend palette.

    Input structure:
    - One background color fills the frame and most empty cells.
    - The top-left corner contains a small square legend whose bottom row and right column share one border color.
    - Away from that legend, several 4-connected multicolor objects sit on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Examine each non-background 4-connected component outside the legend.
    - Preserve a component only if its colors include every legend color.
    - Replace all other non-legend components with the background while leaving the legend untouched.

    Key insight: the legend now defines a positive filter rather than a rejection test, so the output is the subset of objects whose internal palette fully covers the legend palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _is_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if not legend_colors.issubset(cols):
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
NUM_NONMATCH = 3

# -- Generation retry budget --
MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the target swatches identically in input/output."""
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
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], matches: bool) -> np.ndarray:
    """Build an h x w solid component with base color 0 and extra colored pixels."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if matches:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if matches:
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
    """Try once to build a valid input whose output keeps only matching components."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_NONMATCH) + [False] * NUM_NONMATCH
    random.shuffle(roles)

    for matches in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, matches)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if matches:
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
