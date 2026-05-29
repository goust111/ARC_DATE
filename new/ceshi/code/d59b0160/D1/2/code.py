# Variant of d59b0160
# Dimension: D1 — 颜色匹配条件：全部包含 vs 其他逻辑 (match_cond)
# Change: 连通分量必须包含图例中的所有颜色(ALL)才被删除，缺少任何一种图例颜色则保留... → 不包含任何图例颜色则删除

import math
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D1 inversion variant
# - Concrete behavior:
#   The top-left 4x4 legend still shows exactly three target colors inside a color-3 corner frame.
#   Elsewhere, each object is a 4-connected rectangle filled with color 0 plus scattered colored cells.
#   The visible behavioral change is no longer "objects carrying the full legend palette disappear".
#   Instead, an object disappears only when it shares zero colors with the legend.
# - Trace on a small example:
#   If the legend colors are {1,4,6}, then:
#   * object colors {0,5,8} -> original rule kept it (missing all three targets), new rule deletes it
#   * object colors {0,1,5} -> original rule kept it, new rule also keeps it
#   * object colors {0,1,4,6} -> original rule deleted it, new rule keeps it
#   This inversion changes the solver's decision boundary from ALL-target inclusion to ANY-target overlap.
# - Complexity calibration:
#   To make the inversion visible in only a few ARC examples, the generator must not rely on accidental
#   full-target matches. It therefore forces two recurring categories:
#   * deleted objects with zero legend colors
#   * kept objects with one or two legend colors, so "partial overlap is enough to survive" is visible
# - Old implicit assumption that breaks under the variant:
#   The original solver scanned all non-background components, including the legend fragments themselves.
#   Under the old ALL rule this was harmless because the legend border component lacked the full palette.
#   Under the new NONE rule that border component would be wrongly erased, so the solver must now skip any
#   component fully contained inside the detected legend square.
# - Placement legality:
#   Objects are placed as full rectangular patches using random_free_location on an all-occupied mask, so
#   their bounding boxes never overlap and always keep at least one background cell of separation.
#   This guarantees distinct 4-connected components outside the legend, which the solver assumes.
# - Visual observability and cleanliness:
#   The legend remains isolated in the corner, objects remain separated, and target membership is readable
#   directly from colors inside each object. No hidden ordering or color-id arithmetic is used.
# - New solver assumptions and generator guarantees:
#   The solver assumes the only components entirely inside the legend square belong to the legend itself,
#   and that every non-legend object is a separate 4-connected component. The generator enforces both.


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


def _inside_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend fragments live entirely inside the detected top-left legend square."""
    return bool(np.all((pts[:, 0] < sz) & (pts[:, 1] < sz)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those components that avoid the legend colors entirely.

    Input structure:
    - The background color fills most of the canvas and is recovered from the outer border.
    - A small legend square sits in the top-left corner with a shared border color on its bottom and right edges.
    - The legend interior contains several target colors; other 4-connected objects appear elsewhere on the background.

    Rule:
    - Read the legend's target color set from the cells inside the corner frame.
    - Ignore any connected components that belong to the legend region itself.
    - For every other 4-connected component, collect the set of colors present in that object.
    - Erase the component only when its color set has no overlap with the legend colors; otherwise keep it.

    Key insight: The legend is now a "survival palette" rather than a "deletion palette" — sharing even one legend
    color protects an object, while objects completely missing the legend palette are the ones removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _inside_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if cols.isdisjoint(legend_colors):
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

# -- Component fill density (fraction of colored non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box (border color 3) at top-left and place target colors."""
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


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build one rectangular component whose target-color overlap matches its role."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        n_colored = random.randint(lo, hi)
        colors = [random.choice(non_targets) for _ in range(n_colored)]
    else:
        kept_targets = random.sample(targets, random.randint(1, NUM_TARGETS - 1))
        lo = max(lo, len(kept_targets))
        n_colored = random.randint(lo, hi)
        allowed = non_targets + kept_targets
        colors = kept_targets + [random.choice(allowed) for _ in range(n_colored - len(kept_targets))]

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], len(colors))
    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid (input, output) pair; return input or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not deleted:
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
