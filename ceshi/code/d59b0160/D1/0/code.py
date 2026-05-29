# Variant of d59b0160
# Dimension: D1 — 颜色匹配条件：全部包含 vs 其他逻辑 (match_cond)
# Change: 连通分量必须包含图例中的所有颜色(ALL)才被删除，缺少任何一种图例颜色则保留... → 包含任意一种图例颜色即删除

import math
import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break

from arc_utils import components_mask, paste, random_free_location


# Design analysis for this D1 variant
# - Concrete behavior:
#   The top-left 4x4 legend is preserved from the base task: bottom row and right
#   column are green (3), and the 3x3 interior contains exactly three target colors.
#   Outside the legend are several separated rectangular components built from base
#   color 0 with extra sprinkled colors. The variant changes only the matching rule:
#   any non-legend component that contains at least one legend color is erased.
# - Trace of the changed step:
#   If the legend colors are {1,4,8}, then a component with colors {0,1,9} used to
#   survive under ALL because it missed 4 and 8. In this variant it is deleted
#   because its color set intersects the legend. The generator therefore must create
#   partial-hit components, not only full-hit ones, or the samples could collapse
#   back to the original behavior.
# - Hidden assumption audit:
#   The original solver labeled components on the whole non-background mask. That was
#   harmless only because the legend geometry avoided putting all targets into one
#   legend component. Under ANY, that assumption breaks: every legend target pixel
#   would become deletable. The fix is to exclude the detected legend square from the
#   component mask before scanning connected components.
# - Complexity / observability:
#   Grid size, legend size, palette, rectangular object style, densities, and spacing
#   stay unchanged so the task still reads like the same family. To make the rule
#   visible, each generated sample contains at least one deleted component that uses
#   only a strict subset of the legend colors, while kept components use no legend
#   colors at all.
# - Placement legality:
#   Objects are still placed with random_free_location(..., padding=1), so components
#   remain separated from the legend and from each other by background cells. That
#   matches the solver's assumption that every deletable component lies outside the
#   legend box.
# - Clean visuals:
#   No extra markers or overlays are introduced; the input remains "legend + sparse
#   rectangles", and the output differs only by which rectangles disappear.


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
    Remove connected components that share a color with the legend.

    Input structure:
    - A dominant background fills the board.
    - In the top-left corner, a small square legend is framed by a single color on
      its bottom and right edges; the interior lists the target colors to test.
    - Away from the legend, the remaining non-background cells form separate
      4-connected multicolor components.

    Rule:
    - Read the target color set from inside the legend frame.
    - Ignore the legend itself and inspect each other connected component.
    - If a component contains **any** legend color, erase that entire component to
      the background.
    - Keep components that contain none of the legend colors.

    Key insight: The legend is now an overlap filter rather than a full-palette
    requirement, so even a single shared legend color is enough to remove a component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend = _find_legend(g, bg)
    sz, _border_color, legend_colors = legend
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    mask = (g != bg).copy()
    mask[:sz, :sz] = False
    comps = components_mask(mask, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if cols & legend_colors:
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

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


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
    """Sample component dimensions according to the original shape weights."""
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
    - keep: contains no target colors
    - partial_delete: contains 1 or 2 target colors, guaranteeing divergence from ALL
    - full_delete: contains all target colors
    """
    area = h * w
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "keep":
        n_colored = _sample_colored_count(area, 1)
        forced = []
        allowed = non_targets
    elif role == "partial_delete":
        hit_count = random.randint(1, NUM_TARGETS - 1)
        chosen_targets = random.sample(targets, hit_count)
        n_colored = _sample_colored_count(area, hit_count)
        forced = list(chosen_targets)
        allowed = non_targets + chosen_targets
    elif role == "full_delete":
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
    Choose component roles while guaranteeing at least one partial-hit deleted object.
    """
    deleted_count = component_count - NUM_KEPT
    deleted_roles = ["partial_delete"]
    for _ in range(deleted_count - 1):
        deleted_roles.append(random.choices(["partial_delete", "full_delete"], weights=[3, 1])[0])
    roles = deleted_roles + ["keep"] * NUM_KEPT
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is reproduced by main()."""
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
        if role == "keep":
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
