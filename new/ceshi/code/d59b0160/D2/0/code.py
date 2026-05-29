# Variant of d59b0160
# Dimension: D2 — 核心变换操作：删除 vs 其他处理 (transform)
# Change: 匹配的连通分量被删除（所有像素替换为背景色7）... → 重着色为指定颜色而非删除

"""
Design analysis for the D2 variant
==================================

variant_spec:
  preserved_structure:
    grid: 16x16, background color 7
    legend: top-left 4x4 region with bottom row and right column in border color 3
    legend_interior: exactly 3 distinct target colors from {1,2,4,5,6,8,9}
    objects: 4-connected non-background components placed away from each other
    selection_rule: a component is selected iff it contains all legend colors
  changed_behavior:
    old: selected component -> every cell becomes background 7
    new: selected component -> every cell becomes border color 3
  why_border_color:
    - visually explicit: the legend already highlights color 3 as a controlling frame
    - low ambiguity: component geometry stays intact, only state changes
    - no hidden numeric mapping: solvers can infer "matched objects turn green" directly from the picture

behavior_trace:
  example:
    legend_colors: {1,4,8}
    component_A_colors: {0,1,4,8,9}
    component_B_colors: {0,1,4,6}
  original_flow:
    - A contains all legend colors, so A is erased to 7
    - B misses 8, so B is unchanged
  variant_flow:
    - A contains all legend colors, so every pixel of A becomes 3
    - B still remains unchanged
  visible_difference:
    - the matched rectangle does not disappear; it turns into a solid green block with the same footprint

complexity_calibration:
  - Keep the same selection mechanism as the original task so the only conceptual novelty is the output action.
  - Recolored components become solid green rectangles, which is visually stronger than subtle per-pixel recoloring.
  - Keep 1-3 matched components and exactly 3 unmatched ones so multiple examples exhibit both branches.

implicit_assumption_audit:
  - _bg_boundary still works because the outer frame remains dominated by color 7.
  - _find_legend still works because the legend geometry is unchanged.
  - components_mask(g != bg) still finds the same objects because recoloring changes only output, not input parsing.
  - The old assignment `out[r, c] = bg` depended on deletion semantics; in this variant it must become `out[r, c] = border_color`.
  - The solver now assumes the legend border color is the recolor target; the generator guarantees this by keeping border color fixed at 3 and never using 3 inside ordinary components.

placement_legality:
  - The legend stays in the top-left 4x4 area.
  - Components are placed with `random_free_location(..., padding=1, padding_connectivity=8)`, so they do not overlap or touch.
  - Each matched component is generated with all three legend colors present at least once, ensuring the recolor branch is exercised.
  - Each unmatched component omits at least one legend color, preventing accidental recoloring.

visual_observability:
  - Compared with deletion, recoloring leaves large green silhouettes behind; this is easy to spot in 2-3 examples.
  - The border color already serves as a natural visual cue for the replacement color, so no arbitrary color mapping must be guessed.
  - Because color 3 is otherwise absent from generated objects, every new green block is evidence of the rule rather than noise.

main_generator_contract:
  - `main()` recolors every selected component to the legend border color.
  - `generate_input()` constructs inputs where at least one component is recolored and at least three are preserved.
  - Validation uses `np.array_equal(main(inp), out)` on every generated sample before accepting it.
"""

import math
import random

import numpy as np

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor connected components whose palettes match the legend.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a compact legend box whose bottom edge and right edge share one border color; the interior shows the target palette.
    - Elsewhere: multicolored 4-connected objects sit on the background.

    Rule:
    - Read the target color set from the legend interior and read the replacement color from the legend border.
    - For each connected component, test whether it contains **all** target colors.
    - Repaint every matching component uniformly with the legend-border color; leave non-matching components unchanged.
    - Output size stays the same.

    Key insight: The legend still specifies which objects qualify, but instead of removing them, it converts each qualifying shape into a solid copy of the legend's border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = border_color
    return out


# --------------- generator (D2 recolor variant) ---------------

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

# -- Component count & kept/recolored split --
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

    # Keep at least one swatch disconnected from the border so the legend itself never matches.
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


def _component_patch(h: int, w: int, targets: list[int], matched: bool) -> np.ndarray:
    """Build an h x w component with base color 0 plus sprinkled palette colors."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if matched:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if matched:
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


def _recolor_patch(patch: np.ndarray, color: int) -> np.ndarray:
    """Turn an entire component patch into one solid color while preserving its footprint."""
    return np.full_like(patch, int(color))


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid recolor example; return input or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for matched in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, matched)
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
        paste(out, _recolor_patch(patch, BORDER) if matched else patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
