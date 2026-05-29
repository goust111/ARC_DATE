# Variant of d59b0160
# Dimension: D2 — 核心变换操作：删除 vs 其他处理 (transform)
# Change: 匹配的连通分量被删除（所有像素替换为背景色7）... → 匹配分量内部仅保留图例颜色像素

"""
design_analysis:
  source_task: none
  preserved_structure:
    grid: 16x16 with bg=7
    legend:
      position: fixed top-left 4x4
      border: color 3 on bottom row and right column
      payload: exactly 3 distinct target colors from {1,2,4,5,6,8,9}
    objects:
      type: 4-connected rectangular components made of base color 0 with extra colored pixels sprinkled inside
      spacing: at least one bg cell around each placed rectangle
  variant_core:
    old_rule: any non-bg component containing all legend colors is erased completely
    new_rule: any non-bg component containing all legend colors is filtered in place; only pixels whose own color belongs to the legend set survive, every other pixel in that component becomes bg
  behavior_trace:
    small_example:
      legend_colors: [1,4,8]
      component_input:
        - [0, 1, 0]
        - [4, 0, 8]
      original_output:
        - [7, 7, 7]
        - [7, 7, 7]
      variant_output:
        - [7, 1, 7]
        - [4, 7, 8]
    difference:
      - matching still happens at whole-component level
      - transformation is no longer all-or-nothing deletion at pixel level
      - base color 0 and any non-target accent colors disappear only inside matched components
  complexity_calibration:
    - Keep exactly the same legend-reading and component-matching logic so the puzzle still has one clean controlling predicate.
    - Preserve 4-6 components with 1-3 matched components, which gives repeated evidence of the rule across examples.
    - Use solid rectangular components with internal color noise so the filtered result is visibly different from both the input and the original deletion behavior.
  implicit_assumption_review:
    - `_bg_boundary` still assumes the outer border is mostly background. Generator keeps the border clear except where objects may touch edges sparsely; bg remains dominant on the frame.
    - `_find_legend` assumes the first valid bottom/right border square in the top-left corner is the legend. Generator still reserves that exact pattern.
    - `components_mask(g != bg, connectivity=4)` groups multi-color rectangles as single objects. This is still desired because matching depends on the set of colors within an object, not per-color subcomponents.
    - The loop over a matched component can no longer blindly overwrite every cell with bg. The new code must inspect each pixel color and keep target-colored cells.
    - The legend itself must not accidentally contain all 3 target colors in one 4-connected non-bg component. Legend placement keeps one target isolated from the border-connected legend cells, as in the original task.
  placement_legality:
    - Rectangles are sampled first, then placed with `random_free_location(..., padding=1, padding_connectivity=8)` to avoid overlap and touching.
    - The legend occupies only the top-left 4x4, and object placement respects existing occupied cells automatically.
    - Matched rectangles always contain all target colors at least once, so the solver's filtering branch is exercised.
  visual_observability:
    - In matched components, the dense 0-colored body collapses to sparse target-colored pixels. This is visually distinct from complete disappearance.
    - Unmatched components stay unchanged, giving direct side-by-side contrast within one grid.
    - Because target colors are shown in the legend and then literally survive inside matched objects, the color rule is inferable from 2-3 examples without hidden ordering tricks.
  cleanliness:
    - No new overlay symbols or secondary markers are introduced.
    - Output remains uncluttered because filtering only removes pixels; it does not add new colors or geometry.
  new_solver_assumptions_to_guarantee:
    - Every matched component must contain at least one target-colored pixel after filtering. Generator guarantees this by inserting all 3 target colors.
    - Every matched component should also lose some pixels, otherwise the D2 change is not visible. Generator guarantees this because each rectangle has base color 0 over area >= 6, and 0 is never a legend color.
"""

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
    Find the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_target_colors).
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
    Filter matched connected components using a color legend.

    Input structure:
    - Background (bg) occupies most of the grid.
    - The top-left corner contains a square legend whose bottom edge and right edge use one border color, while the interior shows the target color set.
    - Away from the legend, each object is a multicolor 4-connected component on the background.

    Rule:
    - Read the target color set from the legend interior.
    - For each 4-connected non-bg component, test whether its color set contains all legend colors.
    - If a component matches, keep only the pixels inside that component whose own colors belong to the legend set; turn every other pixel of that component into bg.
    - Leave non-matching components unchanged, and keep the output size the same.

    Key insight: The legend still selects components by color-set membership, but the action changes from removing those components entirely to filtering them down to only legend-colored pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                if int(g[r, c]) not in legend_colors:
                    out[r, c] = bg
    return out


# --------------- generator (D2 variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

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

MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the legend while keeping one target isolated from the border-connected cells."""
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
    """Sample a component bounding box according to the original aspect mix."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], matched: bool) -> np.ndarray:
    """
    Build a solid rectangular component whose base is color 0.

    Matched components are guaranteed to contain all target colors.
    Unmatched components may contain some target colors, but never all of them.
    """
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


def _filtered_patch(patch: np.ndarray, targets: list[int]) -> np.ndarray:
    """Keep only legend colors inside a matched component; everything else becomes bg."""
    keep = np.isin(patch, np.array(targets, dtype=int))
    return np.where(keep, patch, BG).astype(int)


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input for the D2 filtering task."""
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
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if matched:
            paste(out, _filtered_patch(patch, targets), r, c)
        else:
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Sample a valid input grid whose transformed output exercises the filtering branch."""
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
