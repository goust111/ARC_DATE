# Variant of d59b0160
# Dimension: D6 — 删除操作的具体实现 (deletion)
# Change: 匹配的连通分量的所有像素被替换为背景色(7/橙色)... → 仅删除图例颜色像素，保留其他颜色

import math
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     old: any non-background 4-connected component containing all legend colors is fully erased
#     new: the same matching test is kept, but only pixels whose own color is in the legend set are erased
#   concrete_grid_effect:
#     - The top-left 4x4 legend stays the same: green border on bottom/right, three target colors inside.
#     - Non-legend objects stay as separated rectangular 4-connected components made of visible black 0-cells
#       plus sprinkled accent colors.
#     - A matching component now survives as a perforated object: target-color cells turn into orange bg=7,
#       while black scaffold cells and non-target colored cells stay in place.
#   behavior_trace:
#     example:
#       legend_colors: {1,4,8}
#       matching_component_colors_before: {0,1,4,5,8}
#       old_output: whole rectangle -> 7
#       new_output: only cells colored 1/4/8 -> 7; cells colored 0/5 remain
#     consequence:
#       the solver still reasons at the component level for the match, but applies deletion at the pixel level
#       inside that component.
#   complexity_calibration:
#     - Matching is still "contains all 3 legend colors", so the conceptual jump is small and local to D6.
#     - To avoid degeneration into "delete these colors everywhere", the generator guarantees that kept
#       components include 0, 1, and 2 legend colors across the three kept objects; these remain unchanged.
#     - To avoid degeneration into the original task, deleted components are constructed with surviving
#       non-target structure, usually including at least one non-target colored pixel in addition to black 0-cells.
#   implicit_assumption_audit:
#     - _bg_boundary and _find_legend do not depend on full-object deletion, so they remain valid.
#     - components_mask(g != bg) must still run on the original input grid, not the progressively edited output;
#       otherwise a partially erased component could split and change later decisions.
#     - The legend itself must not match all target colors inside one connected component. The original placement
#       trick already guarantees one legend swatch is isolated from the border-connected swatches, so the legend
#       stays untouched.
#   placement_legality:
#     - Each rectangular patch is placed with random_free_location(..., padding=1), reserving its whole bbox.
#     - That keeps every non-legend component separated from others and from the legend by at least one bg cell.
#     - Because solver matching uses 4-connectivity on non-bg cells, this placement guarantee is sufficient.
#   visual_observability:
#     - Deleted objects visibly keep their black rectangle body and often some non-target accent pixels, so the
#       result reads as "holes punched in specific colors" rather than total disappearance.
#     - Kept objects carrying only a subset of legend colors give counterexamples against a naive color-wise rule.
#   cleanliness:
#     - Objects remain sparse, axis-aligned rectangles with a consistent legend location, so the scene stays readable.


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
    Punch out legend-colored pixels inside components that realize the full legend palette.

    Input structure:
    - The frame color defines the orange background.
    - In the top-left corner, a small legend square has a green bottom/right border and three interior swatches that specify the target colors.
    - Away from the legend, each object is a 4-connected non-background component made from black structure cells plus sprinkled colored cells.

    Rule:
    - Read the set of target colors from the legend interior.
    - For each non-background 4-connected component, test whether the component contains all target colors somewhere inside it.
    - If it does, erase only the pixels whose own color belongs to the legend set; keep every other pixel of that component unchanged.
    - Components missing at least one legend color remain completely unchanged.

    Key insight:
    - The legend still selects components by palette membership, but deletion is now selective inside a matched object: the full component is identified first, then only its legend-colored cells are turned into background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                if int(g[r, c]) in legend_colors:
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

# -- Component shape weights --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3
KEPT_TARGET_COUNTS = [0, 1, 2]

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


def _sprinkle_coords(h: int, w: int, n_colored: int) -> list[tuple[int, int]]:
    return random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)


def _component_patch(
    h: int,
    w: int,
    targets: list[int],
    deleted: bool,
    kept_target_count: int | None = None,
) -> np.ndarray:
    """Build an h x w 4-connected rectangular component with visible support for the variant rule."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        lo = max(lo, NUM_TARGETS)
        if hi > NUM_TARGETS:
            lo = max(lo, NUM_TARGETS + 1)
        n_colored = random.randint(lo, hi)
        colors = list(targets)
        extra = n_colored - NUM_TARGETS
        if extra > 0:
            colors.append(random.choice(non_targets))
            extra -= 1
        pool = list(targets) * 3 + non_targets
        colors.extend(random.choice(pool) for _ in range(extra))
    else:
        assert kept_target_count is not None
        chosen_targets = random.sample(targets, kept_target_count)
        lo = max(lo, kept_target_count)
        n_colored = random.randint(lo, hi)
        allowed = non_targets + chosen_targets
        colors = list(chosen_targets) + [
            random.choice(allowed) for _ in range(n_colored - kept_target_count)
        ]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    coords = _sprinkle_coords(h, w, n_colored)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _apply_partial_deletion(patch: np.ndarray, targets: list[int]) -> np.ndarray:
    """Replace only legend-colored pixels with bg, preserving the rest of the matched component."""
    out_patch = patch.copy()
    out_patch[np.isin(out_patch, np.array(targets, dtype=int))] = BG
    return out_patch


def _component_specs(component_count: int) -> list[tuple[bool, int | None]]:
    deleted_count = component_count - NUM_KEPT
    specs: list[tuple[bool, int | None]] = [(True, None)] * deleted_count
    kept_counts = KEPT_TARGET_COUNTS.copy()
    random.shuffle(kept_counts)
    specs.extend((False, count) for count in kept_counts)
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose expected output exhibits partial deletion."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    specs = _component_specs(component_count)

    for deleted, kept_target_count in specs:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted, kept_target_count)
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
        paste(out, _apply_partial_deletion(patch, targets) if deleted else patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
