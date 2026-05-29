# Variant of d59b0160
# Dimension: D6 — 删除操作的具体实现 (deletion)
# Change: 匹配的连通分量的所有像素被替换为背景色(7/橙色)... → 删除时保留分量的外接矩形边框
#
# Design analysis (YAML-style):
#   variant_goal:
#     preserve:
#       - legend detection from the top-left bordered box
#       - target-color-set matching over 4-connected non-background components
#       - same background, legend size, palette, component count, and placement regime
#     change_only:
#       - the effect applied to a matching component
#     new_effect:
#       - original: every pixel of a matching component becomes background 7
#       - variant: every pixel of a matching component is cleared to background 7,
#         then the component's bounding rectangle border is drawn in color 3
#   concrete_behavior:
#     input_objects:
#       - legend remains a 4x4 top-left region with border color 3 and exactly 3 target colors inside
#       - each non-legend object is a solid non-background rectangle whose cells are mostly 0,
#         with sparse colored pixels sprinkled inside
#     output_objects:
#       - unmatched rectangles stay identical
#       - matched rectangles disappear internally, but a green rectangular "ghost frame"
#         remains at the same location
#     traced_example:
#       - if a 5x4 component contains all three legend colors, the original task outputs a 5x4
#         orange-background hole
#       - this variant instead outputs a 5x4 green border rectangle with orange interior
#       - the matching test is unchanged; only the deletion rendering differs
#   complexity_calibration:
#     why_not_more_changes:
#       - changing the matching rule as well would blur which dimension changed
#       - keeping the original sparse multicolor rectangles preserves the task identity
#     visibility_guard:
#       - deleted components are generated with height >= 3 and width >= 3 so the hollow-frame
#         effect is usually visible as an actual empty interior, not just a thin strip recolor
#   implicit_assumptions_review:
#     - _bg_boundary still works because the outer border remains dominated by background 7
#     - _find_legend still works because the legend geometry and colors are unchanged
#     - components_mask(g != bg, connectivity=4) still intentionally treats each non-background
#       rectangle as one component and may split the legend into several tiny components; none of
#       those legend pieces contains all target colors, so the legend is preserved
#     - the old solver relied on "delete = fill pts with bg"; that assumption is the only part
#       that must change
#     - the new solver assumes drawing a bounding-box border will not collide with other objects;
#       the generator guarantees at least one background cell of separation around every rectangle
#   placement_legality:
#     - input placement still uses random_free_location with padding=1 under full-rectangle
#       occupancy, so rectangles do not overlap or touch
#     - because the ghost frame is drawn inside the same rectangle's bbox, the output also stays
#       non-overlapping and within bounds
#   observability:
#     - color 3 is excluded from the object palette, so new green rectangles in the output are a
#       clean cue for "this object was removed but marked"
#     - every generated puzzle has at least one deleted component, so the new branch is exercised
#   cleanliness:
#     - the frame uses the component's own bbox, creating a simple rectangular marker instead of
#       noisy per-pixel remnants

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
    Detect the smallest top-left square whose bottom row and right column share one border color.
    Return (size, border_color, set_of_target_colors_inside).
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


def _draw_bbox_border(out: np.ndarray, pts: np.ndarray, color: int) -> None:
    """Draw the inclusive bounding-box border of `pts` into `out`."""
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove legend-matching components but leave a rectangular ghost frame behind.

    Input structure:
    - Background color fills most of the grid.
    - In the top-left corner, a small legend square uses one border color on its bottom edge and
      right edge; the interior shows the target color set.
    - Away from the legend, each object is a 4-connected non-background component on the background.

    Rule:
    - Read the target color set from the legend interior.
    - For each 4-connected non-background component, test whether it contains all target colors.
    - Components that fail the test stay unchanged.
    - Components that pass the test are erased to background, then their bounding-rectangle border
      is redrawn in the legend border color.

    Key insight: The legend still decides which objects are selected, but the selected objects are
    not made fully invisible; they leave a hollow positional marker that records where they were.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, frame_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
            _draw_bbox_border(out, pts, frame_color)
    return out


# --------------- generator (o5x, D6 variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
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
    """Draw the 4x4 legend box at top-left and place the 3 target colors."""
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


def _component_dims(deleted: bool) -> tuple[int, int]:
    """Sample component dimensions, forcing deleted ones to have visible hollow frames."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        h = random.randint(*TALL_H)
        w = random.randint(*TALL_W)
    elif mode == "wide":
        h = random.randint(*WIDE_H)
        w = random.randint(*WIDE_W)
    else:
        h = random.randint(*MED_H)
        w = random.randint(*MED_W)

    if deleted:
        h = max(h, 3)
        w = max(w, 3)
    return h, w


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build a dense non-background rectangle with sparse target/non-target accents."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
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


def _frame_patch(h: int, w: int) -> np.ndarray:
    """Return a hollow rectangular frame in BORDER on BG."""
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BORDER
    patch[-1, :] = BORDER
    patch[:, 0] = BORDER
    patch[:, -1] = BORDER
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output contains the ghost-frame deletion effect."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims(deleted)
        patch = _component_patch(h, w, targets, deleted)
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
        if deleted:
            paste(out, _frame_patch(h, w), r, c, transparent=BG)
        else:
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
