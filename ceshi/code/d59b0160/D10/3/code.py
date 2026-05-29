# Variant of d59b0160
# Dimension: D10 — 多步处理流程 (multi_step)
# Change: 三步流程：Step 1识别图例颜色集合；Step 2找出所有连通分量；Step 3删除包含全部图例颜色的分量... → 多个图例区域定义多轮筛选

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the original 4x4 legend idea, but place two legends in fixed, readable corners.
#      Legend 1 sits at top-left and uses the original orientation: bottom row + right column
#      are border color 3. Legend 2 sits at bottom-right and is mirrored: top row + left
#      column are border color 3. Each interior shows exactly three target colors.
#    - Objects are still 4-connected multicolor rectangles elsewhere on the orange background.
#    - Solver flow changes from one filter to two ordered filters:
#         Round 1: erase components containing all colors from the top-left legend.
#         Round 2: recompute components on the updated grid, then erase components containing
#                  all colors from the bottom-right legend.
#      Small trace:
#         Original: one object with {1,2,4} disappears if the single legend is {1,2,4}.
#         Variant: if TL legend is {1,2,4} and BR legend is {5,6,8}, then an object with
#         {5,6,8} survives round 1 and disappears only in round 2, while an object with
#         {1,2,4} disappears immediately in round 1.
# b) Complexity calibration:
#    - Use exactly two legends and fixed corner order. That is enough to make the new rule
#      visually obvious without introducing ambiguous legend counts or invisible ordering.
#    - Each sample always contains at least one component deleted by legend 1, at least one
#      deleted by legend 2, and at least one kept component. This prevents regression to the
#      original single-filter behavior.
# c) Hidden assumptions from the original code that must change:
#    - Original `_find_legend(...)` assumes a single top-left legend. That fails once a second
#      legend exists, so this variant uses two corner-specific detectors with fixed order.
#    - Original `components_mask(g != bg, ...)` treats legend pixels as ordinary components.
#      With two legends that becomes brittle, so the solver now removes both legend squares from
#      the candidate mask before labeling components.
#    - Original deletion is a single pass over the input grid. The new rule is explicitly
#      multi-round, so the solver recomputes connected components after each round on the
#      current grid state.
# d) Placement legality:
#    - Legends occupy opposite 4x4 corners. Components are placed with `random_free_location`
#      and one-cell padding, so they do not overlap legends or each other and remain separate
#      4-connected components.
#    - Component rectangles stay within bounds; the generator retries if a placement fails.
# e) Visual observability:
#    - Human-solvable cue: two distinct corner legends show two distinct color triplets.
#      Deleted objects visibly split into two palette families, one explained by each corner.
#      Fixed reading order is top-left first, then bottom-right.
# f) Visual cleanliness:
#    - Only two legend boxes exist, both in corners, with a one-cell gap around all objects.
#      This keeps the playfield uncluttered and prevents legend/object tangling.
# g) Color inference:
#    - No ordering or behavior depends on raw color ids. Colors only matter through visible
#      membership in the legend swatches. The two legends use distinct 3-color sets so the two
#      filtering passes are easy to tell apart from examples.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_top_left_legend(g: np.ndarray, bg: int) -> dict:
    """Find the original top-left legend: bottom edge + right edge share one border color."""
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
            return {
                "size": int(sz),
                "border": int(x),
                "colors": colors,
                "bbox": (0, 0, sz - 1, sz - 1),
            }
    raise AssertionError("no top-left legend border found in grid")


def _find_bottom_right_legend(g: np.ndarray, bg: int) -> dict:
    """Find the mirrored bottom-right legend: top edge + left edge share one border color."""
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        r0 = h - sz
        c0 = w - sz
        x = int(g[r0, c0])
        if x == bg:
            continue
        if np.all(g[r0, c0 : c0 + sz] == x) and np.all(g[r0 : r0 + sz, c0] == x):
            inside = g[r0 + 1 : r0 + sz, c0 + 1 : c0 + sz]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return {
                "size": int(sz),
                "border": int(x),
                "colors": colors,
                "bbox": (r0, c0, h - 1, w - 1),
            }
    raise AssertionError("no bottom-right legend border found in grid")


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply two ordered legend filters and erase components that match either legend.

    Input structure:
    - Background (bg) still fills most of the board.
    - The top-left corner contains a 4x4 legend box with border color 3 on its bottom and right edges.
    - The bottom-right corner contains a second 4x4 legend box, mirrored so border color 3 sits on its top and left edges.
    - Elsewhere are multicolored 4-connected components on the bg.

    Rule:
    - Read the target color set from the top-left legend, then read the target color set from the bottom-right legend.
    - Exclude both legend regions from the candidate-object mask.
    - Round 1: erase every component in the current grid that contains all colors from the first legend.
    - Round 2: recompute components after round 1 and erase every remaining component that contains all colors from the second legend.

    Key insight: Each corner legend is its own visible filter specification, and the output comes from applying those filters in corner order rather than using a single global palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legends = [_find_top_left_legend(g, bg), _find_bottom_right_legend(g, bg)]
    for legend in legends:
        assert legend["colors"], "legend interior contains no target colors"

    legend_mask = np.zeros_like(g, dtype=bool)
    for legend in legends:
        legend_mask |= _bbox_mask(g.shape, legend["bbox"])

    out = g.copy()
    for legend in legends:
        comps = components_mask((out != bg) & (~legend_mask), connectivity=4)
        for pts in comps:
            cols = {int(out[r, c]) for r, c in pts}
            if legend["colors"].issubset(cols):
                for r, c in pts:
                    out[r, c] = bg
    return out


# --------------- generator (o5x) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 2, 2]  # medium, tall, wide
TALL_H = (6, 10)
TALL_W = (2, 3)
WIDE_H = (2, 4)
WIDE_W = (6, 10)
MED_H = (3, 6)
MED_W = (3, 7)

DENSITY_LO = 0.12
DENSITY_HI = 0.50

ROLE_BLUEPRINTS = [
    ["first_only", "second_only", "keep", "keep"],
    ["first_only", "second_only", "first_only", "keep", "keep"],
    ["first_only", "second_only", "second_only", "keep", "keep"],
    ["first_only", "second_only", "both", "keep", "keep"],
    ["first_only", "second_only", "both", "first_only", "keep", "keep"],
]

MAX_RETRIES = 400


def _choose_legend_sets() -> tuple[list[int], list[int], list[int]]:
    picks = random.sample(PALETTE, 6)
    first = picks[:NUM_TARGETS]
    second = picks[NUM_TARGETS : 2 * NUM_TARGETS]
    extra = [c for c in PALETTE if c not in first and c not in second]
    return first, second, extra


def _legend_positions_top_left() -> list[tuple[int, int]]:
    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    return positions


def _legend_positions_bottom_right() -> list[tuple[int, int]]:
    isolated = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
    border_side = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    return positions


def _place_legends(inp: np.ndarray, out: np.ndarray, first: list[int], second: list[int]) -> None:
    s = LEGEND_SIZE
    br_r0 = GRID_H - s
    br_c0 = GRID_W - s
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

        g[br_r0:, br_c0:] = BG
        g[br_r0, br_c0:] = BORDER
        g[br_r0:, br_c0] = BORDER

    for color, (r, c) in zip(first, _legend_positions_top_left()):
        inp[r, c] = color
        out[r, c] = color

    for color, (lr, lc) in zip(second, _legend_positions_bottom_right()):
        r = br_r0 + 1 + lr
        c = br_c0 + 1 + lc
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(
    h: int,
    w: int,
    first: list[int],
    second: list[int],
    extras: list[int],
    role: str,
) -> np.ndarray:
    area = h * w
    first_set = set(first)
    second_set = set(second)

    if role == "first_only":
        must_have = list(first)
        missing = random.choice(second)
        allowed = extras + first + [c for c in second if c != missing]
    elif role == "second_only":
        must_have = list(second)
        missing = random.choice(first)
        allowed = extras + second + [c for c in first if c != missing]
    elif role == "both":
        must_have = list(first) + list(second)
        allowed = PALETTE
    elif role == "keep":
        missing_first = random.choice(first)
        missing_second = random.choice(second)
        allowed = extras + [c for c in first if c != missing_first] + [c for c in second if c != missing_second]
        needed = []
        if allowed:
            needed = random.sample(allowed, k=min(2, len(allowed)))
        must_have = needed
    else:
        raise ValueError(f"unknown role: {role}")

    lo = max(len(must_have), math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    colors = list(must_have)
    while len(colors) < n_colored:
        colors.append(random.choice(allowed))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors

    present = {int(v) for v in np.unique(patch) if int(v) != 0}
    has_first = first_set.issubset(present)
    has_second = second_set.issubset(present)
    if role == "first_only":
        assert has_first and not has_second
    elif role == "second_only":
        assert has_second and not has_first
    elif role == "both":
        assert has_first and has_second
    else:
        assert not has_first and not has_second
    return patch


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    first, second, extras = _choose_legend_sets()
    _place_legends(inp, out, first, second)

    roles = random.choice(ROLE_BLUEPRINTS).copy()
    random.shuffle(roles)

    for role in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, first, second, extras, role)
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
