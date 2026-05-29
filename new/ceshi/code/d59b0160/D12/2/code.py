# Variant of d59b0160
# Dimension: D12 — 颜色0在分量中的角色 (color_set_match)
# Change: 颜色0（黑色）是连通分量中的有效颜色，参与连通性计算，但不出现在图例中因此不参与颜色匹配... → 忽略每种颜色的第一个像素进行匹配
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original_match: a component is deleted when every legend color appears at least once.
#     new_match:
#       - scan each 4-connected component in row-major order
#       - ignore the first pixel of every color encountered in that component
#       - a color enters the matching set only when its second pixel is seen
#       - equivalently, only colors with multiplicity >= 2 count toward legend matching
#     effect:
#       - legend colors must be repeated inside a component, not merely present
#       - color 0 still participates in connectivity because components are found on (grid != bg)
#         exactly as before, but color 0 never appears in the legend and therefore never helps match it
#   concretization:
#     grid: unchanged 16x16 orange background (7)
#     legend:
#       - unchanged top-left 4x4 corner box
#       - bottom row and right column are green border color 3
#       - interior still shows exactly 3 distinct target colors once each
#     components:
#       - unchanged rectangular 0-filled bodies with sprinkled non-zero colors
#       - deleted components now contain each legend color at least twice
#       - at least one kept component intentionally contains all legend colors visibly, but one of
#         those colors appears only once, so it survives in the new rule and would have been removed before
#   behavior_trace:
#     example_component_scan:
#       legend = {1,4,8}
#       component pixels in scan order carry colors [0,1,4,8,1,4,5]
#       original_logic:
#         present colors = {0,1,4,5,8} -> legend subset satisfied -> delete
#       new_logic:
#         first 0 ignored, first 1 ignored, first 4 ignored, first 8 ignored,
#         second 1 adds 1, second 4 adds 4, 5 seen only once
#         effective matching set = {1,4}
#         legend color 8 never repeats -> keep
#       difference:
#         the component still visibly shows all three legend colors, but only repeated colors count
#   complexity_calibration:
#     - if kept components merely omitted target colors, many samples would still look compatible with the original task
#     - to make D12 visible, generation always includes >= 1 "near miss" kept component that shows all 3
#       legend colors while failing the repetition test
#     - other kept components may miss colors entirely, which provides contrast without adding extra rule branches
#   implicit_assumptions_from_original_main:
#     - _bg_boundary and _find_legend remain valid because the legend geometry is unchanged
#     - components_mask(g != bg, connectivity=4) must stay unchanged so color 0 still glues a rectangle together
#     - the original line collecting {g[r,c] for pts} is no longer valid because singleton target pixels would
#       incorrectly count; it is replaced with a scan-order repeated-color extractor
#     - deletion still writes bg over the whole component, so generator must keep components disjoint and non-overlapping
#   placement_legality:
#     - legend stays in the top-left corner
#     - every generated component is placed by random_free_location on its full rectangular bbox (all 0s count as occupied),
#       with padding=1, so components do not overlap or touch
#     - deleted components require at least 6 colored cells (2 copies x 3 targets); kept near-miss components require
#       at least 5 colored cells (2,2,1 across the three targets); the sampled bbox ranges support that
#   visual_observability:
#     - humans can compare deleted objects versus near-miss kept objects by counting repeated legend colors inside each box
#     - no hidden ordering by color id is used; scan order only implements "ignore first occurrence", which is visually
#       equivalent to requiring duplicates, a directly observable property
#   cleanliness:
#     - no new overlays or markers are introduced
#     - the same sparse colored-dot rectangles are preserved, avoiding extra visual clutter
#   generator_guarantees_for_new_solver:
#     - every puzzle has at least one deleted component and at least one kept near-miss component
#     - kept near-miss components contain all legend colors, but exactly one chosen target remains singleton
#     - kept missing components omit at least one legend color entirely

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


def _repeated_match_colors(g: np.ndarray, pts: np.ndarray) -> set[int]:
    """
    Colors that survive the variant's matching rule.

    We scan the component in row-major order and ignore the first pixel of each color.
    A color is admitted only when its second occurrence is encountered.
    """
    seen_once: set[int] = set()
    repeated: set[int] = set()
    for r, c in sorted((int(r), int(c)) for r, c in pts):
        col = int(g[r, c])
        if col in seen_once:
            repeated.add(col)
        else:
            seen_once.add(col)
    return repeated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove connected components only when every legend color repeats inside that component.

    Input structure:
    - Most of the board is a single background color.
    - The top-left corner contains a small square legend with a uniform border on its bottom and right edges.
    - Away from the legend, the foreground is split into 4-connected multicolor components carried by color-0 rectangles with colored speckles.

    Rule:
    - Read the target color set from the legend interior.
    - For each foreground component, scan its pixels in row-major order and ignore the first pixel of every color.
    - Treat only colors that appear again later in the same component as part of that component's matching set.
    - Erase the component iff that repeated-color set contains all legend colors; otherwise leave it unchanged.

    Key insight: The legend still specifies which colors matter, but now each target color must recur within a component, not merely appear once.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        match_colors = _repeated_match_colors(g, pts)
        if legend_colors.issubset(match_colors):
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
DENSITY_LO = 0.12
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 400


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


def _sample_colored_count(h: int, w: int, minimum: int) -> int:
    area = h * w
    lo = max(minimum, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    hi = min(hi, area)
    lo = min(lo, hi)
    return random.randint(lo, hi)


def _deleted_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    colors = [color for color in targets for _ in range(2)]
    extras = n_colored - len(colors)
    colors.extend(random.choice(targets + non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _near_miss_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    weak = random.choice(targets)
    strong = [color for color in targets if color != weak]
    colors = [weak] + [strong[0], strong[0], strong[1], strong[1]]
    extras = n_colored - len(colors)
    colors.extend(random.choice(non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _missing_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    used_target_count = random.randint(0, NUM_TARGETS - 1)
    used_targets = random.sample(targets, used_target_count)
    colors: list[int] = []
    for color in used_targets:
        colors.extend([color] * random.randint(1, 2))
    while len(colors) > n_colored:
        colors.pop()
    extras = n_colored - len(colors)
    colors.extend(random.choice(non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _component_patch(h: int, w: int, targets: list[int], style: str) -> np.ndarray:
    """Build an h x w patch filled with color-0 base and colored pixels meeting the requested style."""
    non_targets = [v for v in PALETTE if v not in targets]
    if style == "deleted":
        n_colored = _sample_colored_count(h, w, minimum=2 * NUM_TARGETS)
        colors = _deleted_colors(targets, non_targets, n_colored)
    elif style == "near_miss":
        n_colored = _sample_colored_count(h, w, minimum=5)
        colors = _near_miss_colors(targets, non_targets, n_colored)
    elif style == "missing":
        n_colored = _sample_colored_count(h, w, minimum=1)
        colors = _missing_colors(targets, non_targets, n_colored)
    else:
        raise ValueError(f"unknown component style: {style}")

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_styles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    near_miss_count = random.randint(1, 2)
    missing_count = NUM_KEPT - near_miss_count
    styles = ["deleted"] * deleted_count + ["near_miss"] * near_miss_count + ["missing"] * missing_count
    random.shuffle(styles)
    return styles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose solver output matches the intended keep/delete split."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    styles = _component_styles(component_count)

    for style in styles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, style)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if style != "deleted":
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
