# Variant of d59b0160
# Dimension: D17 — 分量面积对匹配的影响 (meta)
# Change: 不考虑分量面积，任何大小的匹配分量都被删除... → 仅删除面积最大的匹配分量

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete behavior trace
#    - The legend remains a 4x4 top-left box with green border color 3 and three target colors in its 3x3 interior.
#    - Outside the legend, each object is still one 4-connected rectangle-shaped component whose cells are mostly color 0
#      with a sparse sprinkle of palette colors.
#    - Original rule: every component whose color set contains all legend colors is erased.
#    - New rule: first find every component whose color set contains all legend colors, then compare their component areas
#      (number of cells in the 4-connected component, which here equals the whole rectangle area), and erase only the
#      unique largest such component. The other matching components stay.
#    - Example trace:
#      legend = {1, 4, 8}; three outside components have areas 24, 15, 18; the 24-cell and 15-cell components contain
#      {1, 4, 8}, while the 18-cell component is missing 8.
#      Original output would erase both 24 and 15.
#      Variant output erases only 24 and keeps 15 and 18.
# b) Complexity calibration
#    - To make the new competition visible, the generator always creates at least two matching components.
#    - Their areas are forced to be distinct, so there is a unique largest matching candidate.
#    - The number of matching components is kept small (2 or 3), avoiding over-complication while still giving repeated
#      evidence that “matching alone” is no longer sufficient for deletion.
# c) Hidden assumptions review
#    - `_bg_boundary` and `_find_legend` do not depend on the deletion policy and remain valid unchanged.
#    - In `main`, the old loop erased components immediately after detecting a match; that relied on the old invariant
#      that every match should disappear. Under D17 this becomes wrong, because matching components must be collected,
#      compared by area, and only one of them can be erased.
#    - `components_mask(g != bg, connectivity=4)` still works because the variant keeps the same object notion:
#      every non-background rectangle is one 4-connected component.
# d) Placement legality
#    - Objects are placed with `random_free_location(..., padding=1, padding_connectivity=8)`, so components do not
#      overlap or touch, and each remains separately recoverable by the solver.
#    - The legend is still drawn first at top-left; later components are placed around it with at least one background gap.
# e) Visual observability
#    - Samples always contain multiple matching components, and at least one smaller matching component survives.
#      This creates a visible contrast with the original task.
#    - Area comparison uses visible geometry only; no invisible ordering such as color-id ranking is used.
# f) Visual cleanliness
#    - The scene still has one legend region plus a small number of separated rectangular objects.
#    - No new annotations are added, so the variant stays visually close to the source task and easy to parse.
# g) Color inference
#    - Colors keep their original role: the legend specifies a target set, and components either contain all of it or not.
#    - The new choice is driven only by visible component size among the matching set, so no extra color mapping burden is added.
#
# Generator guarantees:
# - exactly 3 non-matching components,
# - 2 or 3 matching components,
# - a unique largest matching component,
# - only that largest matching component is absent from the output.


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
    Remove only the largest matching connected component indicated by the legend.

    Input structure:
    - The background color dominates the outer frame.
    - The top-left corner contains a small legend square whose bottom edge and right edge share one border color.
    - Away from the legend, the grid contains several multicolor 4-connected components on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Find every 4-connected non-background component and test whether it contains all target colors.
    - Among the matching components, compare their areas and erase only the largest one by painting it with the background.
    - Leave every non-matching component and every smaller matching component unchanged.

    Key insight: The legend still defines which components qualify, but qualification is no longer enough by itself;
    the decisive step is selecting the single largest qualifying component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend = _find_legend(g, bg)
    _sz, _x, legend_colors = legend
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    matches: list[np.ndarray] = []
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            matches.append(pts)

    if matches:
        largest = max(matches, key=lambda pts: int(pts.shape[0]))
        for r, c in largest:
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

# -- Variant-specific component mix --
NONMATCH_COUNT = 3
MATCH_COUNTS = [2, 3]
MATCH_WEIGHTS = [4, 1]
MIN_DELETE_AREA = 18
MAX_DELETE_AREA = 36
MIN_OTHER_MATCH_AREA = 6
MIN_MATCH_AREA_GAP = 4
NONMATCH_MAX_AREA = 30

# -- Generation retry budget --
MAX_RETRIES = 500


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


def _sample_dims(*, min_area: int = 1, max_area: int | None = None) -> tuple[int, int]:
    """Sample component dimensions from the original shape families, with optional area bounds."""
    for _ in range(200):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        area = h * w
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        return h, w

    candidates: list[tuple[int, int]] = []
    for h in range(2, GRID_H + 1):
        for w in range(2, GRID_W + 1):
            area = h * w
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            candidates.append((h, w))
    if not candidates:
        raise ValueError("no component dimensions satisfy the requested area bounds")
    return random.choice(candidates)


def _sample_matching_dims(match_count: int) -> list[tuple[int, int]]:
    """Return dimensions for matching components with a guaranteed unique largest area."""
    for _ in range(200):
        delete_h, delete_w = _sample_dims(min_area=MIN_DELETE_AREA, max_area=MAX_DELETE_AREA)
        delete_area = delete_h * delete_w
        max_other_area = delete_area - MIN_MATCH_AREA_GAP
        if max_other_area < MIN_OTHER_MATCH_AREA:
            continue

        dims = [(delete_h, delete_w)]
        used_areas = {delete_area}
        ok = True
        for _ in range(match_count - 1):
            for _ in range(200):
                h, w = _sample_dims(min_area=MIN_OTHER_MATCH_AREA, max_area=max_other_area)
                area = h * w
                if area in used_areas:
                    continue
                dims.append((h, w))
                used_areas.add(area)
                break
            else:
                ok = False
                break
        if ok:
            return dims
    raise RuntimeError("failed to sample matching dimensions with a unique largest area")


def _component_patch(h: int, w: int, targets: list[int], match: bool) -> np.ndarray:
    """Build an h x w patch using full color-0 support and sparse colored pixels."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if match:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if match:
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
    """Try once to build a valid input for the D17 variant; return it or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    match_count = random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0]
    matching_dims = _sample_matching_dims(match_count)
    largest_area = max(h * w for h, w in matching_dims)

    specs: list[dict[str, object]] = []
    for i, (h, w) in enumerate(matching_dims):
        area = h * w
        specs.append(
            {
                "h": h,
                "w": w,
                "match": True,
                "delete": (i == 0 and area == largest_area),
            }
        )

    for _ in range(NONMATCH_COUNT):
        h, w = _sample_dims(min_area=MIN_OTHER_MATCH_AREA, max_area=NONMATCH_MAX_AREA)
        specs.append({"h": h, "w": w, "match": False, "delete": False})

    random.shuffle(specs)

    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        match = bool(spec["match"])
        delete = bool(spec["delete"])
        patch = _component_patch(h, w, targets, match)
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
        if not delete:
            paste(out, patch, r, c)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    bg = _bg_boundary(inp)
    legend_colors = _find_legend(inp, bg)[2]
    match_areas = sorted(
        [
            int(pts.shape[0])
            for pts in components_mask(inp != bg, connectivity=4)
            if legend_colors.issubset({int(inp[r, c]) for r, c in pts})
        ],
        reverse=True,
    )
    if len(match_areas) < 2:
        return None
    if match_areas[0] == match_areas[1]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
