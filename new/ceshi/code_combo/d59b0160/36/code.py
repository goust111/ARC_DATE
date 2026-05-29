# Combo variant of d59b0160 (index 36)
# combo_spec: D10/2+D17/1
# Name: 按面积排序处理+仅删除最大匹配分量

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - The visible grammar stays close to the source task: a 4x4 top-left legend with
#      green border color 3, orange background 7, and several separated rectangular
#      4-components elsewhere.
#    - Original task: compute the legend color set once, then erase every component
#      whose color set contains all legend colors.
#    - D17/1 alone: among all qualifying components, erase only the single largest one.
#    - This combo uses the D17 decision inside a D10-style loop:
#        1. find the current qualifying components,
#        2. sort by area descending,
#        3. erase only the current largest qualifier,
#        4. recompute components on the reduced grid and repeat.
#      Example: legend = {1,4,8}; matching component areas are 24, 15, 10.
#      Round 1 erases 24, round 2 erases 15, round 3 erases 10.
#      The important contrast is therefore with D17/1's one-shot behavior, not with
#      the base task's final mask.
# b) Complexity calibration:
#    - In this task family every deleted object is removed whole, so recomputation does
#      not create new geometry from surviving objects. The only observable effect of
#      D10/2 is the repeated "largest remaining match" choice across rounds.
#    - To keep that effect visible, the generator always creates 2 or 3 matching
#      components with pairwise distinct areas plus exactly 3 non-matching controls.
#      This prevents the combo from collapsing into a D17-style single deletion.
# c) Hidden-assumption audit:
#    - `_bg_boundary(...)` and `_find_legend(...)` are unchanged because the legend and
#      background conventions are preserved.
#    - The original `for pts in comps:` loop assumed all qualifiers can be deleted in
#      one pass. That assumption is invalid here; `main()` must repeatedly relabel the
#      current grid and delete only one component per round.
#    - Area comparison must use the current component sizes, not any cached input-time
#      ordering. The helper `_largest_matching_component(...)` therefore re-reads the
#      live grid each round.
#    - Because objects stay separated by at least one background cell, recomputation is
#      safe and deterministic: deleting one component cannot merge two survivors.
# d) Placement legality:
#    - Components are still axis-aligned rectangles placed with
#      `random_free_location(..., padding=1, padding_connectivity=8)`, so they stay in
#      bounds, do not overlap, and remain separated from the legend and one another.
#    - Matching and non-matching rectangles are sampled under area bounds chosen so a
#      16x16 board can fit 5-6 objects reliably.
# e) Visual observability:
#    - Humans can still infer the legend-palette test exactly as in the base task.
#    - The extra fused behavior is supported by repeated evidence that several
#      qualifying objects exist with different visible sizes, so a D17-style "only the
#      biggest disappears" hypothesis would be contradicted by samples generated from
#      this combo.
# f) Visual cleanliness:
#    - The scene remains a small set of isolated rectangles with sparse colored accents.
#      No new overlays or annotations are introduced.
#    - Distinct matching areas make the size ordering legible without cluttering the
#      board with too many objects.
# g) Change inferability:
#    - No hidden numeric property such as raw color id is used for ranking; the only
#      tie-break signal is visible component area.
#    - The palette stays the original 7 non-background colors, and only 3 legend
#      colors matter per sample, so the color reasoning burden stays modest.
# 9) New solver assumptions checked against the generator:
#    - `main()` assumes at least one matching component can exist; the generator always
#      builds 2 or 3 of them.
#    - `main()` assumes the largest match is unique each round; the generator enforces
#      distinct matching areas.
#    - `main()` assumes repeated relabeling terminates; the generator keeps components
#      disjoint, so each round removes exactly one whole component and strictly reduces
#      the number of matches.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom and right edges share one
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
    raise AssertionError("no legend border found in grid")


def _matching_components(
    g: np.ndarray,
    bg: int,
    legend_colors: set[int],
) -> list[np.ndarray]:
    """Current non-background components whose color sets contain all legend colors."""
    matches: list[np.ndarray] = []
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            matches.append(pts)
    return matches


def _largest_matching_component(
    g: np.ndarray,
    bg: int,
    legend_colors: set[int],
) -> np.ndarray | None:
    """Return the unique largest current matching component, or None if no match exists."""
    matches = _matching_components(g, bg, legend_colors)
    if not matches:
        return None
    matches.sort(key=lambda pts: int(pts.shape[0]), reverse=True)
    return matches[0]


def _erase_component(g: np.ndarray, pts: np.ndarray, bg: int) -> None:
    """Fill one component with the background color."""
    for r, c in pts:
        g[r, c] = bg


def _sequential_largest_output(grid: np.ndarray) -> np.ndarray:
    """Apply the combo rule as a fixed-point loop on a copy of the grid."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    while True:
        largest = _largest_matching_component(out, bg, legend_colors)
        if largest is None:
            break
        _erase_component(out, largest, bg)
    return out


def _single_largest_output(grid: np.ndarray) -> np.ndarray:
    """D17/1 baseline used only for generator-side validation."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    largest = _largest_matching_component(out, bg, legend_colors)
    if largest is not None:
        _erase_component(out, largest, bg)
    return out


def _deletion_round_count(grid: np.ndarray) -> int:
    """Number of combo rounds until no matching component remains."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    rounds = 0
    while True:
        largest = _largest_matching_component(out, bg, legend_colors)
        if largest is None:
            return rounds
        _erase_component(out, largest, bg)
        rounds += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly erase only the current largest legend-matching component.

    Input structure:
    - The board still uses a dominant background determined by the outer frame.
    - A 4x4 legend square sits in the top-left corner, with one border color along its
      bottom and right edges and three target colors inside.
    - Elsewhere, separated multicolor 4-connected components sit on the background.

    Rule:
    - Read the target color set from the legend interior.
    - On the current grid state, find every non-background 4-connected component whose
      palette contains all legend colors.
    - Compare the qualifying components by area and erase only the largest one.
    - Recompute connected components on the updated grid and repeat until no
      qualifying component remains.

    Key insight:
    - The legend still defines qualification, but deletion is now staged: each round
      removes just the largest current match, then the size comparison is redone on
      the reduced grid.
    """
    return _sequential_largest_output(np.asarray(grid, dtype=int))


# --------------- generator (combo 36) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (7, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 6)
MED_W = (2, 8)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

NONMATCH_COUNT = 3
MATCH_COUNTS = [2, 3]
MATCH_WEIGHTS = [3, 2]
MIN_MATCH_AREA = 10
MAX_MATCH_AREA = 30
MIN_OTHER_MATCH_AREA = 6
MIN_MATCH_AREA_GAP = 3
NONMATCH_MAX_AREA = 24

MAX_RETRIES = 700


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


def _sample_dims(*, min_area: int = 1, max_area: int | None = None) -> tuple[int, int]:
    """Sample one rectangle size from the original shape families under area bounds."""
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
    """Return matching component sizes with pairwise distinct areas."""
    for _ in range(300):
        dims: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        current_min = MIN_MATCH_AREA
        for i in range(match_count):
            max_area = MAX_MATCH_AREA - i * MIN_MATCH_AREA_GAP
            if max_area < current_min:
                break
            for _ in range(200):
                h, w = _sample_dims(min_area=current_min, max_area=max_area)
                area = h * w
                if area in used_areas:
                    continue
                dims.append((h, w))
                used_areas.add(area)
                break
            else:
                break
        if len(dims) == match_count:
            dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
            return dims
    raise RuntimeError("failed to sample matching dimensions with distinct areas")


def _component_patch(h: int, w: int, targets: list[int], match: bool) -> np.ndarray:
    """Build a rectangle-shaped component with sparse colored accents on color-0 support."""
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


def _matching_areas(grid: np.ndarray) -> list[int]:
    """Areas of all input matching components, sorted from large to small."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _legend_size, _border_color, legend_colors = _find_legend(g, bg)
    return sorted(
        [int(pts.shape[0]) for pts in _matching_components(g, bg, legend_colors)],
        reverse=True,
    )


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid combo input; return it or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    match_count = random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0]
    matching_dims = _sample_matching_dims(match_count)

    specs: list[dict[str, object]] = []
    for h, w in matching_dims:
        specs.append({"h": h, "w": w, "match": True, "delete": True})

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

    match_areas = _matching_areas(inp)
    if len(match_areas) != match_count:
        return None
    if len(set(match_areas)) != len(match_areas):
        return None

    rounds = _deletion_round_count(inp)
    if rounds != match_count or rounds < 2:
        return None

    if np.array_equal(_single_largest_output(inp), solved):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
