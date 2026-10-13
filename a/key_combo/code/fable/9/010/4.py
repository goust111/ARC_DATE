from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


BG = Color.BLACK
WALL = Color.GREY
WATER = Color.BLUE
CONTROL_COLORS = (Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

CUP_COUNT = 3
HEIGHT_RANGE = (3, 5)
WIDTH_RANGE = (4, 6)
GRID_HEIGHT_RANGE = (11, 18)
MIN_GRID_WIDTH = 14
MAX_GRID_WIDTH = 30
MIN_LEFT_MARGIN = 1
MIN_RIGHT_MARGIN = 1
MIN_CUP_GAP = 2
GENERATION_RETRIES = 500
TIE_BOTTOM_PROB = 0.4


def _cup_capacity(height: int, width: int) -> int:
    return (int(height) - 1) * (int(width) - 2)


def _random_composition(total: int, parts: int) -> list[int]:
    values = [0] * int(parts)
    for _ in range(int(total)):
        values[random.randrange(parts)] += 1
    return values


def _control_budget(g: np.ndarray, bg: int) -> tuple[int, int]:
    top = g[0]
    assert int(top[0]) != bg, "expected a left-aligned control bar in the top row"
    control_color = int(top[0])
    assert control_color not in (bg, WALL, WATER)

    budget = 0
    while budget < g.shape[1] and int(top[budget]) == control_color:
        budget += 1

    assert budget > 0
    assert budget < g.shape[1], "control bar must leave background after its right end"
    assert bool(np.all(top[:budget] == control_color))
    assert bool(np.all(top[budget:] == bg))
    assert not bool(np.any(g[1:] == control_color)), "control color must be confined to the top bar"
    return control_color, budget


def _parse_cup(comp: dict, g: np.ndarray, bg: int) -> dict[str, int]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 4, "each cup must have room for an interior"

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert int(patch[0, 0]) == WALL and int(patch[0, -1]) == WALL
    assert bool(np.all(patch[0, 1:-1] == bg)), "cup must be open along the top"
    assert bool(np.all(patch[1:-1, 0] == WALL))
    assert bool(np.all(patch[1:-1, -1] == WALL))
    assert bool(np.all(patch[1:-1, 1:-1] == bg))
    assert bool(np.all(patch[-1, :] == WALL))

    return {
        "top": r0,
        "left": c0,
        "bottom": r1,
        "right": c1,
        "height": height,
        "width": width,
        "row_width": width - 2,
        "capacity": _cup_capacity(height, width),
    }


def _extract_cups(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[WALL])
    assert len(comps) >= 2, "expected multiple cups"
    cups = [_parse_cup(comp, g, bg) for comp in comps]
    return sorted(cups, key=lambda cup: cup["left"])


def _ordered_cup_indices(cups: list[dict[str, int]]) -> list[int]:
    return sorted(range(len(cups)), key=lambda idx: (-cups[idx]["bottom"], cups[idx]["left"]))


def _fill_single_cup(
    out: np.ndarray,
    cup: dict[str, int],
    units: int,
    *,
    row_order: str = "bottom_up",
    col_order: str = "left_to_right",
) -> int:
    rows = list(range(cup["top"], cup["bottom"]))
    if row_order == "bottom_up":
        rows = rows[::-1]
    else:
        assert row_order == "top_down"

    cols = list(range(cup["left"] + 1, cup["right"]))
    if col_order == "right_to_left":
        cols = cols[::-1]
    else:
        assert col_order == "left_to_right"

    used = 0
    for r in rows:
        for c in cols:
            if used >= units:
                return used
            out[r, c] = WATER
            used += 1
    return used


def _apply_budget(
    grid: np.ndarray,
    cups: list[dict[str, int]],
    budget: int,
    order: list[int],
    *,
    row_order: str = "bottom_up",
    col_order: str = "left_to_right",
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[0, :budget] = BG

    remaining = int(budget)
    for idx in order:
        if remaining <= 0:
            break
        used = _fill_single_cup(out, cups[idx], remaining, row_order=row_order, col_order=col_order)
        remaining -= used
    return out


def _budget_profile(cups: list[dict[str, int]], order: list[int], budget: int) -> dict[str, int | bool | None]:
    remaining = int(budget)
    caps = [cups[idx]["capacity"] for idx in order]
    widths = [cups[idx]["row_width"] for idx in order]
    filled_before = 0
    partial_pos = None
    remainder = 0
    for pos, cap in enumerate(caps):
        if remaining >= cap:
            remaining -= cap
            filled_before += 1
            continue
        if remaining > 0:
            partial_pos = pos
            remainder = remaining
        break
    visible_partial = partial_pos is not None and remainder % widths[int(partial_pos)] != 0
    return {
        "filled_before": filled_before,
        "partial_pos": partial_pos,
        "remainder": remainder,
        "visible_partial": visible_partial,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Use the top-row bar as a shared water budget and pour blue water into lower cups first.

    Input structure:
    - The top row begins with one contiguous non-grey control bar whose length is the total water budget.
    - Below it are several separated grey open-top cup outlines on a black background.
    - Each cup is a rectangular U-shape with black interior cells and no pre-existing blue water.

    Rule:
    - Erase the control bar to background and interpret its length as the total amount of water.
    - Sort cups by bottom row from lower to higher, breaking equal bottoms by the leftmost cup first.
    - Pour water into each cup in that order, filling interior cells blue from the bottom row upward and, within each row, from left to right until the budget is exhausted.

    Key insight:
    - The top bar is one global budget shared across all containers, so both the cross-cup order and the within-cup bottom-up row fill are necessary to determine the unique final water pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 5 <= g.shape[0] <= 30 and 5 <= g.shape[1] <= 30

    bg = get_bg(g)
    fg_colors = set(colors_present(g, exclude={bg}))
    assert WALL in fg_colors, "expected grey cup outlines"
    _control_color, budget = _control_budget(g, bg)
    assert fg_colors == {WALL, _control_color}, "input should contain only the control bar and grey cups"

    cups = _extract_cups(g, bg)
    order = _ordered_cup_indices(cups)
    return _apply_budget(g, cups, budget, order)


def _sample_shapes() -> tuple[list[int], list[int]]:
    for _ in range(200):
        heights = [random.randint(*HEIGHT_RANGE) for _ in range(CUP_COUNT)]
        widths = [random.randint(*WIDTH_RANGE) for _ in range(CUP_COUNT)]
        capacities = [_cup_capacity(h, w) for h, w in zip(heights, widths)]
        if len(set(capacities)) == CUP_COUNT:
            return heights, widths
    raise RuntimeError("failed to sample distinct cup capacities")


def _candidate_bottoms(heights: list[int], capacities: list[int], grid_h: int, *, tie_mode: bool) -> list[tuple[int, int, int]]:
    mins = [h + 1 for h in heights]
    left_order = tuple(range(CUP_COUNT))
    cap_desc = tuple(sorted(range(CUP_COUNT), key=lambda idx: -capacities[idx]))
    cap_asc = tuple(sorted(range(CUP_COUNT), key=lambda idx: capacities[idx]))
    candidates: list[tuple[int, int, int]] = []

    for b0 in range(mins[0], grid_h):
        for b1 in range(mins[1], grid_h):
            for b2 in range(mins[2], grid_h):
                bottoms = (b0, b1, b2)
                unique_count = len(set(bottoms))
                if tie_mode:
                    if unique_count != 2:
                        continue
                    tied_pairs = [(i, j) for i in range(CUP_COUNT) for j in range(i + 1, CUP_COUNT) if bottoms[i] == bottoms[j]]
                    if len(tied_pairs) != 1 or tied_pairs[0][1] - tied_pairs[0][0] != 1:
                        continue
                else:
                    if unique_count != CUP_COUNT:
                        continue

                order = tuple(sorted(range(CUP_COUNT), key=lambda idx: (-bottoms[idx], idx)))
                if order == left_order or order == cap_desc or order == cap_asc:
                    continue
                candidates.append(bottoms)

    return candidates


def _budget_categories(cups: list[dict[str, int]], order: list[int]) -> dict[str, list[int]]:
    total_capacity = sum(cups[idx]["capacity"] for idx in order)
    categories: dict[str, list[int]] = {
        "second_partial_visible": [],
        "third_partial_visible": [],
        "first_partial_visible": [],
        "after_second_exact": [],
        "second_partial_flat": [],
        "any": [],
    }

    for budget in range(1, min(total_capacity, MAX_GRID_WIDTH - 1)):
        profile = _budget_profile(cups, order, budget)
        categories["any"].append(budget)
        if profile["partial_pos"] == 1 and profile["visible_partial"]:
            categories["second_partial_visible"].append(budget)
        if profile["partial_pos"] == 2 and profile["visible_partial"]:
            categories["third_partial_visible"].append(budget)
        if profile["partial_pos"] == 0 and profile["visible_partial"]:
            categories["first_partial_visible"].append(budget)
        if profile["filled_before"] == 2 and profile["partial_pos"] is None:
            categories["after_second_exact"].append(budget)
        if profile["partial_pos"] == 1 and not profile["visible_partial"]:
            categories["second_partial_flat"].append(budget)
    return categories


def _choose_budget(cups: list[dict[str, int]], order: list[int]) -> int | None:
    cats = _budget_categories(cups, order)
    weighted = [
        ("second_partial_visible", 5),
        ("third_partial_visible", 2),
        ("first_partial_visible", 2),
        ("after_second_exact", 2),
        ("second_partial_flat", 1),
        ("any", 1),
    ]
    pool: list[str] = []
    for name, weight in weighted:
        if cats[name]:
            pool.extend([name] * weight)
    if not pool:
        return None
    chosen_name = random.choice(pool)
    return random.choice(cats[chosen_name])


def _layout_width(widths: list[int], budget: int) -> tuple[int, list[int]] | None:
    base_width = MIN_LEFT_MARGIN + MIN_RIGHT_MARGIN + sum(widths) + MIN_CUP_GAP * (CUP_COUNT - 1)
    min_width = max(MIN_GRID_WIDTH, base_width, budget + 1)
    if min_width > MAX_GRID_WIDTH:
        return None
    max_width = min(MAX_GRID_WIDTH, min_width + 6)
    grid_w = random.randint(min_width, max_width)

    extra = grid_w - base_width
    left_extra, gap1_extra, gap2_extra, _right_extra = _random_composition(extra, CUP_COUNT + 1)
    left = MIN_LEFT_MARGIN + left_extra
    positions = [left]
    left += widths[0] + MIN_CUP_GAP + gap1_extra
    positions.append(left)
    left += widths[1] + MIN_CUP_GAP + gap2_extra
    positions.append(left)
    return grid_w, positions


def _build_cups(heights: list[int], widths: list[int], bottoms: tuple[int, int, int], lefts: list[int]) -> list[dict[str, int]]:
    cups = []
    for height, width, bottom, left in zip(heights, widths, bottoms, lefts):
        cups.append(
            {
                "top": bottom - height + 1,
                "left": left,
                "bottom": bottom,
                "right": left + width - 1,
                "height": height,
                "width": width,
                "row_width": width - 2,
                "capacity": _cup_capacity(height, width),
            }
        )
    return cups


def _render_input(grid_h: int, grid_w: int, cups: list[dict[str, int]], budget: int, control_color: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[0, :budget] = control_color

    for cup in cups:
        top = cup["top"]
        left = cup["left"]
        bottom = cup["bottom"]
        right = cup["right"]
        grid[top, left] = WALL
        grid[top, right] = WALL
        grid[top + 1 : bottom, left] = WALL
        grid[top + 1 : bottom, right] = WALL
        grid[bottom, left : right + 1] = WALL
    return grid


def _validate_semantics(inp: np.ndarray, cups: list[dict[str, int]], order: list[int], budget: int) -> bool:
    expected = _apply_budget(inp, cups, budget, order)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    left_order = list(range(CUP_COUNT))
    cap_desc = sorted(range(CUP_COUNT), key=lambda idx: -cups[idx]["capacity"])
    high_first = sorted(range(CUP_COUNT), key=lambda idx: (cups[idx]["bottom"], cups[idx]["left"]))
    if order == left_order or order == cap_desc or order == high_first:
        return False
    if np.array_equal(expected, _apply_budget(inp, cups, budget, left_order)):
        return False
    if np.array_equal(expected, _apply_budget(inp, cups, budget, cap_desc)):
        return False
    if np.array_equal(expected, _apply_budget(inp, cups, budget, high_first)):
        return False

    profile = _budget_profile(cups, order, budget)
    if profile["partial_pos"] is not None and profile["visible_partial"]:
        if np.array_equal(
            expected,
            _apply_budget(inp, cups, budget, order, row_order="bottom_up", col_order="right_to_left"),
        ):
            return False
        if np.array_equal(
            expected,
            _apply_budget(inp, cups, budget, order, row_order="top_down", col_order="left_to_right"),
        ):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    heights, widths = _sample_shapes()
    capacities = [_cup_capacity(h, w) for h, w in zip(heights, widths)]
    grid_h = random.randint(max(GRID_HEIGHT_RANGE[0], max(heights) + 3), GRID_HEIGHT_RANGE[1])

    tie_mode = random.random() < TIE_BOTTOM_PROB
    bottoms_pool = _candidate_bottoms(heights, capacities, grid_h, tie_mode=tie_mode)
    if not bottoms_pool:
        bottoms_pool = _candidate_bottoms(heights, capacities, grid_h, tie_mode=not tie_mode)
    if not bottoms_pool:
        return None
    bottoms = random.choice(bottoms_pool)

    provisional_lefts = [0, 1, 2]
    cups_for_budget = _build_cups(heights, widths, bottoms, provisional_lefts)
    order = _ordered_cup_indices(cups_for_budget)
    budget = _choose_budget(cups_for_budget, order)
    if budget is None:
        return None

    layout = _layout_width(widths, budget)
    if layout is None:
        return None
    grid_w, lefts = layout
    cups = _build_cups(heights, widths, bottoms, lefts)
    order = _ordered_cup_indices(cups)

    control_color = random.choice(CONTROL_COLORS)
    inp = _render_input(grid_h, grid_w, cups, budget, control_color)
    if not _validate_semantics(inp, cups, order, budget):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid water-budget cup scene")
