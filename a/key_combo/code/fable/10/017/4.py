from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLOCKER_COLOR = 1
START_COLOR = 8
BAR_COLOR_POOL = (2, 3, 4, 5, 6, 7, 9)

DIRS = {
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
    "N": (-1, 0),
}
RIGHT_TURN = {"E": "S", "S": "W", "W": "N", "N": "E"}

GEN_H_RANGE = (16, 24)
GEN_W_RANGE = (20, 28)
GEN_BAR_COUNT_RANGE = (4, 6)
GEN_BAR_LENGTH_POOL = tuple(range(3, 10))
GEN_GAP_RANGE = (2, 4)
GEN_ATTEMPTS = 400
GEN_SEARCH_ATTEMPTS = 60


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _step(pos: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return pos[0] + dr, pos[1] + dc


def _right(direction: str) -> str:
    return RIGHT_TURN[direction]


def _extract_legend(
    grid: np.ndarray, bg: int
) -> tuple[str, list[tuple[int, int, int]], set[int]]:
    h, w = grid.shape
    by_side: dict[str, list[tuple[int, int, int]]] = {"left": [], "right": []}
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != bg)

    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 2:
            continue
        rows = np.unique(coords[:, 0])
        if rows.size != 1:
            continue
        row = int(rows[0])
        cols = coords[:, 1]
        c0 = int(cols.min())
        c1 = int(cols.max())
        if coords.shape[0] != c1 - c0 + 1:
            continue
        if c0 == 0:
            by_side["left"].append((row, c1 - c0 + 1, color))
        elif c1 == w - 1:
            by_side["right"].append((row, c1 - c0 + 1, color))

    if len(by_side["left"]) >= 2 and len(by_side["right"]) == 0:
        side = "left"
    elif len(by_side["right"]) >= 2 and len(by_side["left"]) == 0:
        side = "right"
    else:
        side = "left" if len(by_side["left"]) > len(by_side["right"]) else "right"

    legend = sorted(by_side[side], key=lambda item: item[0])
    assert len(legend) >= 2, "expected multiple legend bars"
    rows = [row for row, _length, _color in legend]
    assert len(rows) == len(set(rows)), "legend rows must be distinct"
    colors_set = {color for _row, _length, color in legend}

    return side, legend, colors_set


def _find_start_and_blockers(
    grid: np.ndarray, bg: int, legend_colors: set[int]
) -> tuple[tuple[int, int], np.ndarray]:
    remaining = [
        int(color)
        for color in np.unique(grid)
        if int(color) != bg and int(color) not in legend_colors
    ]
    singleton_colors = [color for color in remaining if int(np.count_nonzero(grid == color)) == 1]
    assert len(singleton_colors) == 1, "expected a unique singleton start marker"
    start_color = singleton_colors[0]
    start = tuple(map(int, np.argwhere(grid == start_color)[0]))

    blockers = np.zeros_like(grid, dtype=bool)
    for color in remaining:
        if color != start_color:
            blockers |= grid == color
    assert np.count_nonzero(blockers) >= 1, "expected blocker pixels"
    return start, blockers


def _simulate(
    grid: np.ndarray,
    start: tuple[int, int],
    blockers: np.ndarray,
    legend: list[tuple[int, int, int]],
    initial_direction: str,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    reserved = grid != _get_bg(grid)
    pos = start
    direction = initial_direction

    for _row, budget, color in legend:
        steps_left = int(budget)
        while steps_left > 0:
            nr, nc = _step(pos, direction)
            if (not _in_bounds(grid.shape, nr, nc)) or bool(blockers[nr, nc]):
                direction = _right(direction)
                tr, tc = _step(pos, direction)
                assert _in_bounds(grid.shape, tr, tc), "turn must lead to an in-bounds cell"
                assert not bool(blockers[tr, tc]), "turn must not face another blocker"
                assert not bool(reserved[tr, tc]), "path should not enter legend/start/blocker cells"
                continue

            assert not bool(reserved[nr, nc]), "path should only paint open-area background"
            pos = (nr, nc)
            out[nr, nc] = color
            steps_left -= 1

        direction = _right(direction)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode edge-attached legend bars into a right-turning polyline that deflects around blockers.

    Input structure:
    - Several single-row horizontal bars of distinct colors touch the same left or right edge and form a top-to-bottom legend.
    - Away from the legend there is one isolated start-marker pixel and some blocker pixels on the background.
    - The legend, start marker, and blockers are already separated from each other in the input.

    Rule:
    - Read the legend from top to bottom; each bar's length is the current step budget and its color is the paint color.
    - Start from the marker facing away from the legend side, move one cell at a time, and paint every traversed open cell with the current bar color.
    - If the next cell would be out of bounds or a blocker, turn right immediately and continue spending the same bar's remaining steps; after a bar's budget is exhausted, turn right once more before the next bar.

    Key insight:
    - The edge bars are a program: color selects the pen, bar length selects the step budget, and blockers are turn triggers that bend a single colored stroke without consuming steps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = _get_bg(g)
    side, legend, legend_colors = _extract_legend(g, bg)
    start, blockers = _find_start_and_blockers(g, bg, legend_colors)
    initial_direction = "E" if side == "left" else "W"
    return _simulate(g, start, blockers, legend, initial_direction)


def _trace_cells(
    start: tuple[int, int], direction: str, steps: int
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    dr, dc = DIRS[direction]
    for k in range(1, steps + 1):
        cells.append((r + dr * k, c + dc * k))
    return cells


def _distance_to_edge(shape: tuple[int, int], pos: tuple[int, int], direction: str) -> int:
    r, c = pos
    h, w = shape
    if direction == "E":
        return w - 1 - c
    if direction == "W":
        return c
    if direction == "S":
        return h - 1 - r
    return r


def _free_cells(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    return all(_in_bounds(shape, r, c) and (r, c) not in forbidden for r, c in cells)


def _command_options(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: str,
    budget: int,
    event_type: str,
    forbidden: set[tuple[int, int]],
) -> list[dict]:
    options: list[dict] = []

    if event_type == "none":
        cells = _trace_cells(pos, direction, budget)
        if _free_cells(cells, shape, forbidden):
            end = cells[-1]
            options.append(
                {
                    "path": cells,
                    "blocker": None,
                    "end": end,
                    "next_dir": _right(direction),
                    "event": "none",
                }
            )
        return options

    if event_type == "obstacle":
        straight_choices = list(range(1, budget))
        random.shuffle(straight_choices)
        turned_direction = _right(direction)
        for straight_steps in straight_choices:
            straight = _trace_cells(pos, direction, straight_steps)
            if not _free_cells(straight, shape, forbidden):
                continue
            pivot = straight[-1]
            blocker = _step(pivot, direction)
            if not _in_bounds(shape, *blocker) or blocker in forbidden:
                continue
            if blocker[0] in (0, shape[0] - 1) or blocker[1] in (0, shape[1] - 1):
                continue
            remaining = budget - straight_steps
            turned = _trace_cells(pivot, turned_direction, remaining)
            turned_forbidden = forbidden | set(straight) | {blocker}
            if not _free_cells(turned, shape, turned_forbidden):
                continue
            options.append(
                {
                    "path": straight + turned,
                    "blocker": blocker,
                    "end": turned[-1],
                    "next_dir": _right(turned_direction),
                    "event": "obstacle",
                }
            )
        return options

    assert event_type == "boundary"
    straight_steps = _distance_to_edge(shape, pos, direction)
    if 1 <= straight_steps < budget:
        straight = _trace_cells(pos, direction, straight_steps)
        turned_direction = _right(direction)
        if _free_cells(straight, shape, forbidden):
            pivot = straight[-1]
            remaining = budget - straight_steps
            turned = _trace_cells(pivot, turned_direction, remaining)
            if _free_cells(turned, shape, forbidden | set(straight)):
                options.append(
                    {
                        "path": straight + turned,
                        "blocker": None,
                        "end": turned[-1],
                        "next_dir": _right(turned_direction),
                        "event": "boundary",
                    }
                )
    return options


def _search_plan(
    shape: tuple[int, int],
    budgets: list[int],
    events: list[str],
    start: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> list[dict] | None:
    def rec(
        idx: int,
        pos: tuple[int, int],
        direction: str,
        occupied: set[tuple[int, int]],
        blockers: set[tuple[int, int]],
        plan: list[dict],
    ) -> list[dict] | None:
        if idx == len(budgets):
            return plan.copy()

        forbidden = reserved | occupied | blockers
        options = _command_options(shape, pos, direction, budgets[idx], events[idx], forbidden)
        random.shuffle(options)
        for option in options:
            new_occupied = occupied | set(option["path"])
            new_blockers = set(blockers)
            if option["blocker"] is not None:
                new_blockers.add(option["blocker"])
            plan.append(option)
            solved = rec(idx + 1, option["end"], option["next_dir"], new_occupied, new_blockers, plan)
            if solved is not None:
                return solved
            plan.pop()
        return None

    return rec(0, start, "E", set(), set(), [])


def _sample_events(bar_count: int) -> list[str]:
    indices = list(range(bar_count))
    obstacle_indices = random.sample(indices, 2)
    events = ["none"] * bar_count
    for idx in obstacle_indices:
        events[idx] = "obstacle"
    remaining = [idx for idx in indices if events[idx] == "none"]
    if remaining and random.random() < 0.72:
        events[random.choice(remaining)] = "boundary"
    return events


def _mirror_grid(grid: np.ndarray) -> np.ndarray:
    return grid[:, ::-1].copy()


def _build_input(
    h: int,
    w: int,
    legend_top: int,
    budgets: list[int],
    colors: list[int],
    start: tuple[int, int],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for i, (budget, color) in enumerate(zip(budgets, colors)):
        grid[legend_top + i, :budget] = color
    grid[start] = START_COLOR
    for r, c in blockers:
        grid[r, c] = BLOCKER_COLOR
    return grid


def _validate_generated(
    grid: np.ndarray,
    budgets: list[int],
    events: list[str],
    plan: list[dict],
) -> bool:
    all_blockers = {option["blocker"] for option in plan if option["blocker"] is not None}
    all_path_cells = {cell for option in plan for cell in option["path"]}
    if np.count_nonzero(grid == START_COLOR) != 1:
        return False
    if np.count_nonzero(grid == BLOCKER_COLOR) < 2:
        return False
    if events.count("obstacle") < 2 or plan is None:
        return False
    if "none" not in events:
        return False
    if all_blockers & all_path_cells:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    bg = _get_bg(grid)
    side, legend, legend_colors = _extract_legend(grid, bg)
    start, _blockers = _find_start_and_blockers(grid, bg, legend_colors)
    empty_blockers = np.zeros_like(grid, dtype=bool)
    if np.array_equal(out, grid):
        return False
    try:
        plain = _simulate(grid, start, empty_blockers, legend, "E" if side == "left" else "W")
    except Exception:
        plain = None
    if plain is not None and np.array_equal(out, plain):
        return False
    if int(np.count_nonzero(out != grid)) < sum(budgets) // 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    bar_count = random.randint(*GEN_BAR_COUNT_RANGE)
    if len(GEN_BAR_LENGTH_POOL) < bar_count:
        return None

    budgets = random.sample(GEN_BAR_LENGTH_POOL, bar_count)
    colors = random.sample(BAR_COLOR_POOL, bar_count)
    legend_top = random.randint(1, 3)
    gap = random.randint(*GEN_GAP_RANGE)
    max_budget = max(budgets)

    if legend_top + bar_count + 4 >= h:
        return None
    start_col_lo = max_budget + gap + 1
    start_col_hi = min(max_budget + gap + 5, w - 5)
    start_row_lo = legend_top + bar_count + 3
    start_row_hi = h - 4
    if start_col_lo > start_col_hi or start_row_lo > start_row_hi:
        return None

    reserved = {(legend_top + i, c) for i, budget in enumerate(budgets) for c in range(budget)}
    starts = [
        (r, c)
        for r in range(start_row_lo, start_row_hi + 1)
        for c in range(start_col_lo, start_col_hi + 1)
    ]
    random.shuffle(starts)

    for _ in range(GEN_SEARCH_ATTEMPTS):
        if not starts:
            break
        start = starts.pop()
        events = _sample_events(bar_count)
        plan = _search_plan((h, w), budgets, events, start, reserved | {start})
        if plan is None:
            continue

        blockers = {option["blocker"] for option in plan if option["blocker"] is not None}
        if len(blockers) < 2:
            continue

        grid = _build_input(h, w, legend_top, budgets, colors, start, blockers)
        if random.random() < 0.5:
            grid = _mirror_grid(grid)
        if _validate_generated(grid, budgets, events, plan):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-polyline ARC input")
