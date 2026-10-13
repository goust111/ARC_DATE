from __future__ import annotations

import random
from collections import deque

import numpy as np

BG = 0
BLOCK = 5
CONTROL = 6
COUNTER = 7
RAY = 8

Coord = tuple[int, int]


def _count_row_counters(row: np.ndarray, c: int) -> tuple[int, int]:
    left = int(np.count_nonzero(row[:c] == COUNTER))
    right = int(np.count_nonzero(row[c + 1 :] == COUNTER))
    return left, right


def _trace_direction(grid: np.ndarray, start: Coord, direction: int, budget: int) -> tuple[list[Coord], list[int], int]:
    h, w = grid.shape
    r, c = map(int, start)
    path: list[Coord] = []
    detours: list[int] = []
    spent = 0

    while spent < budget:
        next_c = c + direction
        if not (0 <= next_c < w):
            break

        while int(grid[r, next_c]) == BLOCK:
            moved = False
            down_r = r + 1
            if down_r < h and int(grid[down_r, c]) != BLOCK:
                nr = down_r
                moved = True
                detours.append(1)
            if not moved:
                up_r = r - 1
                if up_r < 0 or int(grid[up_r, c]) == BLOCK:
                    raise AssertionError("ray is trapped by obstacles or boundary")
                nr = up_r
                detours.append(-1)
            if int(grid[nr, c]) != BG:
                raise AssertionError("vertical detour stepped onto a non-background cell")
            r = nr
            path.append((r, c))

        if int(grid[r, next_c]) != BG:
            raise AssertionError("horizontal ray stepped onto a non-background cell")

        c = next_c
        path.append((r, c))
        spent += 1

    return path, detours, spent


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw budget-controlled cyan horizontal rays that bend vertically around gray blockers.

    Input structure:
    - The grid uses black background with 2-3 magenta control pixels, one per control row.
    - Each control row also contains orange counter pixels strictly to the left and right of the control pixel.
    - Gray obstacle cells are scattered 1x1 or 1x2 wall fragments; valid instances keep ray corridors non-overlapping.

    Rule:
    - Start from an all-black output, then copy every orange counter and gray obstacle to the same location.
    - For each magenta control at (r, c), count orange pixels on its row strictly left of c and strictly right of c; these are the left and right horizontal budgets.
    - Emit a cyan ray left and right from the control pixel. Each horizontal move by one column spends one budget point, but when the next horizontal cell is gray, first move down one row for free; if that is blocked or out of bounds, move up one row for free instead, then continue trying the same horizontal step.
    - Paint every visited ray cell cyan, including the control pixel, while never overwriting the preserved orange counters or gray obstacles.

    Key insight:
    - The orange counters determine only how many horizontal columns each ray may advance, while the gray blockers determine which rows the ray occupies because vertical detours do not consume budget.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.where((g == COUNTER) | (g == BLOCK), g, BG)
    controls = np.argwhere(g == CONTROL)
    assert 2 <= len(controls) <= 3

    seen_rows: set[int] = set()
    for r0, c0 in controls:
        r, c = int(r0), int(c0)
        assert r not in seen_rows
        seen_rows.add(r)
        assert int(np.count_nonzero(g[r] == CONTROL)) == 1

        left_budget, right_budget = _count_row_counters(g[r], c)
        out[r, c] = RAY

        for direction, budget in [(-1, left_budget), (1, right_budget)]:
            path, _detours, _spent = _trace_direction(g, (r, c), direction, budget)
            for pr, pc in path:
                if int(out[pr, pc]) in (COUNTER, BLOCK):
                    raise AssertionError("ray attempted to overwrite a preserved object")
                out[pr, pc] = RAY

    return out


def _connected_component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _wrong_straight_solver(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.where((g == COUNTER) | (g == BLOCK), g, BG)
    for r0, c0 in np.argwhere(g == CONTROL):
        r, c = int(r0), int(c0)
        left_budget, right_budget = _count_row_counters(g[r], c)
        out[r, c] = RAY
        for step in range(1, left_budget + 1):
            cc = c - step
            if cc < 0:
                break
            if int(out[r, cc]) == BG:
                out[r, cc] = RAY
        for step in range(1, right_budget + 1):
            cc = c + step
            if cc >= g.shape[1]:
                break
            if int(out[r, cc]) == BG:
                out[r, cc] = RAY
    return out


def _wrong_paid_vertical_solver(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.where((g == COUNTER) | (g == BLOCK), g, BG)

    for r0, c0 in np.argwhere(g == CONTROL):
        base_r, base_c = int(r0), int(c0)
        left_budget, right_budget = _count_row_counters(g[base_r], base_c)
        out[base_r, base_c] = RAY

        for direction, budget in [(-1, left_budget), (1, right_budget)]:
            r, c = base_r, base_c
            spent = 0
            while spent < budget:
                next_c = c + direction
                if not (0 <= next_c < w):
                    break
                if int(g[r, next_c]) == BLOCK:
                    down_r = r + 1
                    if down_r < h and int(g[down_r, c]) != BLOCK:
                        nr = down_r
                    else:
                        up_r = r - 1
                        if up_r < 0 or int(g[up_r, c]) == BLOCK:
                            break
                        nr = up_r
                    if int(g[nr, c]) != BG:
                        break
                    r = nr
                    out[r, c] = RAY
                    spent += 1
                    continue
                if int(g[r, next_c]) != BG:
                    break
                c = next_c
                out[r, c] = RAY
                spent += 1

    return out


def _choose_rows(height: int, n_controls: int) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = [(0, "down")]
    if n_controls == 3:
        mid = random.randint(max(4, height // 2 - 1), min(height - 5, height // 2 + 1))
        rows.append((mid, "down"))
    rows.append((height - 1, "up"))
    return rows


def _build_side(
    grid: np.ndarray,
    reserved_path_cells: set[Coord],
    start: Coord,
    direction: int,
    budget: int,
    base_shift: int,
    second_shift_step: int | None,
) -> dict:
    h, w = grid.shape
    r, c = map(int, start)
    curr_r, curr_c = r, c
    path: list[Coord] = []
    detours: list[int] = []
    obstacle_cells: list[Coord] = []

    shift_plan = {1: base_shift}
    if second_shift_step is not None:
        shift_plan[int(second_shift_step)] = base_shift

    for step in range(1, budget + 1):
        next_c = curr_c + direction
        if not (0 <= next_c < w):
            raise AssertionError("generator selected an impossible budget")

        if step in shift_plan:
            shift = shift_plan[step]
            ahead = (curr_r, next_c)
            if int(grid[ahead]) != BG or ahead in reserved_path_cells:
                raise AssertionError("obstacle collision during generation")
            grid[ahead] = BLOCK
            obstacle_cells.append(ahead)

            if shift == 1:
                if curr_r + 1 >= h:
                    raise AssertionError("downward detour left the grid")
                if int(grid[curr_r + 1, curr_c]) != BG or (curr_r + 1, curr_c) in reserved_path_cells:
                    raise AssertionError("downward detour blocked during generation")
            else:
                if curr_r + 1 < h:
                    blocker = (curr_r + 1, curr_c)
                    if int(grid[blocker]) != BG or blocker in reserved_path_cells:
                        raise AssertionError("upward forcing blocker collision")
                    grid[blocker] = BLOCK
                    obstacle_cells.append(blocker)
                if curr_r - 1 < 0:
                    raise AssertionError("upward detour left the grid")

            next_r = curr_r + shift
            if int(grid[next_r, curr_c]) != BG or (next_r, curr_c) in reserved_path_cells:
                raise AssertionError("vertical path cell collision during generation")
            curr_r = next_r
            path.append((curr_r, curr_c))
            detours.append(shift)

        if int(grid[curr_r, next_c]) != BG or (curr_r, next_c) in reserved_path_cells:
            raise AssertionError("horizontal path collision during generation")
        curr_c = next_c
        path.append((curr_r, curr_c))

    return {"path": path, "detours": detours, "obstacles": obstacle_cells}


def _sample_counter_positions(grid: np.ndarray, row: int, col: int, budget: int, direction: int) -> list[int]:
    if direction == -1:
        candidates = [c for c in range(col) if int(grid[row, c]) == BG]
    else:
        candidates = [c for c in range(col + 1, grid.shape[1]) if int(grid[row, c]) == BG]
    if len(candidates) < budget:
        raise AssertionError("not enough free cells for counters")
    return random.sample(candidates, budget)


def _place_distractor_blocks(grid: np.ndarray, forbidden: set[Coord]) -> None:
    h, w = grid.shape
    attempts = 0
    extra_count = random.randint(0, 3)
    placed = 0
    while placed < extra_count and attempts < 60:
        attempts += 1
        orient = random.choice(["single", "h2", "v2"])
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        cells = [(r, c)]
        if orient == "h2":
            cells.append((r, c + random.choice([-1, 1])))
        elif orient == "v2":
            cells.append((r + random.choice([-1, 1]), c))

        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
            continue
        if any(int(grid[rr, cc]) != BG for rr, cc in cells):
            continue
        if any((rr, cc) in forbidden for rr, cc in cells):
            continue
        if any(abs(rr - fr) <= 1 and abs(cc - fc) <= 1 for rr, cc in cells for fr, fc in forbidden):
            continue
        for rr, cc in cells:
            grid[rr, cc] = BLOCK
        placed += 1


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    controls = np.argwhere(g == CONTROL)
    if not (2 <= len(controls) <= 3):
        return False
    if len({int(r) for r, _ in controls}) != len(controls):
        return False

    obstacle_sizes = _connected_component_sizes(g == BLOCK)
    if any(size > 2 for size in obstacle_sizes):
        return False

    try:
        out = main(g)
    except Exception:
        return False

    if np.array_equal(out, _wrong_straight_solver(g)):
        return False
    if np.array_equal(out, _wrong_paid_vertical_solver(g)):
        return False

    all_path_cells: set[Coord] = set()
    up_detours = 0
    down_detours = 0
    multi_detour_side_seen = False

    for r0, c0 in controls:
        r, c = int(r0), int(c0)
        left_budget, right_budget = _count_row_counters(g[r], c)
        if min(left_budget, right_budget) < 3:
            return False
        row_path_cells: set[Coord] = set()
        for direction, budget in [(-1, left_budget), (1, right_budget)]:
            try:
                path, detours, spent = _trace_direction(g, (r, c), direction, budget)
            except Exception:
                return False
            if spent != budget:
                return False
            if any(pr == r for pr, _pc in path):
                return False
            if len(detours) >= 2:
                multi_detour_side_seen = True
            up_detours += detours.count(-1)
            down_detours += detours.count(1)
            side_set = set(path)
            overlap = row_path_cells & side_set
            if any(pc != c for _pr, pc in overlap):
                return False
            for cell in path:
                if cell in all_path_cells:
                    return False
                if int(g[cell]) != BG:
                    return False
            row_path_cells.update(side_set)
        all_path_cells.update(row_path_cells)

    if up_detours == 0 or down_detours == 0 or not multi_detour_side_seen:
        return False
    return True


def _generate_once() -> np.ndarray:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    n_controls = random.choice([2, 3, 3])
    grid = np.zeros((h, w), dtype=int)

    row_specs = _choose_rows(h, n_controls)
    reserved_path_cells: set[Coord] = set()
    control_specs: list[dict] = []

    used_down_side = False
    for idx, (row, role) in enumerate(row_specs):
        col = random.randint(4, w - 5)
        max_left = min(5, col - 1)
        max_right = min(5, (w - 1 - col) - 1)
        if min(max_left, max_right) < 3:
            raise AssertionError("grid too small for budgets")
        left_budget = random.randint(3, max_left)
        right_budget = random.randint(3, max_right)

        grid[row, col] = CONTROL
        reserved_path_cells.add((row, col))
        control_specs.append(
            {
                "row": row,
                "col": col,
                "role": role,
                "left_budget": left_budget,
                "right_budget": right_budget,
                "sides": {},
            }
        )

        if role == "down":
            use_second_left = False
            use_second_right = False
            if not used_down_side:
                if random.choice([False, True]):
                    use_second_left = True
                else:
                    use_second_right = True
                used_down_side = True
            else:
                use_second_left = random.choice([False, True])
                use_second_right = random.choice([False, True])

            second_left = random.randint(2, left_budget - 1) if use_second_left else None
            second_right = random.randint(2, right_budget - 1) if use_second_right else None
            base_shift = 1
        else:
            second_left = None
            second_right = None
            base_shift = -1

        left_info = _build_side(
            grid,
            reserved_path_cells,
            (row, col),
            -1,
            left_budget,
            base_shift,
            second_left,
        )
        right_info = _build_side(
            grid,
            reserved_path_cells,
            (row, col),
            1,
            right_budget,
            base_shift,
            second_right,
        )
        reserved_path_cells.update(left_info["path"])
        reserved_path_cells.update(right_info["path"])
        control_specs[-1]["sides"][-1] = left_info
        control_specs[-1]["sides"][1] = right_info

    if not used_down_side:
        raise AssertionError("expected at least one downward double-detour side")

    forbidden_for_distractors = set(reserved_path_cells)
    for spec in control_specs:
        forbidden_for_distractors.add((spec["row"], spec["col"]))
    _place_distractor_blocks(grid, forbidden_for_distractors)

    for spec in control_specs:
        row = int(spec["row"])
        col = int(spec["col"])
        left_positions = _sample_counter_positions(grid, row, col, int(spec["left_budget"]), -1)
        right_positions = _sample_counter_positions(grid, row, col, int(spec["right_budget"]), 1)
        for cc in left_positions + right_positions:
            grid[row, cc] = COUNTER

    if not _validate_semantics(grid):
        raise AssertionError("semantic validation failed")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_once()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid instance for puzzle 8")
