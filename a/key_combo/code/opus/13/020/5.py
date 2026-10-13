from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _guide_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid == GREEN, axis=1))
    assert len(rows) == 1, "expected exactly one horizontal guide row"
    return int(rows[0])


def _left_marker(grid: np.ndarray) -> tuple[int, int]:
    markers = np.argwhere(grid == RED)
    assert markers.shape == (1, 2), "expected exactly one red marker"
    r, c = map(int, markers[0])
    assert c == 0, "red marker must sit on the left border"
    return r, c


def _project_obstacles(grid: np.ndarray, guide_row: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    obstacles: set[tuple[int, int]] = set()

    for r_raw, c_raw in np.argwhere(grid == BLUE):
        r = int(r_raw)
        c = int(c_raw)
        out[r, c] = BLACK
        if r == guide_row:
            continue
        if int(grid[guide_row, c]) != GREEN:
            continue
        landing = guide_row - 1 if r < guide_row else guide_row + 1
        obstacles.add((landing, c))

    for r, c in obstacles:
        out[r, c] = BLUE

    return out, obstacles


def _can_traverse(base: np.ndarray, r: int, c: int) -> bool:
    return _in_bounds(base.shape, r, c) and int(base[r, c]) != BLUE


def _visit(base: np.ndarray, out: np.ndarray, r: int, c: int) -> None:
    if int(base[r, c]) in (BLACK, RED):
        out[r, c] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project scattered blue pixels onto a gapped guide row to create obstacles, then draw the red trace that sidesteps them.

    Input structure:
    - The grid has black background, one horizontal green guide row with internal black gaps, and isolated blue source pixels above and below that row.
    - A single red marker sits on the left border and indicates the horizontal travel corridor.
    - Blue sources never occupy the guide row itself; they are interpreted only through their shared columns with the guide.

    Rule:
    - Each blue source is consumed by a vertical projection onto the guide column. If that guide cell is green, the source becomes a blue obstacle in the cell immediately above or below the guide on the same side; if the guide cell is a black gap, the source disappears.
    - After projection, start from the red marker and move right. On a plain step, advance one cell right. If the next column contains a blue obstacle just above the guide, detour down-right-up. If the next column contains a blue obstacle just below the guide, detour up-right-down.
    - Only visited black cells become red. The green guide, the blue obstacles, and every other non-black cell remain unchanged.

    Key insight:
    - The guide row plays three roles at once: it filters which blue sources survive, fixes the projected obstacle positions, and determines the detour side from whether each obstacle lands above or below it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    guide_row = _guide_row(g)
    start_r, start_c = _left_marker(g)
    base, obstacles = _project_obstacles(g, guide_row)
    out = np.array(base, copy=True)

    if start_r == guide_row or any(r == start_r for r, _ in obstacles):
        path_row = start_r
    else:
        path_row = guide_row

    r, c = path_row, start_c
    if _can_traverse(base, r, c):
        _visit(base, out, r, c)

    while c < g.shape[1] - 1:
        if path_row == guide_row:
            next_col = c + 1
            if (guide_row - 1, next_col) in obstacles:
                steps = [(1, 0), (0, 1), (-1, 0)]
            elif (guide_row + 1, next_col) in obstacles:
                steps = [(-1, 0), (0, 1), (1, 0)]
            else:
                steps = [(0, 1)]
        else:
            next_col = c + 1
            if (path_row, next_col) in obstacles:
                if path_row == guide_row - 1:
                    steps = [(1, 0), (0, 1), (0, 1), (-1, 0)]
                else:
                    steps = [(-1, 0), (0, 1), (0, 1), (1, 0)]
            else:
                steps = [(0, 1)]

        rr, cc = r, c
        ok = True
        for dr, dc in steps:
            rr += dr
            cc += dc
            if not _can_traverse(base, rr, cc):
                ok = False
                break
        if not ok:
            break

        rr, cc = r, c
        for dr, dc in steps:
            rr += dr
            cc += dc
            _visit(base, out, rr, cc)
        r, c = rr, cc

    return out


def _choose_gap_runs(span_cols: list[int]) -> list[int] | None:
    inner_cols = span_cols[1:-1]
    if len(inner_cols) < 2:
        return None

    for _ in range(200):
        gap_count = random.randint(1, min(3, len(inner_cols)))
        gap_cells: set[int] = set()
        runs_made = 0
        for _ in range(gap_count):
            max_width = min(2, len(inner_cols))
            width = random.randint(1, max_width)
            starts = list(range(inner_cols[0], inner_cols[-1] - width + 2))
            random.shuffle(starts)
            placed = False
            for start in starts:
                cells = set(range(start, start + width))
                if not cells <= set(inner_cols):
                    continue
                if cells & gap_cells:
                    continue
                if any((cell - 1 in gap_cells) or (cell + 1 in gap_cells) for cell in cells):
                    continue
                gap_cells.update(cells)
                runs_made += 1
                placed = True
                break
            if not placed:
                break
        if runs_made != gap_count:
            continue
        if gap_cells and len(span_cols) - len(gap_cells) >= 4:
            return sorted(gap_cells)
    return None


def _sample_spaced_columns(candidates: list[int], count: int, blocked: set[int] | None = None) -> list[int] | None:
    blocked = set() if blocked is None else set(blocked)
    for _ in range(250):
        random.shuffle(candidates)
        chosen: list[int] = []
        used = set(blocked)
        for col in candidates:
            if col in used:
                continue
            if any(abs(col - prev) <= 1 for prev in chosen):
                continue
            if any(abs(col - prev) <= 1 for prev in used):
                continue
            chosen.append(col)
            used.add(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _pick_source_row(height: int, guide_row: int, side: int) -> int:
    if side < 0:
        return random.randint(0, guide_row - 2)
    return random.randint(guide_row + 2, height - 1)


def _build_input() -> np.ndarray | None:
    height = random.randint(11, 16)
    width = random.randint(16, 26)
    guide_row = random.randint(3, height - 4)

    span_len = random.randint(max(11, int(round(width * 0.65))), width - 2)
    max_start = width - span_len - 1
    if max_start < 1:
        return None
    guide_start = random.randint(1, max_start)
    guide_end = guide_start + span_len - 1
    span_cols = list(range(guide_start, guide_end + 1))

    gap_cols = _choose_gap_runs(span_cols)
    if not gap_cols:
        return None
    gap_set = set(gap_cols)
    guide_cols = [c for c in span_cols if c not in gap_set]

    valid_count = random.randint(2, min(5, len(guide_cols) // 2))
    valid_cols = _sample_spaced_columns(guide_cols.copy(), valid_count)
    if valid_cols is None:
        return None

    invalid_cap = min(3, len(gap_cols))
    if invalid_cap == 0:
        return None
    invalid_count = random.randint(1, invalid_cap)
    invalid_cols = _sample_spaced_columns(gap_cols.copy(), invalid_count, blocked=set(valid_cols))
    if invalid_cols is None:
        return None

    total_sources = len(valid_cols) + len(invalid_cols)
    if not 4 <= total_sources <= 8:
        return None

    if len(valid_cols) < 2:
        return None

    upper_valid = set(random.sample(valid_cols, random.randint(1, len(valid_cols) - 1)))
    lower_valid = set(valid_cols) - upper_valid
    if not upper_valid or not lower_valid:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[guide_row, guide_start : guide_end + 1] = GREEN
    for c in gap_cols:
        grid[guide_row, c] = BLACK

    grid[guide_row, 0] = RED

    for c in valid_cols:
        side = -1 if c in upper_valid else 1
        r = _pick_source_row(height, guide_row, side)
        grid[r, c] = BLUE

    invalid_sides = {-1, 1}
    for idx, c in enumerate(invalid_cols):
        if idx < len(invalid_sides):
            side = sorted(invalid_sides)[idx]
        else:
            side = random.choice((-1, 1))
        r = _pick_source_row(height, guide_row, side)
        grid[r, c] = BLUE

    # Keep the input visually sparse: no adjacent blue source columns.
    blue_cols = sorted(int(c) for c in np.argwhere(grid == BLUE)[:, 1])
    if any(b - a <= 1 for a, b in zip(blue_cols, blue_cols[1:])):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    projected, obstacles = _project_obstacles(grid, guide_row)
    obstacle_rows = {r for r, _ in obstacles}
    if obstacle_rows != {guide_row - 1, guide_row + 1}:
        return None
    if len(obstacles) != len(valid_cols):
        return None
    if any(abs(c1 - c0) <= 1 for (_, c0), (_, c1) in zip(sorted(obstacles), sorted(obstacles)[1:])):
        return None

    red_added = int(np.count_nonzero((out == RED) & (projected != RED)))
    if red_added < max(4, len(valid_cols) + 1):
        return None

    # Ensure at least one invalid projection was actually erased.
    erased_sources = sum(1 for c in invalid_cols if int(grid[guide_row, c]) == BLACK)
    if erased_sources != len(invalid_cols):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
