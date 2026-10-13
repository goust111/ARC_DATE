from __future__ import annotations

import random

import numpy as np


BG = 0
GUIDE_RIGHT = 4  # yellow
GUIDE_LEFT = 8  # teal
MARBLE_COLORS = (1, 2, 3, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _drop_one(start_col: int, guides: dict[tuple[int, int], int], height: int) -> tuple[int, list[int]]:
    """Simulate one marble and consume every guide it triggers."""
    row = 0
    col = int(start_col)
    used: list[int] = []
    while row < height - 1:
        guide = guides.pop((row + 1, col), None)
        row += 1
        if guide is None:
            continue
        used.append(int(guide))
        if guide == GUIDE_LEFT:
            col -= 1
        elif guide == GUIDE_RIGHT:
            col += 1
        else:
            raise AssertionError(f"unexpected guide color: {guide}")
    return col, used


def _simulate_sequence(grid: np.ndarray, *, consume: bool) -> tuple[list[int], list[int]]:
    """Return landing columns and guide colors used in left-to-right release order."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    marbles = [(int(col), int(g[0, col])) for col in np.flatnonzero(g[0] != BG)]
    assert marbles, "expected at least one top-row marble"

    guides = {
        (int(r), int(c)): int(g[r, c])
        for r in range(1, height)
        for c in range(width)
        if int(g[r, c]) != BG
    }
    assert all(color in (GUIDE_LEFT, GUIDE_RIGHT) for color in guides.values()), "interior cells must be guide blocks"

    landings: list[int] = []
    used_colors: list[int] = []
    for start_col, _color in marbles:
        guide_state = guides if consume else dict(guides)
        landing, used = _drop_one(start_col, guide_state, height)
        assert 0 <= landing < width, "marble was pushed out of bounds"
        landings.append(landing)
        used_colors.extend(used)
    return landings, used_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the top-row marbles through one-shot guide blocks and keep only their bottom-row landing pixels.

    Input structure:
    - Black background with several colored single-pixel marbles on the top row.
    - Interior cells contain isolated single-pixel guide blocks: teal means "push left" and yellow means "push right".
    - Guide blocks are the only non-black cells below the top row, and the layout keeps every push inside the grid.

    Rule:
    - Release the marbles one by one from left to right according to their starting columns.
    - Each marble falls straight down; when the cell directly below it is a guide block, teal shifts the marble one column left and yellow shifts it one column right, and that guide block disappears immediately.
    - Continue until the marble reaches the bottom row. The output is an all-black grid except for the bottom-row landing cells, which keep the original marble colors.

    Key insight:
    - The guide field is stateful: earlier marbles consume guide blocks, so later marbles can fall through places that would have redirected them in the untouched layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    marbles = sorted((int(col), int(g[0, col])) for col in np.flatnonzero(g[0] != BG))
    assert marbles, "expected at least one top-row marble"

    guides = {
        (int(r), int(c)): int(g[r, c])
        for r in range(1, height)
        for c in range(width)
        if int(g[r, c]) != BG
    }
    assert all(color in (GUIDE_LEFT, GUIDE_RIGHT) for color in guides.values()), "interior cells must be guide blocks"

    out = np.zeros_like(g)
    for start_col, color in marbles:
        landing_col, _used = _drop_one(start_col, guides, height)
        assert out[height - 1, landing_col] == BG, "landing columns must be distinct"
        out[height - 1, landing_col] = int(color)
    return out


def _sample_spaced_rows(height: int, count: int, gap: int = 2) -> list[int]:
    rows: list[int] = []
    candidates = list(range(1, height - 1))
    random.shuffle(candidates)
    for row in candidates:
        if all(abs(row - existing) >= gap for existing in rows):
            rows.append(row)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError("could not sample enough spaced rows")
    rows.sort()
    return rows


def _guide_isolated(existing: dict[tuple[int, int], int], row: int, col: int, height: int, width: int) -> bool:
    if not (1 <= row < height - 1 and 0 <= col < width):
        return False
    for other_row, other_col in existing:
        if abs(other_row - row) <= 1 and abs(other_col - col) <= 1:
            return False
    return True


def _visited_below_cells(starts: list[int], guides: dict[tuple[int, int], int], height: int) -> set[tuple[int, int]]:
    """Cells that would trigger if a guide were placed there during sequential play."""
    guide_state = dict(guides)
    seen: set[tuple[int, int]] = set()
    for start_col in starts:
        row = 0
        col = int(start_col)
        while row < height - 1:
            seen.add((row + 1, col))
            guide = guide_state.pop((row + 1, col), None)
            row += 1
            if guide == GUIDE_LEFT:
                col -= 1
            elif guide == GUIDE_RIGHT:
                col += 1
    return seen


def _reserve_extra_start(
    width: int,
    taken: list[int],
    core_left: int,
    core_right: int,
) -> int:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    for col in candidates:
        if core_left - 2 <= col <= core_right + 2:
            continue
        if all(abs(col - other) >= 4 for other in taken):
            return col
    raise ValueError("could not place an extra marble start")


def _build_candidate() -> np.ndarray | None:
    marble_count = random.randint(3, 5)
    merge_steps = random.randint(2, 3)

    min_width = max(12, 6 + 4 * (marble_count - 2) + merge_steps)
    width = random.randint(min_width, min(30, min_width + 8))
    height = random.randint(12, 22)

    grid = np.zeros((height, width), dtype=int)
    guides: dict[tuple[int, int], int] = {}

    core_left = random.randint(1, width - merge_steps - 3)
    core_right = core_left + merge_steps
    core_rows = _sample_spaced_rows(height, merge_steps + 1, gap=2)

    for step, row in enumerate(core_rows[:-1]):
        col = core_left + step
        if not _guide_isolated(guides, row, col, height, width):
            return None
        guides[(row, col)] = GUIDE_RIGHT

    contest_row = core_rows[-1]
    if not _guide_isolated(guides, contest_row, core_right, height, width):
        return None
    guides[(contest_row, core_right)] = GUIDE_LEFT

    starts = [core_left, core_right]
    lane_current_cols: dict[int, int] = {core_left: core_left, core_right: core_right}

    for _ in range(marble_count - 2):
        try:
            start = _reserve_extra_start(width, starts, core_left, core_right)
        except ValueError:
            return None
        starts.append(start)
        lane_current_cols[start] = start

    starts.sort()

    for start in starts:
        if start in (core_left, core_right):
            continue
        num_guides = random.randint(0, 2)
        if num_guides == 0:
            continue
        lane_rows = _sample_spaced_rows(height, num_guides, gap=2)
        lane_rows.sort()
        current = lane_current_cols[start]
        left_limit = max(0, start - 2)
        right_limit = min(width - 1, start + 2)
        for row in lane_rows:
            directions = []
            if current > left_limit:
                directions.append(GUIDE_LEFT)
            if current < right_limit:
                directions.append(GUIDE_RIGHT)
            random.shuffle(directions)
            placed = False
            for direction in directions:
                if not _guide_isolated(guides, row, current, height, width):
                    continue
                next_col = current - 1 if direction == GUIDE_LEFT else current + 1
                if not (0 <= next_col < width):
                    continue
                guides[(row, current)] = direction
                current = next_col
                placed = True
                break
            if not placed:
                return None

    sequential_starts = sorted(starts)
    trigger_cells = _visited_below_cells(sequential_starts, guides, height)
    decoy_count = random.randint(0, 2)
    for _ in range(decoy_count):
        placed = False
        candidates = [(r, c) for r in range(1, height - 1) for c in range(width)]
        random.shuffle(candidates)
        for row, col in candidates:
            if (row, col) in trigger_cells:
                continue
            if not _guide_isolated(guides, row, col, height, width):
                continue
            guides[(row, col)] = random.choice((GUIDE_LEFT, GUIDE_RIGHT))
            placed = True
            break
        if not placed:
            break

    colors = random.sample(MARBLE_COLORS, k=marble_count)
    for col, color in zip(sequential_starts, colors):
        grid[0, col] = int(color)
    for (row, col), color in guides.items():
        grid[row, col] = int(color)

    landings, used_colors = _simulate_sequence(grid, consume=True)
    static_landings, _ = _simulate_sequence(grid, consume=False)

    if len(set(landings)) != marble_count:
        return None
    if landings == sorted(sequential_starts):
        return None
    if landings == static_landings:
        return None
    if GUIDE_LEFT not in used_colors or GUIDE_RIGHT not in used_colors:
        return None
    if np.count_nonzero(grid[1:]) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
