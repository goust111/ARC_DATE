from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
CONTROL_PALETTE = (3, 4, 7)

GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (11, 14)
GRID_WIDTH_RANGE = (11, 14)
CONTROL_COUNT_CHOICES = (2, 3)
CONTROL_GAP_MIN = 3
CONTROL_MARGIN = 1
OPEN_FRACTION_RANGE = (0.43, 0.60)
BRANCH_GROWTH_ATTEMPTS = 500
LOOP_DEPTH_CHOICES = (1, 2, 3)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _top_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) not in (BLACK, BLUE)]


def _shortest_path_data(grid: np.ndarray, start: tuple[int, int]):
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    count = np.zeros((h, w), dtype=np.int64)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    count[start] = 1

    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) == BLUE:
                continue
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                count[nr, nc] = count[r, c]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                count[nr, nc] += count[r, c]
    return dist, count


def _unique_shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    dist, count = _shortest_path_data(grid, start)
    assert dist[goal] >= 0, "expected the red endpoints to be connected"
    assert int(count[goal]) == 1, "expected a unique shortest path"

    path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if in_bounds(grid.shape, nr, nc) and dist[nr, nc] == dist[r, c] - 1
        ]
        assert len(preds) == 1, "expected a unique predecessor chain"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restrict each top-row control ray to the unique shortest endpoint path, then toggle-repaint only that path-column intersection.

    Input structure:
    - The grid has black background, blue obstacle pixels, exactly two red endpoints, and 2-3 distinct control pixels on the top row.
    - Each control pixel defines one column and one repaint color; blue cells are impassable and control columns themselves contain no blue.
    - The two red endpoints are connected by a unique shortest 4-neighbor path through non-blue cells, with black interior path cells.

    Rule:
    - Find the unique shortest 4-neighbor path between the two red endpoints while treating blue as blocked.
    - Process the top-row control pixels from left to right.
    - For each control column, only the path pixels in that same column are affected: if one currently equals the control color, erase it to black; otherwise repaint it to the control color.
    - Leave every path pixel outside the control columns unchanged, and leave every non-path pixel unchanged.

    Key insight:
    - The vertical control rays are not applied to whole columns; each ray is masked down to the shortest-path trace, so solving the maze and locating the control columns are both necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    controls = _top_controls(g)
    assert 2 <= len(controls) <= 3, "expected 2-3 top-row control pixels"
    control_cols = [c for c, _ in controls]
    control_colors = [color for _, color in controls]
    assert len(set(control_cols)) == len(control_cols)
    assert len(set(control_colors)) == len(control_colors)
    assert set(control_colors).issubset(CONTROL_PALETTE)
    for c in control_cols:
        assert not np.any(g[:, c] == BLUE), "control columns must be free of blue obstacles"

    reds = np.argwhere(g == RED)
    assert reds.shape == (2, 2), "expected exactly two red endpoints"
    assert np.all(reds[:, 0] > 0), "endpoints must be below the control row"
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))

    path = _unique_shortest_path(g, start, goal)
    assert all(r > 0 for r, _ in path), "generated tasks keep the shortest path off the control row"
    for r, c in path[1:-1]:
        assert int(g[r, c]) == BLACK, "interior shortest-path cells must start as black"

    path_by_col: dict[int, list[tuple[int, int]]] = {}
    for r, c in path:
        if r > 0:
            path_by_col.setdefault(c, []).append((r, c))

    out = g.copy()
    for c, color in controls:
        for r, cc in path_by_col.get(c, []):
            out[r, cc] = BLACK if int(out[r, cc]) == color else color
    return out


def _choose_control_columns(width: int, count: int) -> tuple[int, ...] | None:
    cols = range(CONTROL_MARGIN, width - CONTROL_MARGIN)
    choices = [
        combo
        for combo in itertools.combinations(cols, count)
        if min(b - a for a, b in zip(combo, combo[1:])) >= CONTROL_GAP_MIN
    ]
    return random.choice(choices) if choices else None


def _split_rows(height: int) -> tuple[list[int], list[int]]:
    low = list(range(2, max(3, height // 2)))
    high = list(range(max(height // 2 + 1, 3), height - 1))
    if not low:
        low = list(range(1, height - 1))
    if not high:
        high = list(range(1, height - 1))
    return low, high


def _pick_snaking_rows(height: int, count: int) -> tuple[list[int], int, int] | None:
    low, high = _split_rows(height)
    connectors: list[int] = []
    start_high = random.choice((False, True))
    for i in range(count - 1):
        preferred = high if (i % 2 == 0) == start_high else low
        choices = [r for r in preferred if all(abs(r - prev) >= 3 for prev in connectors)]
        if not choices:
            choices = [r for r in range(1, height - 1) if all(abs(r - prev) >= 3 for prev in connectors)]
        if not choices:
            return None
        connectors.append(random.choice(choices))

    first_pool = low if connectors[0] in high else high
    start_choices = [r for r in first_pool if abs(r - connectors[0]) >= 2]
    if not start_choices:
        start_choices = [r for r in range(1, height - 1) if abs(r - connectors[0]) >= 2]
    last_pool = low if connectors[-1] in high else high
    goal_choices = [r for r in last_pool if abs(r - connectors[-1]) >= 2]
    if not goal_choices:
        goal_choices = [r for r in range(1, height - 1) if abs(r - connectors[-1]) >= 2]
    if not start_choices or not goal_choices:
        return None
    return connectors, random.choice(start_choices), random.choice(goal_choices)


def _carve_main_structure(
    grid: np.ndarray, control_cols: tuple[int, ...], connector_rows: list[int]
) -> set[tuple[int, int]]:
    open_cells: set[tuple[int, int]] = set()
    for c in control_cols:
        for r in range(1, grid.shape[0]):
            grid[r, c] = BLACK
            open_cells.add((r, c))
    for row, c0, c1 in zip(connector_rows, control_cols, control_cols[1:]):
        for c in range(c0, c1 + 1):
            grid[row, c] = BLACK
            open_cells.add((row, c))
    return open_cells


def _add_longer_loop(
    grid: np.ndarray, c0: int, c1: int, main_row: int, blocked_rows: set[int]
) -> int | None:
    options = []
    for depth in LOOP_DEPTH_CHOICES:
        for sign in (-1, 1):
            side_row = main_row + sign * depth
            if not (1 <= side_row < grid.shape[0]):
                continue
            if side_row in blocked_rows:
                continue
            if np.all(grid[side_row, c0 + 1 : c1] == BLUE):
                options.append(side_row)
    if not options:
        return None
    side_row = random.choice(options)
    grid[side_row, c0 : c1 + 1] = BLACK
    return side_row


def _open_neighbor_count(grid: np.ndarray, r: int, c: int) -> int:
    return sum(
        1
        for nr, nc in neighbors4(r, c)
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BLUE
    )


def _grow_tree_branch(grid: np.ndarray, anchor: tuple[int, int], target_len: int) -> int:
    grown = 0
    current = anchor
    for _ in range(target_len):
        candidates = []
        for nr, nc in neighbors4(*current):
            if not in_bounds(grid.shape, nr, nc) or nr == 0:
                continue
            if int(grid[nr, nc]) != BLUE:
                continue
            if _open_neighbor_count(grid, nr, nc) != 1:
                continue
            candidates.append((nr, nc))
        if not candidates:
            break
        current = random.choice(candidates)
        grid[current] = BLACK
        grown += 1
    return grown


def _connected_without_edge(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_edge: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for nxt in neighbors4(*cur):
            if not in_bounds(grid.shape, *nxt):
                continue
            if int(grid[nxt]) == BLUE or nxt in seen:
                continue
            if (cur, nxt) == blocked_edge or (nxt, cur) == blocked_edge:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def _has_alternate_route(
    grid: np.ndarray, path: list[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    return any(
        _connected_without_edge(grid, start, goal, (a, b))
        for a, b in zip(path, path[1:])
    )


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        controls = _top_controls(grid)
        if len(controls) not in (2, 3):
            return False
        control_cols = {c for c, _ in controls}
        control_colors = [color for _, color in controls]
        if len(control_cols) != len(controls) or len(set(control_colors)) != len(controls):
            return False
        if not set(control_colors).issubset(CONTROL_PALETTE):
            return False
        if any(np.any(grid[:, c] == BLUE) for c in control_cols):
            return False

        reds = np.argwhere(grid == RED)
        if reds.shape != (2, 2) or not np.all(reds[:, 0] > 0):
            return False
        start = tuple(map(int, reds[0]))
        goal = tuple(map(int, reds[1]))

        path = _unique_shortest_path(grid, start, goal)
        if any(r == 0 for r, _ in path):
            return False
        if any(int(grid[r, c]) != BLACK for r, c in path[1:-1]):
            return False
        path_cols = {c for _, c in path}
        if path_cols & control_cols != control_cols:
            return False
        if len(path_cols & control_cols) < 2:
            return False
        if not _has_alternate_route(grid, path, start, goal):
            return False

        out = main(grid)
        changed = {tuple(map(int, rc)) for rc in np.argwhere(out != grid)}
        expected_changed = {(r, c) for r, c in path if r > 0 and c in control_cols}
        if changed != expected_changed:
            return False
        if not expected_changed:
            return False
        for c, color in controls:
            hits = [(r, cc) for r, cc in path if r > 0 and cc == c]
            if not hits:
                return False
            for r, cc in hits:
                expected = BLACK if int(grid[r, cc]) == color else color
                if int(out[r, cc]) != expected:
                    return False
        if np.array_equal(out, grid):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(CONTROL_COUNT_CHOICES)
    control_cols = _choose_control_columns(width, count)
    if control_cols is None:
        return None

    row_pick = _pick_snaking_rows(height, count)
    if row_pick is None:
        return None
    connector_rows, start_row, goal_row = row_pick

    grid = np.full((height, width), BLUE, dtype=int)
    open_cells = _carve_main_structure(grid, control_cols, connector_rows)

    blocked_rows = set(connector_rows) | {start_row, goal_row}
    loop_rows: list[int] = []
    loop_count = 0
    segment_indices = list(range(len(control_cols) - 1))
    random.shuffle(segment_indices)
    desired_loops = 1 if count == 2 else random.choice((1, 2))
    for idx in segment_indices:
        if loop_count >= desired_loops:
            break
        side_row = _add_longer_loop(
            grid,
            control_cols[idx],
            control_cols[idx + 1],
            connector_rows[idx],
            blocked_rows,
        )
        if side_row is not None:
            loop_rows.append(side_row)
            blocked_rows.add(side_row)
            for c in range(control_cols[idx], control_cols[idx + 1] + 1):
                open_cells.add((side_row, c))
            loop_count += 1
    if loop_count == 0:
        return None

    min_open = int(np.ceil(OPEN_FRACTION_RANGE[0] * height * width))
    max_open = int(np.floor(OPEN_FRACTION_RANGE[1] * height * width))
    target_open = random.randint(max(min_open, len(open_cells)), max(max_open, len(open_cells)))
    growth_attempts = 0
    while len(open_cells) < target_open and growth_attempts < BRANCH_GROWTH_ATTEMPTS:
        growth_attempts += 1
        anchor = random.choice(tuple(open_cells))
        before = np.count_nonzero(grid != BLUE)
        grown = _grow_tree_branch(grid, anchor, random.randint(2, 5))
        if grown:
            after_cells = {(int(r), int(c)) for r, c in np.argwhere(grid != BLUE)}
            open_cells = after_cells
        if np.count_nonzero(grid != BLUE) == before and growth_attempts > BRANCH_GROWTH_ATTEMPTS // 2:
            break

    start = (start_row, control_cols[0])
    goal = (goal_row, control_cols[-1])
    grid[start] = RED
    grid[goal] = RED
    colors = random.sample(list(CONTROL_PALETTE), count)
    for c in range(width):
        if c not in control_cols:
            grid[0, c] = BLUE
    for c, color in zip(control_cols, colors):
        grid[0, c] = color

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
