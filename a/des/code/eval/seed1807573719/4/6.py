import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

COLOR_TO_STEP = {
    RED: 1,
    GREEN: 2,
    BLUE: 3,
    YELLOW: 4,
}
STEP_TO_COLOR = {step: color for color, step in COLOR_TO_STEP.items()}

GEN_RETRIES = 500
PATH_RETRIES = 300
STEP_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(vals))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(1, w - 1)
        if np.all(grid[:, c] == AZURE)
    ]
    if not candidates:
        raise AssertionError("expected a full-height azure divider")
    if len(candidates) == 1:
        return int(candidates[0])

    for c in candidates:
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if np.all(np.isin(left, list(COLOR_TO_STEP))) and np.any(right == GREY):
            return int(c)
    return int(candidates[0])


def _trace_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(path_mask)}
    if not cells:
        raise AssertionError("expected a grey path")

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        neighbors = [nb for nb in _neighbors4(r, c) if nb in cells]
        if len(neighbors) > 2:
            raise AssertionError("path must be branchless")
        neighbor_map[(r, c)] = neighbors

    if len(cells) == 1:
        start = next(iter(cells))
    else:
        endpoints = [cell for cell, neighbors in neighbor_map.items() if len(neighbors) == 1]
        if len(endpoints) != 2:
            raise AssertionError("path must have exactly two endpoints")
        left_edge_endpoints = [cell for cell in endpoints if cell[1] == 0]
        if left_edge_endpoints:
            start = min(left_edge_endpoints, key=lambda rc: (rc[0], rc[1]))
        else:
            start = min(endpoints, key=lambda rc: (rc[0], rc[1]))

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        if cur in seen:
            raise AssertionError("path traversal looped")
        ordered.append(cur)
        seen.add(cur)
        next_cells = [nb for nb in neighbor_map[cur] if nb != prev]
        if not next_cells:
            break
        if len(next_cells) != 1:
            raise AssertionError("path traversal became ambiguous")
        prev, cur = cur, next_cells[0]

    if seen != cells:
        raise AssertionError("path is disconnected")
    return ordered


def _turn_count(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(d0 != d1 for d0, d1 in zip(dirs, dirs[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode left-side color stripes into step lengths, then color the right-side grey path in order.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape

    divider = _find_divider_col(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    if left.size == 0 or right.size == 0:
        raise AssertionError("expected non-empty instruction and path regions")

    instructions: list[int] = []
    for r in range(h):
        row = left[r]
        vals, counts = np.unique(row, return_counts=True)
        ranked = sorted(
            ((int(cnt), int(val)) for val, cnt in zip(vals, counts) if int(val) in COLOR_TO_STEP),
            reverse=True,
        )
        if not ranked:
            raise AssertionError("each left-row stripe must use an instruction color")
        instructions.append(ranked[0][1])

    path = _trace_path(right == GREY)

    out = g.copy()
    path_idx = 0
    for color in instructions:
        step_count = COLOR_TO_STEP[int(color)]
        for _ in range(step_count):
            if path_idx >= len(path):
                return out
            rr, cc = path[path_idx]
            out[rr, divider + 1 + cc] = int(color)
            path_idx += 1

    return out


def _append_line(path: list[tuple[int, int]], target: tuple[int, int]) -> None:
    r, c = path[-1]
    tr, tc = target
    if r != tr and c != tc:
        raise ValueError("segments must be axis-aligned")
    dr = 0 if r == tr else (1 if tr > r else -1)
    dc = 0 if c == tc else (1 if tc > c else -1)
    while (r, c) != (tr, tc):
        r += dr
        c += dc
        path.append((r, c))


def _choose_target_col(current_col: int, width: int, max_span: int) -> int:
    span = min(width - 1, max_span)
    candidates = set()
    for dist in range(1, span + 1):
        if current_col - dist >= 0:
            candidates.add(current_col - dist)
        if current_col + dist < width:
            candidates.add(current_col + dist)
    if not candidates:
        candidates = {c for c in range(width) if c != current_col}
    return int(random.choice(sorted(candidates)))


def _generate_path(height: int, width: int) -> list[tuple[int, int]]:
    for _ in range(PATH_RETRIES):
        first_horizontal = random.choice([True, False])
        horiz_count = random.randint(1, 3)

        if first_horizontal:
            if horiz_count == 1:
                rows = [0]
            else:
                rows = [0] + sorted(random.sample(range(1, height), horiz_count - 1))
        else:
            rows = sorted(random.sample(range(1, height), horiz_count))

        end_with_vertical = random.choice([True, False])
        if first_horizontal and horiz_count == 1:
            end_with_vertical = True
        if rows[-1] == height - 1:
            if first_horizontal and horiz_count == 1:
                continue
            end_with_vertical = False

        path = [(0, 0)]
        current_row, current_col = 0, 0

        for row in rows:
            if current_row < row:
                _append_line(path, (row, current_col))
                current_row = row

            target_col = _choose_target_col(current_col, width, max_span=height)
            _append_line(path, (current_row, target_col))
            current_col = target_col

        if end_with_vertical:
            end_row = random.randint(rows[-1] + 1, height - 1)
            _append_line(path, (end_row, current_col))

        if len(set(path)) != len(path):
            continue
        if not (height + 2 <= len(path) <= 4 * height - 2):
            continue
        if _turn_count(path) < 1:
            continue
        return path

    raise RuntimeError("failed to generate a valid path")


def _sample_steps(height: int, path_len: int, should_exhaust: bool) -> list[int]:
    min_distinct = 3 if height >= 4 and random.random() < 0.6 else 2

    for _ in range(STEP_RETRIES):
        steps = [random.randint(1, 4) for _ in range(height)]
        if len(set(steps)) < min_distinct:
            continue
        total = sum(steps)
        prefix = set(int(v) for v in np.cumsum(steps))
        if should_exhaust:
            if total <= path_len:
                continue
            if path_len in prefix:
                continue
        else:
            if total >= path_len:
                continue
        return steps

    raise RuntimeError("failed to sample instruction steps")


def _build_grid(
    height: int,
    left_width: int,
    right_width: int,
    path: list[tuple[int, int]],
    steps: list[int],
) -> np.ndarray:
    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = np.array([STEP_TO_COLOR[step] for step in steps], dtype=int)[:, None]
    grid[:, left_width] = AZURE
    for r, c in path:
        grid[r, left_width + 1 + c] = GREY
    return grid


def _validate_instance(grid: np.ndarray, should_exhaust: bool) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert get_bg(grid) == BG

    divider = _find_divider_col(grid)
    left = grid[:, :divider]
    right = grid[:, divider + 1 :]
    assert left.shape[1] >= 1
    assert right.shape[1] >= 2
    assert np.all(grid[:, divider] == AZURE)

    row_colors = []
    for r in range(grid.shape[0]):
        assert np.all(left[r] == left[r, 0])
        assert int(left[r, 0]) in COLOR_TO_STEP
        row_colors.append(int(left[r, 0]))

    path = _trace_path(right == GREY)
    assert path[0] == (0, 0)
    assert _turn_count(path) >= 1

    steps = [COLOR_TO_STEP[color] for color in row_colors]
    total_steps = sum(steps)
    if should_exhaust:
        assert total_steps > len(path)
        assert len(path) not in set(int(v) for v in np.cumsum(steps))
    else:
        assert total_steps < len(path)

    out = main(grid)
    assert np.array_equal(out[:, : divider + 1], grid[:, : divider + 1])
    transformed_right = out[:, divider + 1 :]

    if should_exhaust:
        assert not np.any(transformed_right == GREY)
    else:
        assert np.any(transformed_right == GREY)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(8, 14)
        left_width = random.randint(2, 5)
        right_width = random.randint(12, 18)
        should_exhaust = random.choice([True, False])

        try:
            path = _generate_path(height, right_width)
            steps = _sample_steps(height, len(path), should_exhaust)
        except RuntimeError:
            continue

        grid = _build_grid(height, left_width, right_width, path, steps)

        try:
            _validate_instance(grid, should_exhaust)
        except AssertionError:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
