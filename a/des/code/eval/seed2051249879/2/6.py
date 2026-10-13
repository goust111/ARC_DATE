from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

COLOR_TO_STEP = {
    RED: 2,
    BLUE: 3,
    GREEN: 1,
    YELLOW: 4,
}
STEP_TO_COLOR = {step: color for color, step in COLOR_TO_STEP.items()}
STEP_VALUES = tuple(sorted(STEP_TO_COLOR))

GEN_RETRY_LIMIT = 400
PATH_RETRY_LIMIT = 250


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _read_instruction_sequence(grid: np.ndarray) -> list[int]:
    coords = np.argwhere(np.isin(grid, tuple(COLOR_TO_STEP)))
    assert coords.size > 0
    left_col = int(coords[:, 1].min())
    seq = [
        (int(r), int(grid[int(r), int(c)]))
        for r, c in coords
        if int(c) == left_col
    ]
    seq.sort()
    assert 1 <= len(seq) <= 6
    return [color for _row, color in seq]


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    path_mask = (g == GREY) | (g == MAROON)
    coords = [(int(r), int(c)) for r, c in np.argwhere(path_mask)]
    assert coords

    start_cells = np.argwhere(g == MAROON)
    assert start_cells.shape[0] == 1
    start = tuple(map(int, start_cells[0]))
    assert path_mask[start]

    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in coords:
        r, c = cell
        linked = [nb for nb in _neighbors4(r, c, h, w) if path_mask[nb]]
        nbrs[cell] = linked
        if len(linked) == 1:
            endpoints.append(cell)
        else:
            assert len(linked) == 2

    assert len(endpoints) == 2
    assert start in endpoints

    path = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        options = [nb for nb in nbrs[cur] if nb != prev]
        if not options:
            break
        assert len(options) == 1
        nxt = options[0]
        assert nxt not in seen
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(coords)
    return path


def _apply_sequence(
    grid: np.ndarray, path: list[tuple[int, int]], instructions: list[int]
) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    index = 0
    for color in instructions:
        step = COLOR_TO_STEP[int(color)]
        if index + step >= len(path):
            continue
        index += step
        r, c = path[index]
        if out[r, c] != MAROON:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Execute the left instruction column along the unique grey path."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    instructions = _read_instruction_sequence(g)
    path = _trace_path(g)
    return _apply_sequence(g, path, instructions)


def _simulate_steps(steps: list[int], capacity: int) -> list[bool]:
    pos = 0
    executed: list[bool] = []
    for step in steps:
        if pos + step <= capacity:
            pos += step
            executed.append(True)
        else:
            executed.append(False)
    return executed


def _sample_mode() -> tuple[list[int], int]:
    modes = (_sample_fit_mode, _sample_skip_stop_mode, _sample_skip_continue_mode)
    for _ in range(200):
        steps, capacity = random.choice(modes)()
        if steps:
            return steps, capacity
    raise RuntimeError("failed to sample instruction mode")


def _sample_fit_mode() -> tuple[list[int], int]:
    for _ in range(200):
        count = random.randint(2, 6)
        steps = [random.choice(STEP_VALUES) for _ in range(count)]
        total = sum(steps)
        if total > 18:
            continue
        capacity = random.randint(max(6, total), min(18, total + 4))
        if all(_simulate_steps(steps, capacity)):
            return steps, capacity
    return [], 0


def _sample_skip_stop_mode() -> tuple[list[int], int]:
    for _ in range(200):
        prefix_len = random.randint(1, 4)
        prefix = [random.choice(STEP_VALUES) for _ in range(prefix_len)]
        remaining = random.randint(0, 3)
        capacity = sum(prefix) + remaining
        if not (6 <= capacity <= 18):
            continue

        too_large = [step for step in STEP_VALUES if step > remaining]
        skip_step = random.choice(too_large)

        tail_len = random.randint(0, max(0, 6 - prefix_len - 1))
        tail = [random.choice(too_large) for _ in range(tail_len)]
        steps = prefix + [skip_step] + tail
        executed = _simulate_steps(steps, capacity)
        if any(executed) and not all(executed) and not any(executed[prefix_len + 1 :]):
            return steps, capacity
    return [], 0


def _sample_skip_continue_mode() -> tuple[list[int], int]:
    for _ in range(200):
        prefix_len = random.randint(1, 3)
        prefix = [random.choice(STEP_VALUES) for _ in range(prefix_len)]
        remaining = random.randint(1, 3)
        capacity = sum(prefix) + remaining
        if not (6 <= capacity <= 18):
            continue

        skip_step = random.choice([step for step in STEP_VALUES if step > remaining])
        first_tail = random.randint(1, remaining)
        extra_len = random.randint(0, max(0, 6 - prefix_len - 2))
        extra = [random.choice(STEP_VALUES) for _ in range(extra_len)]
        steps = prefix + [skip_step, first_tail] + extra
        executed = _simulate_steps(steps, capacity)
        if any(executed) and not all(executed) and any(executed[prefix_len + 1 :]):
            return steps, capacity
    return [], 0


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        d1 = (
            path[i - 1][0] - path[i - 2][0],
            path[i - 1][1] - path[i - 2][1],
        )
        d2 = (
            path[i][0] - path[i - 1][0],
            path[i][1] - path[i - 1][1],
        )
        turns += d1 != d2
    return turns


def _can_step(
    cell: tuple[int, int],
    prev: tuple[int, int],
    visited: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in visited:
        return False
    visited_neighbors = sum((nb in visited) for nb in _neighbors4(r, c, h, w))
    return visited_neighbors == 1 and prev in _neighbors4(r, c, h, w)


def _future_options(
    cell: tuple[int, int], visited: set[tuple[int, int]], h: int, w: int
) -> int:
    return sum(
        _can_step(nb, cell, visited, h, w)
        for nb in _neighbors4(cell[0], cell[1], h, w)
    )


def _generate_path(height: int, width: int, length: int) -> list[tuple[int, int]] | None:
    min_turns = 1 if length < 10 else 2
    for _ in range(PATH_RETRY_LIMIT):
        start = (random.randrange(height), random.randrange(width))
        path = [start]
        visited = {start}

        while len(path) < length:
            cur = path[-1]
            prev = path[-2] if len(path) >= 2 else None
            prev_dir = None
            if prev is not None:
                prev_dir = (cur[0] - prev[0], cur[1] - prev[1])

            choices: list[tuple[tuple[int, int], float]] = []
            for nb in _neighbors4(cur[0], cur[1], height, width):
                if not _can_step(nb, cur, visited, height, width):
                    continue
                next_visited = visited | {nb}
                future = _future_options(nb, next_visited, height, width)
                if len(path) + 1 < length and future == 0:
                    continue
                new_dir = (nb[0] - cur[0], nb[1] - cur[1])
                turn_bonus = 2 if prev_dir is not None and new_dir != prev_dir else 0
                edge_bonus = 1 if 0 < nb[0] < height - 1 and 0 < nb[1] < width - 1 else 0
                weight = 1.0 + 3.0 * future + turn_bonus + edge_bonus
                choices.append((nb, weight))

            if not choices:
                break

            total = sum(weight for _cell, weight in choices)
            pick = random.random() * total
            upto = 0.0
            nxt = choices[-1][0]
            for cell, weight in choices:
                upto += weight
                if pick <= upto:
                    nxt = cell
                    break

            path.append(nxt)
            visited.add(nxt)

        if len(path) == length and _turn_count(path) >= min_turns:
            return path
    return None


def _build_grid(
    height: int,
    width: int,
    instruction_col: int,
    instruction_rows: list[int],
    instructions: list[int],
    path_col0: int,
    path: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row, color in zip(instruction_rows, instructions):
        grid[int(row), int(instruction_col)] = int(color)
    for idx, (r, c) in enumerate(path):
        grid[int(r), int(path_col0 + c)] = MAROON if idx == 0 else GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    steps, capacity = _sample_mode()
    instructions = [STEP_TO_COLOR[step] for step in steps]
    path_len = capacity + 1

    height = random.randint(max(8, len(instructions) + 2), 16)
    left_block = random.randint(3, 5)
    instruction_col = random.randint(1, left_block - 2)
    gap = random.randint(2, 4)

    min_path_width = max(6, (3 * path_len + height - 1) // height)
    path_width = random.randint(min_path_width, min_path_width + 8)
    path = _generate_path(height, path_width, path_len)
    if path is None:
        return None

    if random.random() < 0.5:
        path = list(reversed(path))

    right_margin = random.randint(0, 2)
    path_col0 = left_block + gap
    width = path_col0 + path_width + right_margin

    instruction_rows = sorted(random.sample(range(height), len(instructions)))
    grid = _build_grid(
        height,
        width,
        instruction_col,
        instruction_rows,
        instructions,
        path_col0,
        path,
    )

    expected = _apply_sequence(grid, _trace_path(grid), instructions)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
