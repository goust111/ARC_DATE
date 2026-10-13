from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GRID_SIZE = 30
GEN_MAX_ATTEMPTS = 2000

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIR_BY_COLOR: dict[int, tuple[int, int]] = {
    YELLOW: (-1, 0),
    RED: (0, 1),
    BLUE: (1, 0),
    GREEN: (0, -1),
}
COLOR_BY_DIR = {delta: color for color, delta in DIR_BY_COLOR.items()}
DIRECTIONS = list(COLOR_BY_DIR)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _add(pos: tuple[int, int], delta: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return pos[0] + delta[0] * scale, pos[1] + delta[1] * scale


def _opposite(delta: tuple[int, int]) -> tuple[int, int]:
    return -delta[0], -delta[1]


def _cells_along_segment(
    start: tuple[int, int],
    delta: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]]:
    return [_add(start, delta, step) for step in range(1, steps + 1)]


def _ray_to_edge(shape: tuple[int, int], start: tuple[int, int], delta: tuple[int, int]) -> list[tuple[int, int]]:
    ray: list[tuple[int, int]] = []
    r, c = _add(start, delta)
    while _in_bounds(shape, r, c):
        ray.append((r, c))
        r += delta[0]
        c += delta[1]
    return ray


def _extract_motion_positions(grid: np.ndarray) -> dict[tuple[int, int], int]:
    positions: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(np.isin(grid, list(DIR_BY_COLOR))):
        positions[(int(r), int(c))] = int(grid[r, c])
    return positions


def _find_trigger(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GRAY)
    if len(coords) != 1:
        raise ValueError("expected exactly one gray trigger")
    return int(coords[0, 0]), int(coords[0, 1])


def _find_start_pixel(
    trigger: tuple[int, int],
    positions: dict[tuple[int, int], int],
) -> tuple[int, int]:
    tr, tc = trigger
    starts = [pos for pos in positions if abs(pos[0] - tr) + abs(pos[1] - tc) == 1]
    if len(starts) != 1:
        raise ValueError("expected exactly one motion pixel adjacent to the trigger")
    return starts[0]


def _run_relay(
    shape: tuple[int, int],
    positions: dict[tuple[int, int], int],
    start: tuple[int, int],
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    active = start
    activated: list[tuple[int, int]] = []

    for _ in range(len(positions) + 1):
        activated.append(active)
        color = positions[active]
        dr, dc = DIR_BY_COLOR[color]
        r, c = active
        last_free = active
        hit: tuple[int, int] | None = None

        nr, nc = r + dr, c + dc
        while _in_bounds(shape, nr, nc):
            if (nr, nc) in positions:
                hit = (nr, nc)
                break
            last_free = (nr, nc)
            nr += dr
            nc += dc

        if last_free != active:
            del positions[active]
            positions[last_free] = color

        if hit is None:
            return positions, activated
        active = hit

    raise ValueError("relay did not terminate on a valid edge-reaching move")


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Start the unique trigger-adjacent motion pixel, relay movement on collisions,
    and stop once some activated pixel slides unobstructed to the border.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _get_bg(grid)
    trigger = _find_trigger(grid)
    positions = _extract_motion_positions(grid)
    start = _find_start_pixel(trigger, positions)
    final_positions, activated = _run_relay(grid.shape, positions, start)

    out = grid.copy()
    for r, c in activated:
        out[r, c] = bg
    for (r, c), color in final_positions.items():
        out[r, c] = int(color)
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_MAX_ATTEMPTS):
        height = random.randint(8, 16)
        width = random.randint(8, 16)
        shape = (height, width)
        chain_len = random.randint(1, 5)

        start = (random.randint(1, height - 2), random.randint(1, width - 2))
        chain_positions = [start]
        chain_dirs: list[tuple[int, int]] = []
        used = {start}

        ok = True
        current = start
        for step_idx in range(chain_len - 1):
            direction_choices = DIRECTIONS[:]
            random.shuffle(direction_choices)
            if chain_dirs:
                direction_choices = [d for d in direction_choices if d != _opposite(chain_dirs[-1])]

            candidates: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
            for delta in direction_choices:
                max_steps = 0
                probe = _add(current, delta)
                while _in_bounds(shape, probe[0], probe[1]):
                    max_steps += 1
                    probe = _add(probe, delta)
                step_values = list(range(2, max_steps + 1))
                random.shuffle(step_values)
                for steps in step_values:
                    nxt = _add(current, delta, steps)
                    if step_idx < chain_len - 2 and (
                        nxt[0] in (0, height - 1) or nxt[1] in (0, width - 1)
                    ):
                        continue
                    segment = _cells_along_segment(current, delta, steps)
                    if any(cell in used for cell in segment):
                        continue
                    candidates.append((delta, nxt, segment))

            if not candidates:
                ok = False
                break

            delta, nxt, segment = random.choice(candidates)
            chain_dirs.append(delta)
            chain_positions.append(nxt)
            used.update(segment)
            current = nxt

        if not ok:
            continue

        last_direction_choices = DIRECTIONS[:]
        random.shuffle(last_direction_choices)
        if chain_dirs:
            last_direction_choices = [d for d in last_direction_choices if d != _opposite(chain_dirs[-1])]

        tail_candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for delta in last_direction_choices:
            ray = _ray_to_edge(shape, current, delta)
            if not ray:
                continue
            if any(cell in used for cell in ray):
                continue
            tail_candidates.append((delta, ray))

        if not tail_candidates:
            continue

        last_delta, last_ray = random.choice(tail_candidates)
        chain_dirs.append(last_delta)

        trigger_forbidden = used | set(last_ray)
        trigger_candidates = []
        for nbr in _neighbors4(start[0], start[1]):
            if not _in_bounds(shape, nbr[0], nbr[1]):
                continue
            if nbr in trigger_forbidden:
                continue
            adjacent_motion = sum(
                1 for pos in chain_positions if abs(pos[0] - nbr[0]) + abs(pos[1] - nbr[1]) == 1
            )
            if adjacent_motion != 1:
                continue
            trigger_candidates.append(nbr)

        if not trigger_candidates:
            continue

        trigger = random.choice(trigger_candidates)
        forbidden = trigger_forbidden | {trigger}

        grid = np.full(shape, BG, dtype=int)
        grid[trigger] = GRAY

        for pos, delta in zip(chain_positions, chain_dirs):
            grid[pos] = COLOR_BY_DIR[delta]

        free_cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in forbidden and grid[r, c] == BG
        ]
        random.shuffle(free_cells)
        extra_count = random.randint(0, min(4, len(free_cells)))
        for pos in free_cells[:extra_count]:
            if abs(pos[0] - trigger[0]) + abs(pos[1] - trigger[1]) == 1:
                continue
            grid[pos] = random.choice(list(DIR_BY_COLOR))

        expected = grid.copy()
        for pos in chain_positions:
            expected[pos] = BG

        for idx, (pos, delta) in enumerate(zip(chain_positions, chain_dirs)):
            if idx < len(chain_positions) - 1:
                final_pos = _add(chain_positions[idx + 1], _opposite(delta))
            else:
                final_pos = last_ray[-1]
            expected[final_pos] = COLOR_BY_DIR[delta]

        if expected.ndim != 2:
            continue
        if expected.shape[0] > MAX_GRID_SIZE or expected.shape[1] > MAX_GRID_SIZE:
            continue
        return grid, expected

    raise RuntimeError("failed to generate a valid relay case")


def generate_input() -> np.ndarray:
    return _build_case()[0]


if __name__ == "__main__":
    for _ in range(12):
        generated_input, expected_output = _build_case()
        actual_output = main(generated_input)
        assert isinstance(generated_input, np.ndarray) and generated_input.ndim == 2
        assert isinstance(actual_output, np.ndarray) and actual_output.ndim == 2
        assert actual_output.shape == expected_output.shape
        assert actual_output.shape[0] <= MAX_GRID_SIZE and actual_output.shape[1] <= MAX_GRID_SIZE
        assert np.array_equal(actual_output, expected_output)
