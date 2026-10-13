from __future__ import annotations

from collections import Counter

import numpy as np


Coord = tuple[int, int]
DIRS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (12, 25)
GRID_W_RANGE = (12, 25)
MIN_SEGMENTS = 2
MAX_SEGMENTS = 6


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _add(a: Coord, b: Coord) -> Coord:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _turn_left(direction: Coord) -> Coord:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: Coord) -> Coord:
    dr, dc = direction
    return dc, -dr


def _is_boundary(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = map(int, cell)
    h, w = map(int, shape)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_start_bar(grid: np.ndarray, bg: int) -> tuple[int, Coord, Coord]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, Coord, Coord]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 2:
            continue
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue
        a_edge = _is_boundary((h, w), a)
        b_edge = _is_boundary((h, w), b)
        if a_edge == b_edge:
            continue
        boundary = a if a_edge else b
        inner = b if a_edge else a
        candidates.append((color, boundary, inner))
    assert len(candidates) == 1, "expected a unique 2-pixel start bar"
    return candidates[0]


def _trace_segments(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    start_color, boundary, inner = _find_start_bar(g, bg)
    blocker_mask = (g != bg) & (g != start_color)

    direction = (int(inner[0] - boundary[0]), int(inner[1] - boundary[1]))
    current = boundary
    visited = {boundary}
    lengths: list[int] = []
    row_colors: list[int] = []
    segment_len = 1

    while True:
        forward = _add(current, direction)
        if not in_bounds(g.shape, forward[0], forward[1]):
            lengths.append(segment_len)
            row_colors.append(start_color)
            break

        if blocker_mask[forward]:
            options: list[tuple[Coord, int, Coord]] = []
            for new_direction, color in ((_turn_left(direction), 1), (_turn_right(direction), 2)):
                nxt = _add(current, new_direction)
                if in_bounds(g.shape, nxt[0], nxt[1]) and not blocker_mask[nxt]:
                    options.append((new_direction, color, nxt))
            assert len(options) == 1, "turn must have exactly one open perpendicular branch"
            lengths.append(segment_len)
            direction, color, current = options[0]
            row_colors.append(color)
            assert current not in visited, "corridor path should not revisit cells"
            visited.add(current)
            segment_len = 1
            continue

        current = forward
        assert current not in visited, "corridor path should not loop"
        visited.add(current)
        segment_len += 1

    return bg, lengths, row_colors


def main(input_grid: np.ndarray) -> np.ndarray:
    """Serialize the walked corridor into one horizontal bar per straight segment."""
    bg, lengths, row_colors = _trace_segments(np.asarray(input_grid, dtype=int))
    out = np.full((len(lengths), max(lengths)), int(bg), dtype=int)
    for row, (length, color) in enumerate(zip(lengths, row_colors)):
        out[row, :length] = int(color)
    return out


def _can_append(cell: Coord, occupied: set[Coord], shape: tuple[int, int], prev: Coord) -> bool:
    if not in_bounds(shape, cell[0], cell[1]) or cell in occupied:
        return False
    for dr, dc in DIRS:
        nbr = (cell[0] + dr, cell[1] + dc)
        if nbr in occupied and nbr != prev:
            return False
    return True


def _finish_path(current: Coord, direction: Coord, occupied: set[Coord], shape: tuple[int, int]) -> list[Coord] | None:
    added: list[Coord] = []
    local_occupied = set(occupied)
    prev = current
    cell = _add(current, direction)
    while in_bounds(shape, cell[0], cell[1]):
        if not _can_append(cell, local_occupied, shape, prev):
            return None
        added.append(cell)
        local_occupied.add(cell)
        prev = cell
        cell = _add(cell, direction)
    return added


def _turn_candidates(
    current: Coord,
    direction: Coord,
    occupied: set[Coord],
    shape: tuple[int, int],
) -> list[tuple[list[Coord], Coord, Coord]]:
    candidates: list[tuple[list[Coord], Coord, Coord]] = []
    end = current
    straight_cells: list[Coord] = []
    local_occupied = set(occupied)

    while True:
        forward = _add(end, direction)
        if in_bounds(shape, forward[0], forward[1]) and forward not in local_occupied:
            left_direction = _turn_left(direction)
            right_direction = _turn_right(direction)
            for new_direction, other_direction in (
                (left_direction, right_direction),
                (right_direction, left_direction),
            ):
                turn_cell = _add(end, new_direction)
                other_cell = _add(end, other_direction)
                if not _can_append(turn_cell, local_occupied, shape, end):
                    continue
                if other_cell in local_occupied:
                    continue
                candidates.append((list(straight_cells), turn_cell, new_direction))

        if not _can_append(forward, local_occupied, shape, end):
            break
        straight_cells.append(forward)
        local_occupied.add(forward)
        end = forward

    return candidates


def _sample_start(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[Coord, Coord]:
    h, w = map(int, shape)
    side = int(rng.integers(4))
    if side == 0:
        return (0, int(rng.integers(1, w - 1))), (1, 0)
    if side == 1:
        return (h - 1, int(rng.integers(1, w - 1))), (-1, 0)
    if side == 2:
        return (int(rng.integers(1, h - 1)), 0), (0, 1)
    return (int(rng.integers(1, h - 1)), w - 1), (0, -1)


def _grow_path(
    rng: np.random.Generator,
    current: Coord,
    direction: Coord,
    occupied: set[Coord],
    shape: tuple[int, int],
    segments_used: int,
    max_segments: int,
) -> list[Coord] | None:
    if segments_used >= MIN_SEGMENTS:
        finish_cells = _finish_path(current, direction, occupied, shape)
        if finish_cells is not None and (segments_used == max_segments or rng.random() < 0.4):
            return finish_cells
    if segments_used >= max_segments:
        return None

    candidates = _turn_candidates(current, direction, occupied, shape)
    if not candidates:
        return None
    order = rng.permutation(len(candidates))
    for idx in order:
        straight_cells, turn_cell, new_direction = candidates[int(idx)]
        added = list(straight_cells) + [turn_cell]
        new_occupied = set(occupied)
        new_occupied.update(added)
        tail = _grow_path(
            rng,
            turn_cell,
            new_direction,
            new_occupied,
            shape,
            segments_used + 1,
            max_segments,
        )
        if tail is not None:
            return added + tail
    return None


def _build_blockers(path: list[Coord], shape: tuple[int, int]) -> set[Coord]:
    path_set = set(path)
    blockers: set[Coord] = set()
    for index, cell in enumerate(path):
        used: set[Coord] = set()
        if index > 0:
            prev = path[index - 1]
            used.add((prev[0] - cell[0], prev[1] - cell[1]))
        if index + 1 < len(path):
            nxt = path[index + 1]
            used.add((nxt[0] - cell[0], nxt[1] - cell[1]))
        for direction in DIRS:
            if direction in used:
                continue
            wall = _add(cell, direction)
            if in_bounds(shape, wall[0], wall[1]) and wall not in path_set:
                blockers.add(wall)
    return blockers


def _path_segment_lengths(path: list[Coord]) -> tuple[list[int], list[int]]:
    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    lengths: list[int] = []
    colors: list[int] = []
    start = 0
    for i in range(1, len(directions)):
        if directions[i] == directions[i - 1]:
            continue
        lengths.append(i - start + 1)
        if directions[i] == _turn_left(directions[i - 1]):
            colors.append(1)
        else:
            colors.append(2)
        start = i + 1
    lengths.append(len(path) - start)
    colors.append(-1)
    return lengths, colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1)),
        int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1)),
    )
    start = _sample_start(rng, shape)
    boundary, direction = start
    inner = _add(boundary, direction)
    path = [boundary, inner]
    occupied = {boundary, inner}
    max_segments = int(rng.integers(MIN_SEGMENTS, MAX_SEGMENTS + 1))
    tail = _grow_path(rng, inner, direction, occupied, shape, 1, max_segments)
    if tail is None:
        return None
    path.extend(tail)

    segment_lengths, turn_colors = _path_segment_lengths(path)
    if len(segment_lengths) < MIN_SEGMENTS or max(segment_lengths) > 30:
        return None

    blockers = _build_blockers(path, shape)
    if not blockers:
        return None

    bg = 0
    palette = [3, 4, 5, 6, 7, 8, 9]
    start_color = int(palette[int(rng.integers(len(palette)))])
    blocker_choices = [color for color in range(1, 10) if color != start_color]
    blocker_color = int(blocker_choices[int(rng.integers(len(blocker_choices)))])

    grid = np.full(shape, bg, dtype=int)
    for r, c in blockers:
        grid[r, c] = blocker_color
    grid[path[0]] = start_color
    grid[path[1]] = start_color

    if get_bg(grid) != bg:
        return None

    out = main(grid)
    if out.ndim != 2 or out.shape[0] != len(segment_lengths) or out.shape[1] != max(segment_lengths):
        return None
    if not np.array_equal(out[-1, : segment_lengths[-1]], np.full(segment_lengths[-1], start_color, dtype=int)):
        return None
    if turn_colors[:-1]:
        expected_turns = np.array(turn_colors[:-1], dtype=int)
        actual_turns = np.array([int(out[i, 0]) for i in range(len(turn_colors) - 1)], dtype=int)
        if not np.array_equal(expected_turns, actual_turns):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-log instance")


def _self_check() -> None:
    for _ in range(8):
        grid = generate_input()
        out = main(grid)
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert isinstance(out, np.ndarray) and out.ndim == 2
        assert grid.shape[0] <= 30 and grid.shape[1] <= 30
        assert out.shape[0] <= 30 and out.shape[1] <= 30
        bg, lengths, row_colors = _trace_segments(grid)
        expected = np.full((len(lengths), max(lengths)), bg, dtype=int)
        for row, (length, color) in enumerate(zip(lengths, row_colors)):
            expected[row, :length] = color
        assert np.array_equal(out, expected)


if __name__ == "__main__":
    _self_check()
