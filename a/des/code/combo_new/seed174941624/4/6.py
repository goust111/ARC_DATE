from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GRID_SIZE_RANGE = (12, 30)
PATH_COUNT_RANGE = (1, 4)
TURN_COUNT_RANGE = (1, 4)
SEGMENT_LENGTH_RANGE = (2, 5)
GENERATION_RETRIES = 400
PATH_ATTEMPTS = 250
PLACEMENT_ATTEMPTS = 250

Coord = tuple[int, int]


def _neighbors4(shape: tuple[int, int], cell: Coord) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _collect_component(mask: np.ndarray, start: Coord, seen: np.ndarray) -> list[Coord]:
    q: deque[Coord] = deque([start])
    seen[start] = True
    component: list[Coord] = []
    while q:
        cell = q.popleft()
        component.append(cell)
        for nbr in _neighbors4(mask.shape, cell):
            if seen[nbr] or not bool(mask[nbr]):
                continue
            seen[nbr] = True
            q.append(nbr)
    return component


def _turn_marker(prev_cell: Coord, cell: Coord, next_cell: Coord) -> int | None:
    dr1 = cell[0] - prev_cell[0]
    dc1 = cell[1] - prev_cell[1]
    dr2 = next_cell[0] - cell[0]
    dc2 = next_cell[1] - cell[1]
    if (dr1, dc1) == (dr2, dc2):
        return None

    cross = dr1 * dc2 - dc1 * dr2
    assert cross != 0, "path steps must be cardinal and cannot reverse"
    return BLUE if cross > 0 else ORANGE


def _trace_path(grid: np.ndarray, component: list[Coord]) -> list[Coord]:
    component_set = set(component)
    starts: list[Coord] = []
    ends: list[Coord] = []

    for cell in component:
        color = int(grid[cell])
        degree = sum(1 for nbr in _neighbors4(grid.shape, cell) if nbr in component_set)
        if color == GREEN:
            assert degree == 1
            starts.append(cell)
        elif color == RED:
            assert degree == 1
            ends.append(cell)
        else:
            assert color == GREY
            assert degree == 2

    assert len(starts) == 1
    assert len(ends) == 1

    start = starts[0]
    end = ends[0]
    ordered = [start]
    prev: Coord | None = None
    cur = start

    while cur != end:
        next_cells = [nbr for nbr in _neighbors4(grid.shape, cur) if nbr in component_set and nbr != prev]
        assert len(next_cells) == 1
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
        assert len(ordered) <= len(component)

    assert len(ordered) == len(component)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each turn along every green-to-red path by whether it is a left turn or a right turn.

    Input structure:
    - The background is black.
    - Each foreground component is a single-pixel-wide 4-connected path whose interior cells are grey.
    - Every path has exactly one green endpoint, exactly one red endpoint, and zero or more 90-degree turns.

    Rule:
    - For each connected path, start at the green endpoint and walk cell by cell until the red endpoint.
    - Leave the green start, the red end, and every straight grey segment unchanged.
    - At each turning cell, compare the incoming step with the outgoing step.
    - Recolor a left turn blue and a right turn orange.

    Key insight:
    - The corner color depends on traversal direction from green to red, so identical geometric corners can map to opposite colors when the start and end are swapped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(int(x) for x in np.unique(g)).issubset({BG, GREY, GREEN, RED})

    out = g.copy()
    fg = g != BG
    seen = np.zeros(g.shape, dtype=bool)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not bool(fg[r, c]) or bool(seen[r, c]):
                continue
            component = _collect_component(fg, (r, c), seen)
            path = _trace_path(g, component)
            for idx in range(1, len(path) - 1):
                marker = _turn_marker(path[idx - 1], path[idx], path[idx + 1])
                if marker is not None:
                    out[path[idx]] = marker

    return out


def _normalize_path(path: list[Coord]) -> list[Coord]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    return [(r - min_r, c - min_c) for r, c in path]


def _path_dims(path: list[Coord]) -> tuple[int, int]:
    return max(r for r, _ in path) + 1, max(c for _, c in path) + 1


def _apply_turn(dir_idx: int, side: str) -> int:
    if side == "L":
        return (dir_idx - 1) % 4
    assert side == "R"
    return (dir_idx + 1) % 4


def _can_add_step(path: list[Coord], occupied: set[Coord], cell: Coord) -> bool:
    if cell in occupied:
        return False

    prev = path[-1]
    prev_prev = path[-2] if len(path) >= 2 else None
    for rr, cc in occupied:
        if max(abs(rr - cell[0]), abs(cc - cell[1])) > 1:
            continue
        if (rr, cc) == prev:
            continue
        if prev_prev is not None and (rr, cc) == prev_prev:
            continue
        return False
    return True


def _build_abstract_path(turn_sides: list[str]) -> list[Coord] | None:
    for _ in range(PATH_ATTEMPTS):
        dir_idx = random.randrange(4)
        segment_lengths = [
            random.randint(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1])
            for _ in range(len(turn_sides) + 1)
        ]
        path = [(0, 0)]
        occupied = {(0, 0)}
        ok = True

        for seg_idx, seg_len in enumerate(segment_lengths):
            dr, dc = DIRS[dir_idx]
            for _step in range(seg_len):
                cur_r, cur_c = path[-1]
                nxt = (cur_r + dr, cur_c + dc)
                if not _can_add_step(path, occupied, nxt):
                    ok = False
                    break
                path.append(nxt)
                occupied.add(nxt)
            if not ok:
                break
            if seg_idx < len(turn_sides):
                dir_idx = _apply_turn(dir_idx, turn_sides[seg_idx])

        if not ok:
            continue

        path = _normalize_path(path)
        path_h, path_w = _path_dims(path)
        if max(path_h, path_w) > 18:
            continue
        return path

    return None


def _placement_range(extra: int) -> tuple[int, int]:
    if extra >= 2:
        return 1, extra - 1
    return 0, extra


def _expanded_cells(cells: list[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _place_paths(paths: list[list[Coord]], shape: tuple[int, int]) -> list[list[Coord]] | None:
    h, w = shape
    order = list(range(len(paths)))
    random.shuffle(order)
    order.sort(key=lambda idx: _path_dims(paths[idx])[0] * _path_dims(paths[idx])[1], reverse=True)

    blocked: set[Coord] = set()
    placed: list[list[Coord] | None] = [None] * len(paths)

    for idx in order:
        path = paths[idx]
        path_h, path_w = _path_dims(path)
        if path_h > h or path_w > w:
            return None

        row_lo, row_hi = _placement_range(h - path_h)
        col_lo, col_hi = _placement_range(w - path_w)
        success = False

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            shifted = [(r + r0, c + c0) for r, c in path]
            expanded = _expanded_cells(shifted, shape)
            if expanded & blocked:
                continue
            blocked |= expanded
            placed[idx] = shifted
            success = True
            break

        if not success:
            return None

    return [path for path in placed if path is not None]


def _render_input(shape: tuple[int, int], paths: list[list[Coord]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in paths:
        for idx, cell in enumerate(path):
            if idx == 0:
                color = GREEN
            elif idx == len(path) - 1:
                color = RED
            else:
                color = GREY
            assert int(grid[cell]) == BG
            grid[cell] = color
    return grid


def _oracle_output(grid: np.ndarray, paths: list[list[Coord]]) -> tuple[np.ndarray, int, int]:
    out = grid.copy()
    left_turns = 0
    right_turns = 0
    for path in paths:
        for idx in range(1, len(path) - 1):
            marker = _turn_marker(path[idx - 1], path[idx], path[idx + 1])
            if marker is None:
                continue
            out[path[idx]] = marker
            if marker == BLUE:
                left_turns += 1
            else:
                right_turns += 1
    return out, left_turns, right_turns


def _choose_turn_counts(path_count: int) -> list[int]:
    counts = [random.randint(TURN_COUNT_RANGE[0], TURN_COUNT_RANGE[1]) for _ in range(path_count)]
    if sum(counts) < 2:
        counts[random.randrange(path_count)] += 1
    return counts


def _balanced_turn_sides(total_turns: int) -> list[str]:
    assert total_turns >= 2
    left_count = total_turns // 2
    if total_turns % 2 == 1 and random.random() < 0.5:
        left_count += 1
    right_count = total_turns - left_count
    assert left_count > 0 and right_count > 0
    sides = ["L"] * left_count + ["R"] * right_count
    random.shuffle(sides)
    return sides


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(PATH_COUNT_RANGE[0], PATH_COUNT_RANGE[1])
    turn_counts = _choose_turn_counts(path_count)
    all_turns = _balanced_turn_sides(sum(turn_counts))

    paths: list[list[Coord]] = []
    cursor = 0
    for turn_count in turn_counts:
        turn_sides = all_turns[cursor : cursor + turn_count]
        cursor += turn_count
        path = _build_abstract_path(turn_sides)
        if path is None:
            return None
        paths.append(path)

    dims = [_path_dims(path) for path in paths]
    min_h = max(max(path_h for path_h, _ in dims) + 2, GRID_SIZE_RANGE[0])
    min_w = max(max(path_w for _, path_w in dims) + 2, GRID_SIZE_RANGE[0])
    padded_area = sum((path_h + 2) * (path_w + 2) for path_h, path_w in dims)

    for _ in range(80):
        h = random.randint(min_h, GRID_SIZE_RANGE[1])
        min_fit_w = max(min_w, math.ceil(padded_area / h))
        if min_fit_w > GRID_SIZE_RANGE[1]:
            continue
        w = random.randint(min_fit_w, GRID_SIZE_RANGE[1])
        if h * w < padded_area:
            continue

        placed = _place_paths(paths, (h, w))
        if placed is None:
            continue

        inp = _render_input((h, w), placed)
        out, left_turns, right_turns = _oracle_output(inp, placed)
        if left_turns == 0 or right_turns == 0:
            continue
        if np.array_equal(inp, out):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
