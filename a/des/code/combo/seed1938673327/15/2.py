from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PATH_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


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


def _is_edge_cell(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert cells, "expected a non-empty path"

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in cells:
        r, c = cell
        nbrs = [nbr for nbr in _neighbors4(r, c, h, w) if nbr in cells]
        assert 1 <= len(nbrs) <= 2, "path must be 1-pixel wide and non-branching"
        adjacency[cell] = nbrs
        if len(nbrs) == 1:
            endpoints.append(cell)

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(_is_edge_cell(cell, h, w) for cell in endpoints), "endpoints must lie on the grid edge"

    start = min(endpoints, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    goal = endpoints[1] if endpoints[0] == start else endpoints[0]

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != goal:
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(nxts) == 1, "path traversal must be unambiguous"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(cells), "path must not self-intersect or contain loops"
    return path


def _turn_sequence(path: list[tuple[int, int]]) -> list[str]:
    turns: list[str] = []
    for i in range(1, len(path) - 1):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        nr, nc = path[i + 1]
        incoming = (cr - pr, cc - pc)
        outgoing = (nr - cr, nc - cc)
        if incoming == outgoing:
            continue
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        assert abs(cross) == 1, "turns must be 90-degree bends"
        turns.append("L" if cross > 0 else "R")
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """Encode the path's turn sequence as colored rows in a 2-column output table."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    fg_colors = sorted({int(v) for v in np.unique(g) if int(v) != BG})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    path = _ordered_path(g != BG)
    turns = _turn_sequence(path)

    out = np.zeros((len(turns), 2), dtype=int)
    for idx, turn in enumerate(turns):
        if turn == "L":
            out[idx, 0] = RED
        else:
            out[idx, 1] = BLUE
    return out


def _append_segment(points: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r0, c0 = points[-1]
    r1, c1 = end
    assert (r0 == r1) ^ (c0 == c1), "segments must be axis-aligned"

    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    r, c = r0, c0
    while (r, c) != (r1, c1):
        r += dr
        c += dc
        points.append((r, c))


def _choose_spaced_positions(count: int, first_hi: int, last_max: int) -> list[int]:
    assert count >= 1
    for _ in range(100):
        upper = min(first_hi, last_max - 2 * (count - 1))
        if upper < 1:
            break
        positions = [random.randint(1, upper)]
        ok = True
        for idx in range(1, count):
            remaining = count - idx - 1
            choices = []
            for gap in (2, 3, 4):
                nxt = positions[-1] + gap
                if nxt <= last_max - 2 * remaining:
                    choices.append(nxt)
            if not choices:
                ok = False
                break
            positions.append(random.choice(choices))
        if ok:
            return positions
    raise RuntimeError("failed to choose spaced positions")


def _horizontal_snake(h: int, w: int) -> list[tuple[int, int]]:
    max_runs = min(5, 1 + (h - 3) // 2)
    run_count = random.randint(1, max_runs)
    end_horizontally = random.choice((True, False))
    if run_count == 1:
        end_horizontally = False

    rows = _choose_spaced_positions(run_count, min(3, h - 2), h - 2)
    left_inner = random.randint(1, max(1, min(w - 4, w // 3)))
    right_inner = random.randint(max(left_inner + 2, w // 2), w - 2)

    points = [(rows[0], 0)]
    moving_east = True
    for idx, row in enumerate(rows):
        if idx == run_count - 1 and end_horizontally:
            target_col = w - 1 if moving_east else 0
        else:
            target_col = right_inner if moving_east else left_inner
        _append_segment(points, (row, target_col))

        if idx == run_count - 1:
            if not end_horizontally:
                _append_segment(points, (h - 1, target_col))
            break

        _append_segment(points, (rows[idx + 1], target_col))
        moving_east = not moving_east

    return points


def _vertical_snake(h: int, w: int) -> list[tuple[int, int]]:
    max_runs = min(5, 1 + (w - 3) // 2)
    run_count = random.randint(1, max_runs)
    end_on_right = random.choice((True, False))
    if run_count == 1 or run_count % 2 == 0:
        end_on_right = True

    cols = _choose_spaced_positions(run_count, min(3, w - 2), w - 2)
    top_inner = random.randint(1, max(1, min(h - 4, h // 3)))
    bottom_inner = random.randint(max(top_inner + 2, h // 2), h - 2)

    points = [(0, cols[0])]
    moving_down = True
    for idx, col in enumerate(cols):
        if idx == run_count - 1 and not end_on_right:
            target_row = h - 1
        else:
            target_row = bottom_inner if moving_down else top_inner
        _append_segment(points, (target_row, col))

        if idx == run_count - 1:
            if end_on_right:
                _append_segment(points, (target_row, w - 1))
            break

        _append_segment(points, (target_row, cols[idx + 1]))
        moving_down = not moving_down

    return points


def _render_path(shape: tuple[int, int], points: list[tuple[int, int]], color: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    seen: set[tuple[int, int]] = set()
    for r, c in points:
        if (r, c) in seen:
            raise ValueError("path revisited a cell")
        seen.add((r, c))
        grid[r, c] = color
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        path = _ordered_path(grid != BG)
        turns = _turn_sequence(path)
        if not turns:
            return False
        if len({int(v) for v in np.unique(grid) if int(v) != BG}) != 1:
            return False
        start, end = path[0], path[-1]
        if start[0] + start[1] == end[0] + end[1]:
            return False
        out = main(grid)
        if out.shape != (len(turns), 2):
            return False
        return True
    except (AssertionError, RuntimeError, ValueError):
        return False


def _build_candidate() -> np.ndarray:
    h = random.randint(8, 18)
    w = random.randint(8, 18)
    color = random.choice(PATH_COLORS)

    if random.choice((False, True)):
        points = _horizontal_snake(h, w)
    else:
        points = _vertical_snake(h, w)

    return _render_path((h, w), points, color)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid path puzzle")
