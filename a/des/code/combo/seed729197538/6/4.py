from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BG = 0
START = 4
GATE = 9
PATH_COLORS = [1, 2, 3, 5, 6, 7, 8]
SIDES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(coords)
    return out


def _bar_ray(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _choose_bar_direction(
    grid: np.ndarray,
    comp_set: set[tuple[int, int]],
    start: tuple[int, int],
    step: tuple[int, int],
    length: int,
) -> tuple[int, int]:
    dr, dc = step
    candidates = [(-dc, dr), (dc, -dr)]
    height, width = grid.shape

    def is_clear(direction: tuple[int, int], check_any: bool) -> bool:
        for rr, cc in _bar_ray(start, direction, length):
            if not (0 <= rr < height and 0 <= cc < width):
                return False
            if (rr, cc) in comp_set:
                return False
            if check_any and int(grid[rr, cc]) != BG:
                return False
        return True

    for direction in candidates:
        if is_clear(direction, check_any=True):
            return direction
    for direction in candidates:
        if is_clear(direction, check_any=False):
            return direction
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count maroon gate pixels on each colored path and draw a perpendicular bar at the yellow endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    out = g.copy()
    for comp in _components(g != BG):
        comp_set = set(comp)
        start_cells = [(r, c) for r, c in comp if int(g[r, c]) == START]
        assert len(start_cells) == 1
        start = start_cells[0]

        base_colors = {int(g[r, c]) for r, c in comp} - {START, GATE}
        assert len(base_colors) == 1
        base_color = next(iter(base_colors))

        degrees = {
            (r, c): sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c, height, width))
            for r, c in comp
        }
        endpoints = [coord for coord, degree in degrees.items() if degree == 1]
        assert len(endpoints) == 2
        assert start in endpoints

        start_neighbors = [coord for coord in _neighbors4(start[0], start[1], height, width) if coord in comp_set]
        assert len(start_neighbors) == 1
        step = (start_neighbors[0][0] - start[0], start_neighbors[0][1] - start[1])

        gate_count = sum(int(g[r, c]) == GATE for r, c in comp)
        if gate_count == 0:
            continue

        bar_dir = _choose_bar_direction(g, comp_set, start, step, gate_count)
        for rr, cc in _bar_ray(start, bar_dir, gate_count):
            if 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = base_color

    return out


def _build_top_left_snake(height: int, width: int) -> list[tuple[int, int]]:
    max_levels = min(4, 1 + (height - 1) // 2)
    levels = random.randint(1, max_levels)
    row_options = [
        (0, *tail)
        for tail in itertools.combinations(range(1, height), levels - 1)
        if (not tail or tail[0] >= 2) and all(b - a >= 2 for a, b in zip((0, *tail[:-1]), tail))
    ]
    rows = list(random.choice(row_options))

    near_hi = min(width - 2, 2)
    near_col = random.randint(1, near_hi)
    far_lo = max(near_col + 1, width - 2)
    far_col = random.randint(far_lo, width - 1)

    coords = [(0, 0)]
    current_col = 0
    current_row = 0
    for i, row in enumerate(rows):
        if row > current_row:
            for rr in range(current_row + 1, row + 1):
                coords.append((rr, current_col))
            current_row = row

        target_col = far_col if i % 2 == 0 else near_col
        step = 1 if target_col > current_col else -1
        for cc in range(current_col + step, target_col + step, step):
            coords.append((current_row, cc))
        current_col = target_col

    return coords


def _mirror_h(coords: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    return [(r, width - 1 - c) for r, c in coords]


def _mirror_v(coords: list[tuple[int, int]], height: int) -> list[tuple[int, int]]:
    return [(height - 1 - r, c) for r, c in coords]


def _transpose(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, r) for r, c in coords]


def _make_local_path(side: str, box_h: int, box_w: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    start_from_left = bool(random.getrandbits(1))

    if side == "top":
        coords = _build_top_left_snake(box_h, box_w)
        if not start_from_left:
            coords = _mirror_h(coords, box_w)
        return coords, (1, 0)

    if side == "bottom":
        coords = _build_top_left_snake(box_h, box_w)
        if not start_from_left:
            coords = _mirror_h(coords, box_w)
        return _mirror_v(coords, box_h), (-1, 0)

    coords = _build_top_left_snake(box_w, box_h)
    if not start_from_left:
        coords = _mirror_h(coords, box_h)
    coords = _transpose(coords)
    if side == "left":
        return coords, (0, 1)
    assert side == "right"
    return _mirror_h(coords, box_w), (0, -1)


def _place_path(
    grid: np.ndarray,
    expected: np.ndarray,
    side: str,
    rect: tuple[int, int, int, int],
    color: int,
) -> bool:
    r0, r1, c0, c1 = rect
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    local_path, bar_dir = _make_local_path(side, box_h, box_w)
    path = [(r0 + r, c0 + c) for r, c in local_path]

    start = path[0]
    capacity = box_h - 1 if side in ("top", "bottom") else box_w - 1
    max_gates = min(capacity, len(path) - 2, 4)
    if max_gates <= 0:
        return False
    gate_count = random.randint(1, max_gates)
    gate_idx = set(random.sample(range(1, len(path) - 1), gate_count))

    for idx, (r, c) in enumerate(path):
        if int(grid[r, c]) != BG:
            return False
        grid[r, c] = GATE if idx in gate_idx else color
    grid[start] = START

    for rr, cc in _bar_ray(start, bar_dir, gate_count):
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return False
        if int(grid[rr, cc]) != BG:
            return False
        expected[rr, cc] = color

    return True


def _sample_layout():
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    top_h = random.randint(4, 6)
    bottom_h = random.randint(4, 6)
    left_w = random.randint(4, 6)
    right_w = random.randint(4, 6)

    if height - top_h - bottom_h < 6:
        return None
    if width - left_w - right_w < 6:
        return None

    return height, width, {
        "top": (0, top_h - 1, left_w + 1, width - right_w - 2),
        "bottom": (height - bottom_h, height - 1, left_w + 1, width - right_w - 2),
        "left": (top_h + 1, height - bottom_h - 2, 0, left_w - 1),
        "right": (top_h + 1, height - bottom_h - 2, width - right_w, width - 1),
    }


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    height, width, rects = layout

    count = random.randint(2, 4)
    sides = random.sample(list(SIDES), count)
    colors = random.sample(PATH_COLORS, count)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    for side, color in zip(sides, colors):
        if not _place_path(grid, expected, side, rects[side], color):
            return None

    expected[(grid != BG)] = grid[(grid != BG)]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
