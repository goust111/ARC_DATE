from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SQUARE_SIZE = 3
RAY_MARKER_LIMIT = 6
GEN_RETRIES = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RAY_SPECS = [
    ((0, 0), (-1, -1)),
    ((0, 2), (-1, 1)),
    ((2, 2), (1, 1)),
    ((2, 0), (1, -1)),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_square_top_left(grid: np.ndarray) -> tuple[int, int]:
    blue_components = _find_components_of_color(grid, BLUE)
    matches: list[tuple[int, int]] = []
    for comp in blue_components:
        if len(comp) != SQUARE_SIZE * SQUARE_SIZE:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (SQUARE_SIZE, SQUARE_SIZE):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE):
            matches.append((r0, c0))
    if len(matches) != 1:
        raise ValueError("expected exactly one solid 3x3 blue square")
    return matches[0]


def _ray_cells(shape: tuple[int, int], square_top_left: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = square_top_left
    cells: list[tuple[int, int]] = []
    for (corner_dr, corner_dc), (step_r, step_c) in RAY_SPECS:
        cr = sr + corner_dr
        cc = sc + corner_dc
        for step in range(1, length + 1):
            rr = cr + step_r * step
            cc2 = cc + step_c * step
            if not _in_bounds(shape, rr, cc2):
                break
            cells.append((rr, cc2))
    return cells


def _available_steps(shape: tuple[int, int], square_top_left: tuple[int, int]) -> list[int]:
    h, w = shape
    sr, sc = square_top_left
    return [
        min(sr, sc),
        min(sr, w - 1 - (sc + 2)),
        min(h - 1 - (sr + 2), w - 1 - (sc + 2)),
        min(h - 1 - (sr + 2), sc),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red components to set diagonal ray lengths, then recolor the yellow markers those rays touch.

    Input structure:
    - Black cells form the background of a small rectangular grid.
    - One solid blue 3x3 square sits near the middle, while 2-5 disjoint red 4-connected components are scattered away from it.
    - Yellow singleton markers appear both on and off the four outward diagonals from the square's corners.

    Rule:
    - Count the number N of red 4-connected components.
    - From each blue-square corner, extend one diagonal ray in that corner's natural outward direction for exactly N steps, truncating at the grid boundary.
    - Color every traversed background cell blue, recolor traversed yellow markers green, and leave untouched yellow markers, red components, and the blue square unchanged.

    Key insight:
    - The red shapes are only a counting device: their component count controls ray geometry, and that geometry alone decides which yellow markers change color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    square_top_left = _find_square_top_left(g)
    ray_length = len(_find_components_of_color(g, RED))
    if not 2 <= ray_length <= 5:
        raise ValueError("expected 2-5 red components")

    out = g.copy()
    for r, c in _ray_cells(g.shape, square_top_left, ray_length):
        if int(out[r, c]) == RED:
            continue
        if int(out[r, c]) == YELLOW:
            out[r, c] = GREEN
        else:
            out[r, c] = BLUE
    return out


def _grow_connected_shape(area: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        cells.append((nr, nc))
        for dr, dc in DIR4:
            nxt = (nr + dr, nc + dc)
            if nxt not in cells:
                frontier.add(nxt)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, min_c = min(rows), min(cols)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _component_touches_forbidden(
    grid: np.ndarray, component_cells: list[tuple[int, int]], top: int, left: int, forbidden: set[tuple[int, int]]
) -> bool:
    h, w = grid.shape
    occupied = {(r, c) for r, c in zip(*np.where(grid != BLACK))}
    for dr, dc in component_cells:
        r = top + dr
        c = left + dc
        if not _in_bounds(grid.shape, r, c):
            return True
        if (r, c) in forbidden or int(grid[r, c]) != BLACK:
            return True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                    return True
    return False


def _marker_ok(grid: np.ndarray, r: int, c: int) -> bool:
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
        return False
    h, w = grid.shape
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLACK:
                return False
    return True


def _choose_square_position(h: int, w: int, n_components: int, want_clipping: bool) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for sr in range(1, h - SQUARE_SIZE):
        for sc in range(1, w - SQUARE_SIZE):
            avail = _available_steps((h, w), (sr, sc))
            if max(avail) < n_components + 1:
                continue
            if max(avail) < 1:
                continue
            if want_clipping and not any(step < n_components for step in avail):
                continue
            if (not want_clipping) and any(step < 1 for step in avail):
                continue
            center_penalty = abs((sr + 1) - (h - 1) / 2) + abs((sc + 1) - (w - 1) / 2)
            candidates.append((int(center_penalty * 10), sr, sc))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    top_band = candidates[: max(6, len(candidates) // 3)]
    _, sr, sc = random.choice(top_band)
    return sr, sc


def _place_red_components(
    grid: np.ndarray, n_components: int, forbidden: set[tuple[int, int]]
) -> bool:
    h, w = grid.shape
    for _ in range(100):
        temp = grid.copy()
        ok = True
        for _comp_idx in range(n_components):
            area = random.randint(2, 5)
            shape = _grow_connected_shape(area)
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            random.shuffle(positions)
            placed = False
            for top, left in positions:
                if _component_touches_forbidden(temp, shape, top, left, forbidden):
                    continue
                for dr, dc in shape:
                    temp[top + dr, left + dc] = RED
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            grid[:, :] = temp
            return True
    return False


def _place_ray_markers(
    grid: np.ndarray, square_top_left: tuple[int, int], ray_length: int
) -> tuple[bool, int, int]:
    sr, sc = square_top_left
    ray_positions: dict[int, list[tuple[int, int, int]]] = {}
    hit_options: list[tuple[int, int, int]] = []
    far_options: list[tuple[int, int, int]] = []
    for ray_idx, ((corner_dr, corner_dc), (step_r, step_c)) in enumerate(RAY_SPECS):
        cr = sr + corner_dr
        cc = sc + corner_dc
        positions: list[tuple[int, int, int]] = []
        for step in range(1, RAY_MARKER_LIMIT + 1):
            rr = cr + step_r * step
            cc2 = cc + step_c * step
            if not _in_bounds(grid.shape, rr, cc2):
                break
            positions.append((rr, cc2, step))
            if step <= ray_length:
                hit_options.append((rr, cc2, step))
            else:
                far_options.append((rr, cc2, step))
        ray_positions[ray_idx] = positions

    if not hit_options or not far_options:
        return False, 0, 0

    hit_count = random.randint(1, min(4, len(hit_options)))
    far_count = random.randint(1, min(3, len(far_options)))
    hit_cells = random.sample(hit_options, hit_count)
    far_cells = random.sample(far_options, far_count)

    used = {(r, c) for r, c, _ in hit_cells + far_cells}
    extra_line_pool = [
        cell
        for positions in ray_positions.values()
        for cell in positions
        if (cell[0], cell[1]) not in used
    ]
    extra_line_count = random.randint(0, min(2, len(extra_line_pool)))
    extra_line_cells = random.sample(extra_line_pool, extra_line_count)

    for r, c, _step in hit_cells + far_cells + extra_line_cells:
        if not _marker_ok(grid, r, c):
            return False, 0, 0
        grid[r, c] = YELLOW

    return True, len(hit_cells), len(far_cells)


def _place_off_diagonal_markers(
    grid: np.ndarray, forbidden: set[tuple[int, int]], count: int
) -> bool:
    choices = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r, c) not in forbidden and _marker_ok(grid, r, c)
    ]
    if len(choices) < count:
        return False
    for r, c in random.sample(choices, count):
        grid[r, c] = YELLOW
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    n_components = random.randint(2, 5)
    want_clipping = random.random() < 0.45

    square_top_left = _choose_square_position(h, w, n_components, want_clipping)
    if square_top_left is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    sr, sc = square_top_left
    grid[sr : sr + SQUARE_SIZE, sc : sc + SQUARE_SIZE] = BLUE

    ray_reserved = set(_ray_cells(grid.shape, square_top_left, RAY_MARKER_LIMIT))
    square_reserved = {(r, c) for r in range(sr, sr + SQUARE_SIZE) for c in range(sc, sc + SQUARE_SIZE)}
    forbidden = ray_reserved | square_reserved

    if not _place_red_components(grid, n_components, forbidden):
        return None

    placed, hit_count, far_count = _place_ray_markers(grid, square_top_left, n_components)
    if not placed:
        return None

    off_count = random.randint(1, 4)
    if not _place_off_diagonal_markers(grid, forbidden, off_count):
        return None

    out = main(grid)
    changed_to_green = int(np.count_nonzero((grid == YELLOW) & (out == GREEN)))
    unchanged_yellow = int(np.count_nonzero((grid == YELLOW) & (out == YELLOW)))
    if changed_to_green < 1 or unchanged_yellow < 1:
        return None
    if changed_to_green < hit_count or unchanged_yellow < far_count:
        return None
    if len(_find_components_of_color(grid, RED)) != n_components:
        return None
    if np.count_nonzero(grid == BLUE) != SQUARE_SIZE * SQUARE_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
