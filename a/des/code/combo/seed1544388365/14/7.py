from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

_PRIORITY_ORDER = (RED, BLUE, YELLOW, ORANGE, MAGENTA, AZURE)
_PRIORITY_RANK = {color: rank for rank, color in enumerate(reversed(_PRIORITY_ORDER), start=1)}
_FRAGMENT_COLORS = _PRIORITY_ORDER


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _color_rank(color: int) -> int:
    color = int(color)
    if color == BLACK:
        return -1
    return _PRIORITY_RANK.get(color, 0)


def _find_color_components(grid: np.ndarray, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    steps = _neighbors8 if connectivity == 8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in steps(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_fragments(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    excluded = {BLACK, GREEN, MAROON}
    visited = np.zeros((h, w), dtype=bool)
    fragments: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in excluded:
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            fragments.append({"color": color, "cells": cells})
    return fragments


def _detect_arrow(grid: np.ndarray) -> tuple[str, set[tuple[int, int]]]:
    h, w = grid.shape
    green_components = [set(comp) for comp in _find_color_components(grid, GREEN, connectivity=8)]
    candidates: list[tuple[str, set[tuple[int, int]]]] = []

    if w >= 2:
        for r in range(1, h - 1):
            candidates.append(("right", {(r, 0), (r - 1, 1), (r + 1, 1)}))
            candidates.append(("left", {(r, w - 1), (r - 1, w - 2), (r + 1, w - 2)}))
    if h >= 2:
        for c in range(1, w - 1):
            candidates.append(("down", {(0, c), (1, c - 1), (1, c + 1)}))
            candidates.append(("up", {(h - 1, c), (h - 2, c - 1), (h - 2, c + 1)}))

    matches = [(direction, cells) for direction, cells in candidates if cells in green_components]
    assert len(matches) == 1, "expected exactly one green edge arrow"
    return matches[0]


def _detect_barrier(grid: np.ndarray, direction: str) -> np.ndarray:
    mask = grid == MAROON
    coords = np.argwhere(mask)
    assert len(coords) >= 1, "expected a maroon barrier"
    rows = sorted(set(int(r) for r, _ in coords))
    cols = sorted(set(int(c) for _, c in coords))
    if direction in ("left", "right"):
        assert len(cols) == 1, "horizontal slide needs a vertical barrier"
        assert rows == list(range(rows[0], rows[-1] + 1)), "barrier must be contiguous"
    else:
        assert len(rows) == 1, "vertical slide needs a horizontal barrier"
        assert cols == list(range(cols[0], cols[-1] + 1)), "barrier must be contiguous"
    return mask


def _fragment_sort_key(fragment: dict[str, object], direction: str) -> tuple[int, int, int]:
    cells = fragment["cells"]
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if direction == "right":
        return (-max(cols), min(rows), min(cols))
    if direction == "left":
        return (min(cols), min(rows), min(cols))
    if direction == "down":
        return (-max(rows), min(cols), min(rows))
    return (min(rows), min(cols), min(rows))


def _stripe_shifts(
    cells: list[tuple[int, int]],
    direction: str,
    occupied: np.ndarray,
) -> dict[int, int]:
    stripes: dict[int, list[int]] = defaultdict(list)
    horizontal = direction in ("left", "right")
    for r, c in cells:
        key = r if horizontal else c
        axis = c if horizontal else r
        stripes[key].append(axis)

    shifts: dict[int, int] = {}
    h, w = occupied.shape
    for key, axis_positions in stripes.items():
        if direction == "right":
            front = max(axis_positions)
            shift = 0
            while front + shift + 1 < w and not occupied[key, front + shift + 1]:
                shift += 1
        elif direction == "left":
            front = min(axis_positions)
            shift = 0
            while front + shift - 1 >= 0 and not occupied[key, front + shift - 1]:
                shift -= 1
        elif direction == "down":
            front = max(axis_positions)
            shift = 0
            while front + shift + 1 < h and not occupied[front + shift + 1, key]:
                shift += 1
        else:
            front = min(axis_positions)
            shift = 0
            while front + shift - 1 >= 0 and not occupied[front + shift - 1, key]:
                shift -= 1
        shifts[key] = shift
    return shifts


def _solve(grid: np.ndarray, collect: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    direction, arrow_cells = _detect_arrow(g)
    barrier_mask = _detect_barrier(g, direction)
    fragments = sorted(_find_fragments(g), key=lambda fragment: _fragment_sort_key(fragment, direction))

    visible = np.full((h, w), BLACK, dtype=int)
    occupied_fragments = np.zeros((h, w), dtype=bool)
    occupied = barrier_mask.copy()

    overlap_cells: set[tuple[int, int]] = set()
    priority_overwrites: set[tuple[int, int]] = set()
    split_fragments = 0

    for fragment in fragments:
        color = int(fragment["color"])
        cells = list(fragment["cells"])
        shifts = _stripe_shifts(cells, direction, occupied)
        if len(set(shifts.values())) > 1:
            split_fragments += 1

        final_cells: list[tuple[int, int]] = []
        if direction in ("left", "right"):
            for r, c in cells:
                final_cells.append((r, c + shifts[r]))
        else:
            for r, c in cells:
                final_cells.append((r + shifts[c], c))

        for r, c in final_cells:
            if occupied_fragments[r, c]:
                overlap_cells.add((r, c))
                if _color_rank(color) > _color_rank(int(visible[r, c])):
                    priority_overwrites.add((r, c))

            occupied_fragments[r, c] = True
            current = int(visible[r, c])
            if current == BLACK or current == color or _color_rank(color) > _color_rank(current):
                visible[r, c] = color

        occupied = barrier_mask | occupied_fragments

    out = visible
    out[barrier_mask] = MAROON
    for r, c in arrow_cells:
        out[r, c] = BLACK

    if not collect:
        return out

    info = {
        "direction": direction,
        "fragment_count": len(fragments),
        "split_fragments": split_fragments,
        "overlap_cells": len(overlap_cells),
        "priority_overwrites": len(priority_overwrites),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """Slide monochrome fragments along the arrow direction, splitting at the partial barrier and stacking by color priority."""
    return _solve(grid, collect=False)


_CORE_A = (
    (0, 0),
    (0, 5),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 5),
)
_CORE_B = (
    (0, 0),
    (0, 4),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 0),
)
_EXTRA_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)
_GEN_ATTEMPTS = 300


def _flip_vertical(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _ in shape) + 1
    return tuple(sorted((height - 1 - r, c) for r, c in shape))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _can_place_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    gap: int = 1,
) -> bool:
    h, w = grid.shape
    coords = {(top + dr, left + dc) for dr, dc in shape}
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in coords:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in coords and int(grid[rr, cc]) != BLACK:
                    return False
    return True


def _rotate_from_right(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k).copy()


def _choose_core_colors(rng: np.random.Generator) -> tuple[int, int]:
    low_idx = int(rng.integers(2, len(_PRIORITY_ORDER)))
    high_idx = int(rng.integers(0, low_idx))
    return _PRIORITY_ORDER[low_idx], _PRIORITY_ORDER[high_idx]


def _candidate_arrow_rows(h: int, shared_row: int) -> list[int]:
    blocked = set(range(shared_row - 3, shared_row + 4))
    return [r for r in range(1, h - 1) if r not in blocked]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 17))
    w = int(rng.integers(16, 23))
    grid = np.full((h, w), BLACK, dtype=int)

    shared_row = int(rng.integers(3, h - 3))
    c_b = int(rng.integers(2, w - 11))
    c_a = c_b + 1
    barrier_col = c_b + 8

    arrow_rows = _candidate_arrow_rows(h, shared_row)
    if not arrow_rows:
        return None
    arrow_r = int(rng.choice(arrow_rows))

    barrier_rows = [shared_row - 1, shared_row, shared_row + 1]
    grid[barrier_rows, barrier_col] = MAROON
    grid[arrow_r, 0] = GREEN
    grid[arrow_r - 1, 1] = GREEN
    grid[arrow_r + 1, 1] = GREEN

    color_a, color_b = _choose_core_colors(rng)
    shape_a = _CORE_A
    shape_b = _flip_vertical(_CORE_B)

    top_a, left_a = shared_row, c_a
    top_b, left_b = shared_row - 2, c_b

    if not _can_place_shape(grid, shape_a, top_a, left_a, gap=0):
        return None
    _paint_shape(grid, shape_a, top_a, left_a, color_a)

    if not _can_place_shape(grid, shape_b, top_b, left_b, gap=0):
        return None
    _paint_shape(grid, shape_b, top_b, left_b, color_b)

    extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        shape = _EXTRA_SHAPES[int(rng.integers(len(_EXTRA_SHAPES)))]
        sh, sw = _shape_size(shape)
        placed = False
        for _attempt in range(40):
            band = "top" if rng.random() < 0.5 else "bottom"
            if band == "top":
                row_min = 1
                row_max = shared_row - 3 - sh
            else:
                row_min = shared_row + 3
                row_max = h - 2 - sh
            if row_min > row_max:
                break

            side = "left" if rng.random() < 0.6 else "right"
            if side == "left":
                col_min = 2
                col_max = barrier_col - sw - 2
            else:
                col_min = barrier_col + 2
                col_max = w - sw - 2
            if col_min > col_max:
                continue

            top = int(rng.integers(row_min, row_max + 1))
            left = int(rng.integers(col_min, col_max + 1))
            color = int(rng.choice(_FRAGMENT_COLORS))
            if color in (color_a, color_b) and rng.random() < 0.5:
                continue
            if _can_place_shape(grid, shape, top, left, gap=1):
                _paint_shape(grid, shape, top, left, color)
                placed = True
                break
        if not placed:
            continue

    rotation_k = int(rng.integers(0, 4))
    rotated = _rotate_from_right(grid, rotation_k)
    _, info = _solve(rotated, collect=True)
    if info["fragment_count"] < 2:
        return None
    if info["split_fragments"] < 2:
        return None
    if info["overlap_cells"] < 1 or info["priority_overwrites"] < 1:
        return None
    return rotated


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
