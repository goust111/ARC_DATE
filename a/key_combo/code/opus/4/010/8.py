from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
MAGENTA = 6
PALETTE = (BLUE, RED, GREEN)
FOUR_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbors4(r: int, c: int):
    """ARC `(r, c)` 4-neighborhood, matching the shared toolbox convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _corners(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _nearest_corner(shape: tuple[int, int], coord: tuple[int, int]) -> tuple[tuple[int, int], int]:
    r, c = map(int, coord)
    best: tuple[int, int] | None = None
    best_dist: int | None = None
    tied = False
    for corner in _corners(shape):
        dist = abs(r - corner[0]) + abs(c - corner[1])
        if best_dist is None or dist < best_dist:
            best = corner
            best_dist = dist
            tied = False
        elif dist == best_dist:
            tied = True
    assert best is not None and best_dist is not None and not tied, "nearest corner must be unique"
    return best, best_dist


def _pixel_items(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    coords = np.argwhere((grid != BLACK) & (grid != MAGENTA))
    items: list[dict[str, int | tuple[int, int]]] = []
    for r, c in coords:
        corner, dist = _nearest_corner(grid.shape, (int(r), int(c)))
        items.append(
            {
                "r": int(r),
                "c": int(c),
                "color": int(grid[r, c]),
                "corner": corner,
                "dist": int(dist),
            }
        )
    return items


def _sorted_order(items: list[dict[str, int | tuple[int, int]]]) -> list[int]:
    return sorted(range(len(items)), key=lambda i: (int(items[i]["dist"]), int(items[i]["r"]), int(items[i]["c"])))


def _paint_rectangles(
    base: np.ndarray,
    items: list[dict[str, int | tuple[int, int]]],
    order: list[int] | None = None,
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    if order is None:
        order = _sorted_order(items)
    for idx in order:
        item = items[idx]
        r = int(item["r"])
        c = int(item["c"])
        cr, cc = item["corner"]  # type: ignore[misc]
        color = int(item["color"])
        r0, r1 = sorted((r, int(cr)))
        c0, c1 = sorted((c, int(cc)))
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _marker_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MAGENTA)
    assert coords.shape == (1, 2), "expected exactly one magenta marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _marker_component(grid: np.ndarray) -> set[tuple[int, int]]:
    h, w = map(int, grid.shape)
    start = _marker_coord(grid)
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in (BLACK, MAGENTA):
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return seen


def _solve_with_order(grid: np.ndarray, order: list[int] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    items = _pixel_items(g)
    painted = _paint_rectangles(g, items, order=order)
    region = _marker_component(painted)
    out = painted.copy()
    for r, c in region:
        if out[r, c] == BLACK:
            out[r, c] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Paint corner-anchored rectangles in sorted order, then flood-fill the marked chamber.

    Input structure:
    - A black grid contains 3-4 singleton foreground pixels colored blue, red, or green.
    - Exactly one magenta singleton marker sits near the center and is guaranteed to remain outside every painted rectangle.
    - Each non-marker singleton has a unique nearest grid corner under Manhattan distance.

    Rule:
    - Ignore the magenta marker when constructing rectangles.
    - Sort the other singleton pixels by nearest-corner Manhattan distance, then by row and column.
    - In that order, fill the full axis-aligned rectangle between each pixel and its nearest corner, letting later rectangles overwrite earlier overlaps.
    - Treat every non-black, non-magenta cell as a wall, find the 4-connected open region containing the magenta marker, and recolor only the black cells in that region to magenta.

    Key insight:
    - The visible singleton pixels are only seeds: each one expands to a corner rectangle, and the marker identifies which zero-valued chamber created by those rectangles must be flood-filled after the ordered overlaps are drawn.
    """
    return _solve_with_order(grid)


def _rectangles_overlap(items: list[dict[str, int | tuple[int, int]]]) -> bool:
    rects: list[tuple[int, int, int, int]] = []
    for item in items:
        r = int(item["r"])
        c = int(item["c"])
        cr, cc = item["corner"]  # type: ignore[misc]
        r0, r1 = sorted((r, int(cr)))
        c0, c1 = sorted((c, int(cc)))
        rects.append((r0, c0, r1, c1))
    for i in range(len(rects)):
        a = rects[i]
        for j in range(i + 1, len(rects)):
            b = rects[j]
            if max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3]):
                return True
    return False


def _component_touches_edge(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)


def _central_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    cells = [(r, c) for r in range(h) for c in range(w)]
    cells.sort(key=lambda rc: (abs(rc[0] - cr) + abs(rc[1] - cc), rc[0], rc[1]))
    return cells


def _wrong_orders(items: list[dict[str, int | tuple[int, int]]]) -> list[list[int]]:
    canonical = _sorted_order(items)
    by_scan = sorted(range(len(items)), key=lambda i: (int(items[i]["r"]), int(items[i]["c"])))
    reverse = list(reversed(canonical))
    swaps: list[list[int]] = []
    for i in range(len(canonical)):
        for j in range(i + 1, len(canonical)):
            trial = canonical.copy()
            trial[i], trial[j] = trial[j], trial[i]
            swaps.append(trial)
    orders = [by_scan, reverse, *swaps]
    dedup: list[list[int]] = []
    seen = set()
    for order in orders:
        key = tuple(order)
        if key == tuple(canonical) or key in seen:
            continue
        seen.add(key)
        dedup.append(order)
    return dedup


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    marker_cells = np.argwhere(g == MAGENTA)
    if marker_cells.shape != (1, 2):
        return False
    pixels = np.argwhere((g != BLACK) & (g != MAGENTA))
    if pixels.shape[0] != 4:
        return False
    if any(int(g[r, c]) not in PALETTE for r, c in pixels):
        return False

    items = _pixel_items(g)
    if not _rectangles_overlap(items):
        return False

    marker = tuple(map(int, marker_cells[0]))
    for item in items:
        r = int(item["r"])
        c = int(item["c"])
        cr, cc = item["corner"]  # type: ignore[misc]
        r0, r1 = sorted((r, int(cr)))
        c0, c1 = sorted((c, int(cc)))
        if r0 <= marker[0] <= r1 and c0 <= marker[1] <= c1:
            return False

    canonical = _solve_with_order(g)
    chamber = _marker_component(_paint_rectangles(g, items))
    zero_cells = [coord for coord in chamber if int(_paint_rectangles(g, items)[coord]) == BLACK]
    if len(zero_cells) < 4 or _component_touches_edge(chamber, g.shape):
        return False

    if np.array_equal(canonical, g):
        return False

    if canonical[marker] != MAGENTA or not all(canonical[r, c] == MAGENTA for r, c in zero_cells):
        return False

    if not any(not np.array_equal(canonical, _solve_with_order(g, order=order)) for order in _wrong_orders(items)):
        return False

    return True


def _sample_distinct_pair(values: tuple[int, ...]) -> tuple[int, int]:
    a = random.choice(values)
    options = [v for v in values if v != a]
    b = random.choice(options)
    return a, b


def _canonical_grid() -> np.ndarray:
    h = random.choice(range(12, 21, 2))
    w = random.choice(range(12, 21, 2))
    mid_r = h // 2
    mid_c = w // 2

    top_choices = list(range(1, mid_r - 2))
    left_choices = list(range(1, mid_c - 2))
    assert top_choices and left_choices

    want_horizontal_first = random.random() < 0.5
    pairs = [(tr, lc) for tr in top_choices for lc in left_choices if (tr <= lc) == want_horizontal_first]
    if not pairs:
        pairs = [(tr, lc) for tr in top_choices for lc in left_choices]
    top_r, left_c = random.choice(pairs)

    chamber_rows = list(range(top_r + 1, mid_r))
    chamber_cols = list(range(left_c + 1, mid_c))
    assert len(chamber_rows) >= 2 and len(chamber_cols) >= 2

    color_top, color_left = _sample_distinct_pair(PALETTE)
    color_right = random.choice(PALETTE)
    color_bottom = random.choice(PALETTE)

    grid = np.zeros((h, w), dtype=int)
    grid[top_r, mid_c - 1] = color_top
    grid[mid_r - 1, left_c] = color_left
    grid[mid_r - 1, mid_c] = color_right
    grid[mid_r, mid_c - 1] = color_bottom

    center_order = _central_candidates(grid.shape)
    chamber_set = {(r, c) for r in chamber_rows for c in chamber_cols}
    marker = next(cell for cell in center_order if cell in chamber_set)
    grid[marker] = MAGENTA
    return grid


def generate_input() -> np.ndarray:
    # We always emit four foreground pixels: one duplicated corner creates the required overlap,
    # while the two adjacent corners finish the opposite walls of the chamber.
    for _ in range(300):
        grid = _canonical_grid()
        grid = np.rot90(grid, k=random.randrange(4))
        if random.random() < 0.5:
            grid = np.fliplr(grid)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
