from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    border = np.concatenate([grid[0, :], grid[h - 1, :], grid[1 : h - 1, 0], grid[1 : h - 1, w - 1]])
    border = [int(v) for v in border if int(v) != bg]
    if border:
        return Counter(border).most_common(1)[0][0]
    if WALL in np.unique(grid):
        return WALL
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected a non-background wall color"
    return counts.most_common(1)[0][0]


def _region_marker_color(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> int:
    colors = {int(grid[r, c]) for r, c in cells if int(grid[r, c]) != bg}
    assert len(colors) == 1, "each region must contain exactly one marker color"
    return colors.pop()


def _region_markers(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> list[tuple[int, int]]:
    pts = sorted((r, c) for r, c in cells if int(grid[r, c]) == color)
    assert 2 <= len(pts) <= 3, "each region must contain 2 or 3 markers"
    return pts


def _orientation(markers: list[tuple[int, int]]) -> tuple[str, int]:
    rows = {r for r, _ in markers}
    cols = {c for _, c in markers}
    if len(rows) == 1:
        return ("h", markers[0][0])
    if len(cols) == 1:
        return ("v", markers[0][1])

    diffs = {r - c for r, c in markers}
    if len(diffs) == 1:
        return ("d1", next(iter(diffs)))

    sums = {r + c for r, c in markers}
    if len(sums) == 1:
        return ("d2", next(iter(sums)))

    raise AssertionError("markers must be horizontal, vertical, or 45-degree diagonal")


def _stripe_mask(cells: list[tuple[int, int]], orientation: tuple[str, int]) -> set[tuple[int, int]]:
    kind, anchor = orientation
    chosen: set[tuple[int, int]] = set()
    for r, c in cells:
        if kind == "h" and (r - anchor) % 2 == 0:
            chosen.add((r, c))
        elif kind == "v" and (c - anchor) % 2 == 0:
            chosen.add((r, c))
        elif kind == "d1" and (r - c - anchor) % 2 == 0:
            chosen.add((r, c))
        elif kind == "d2" and (r + c - anchor) % 2 == 0:
            chosen.add((r, c))
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    wall = _wall_color(g, bg)
    out = g.copy()

    for cells in _components(g != wall):
        if _touches_border(cells, g.shape):
            continue
        marker_color = _region_marker_color(g, cells, bg)
        markers = _region_markers(g, cells, marker_color)
        fill = _stripe_mask(cells, _orientation(markers))
        for r, c in fill:
            out[r, c] = marker_color

    return out


def _room_bounds(sizes: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 1
    for size in sizes:
        bounds.append((start, start + size - 1))
        start += size + 1
    return bounds


def _marker_positions(
    bounds: tuple[int, int, int, int], orientation: str, count: int, rng: random.Random
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if orientation == "h":
        row = rng.randint(r0, r1)
        max_step = max(1, (width - 1) // max(1, count - 1))
        step = rng.choice([s for s in (1, 2) if s <= max_step])
        span = 1 + (count - 1) * step
        start = rng.randint(c0, c1 - span + 1)
        return [(row, start + i * step) for i in range(count)]

    if orientation == "v":
        col = rng.randint(c0, c1)
        max_step = max(1, (height - 1) // max(1, count - 1))
        step = rng.choice([s for s in (1, 2) if s <= max_step])
        span = 1 + (count - 1) * step
        start = rng.randint(r0, r1 - span + 1)
        return [(start + i * step, col) for i in range(count)]

    max_step = max(1, min(height - 1, width - 1) // max(1, count - 1))
    step = rng.choice([s for s in (1, 2) if s <= max_step])
    span = 1 + (count - 1) * step

    if orientation == "d1":
        start_r = rng.randint(r0, r1 - span + 1)
        start_c = rng.randint(c0, c1 - span + 1)
        return [(start_r + i * step, start_c + i * step) for i in range(count)]

    if orientation == "d2":
        start_r = rng.randint(r0, r1 - span + 1)
        start_c = rng.randint(c0 + span - 1, c1)
        return [(start_r + i * step, start_c - i * step) for i in range(count)]

    raise AssertionError("unknown orientation")


def _make_grid(rng: random.Random) -> np.ndarray:
    room_layout = rng.choice([(2, 2), (2, 3), (3, 2)])
    n_rows, n_cols = room_layout
    row_sizes = [rng.randint(4, 7) for _ in range(n_rows)]
    col_sizes = [rng.randint(4, 7) for _ in range(n_cols)]

    h = sum(row_sizes) + n_rows + 1
    w = sum(col_sizes) + n_cols + 1
    grid = np.full((h, w), BG, dtype=int)

    row_edges = [0]
    for size in row_sizes:
        row_edges.append(row_edges[-1] + size + 1)
    col_edges = [0]
    for size in col_sizes:
        col_edges.append(col_edges[-1] + size + 1)

    for r in row_edges:
        grid[r, :] = WALL
    grid[h - 1, :] = WALL
    for c in col_edges:
        grid[:, c] = WALL
    grid[:, w - 1] = WALL

    row_bounds = _room_bounds(row_sizes)
    col_bounds = _room_bounds(col_sizes)
    rooms = [(rr0, rr1, cc0, cc1) for rr0, rr1 in row_bounds for cc0, cc1 in col_bounds]
    assert len(rooms) >= 4

    orientations = ["h", "v", "d1"]
    while len(orientations) < len(rooms):
        orientations.append(rng.choice(["h", "v", "d1", "d2"]))
    rng.shuffle(orientations)

    colors = list(PALETTE)
    rng.shuffle(colors)
    while len(colors) < len(rooms):
        colors.extend(colors)

    for idx, bounds in enumerate(rooms):
        orientation = orientations[idx]
        color = colors[idx]
        count = rng.choice([2, 3])
        for r, c in _marker_positions(bounds, orientation, count, rng):
            grid[r, c] = color

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_grid(rng)
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        colors = set(int(v) for v in np.unique(grid))
        if WALL not in colors or BG not in colors:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
