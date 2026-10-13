from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

WINDOW_SIZE = 4
BASE_COLORS = (RED, TEAL, YELLOW, GREEN)
GROUP_THRESHOLD = 3
GEN_RETRIES = 300


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


GLYPH_SEQUENCE = [
    (
        RED,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        TEAL,
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        ),
    ),
    (
        YELLOW,
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    (
        GREEN,
        np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=bool,
        ),
    ),
]

GLYPH_TO_COLOR = {_mask_key(mask): color for color, mask in GLYPH_SEQUENCE}
COLOR_TO_GLYPH = {color: mask for color, mask in GLYPH_SEQUENCE}


def _true_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = int(len(flags))
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]):
            j += 1
        intervals.append((int(i), int(j)))
        i = j + 1
    return intervals


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _window_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_separators = np.all(grid == GREY, axis=1)
    col_separators = np.all(grid == GREY, axis=0)
    row_bands = _true_intervals(~row_separators)
    col_bands = _true_intervals(~col_separators)

    assert len(row_bands) == len(col_bands), "window lattice must be square"
    assert len(row_bands) in (3, 4), "expected a 3x3 or 4x4 window lattice"
    assert all(r1 - r0 + 1 == WINDOW_SIZE for r0, r1 in row_bands)
    assert all(c1 - c0 + 1 == WINDOW_SIZE for c0, c1 in col_bands)
    return row_bands, col_bands


def _decode_layout(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input must contain only black glyph holes and grey separators"

    row_bands, col_bands = _window_bands(g)
    layout = np.zeros((len(row_bands), len(col_bands)), dtype=int)

    for wr, (r0, r1) in enumerate(row_bands):
        for wc, (c0, c1) in enumerate(col_bands):
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            assert window.shape == (WINDOW_SIZE, WINDOW_SIZE)
            key = _mask_key(window == BLACK)
            assert key in GLYPH_TO_COLOR, "unknown glyph mask"
            layout[wr, wc] = GLYPH_TO_COLOR[key]

    return layout, row_bands, col_bands


def _recolor_large_groups(layout: np.ndarray) -> np.ndarray:
    h, w = map(int, layout.shape)
    out = layout.copy()
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(layout[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            group = []
            while queue:
                cr, cc = queue.popleft()
                group.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(layout[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            if len(group) >= GROUP_THRESHOLD:
                for gr, gc in group:
                    out[gr, gc] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each 4x4 glyph window into a fill color, then merge and recolor large same-color window groups.

    Input structure:
    - The grid contains only grey (5) separator lines and black (0) glyph holes.
    - Complete grey rows and columns partition the canvas into a square 3x3 or 4x4 lattice of 4x4 windows.
    - Inside each window, black pixels form one of four fixed glyph masks on a grey background.

    Rule:
    - Detect the separator lattice and extract every 4x4 window.
    - Decode each exact black-hole mask into one of four colors: red (2), teal (8), yellow (4), or green (3).
    - Replace every window with a solid block of its decoded color while keeping all grey separator lines unchanged.
    - On the window lattice, find 4-connected groups of adjacent windows that share the same decoded color.
    - Recolor every group of size at least three to pink (6); groups of size one or two keep their decoded color.

    Key insight:
    - The visible output color of a window depends on two stages: first decode its local glyph, then reason on the coarser window adjacency graph to see whether that decoded color belongs to a large connected cluster.
    """
    decoded, row_bands, col_bands = _decode_layout(grid)
    final_layout = _recolor_large_groups(decoded)

    out = np.full_like(np.asarray(grid, dtype=int), GREY)
    for wr, (r0, r1) in enumerate(row_bands):
        for wc, (c0, c1) in enumerate(col_bands):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(final_layout[wr, wc])
    return out


def _random_connected_cells(side: int, size: int) -> set[tuple[int, int]]:
    start = (random.randrange(side), random.randrange(side))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], side, side))

    while len(cells) < size:
        choice = random.choice(sorted(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nbr in _neighbors4(choice[0], choice[1], side, side):
            if nbr not in cells:
                frontier.add(nbr)
    return cells


def _group_sizes(layout: np.ndarray) -> list[int]:
    h, w = map(int, layout.shape)
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(layout[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while queue:
                cr, cc = queue.popleft()
                size += 1
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(layout[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            sizes.append(size)
    return sizes


def _valid_layout(layout: np.ndarray) -> bool:
    colors = set(map(int, layout.reshape(-1)))
    if colors != set(BASE_COLORS):
        return False

    sizes = _group_sizes(layout)
    if not any(size >= GROUP_THRESHOLD for size in sizes):
        return False
    if not any(size <= 2 for size in sizes):
        return False

    recolored = _recolor_large_groups(layout)
    if len(set(map(int, recolored.reshape(-1)))) < 2:
        return False
    return True


def _sample_layout(side: int) -> np.ndarray | None:
    for _ in range(200):
        layout = np.full((side, side), -1, dtype=int)

        large_color = random.choice(BASE_COLORS)
        max_large = min(side * side - 3, 2 * side + 1)
        if max_large < GROUP_THRESHOLD:
            continue
        large_size = random.randint(GROUP_THRESHOLD, max_large)
        large_cells = _random_connected_cells(side, large_size)
        for r, c in large_cells:
            layout[r, c] = large_color

        free_cells = [(r, c) for r in range(side) for c in range(side) if layout[r, c] < 0]
        random.shuffle(free_cells)
        if len(free_cells) < len(BASE_COLORS) - 1:
            continue

        missing_colors = [color for color in BASE_COLORS if color != large_color]
        for color in missing_colors:
            r, c = free_cells.pop()
            layout[r, c] = color

        for r, c in free_cells:
            layout[r, c] = random.choice(BASE_COLORS)

        if _valid_layout(layout):
            return layout
    return None


def _render_input(layout: np.ndarray) -> np.ndarray:
    side = int(layout.shape[0])
    size = side * WINDOW_SIZE + side + 1
    grid = np.full((size, size), GREY, dtype=int)

    for wr in range(side):
        for wc in range(side):
            r0 = 1 + wr * (WINDOW_SIZE + 1)
            c0 = 1 + wc * (WINDOW_SIZE + 1)
            glyph = COLOR_TO_GLYPH[int(layout[wr, wc])]
            window = np.full((WINDOW_SIZE, WINDOW_SIZE), GREY, dtype=int)
            window[glyph] = BLACK
            grid[r0 : r0 + WINDOW_SIZE, c0 : c0 + WINDOW_SIZE] = window
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        side = random.choice((3, 4))
        layout = _sample_layout(side)
        if layout is None:
            continue
        inp = _render_input(layout)
        try:
            out = main(inp)
        except Exception:
            continue
        expected_layout = _recolor_large_groups(layout)
        expected = np.full_like(inp, GREY)
        for wr in range(side):
            for wc in range(side):
                r0 = 1 + wr * (WINDOW_SIZE + 1)
                c0 = 1 + wc * (WINDOW_SIZE + 1)
                expected[r0 : r0 + WINDOW_SIZE, c0 : c0 + WINDOW_SIZE] = expected_layout[wr, wc]
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid glyph-window puzzle")
