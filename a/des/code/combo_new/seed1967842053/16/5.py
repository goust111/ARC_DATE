import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors(pos, shape):
    r, c = pos
    h, w = shape
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _legend_order(grid):
    order = []
    seen = set()
    if grid.shape[1] == 0:
        return order
    last = None
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color not in (BLACK, GREY) and color != last and color not in seen:
            order.append(color)
            seen.add(color)
        last = color
    return order


def _source_starts(grid, legend):
    legend_colors = set(legend)
    starts = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            if c == 0:
                continue
            if legend_colors and color not in legend_colors:
                continue
            grey_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors((r, c), grid.shape)
                if int(grid[nr, nc]) == GREY
            ]
            for first in grey_neighbors:
                starts.append((color, (r, c), first))
    return starts


def _trace_path(grid, source, first):
    path = []
    previous = source
    current = first
    seen_edges = set()
    limit = grid.shape[0] * grid.shape[1] * 4

    for _ in range(limit):
        if int(grid[current]) != GREY:
            break
        path.append(current)
        edge = (previous, current)
        if edge in seen_edges:
            break
        seen_edges.add(edge)

        candidates = [
            nb
            for nb in _neighbors(current, grid.shape)
            if nb != previous and int(grid[nb]) == GREY
        ]
        if not candidates:
            break

        incoming = (current[0] - previous[0], current[1] - previous[1])
        straight = (current[0] + incoming[0], current[1] + incoming[1])
        if straight in candidates:
            nxt = straight
        elif len(candidates) == 1:
            nxt = candidates[0]
        else:
            nxt = min(candidates)
        previous, current = current, nxt

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each source color along grey paths and resolve crossings by the left-edge layer legend.

    Input structure:
    The grid has a black background, grey one-pixel-wide paths, unique colored source endpoints on those paths, and a left-edge vertical legend listing the source colors from highest to lowest layer priority.
    Rule:
    For every colored source endpoint, follow the adjacent grey path, replacing each reached grey pixel with that source color. At a grey pixel reached by multiple source paths, choose the reached color whose legend block is higher on the left edge. Colored source endpoints and legend blocks stay unchanged, and black background stays black.
    Key insight:
    The path ownership and the layer priority are separate clues: first trace which source can reach each grey path pixel, then use the visible legend order only where multiple traces claim the same crossing pixel.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    legend = _legend_order(grid)
    starts = _source_starts(grid, legend)

    priority = {}
    for color in legend:
        if color not in priority:
            priority[color] = len(priority)
    for color, _, _ in starts:
        if color not in priority:
            priority[color] = len(priority)

    claims = defaultdict(list)
    for color, source, first in starts:
        for cell in _trace_path(grid, source, first):
            if color not in claims[cell]:
                claims[cell].append(color)

    for cell, colors in claims.items():
        out[cell] = min(colors, key=lambda color: priority.get(color, len(priority)))
    return out


def _sample_spaced(values, count):
    values = list(values)
    for _ in range(200):
        random.shuffle(values)
        chosen = []
        for value in values:
            if all(abs(value - other) >= 2 for other in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    raise ValueError("not enough spaced positions")


def _build_candidate():
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_paths = random.randint(2, 5)
    n_horizontal = random.randint(1, n_paths - 1)
    n_vertical = n_paths - n_horizontal

    h_rows = _sample_spaced(range(4, h - 4), n_horizontal)
    v_cols = _sample_spaced(range(5, w - 4), n_vertical)

    min_row, max_row = min(h_rows), max(h_rows)
    min_col, max_col = min(v_cols), max(v_cols)
    grid = np.zeros((h, w), dtype=int)

    source_colors = random.sample(COLORS, n_paths)
    legend = source_colors[:]
    random.shuffle(legend)
    for idx, color in enumerate(legend, start=1):
        grid[idx, 0] = color

    color_index = 0
    sources = []

    for row in h_rows:
        color = source_colors[color_index]
        color_index += 1
        left = random.randint(3, min_col - 2)
        right = random.randint(max_col + 2, w - 3)
        source_on_left = random.choice([True, False])
        source = (row, left if source_on_left else right)
        for col in range(left, right + 1):
            grid[row, col] = GREY
        grid[source] = color
        sources.append(source)

    for col in v_cols:
        color = source_colors[color_index]
        color_index += 1
        top = random.randint(2, min_row - 2)
        bottom = random.randint(max_row + 2, h - 3)
        source_on_top = random.choice([True, False])
        source = (top if source_on_top else bottom, col)
        for row in range(top, bottom + 1):
            if int(grid[row, col]) in (BLACK, GREY):
                grid[row, col] = GREY
        grid[source] = color
        sources.append(source)

    for source in sources:
        grey_neighbors = sum(
            1 for nr, nc in _neighbors(source, grid.shape) if int(grid[nr, nc]) == GREY
        )
        if grey_neighbors != 1:
            raise ValueError("ambiguous source")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        try:
            return _build_candidate()
        except ValueError:
            continue
    return _build_candidate()
