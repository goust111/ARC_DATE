import random
from collections import deque

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIDES = ("top", "bottom", "left", "right")
_COLORS = tuple(range(10))
_GEN_RETRIES = 500
_PATH_RETRIES = 800


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in _DIRS]


def _edge_cells(grid: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    if side == "top":
        return [(0, c) for c in range(w) if int(grid[0, c]) != bg]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if int(grid[h - 1, c]) != bg]
    if side == "left":
        return [(r, 0) for r in range(h) if int(grid[r, 0]) != bg]
    if side == "right":
        return [(r, w - 1) for r in range(h) if int(grid[r, w - 1]) != bg]
    raise ValueError(side)


def _path_endpoints(cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(cells) < 3:
        return None

    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        pos = q.popleft()
        for nb in _neighbors(pos):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    if len(seen) != len(cells):
        return None

    degrees = {pos: sum(nb in cells for nb in _neighbors(pos)) for pos in cells}
    endpoints = [pos for pos, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        return None
    if any(degree not in (1, 2) for degree in degrees.values()):
        return None
    return endpoints


def _legend_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for side in _SIDES:
        legend_cells = _edge_cells(grid, side, bg)
        if not (2 <= len(legend_cells) <= 6):
            continue

        legend_order = [int(grid[r, c]) for r, c in legend_cells]
        marker_set = set(legend_order)
        if len(marker_set) != len(legend_order):
            continue
        if any(int(np.count_nonzero(grid == color)) != 2 for color in marker_set):
            continue

        path_mask = grid != bg
        for r, c in legend_cells:
            path_mask[r, c] = False

        remaining = grid[path_mask]
        if remaining.size <= len(legend_order):
            continue
        if any(int(np.count_nonzero(remaining == color)) != 1 for color in marker_set):
            continue

        path_colors = set(map(int, np.unique(remaining))) - marker_set
        if len(path_colors) != 1:
            continue
        path_color = next(iter(path_colors))
        if int(np.count_nonzero(remaining == path_color)) < 2:
            continue

        cells = {(int(r), int(c)) for r, c in np.argwhere(path_mask)}
        endpoints = _path_endpoints(cells)
        if endpoints is None:
            continue

        candidates.append(
            {
                "legend_cells": legend_cells,
                "legend_order": legend_order,
                "path_cells": cells,
                "endpoints": endpoints,
                "path_color": path_color,
            }
        )
    return candidates


def _trace_from(cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    prev = None
    cur = start
    while True:
        path.append(cur)
        nxt = [nb for nb in _neighbors(cur) if nb in cells and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
    assert len(path) == len(cells), "path traversal must cover all path cells"
    return path


def _orient_path(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    endpoints: list[tuple[int, int]],
    marker_set: set[int],
    legend_order: list[int],
) -> list[tuple[int, int]]:
    orders = [_trace_from(cells, endpoint) for endpoint in endpoints]
    legend_rank = {color: i for i, color in enumerate(legend_order)}

    def score(path: list[tuple[int, int]]) -> tuple[int, int, int, int, int]:
        marker_positions = [
            i for i, pos in enumerate(path) if int(grid[pos[0], pos[1]]) in marker_set
        ]
        first_marker = marker_positions[0] if marker_positions else 10**6
        start_color = int(grid[path[0][0], path[0][1]])
        start_rank = legend_rank.get(start_color, len(legend_order))
        return (0 if first_marker == 0 else 1, first_marker, start_rank, path[0][0], path[0][1])

    path = min(orders, key=score)
    assert int(grid[path[0][0], path[0][1]]) in marker_set, "first path endpoint must be a marker"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder marked path segments by the color order in an edge legend.

    Input structure:
    - A dominant background surrounds one 4-connected one-pixel-wide path in a
      single path color, with no branches.
    - Several unique marker pixels sit on the path; each marker uses a color
      different from the path color and from the other markers.
    - One edge row or edge column contains a legend strip whose colored one-pixel
      entries are exactly the same colors as the path markers, ordered left to
      right or top to bottom.

    Rule:
    - Remove the edge legend and trace the path from its marked endpoint to the
      other endpoint.
    - The marker pixels split the traced path into segments; each marker belongs
      to the segment after it, so the segment for a marker runs from that marker
      up to the cell before the next marker, or to the path endpoint.
    - Read the legend order and lay the corresponding segments as one tight
      horizontal line starting at row 1, column 0.
    - Recolor every straightened segment with its marker color and clear the
      original path and legend to background.

    Key insight:
    The same marker pixels both cut the path into pieces and link those pieces
    to the ordering encoded by the separate edge legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w >= 2

    bg = _background(g)
    candidates = _legend_candidates(g, bg)
    assert candidates, "one unambiguous edge legend must be present"
    candidates.sort(key=lambda item: (len(item["legend_order"]), len(item["path_cells"])), reverse=True)
    scene = candidates[0]

    legend_order = scene["legend_order"]
    marker_set = set(legend_order)
    path = _orient_path(g, scene["path_cells"], scene["endpoints"], marker_set, legend_order)

    marker_positions = []
    for i, (r, c) in enumerate(path):
        color = int(g[r, c])
        if color in marker_set:
            marker_positions.append((i, color))

    assert len(marker_positions) == len(marker_set), "each legend color must mark one path point"
    assert len({color for _, color in marker_positions}) == len(marker_positions)
    assert marker_positions[0][0] == 0, "the first segment must start on a marker"

    segment_lengths: dict[int, int] = {}
    for i, (start, color) in enumerate(marker_positions):
        end = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(path)
        assert end > start
        segment_lengths[color] = end - start

    total = sum(segment_lengths[color] for color in legend_order)
    assert total <= w, "straightened path must fit on the output row"

    out = np.full_like(g, bg)
    col = 0
    for color in legend_order:
        length = segment_lengths[color]
        out[1, col : col + length] = color
        col += length
    return out


def _self_avoiding_path(h: int, w: int, length: int) -> list[tuple[int, int]] | None:
    row_min, row_max = 2, h - 2
    col_min, col_max = 1, w - 2
    if row_min > row_max or col_min > col_max:
        return None

    for _ in range(_PATH_RETRIES):
        start = (random.randint(row_min, row_max), random.randint(col_min, col_max))
        path = [start]
        cells = {start}

        for _step in range(length - 1):
            cur = path[-1]
            options = []
            dirs = list(_DIRS)
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = cur[0] + dr, cur[1] + dc
                nxt = (nr, nc)
                if not (row_min <= nr <= row_max and col_min <= nc <= col_max):
                    continue
                if nxt in cells:
                    continue
                if any(nb in cells and nb != cur for nb in _neighbors(nxt)):
                    continue
                options.append(nxt)
            if not options:
                break
            nxt = random.choice(options)
            path.append(nxt)
            cells.add(nxt)
        if len(path) == length:
            return path
    return None


def _place_legend(grid: np.ndarray, side: str, colors: list[int]) -> None:
    h, w = grid.shape
    n = len(colors)
    if side == "top":
        start = random.randint(0, w - n)
        for i, color in enumerate(colors):
            grid[0, start + i] = color
    elif side == "bottom":
        start = random.randint(0, w - n)
        for i, color in enumerate(colors):
            grid[h - 1, start + i] = color
    elif side == "left":
        start = random.randint(0, h - n)
        for i, color in enumerate(colors):
            grid[start + i, 0] = color
    elif side == "right":
        start = random.randint(0, h - n)
        for i, color in enumerate(colors):
            grid[start + i, w - 1] = color
    else:
        raise ValueError(side)


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    legend_order: list[int],
    segment_lengths: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    c = 0
    for color in legend_order:
        length = segment_lengths[color]
        out[1, c : c + length] = color
        c += length
    return out


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(3, 5)
    lengths = [random.randint(2, 5) for _ in range(marker_count)]
    if random.random() < 0.35:
        lengths[random.randrange(marker_count)] = random.randint(5, 7)
    total_len = sum(lengths)
    if total_len > 27:
        return None

    w = random.randint(max(12, total_len), 30)
    h = random.randint(8, 18)
    path = _self_avoiding_path(h, w, total_len)
    if path is None:
        return None

    sampled = random.sample(_COLORS, marker_count + 2)
    bg, path_color = sampled[0], sampled[1]
    marker_colors = sampled[2:]

    grid = np.full((h, w), bg, dtype=int)
    for r, c in path:
        grid[r, c] = path_color

    marker_indices = []
    offset = 0
    for length in lengths:
        marker_indices.append(offset)
        offset += length

    segment_lengths = {}
    for color, index, length in zip(marker_colors, marker_indices, lengths):
        r, c = path[index]
        grid[r, c] = color
        segment_lengths[color] = length

    legend_order = marker_colors[:]
    random.shuffle(legend_order)
    _place_legend(grid, random.choice(_SIDES), legend_order)

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != bg:
        return None

    expected = _expected_output(grid.shape, bg, legend_order, segment_lengths)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid marked-path legend input")
