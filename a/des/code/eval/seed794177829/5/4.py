from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _on_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert len(coords) >= 2, "expected a non-trivial path"

    endpoints: list[tuple[int, int]] = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        degree = len(_neighbors4(mask, r, c))
        assert degree in (1, 2), "path must be single-width and unbranched"
        if degree == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(_on_edge(mask.shape, r, c) for r, c in endpoints), "path endpoints must touch the grid edge"

    path = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while True:
        nxt = [pos for pos in _neighbors4(mask, cur[0], cur[1]) if pos != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]
        path.append(cur)

    assert len(path) == int(mask.sum()), "path must be a single connected component"
    assert path[-1] == endpoints[1], "traced path must end at the opposite endpoint"
    return path


def _segment_path(path: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if len(path) == 1:
        return [path[:]]

    segments: list[list[tuple[int, int]]] = []
    start = 0
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_dir != next_dir:
            segments.append(path[start:i])
            start = i
    segments.append(path[start:])
    assert segments and all(segment for segment in segments), "every segment must contain at least one pixel"
    return segments


def _segment_marker_color(grid: np.ndarray, segment: list[tuple[int, int]]) -> int | None:
    markers = {int(grid[r, c]) for r, c in segment if int(grid[r, c]) not in (BG, GREY)}
    assert len(markers) <= 1, "each straight segment may contain at most one marker color"
    return None if not markers else next(iter(markers))


def _resolve_segment_colors(grid: np.ndarray, segments: list[list[tuple[int, int]]]) -> list[int]:
    colors: list[int | None] = [_segment_marker_color(grid, segment) for segment in segments]
    i = 0
    while i < len(colors):
        if colors[i] is not None:
            i += 1
            continue

        j = i
        while j < len(colors) and colors[j] is None:
            j += 1

        left = colors[i - 1] if i > 0 else None
        right = colors[j] if j < len(colors) else None
        if left is None and right is None:
            fill = GREY
        elif left is None:
            fill = int(right)
        elif right is None:
            fill = int(left)
        elif left == right:
            fill = int(left)
        else:
            fill = GREY

        for k in range(i, j):
            colors[k] = fill
        i = j

    return [int(color) for color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the single-pixel path at every turn. Marker segments adopt their marker color;
    unmarked runs inherit a shared neighboring color, otherwise they stay grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    path_mask = g != BG
    assert np.any(path_mask), "expected a foreground path"
    path = _trace_path(path_mask)
    segments = _segment_path(path)
    segment_colors = _resolve_segment_colors(g, segments)

    out = np.full_like(g, BG)
    for segment, color in zip(segments, segment_colors, strict=True):
        for r, c in segment:
            out[r, c] = int(color)
    return out


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    pts = [(r0, c0)]
    r = r0
    c = c0
    while (r, c) != (r1, c1):
        r += dr
        c += dc
        pts.append((r, c))
    return pts


def _partition_total(total: int, count: int, minimum: int) -> list[int]:
    parts: list[int] = []
    remaining = total
    for i in range(count - 1):
        rest = count - i - 1
        hi = remaining - minimum * rest
        part = random.randint(minimum, hi)
        parts.append(part)
        remaining -= part
    parts.append(remaining)
    return parts


def _sample_rows(height: int) -> list[int] | None:
    max_len = min(6, height - 3)
    for _ in range(200):
        vertical_lengths = [random.randint(3, max_len) for _ in range(3)]
        directions = [random.choice((-1, 1)) for _ in range(3)]
        rows = [random.randint(1, height - 2)]
        ok = True
        for step, direction in zip(vertical_lengths, directions, strict=True):
            nxt = rows[-1] + direction * step
            if not (1 <= nxt <= height - 2):
                ok = False
                break
            rows.append(nxt)
        if ok:
            return rows
    return None


def _build_canonical_path() -> tuple[int, int, list[tuple[int, int]]] | None:
    height = random.randint(12, 18)
    width = random.randint(15, 23)

    rows = _sample_rows(height)
    if rows is None:
        return None

    horizontal_lengths = _partition_total(width, 4, 3)
    x1 = horizontal_lengths[0]
    x2 = horizontal_lengths[0] + horizontal_lengths[1]
    x3 = horizontal_lengths[0] + horizontal_lengths[1] + horizontal_lengths[2]

    vertices = [
        (rows[0], 0),
        (rows[0], x1),
        (rows[1], x1),
        (rows[1], x2),
        (rows[2], x2),
        (rows[2], x3),
        (rows[3], x3),
        (rows[3], width - 1),
    ]

    path = [vertices[0]]
    for start, end in zip(vertices, vertices[1:]):
        path.extend(_line_points(start, end)[1:])

    segments = _segment_path(path)
    if len(segments) != 7:
        return None
    if min(len(segments[1]), len(segments[3]), len(segments[5])) < 3:
        return None
    return height, width, path


def _marker_index(segment: list[tuple[int, int]]) -> int:
    if len(segment) >= 3:
        return random.randint(1, len(segment) - 2)
    return random.randrange(len(segment))


def _place_markers(grid: np.ndarray, segments: list[list[tuple[int, int]]]) -> None:
    color_a, color_b = random.sample(list(MARKER_COLORS), 2)
    if random.random() < 0.5:
        marker_map = {1: color_a, 3: color_a, 5: color_b}
    else:
        marker_map = {1: color_a, 3: color_b, 5: color_b}

    for seg_idx, color in marker_map.items():
        r, c = segments[seg_idx][_marker_index(segments[seg_idx])]
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    spec = _build_canonical_path()
    if spec is None:
        return None
    height, width, path = spec

    grid = np.full((height, width), BG, dtype=int)
    for r, c in path:
        grid[r, c] = GREY

    segments = _segment_path(path)
    _place_markers(grid, segments)
    grid = np.rot90(grid, k=random.randrange(4))

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
