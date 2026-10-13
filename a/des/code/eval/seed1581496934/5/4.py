from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _split_segments(path: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if len(path) <= 1:
        return [path[:]]
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    segments: list[list[tuple[int, int]]] = []
    current_dir = dirs[0]
    current: list[tuple[int, int]] = []
    for i, step_dir in enumerate(dirs):
        if i > 0 and step_dir != current_dir:
            segments.append(current)
            current = []
            current_dir = step_dir
        current.append(path[i])
    current.append(path[-1])
    segments.append(current)
    return segments


def _find_palette(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    path_pts = np.argwhere(grid == GREY)
    assert path_pts.size > 0, "grey path missing"
    path_max_c = int(path_pts[:, 1].max())

    h, w = grid.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for c in range(path_max_c + 1, w):
        rows = [r for r in range(h) if int(grid[r, c]) not in (BLACK, GREY)]
        if len(rows) < 2:
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((len(rows), c, rows))

    assert candidates, "palette column not found"
    _count, palette_col, palette_rows = max(candidates, key=lambda item: (item[0], item[1]))
    palette_rows = sorted(palette_rows)
    palette = [int(grid[r, palette_col]) for r in palette_rows]
    return palette_col, palette_rows, palette


def _ordered_path_from_start(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    path_set = {tuple(map(int, pt)) for pt in np.argwhere(grid == GREY)}
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [pt for pt in _neighbors4(cur[0], cur[1], h, w) if pt in path_set and pt != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path branches are not allowed"
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    assert len(ordered) == len(path_set), "path traversal did not cover all pixels"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    palette_col, palette_rows, palette = _find_palette(g)
    palette_coords = {(r, palette_col) for r in palette_rows}
    palette_rank = {color: idx for idx, color in enumerate(palette)}

    path_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == GREY)]
    assert path_pts, "grey path missing"

    endpoints: list[tuple[int, int]] = []
    for r, c in path_pts:
        degree = sum((nr, nc) in set(path_pts) for nr, nc in _neighbors4(r, c, h, w))
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start_info: list[tuple[int, tuple[int, int]]] = []
    for endpoint in endpoints:
        marker_colors = []
        for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
            if (nr, nc) in palette_coords:
                continue
            color = int(g[nr, nc])
            if color not in (BLACK, GREY):
                marker_colors.append(color)
        assert len(marker_colors) == 1, "each endpoint needs one marker"
        marker_color = marker_colors[0]
        assert marker_color in palette_rank, "marker color must appear in palette"
        start_info.append((palette_rank[marker_color], endpoint))

    start_info.sort(key=lambda item: item[0])
    assert start_info[0][0] != start_info[1][0], "marker colors must differ in palette rank"
    ordered_path = _ordered_path_from_start(g, start_info[0][1])
    segments = _split_segments(ordered_path)

    out = np.full((h, w), BLACK, dtype=int)
    for r in palette_rows:
        out[r, palette_col] = int(g[r, palette_col])
    for segment in segments:
        color = palette[min(len(segment), len(palette)) - 1]
        for r, c in segment:
            out[r, c] = color
    return out


def _build_path(seg_lengths: list[int], dirs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    start = (0, 0)
    for length, (dr, dc) in zip(seg_lengths, dirs):
        for t in range(length):
            path.append((start[0] + dr * t, start[1] + dc * t))
        start = (start[0] + dr * length, start[1] + dc * length)
    return path


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    if len(path) < 2 or len(set(path)) != len(path):
        return False
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False

    pts = set(path)
    endpoint_count = 0
    for r, c in pts:
        degree = sum((r + dr, c + dc) in pts for dr, dc in DIRS4)
        if degree == 1:
            endpoint_count += 1
        elif degree != 2:
            return False
    return endpoint_count == 2


def _random_path(seg_lengths: list[int]) -> list[tuple[int, int]] | None:
    for _ in range(300):
        dirs = [random.choice(DIRS4)]
        for _ in range(len(seg_lengths) - 1):
            prev_dr, prev_dc = dirs[-1]
            if prev_dr == 0:
                choices = [(1, 0), (-1, 0)]
            else:
                choices = [(0, 1), (0, -1)]
            dirs.append(random.choice(choices))

        path = _build_path(seg_lengths, dirs)
        if not _is_simple_path(path):
            continue

        rows = [r for r, _ in path]
        cols = [c for _, c in path]
        if max(rows) - min(rows) > 14 or max(cols) - min(cols) > 14:
            continue
        return path
    return None


def _translate_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in path]


def _marker_candidates(
    endpoint: tuple[int, int],
    path_set: set[tuple[int, int]],
    h: int,
    w: int,
    palette_coords: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
        if (nr, nc) in path_set or (nr, nc) in palette_coords:
            continue
        touched = [pt for pt in _neighbors4(nr, nc, h, w) if pt in path_set]
        if len(touched) == 1 and touched[0] == endpoint:
            candidates.append((nr, nc))
    return candidates


def _render_expected(
    shape: tuple[int, int],
    palette_entries: list[tuple[int, int, int]],
    ordered_path: list[tuple[int, int]],
    palette: list[int],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r, c, color in palette_entries:
        out[r, c] = color
    for segment in _split_segments(ordered_path):
        color = palette[min(len(segment), len(palette)) - 1]
        for r, c in segment:
            out[r, c] = color
    return out


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        palette_col, palette_rows, _palette = _find_palette(g)
    except AssertionError:
        return False

    path_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == GREY)]
    if not _is_simple_path(path_pts):
        return False

    palette_coords = {(r, palette_col) for r in palette_rows}
    colored_off_palette = [
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in (BLACK, GREY) and (r, c) not in palette_coords
    ]
    if len(colored_off_palette) != 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 6)
    palette = random.sample(PALETTE_POOL, k)

    n_segments = random.randint(2, 5)
    short_max = min(k, 5)
    seg_lengths = [random.randint(1, short_max) for _ in range(n_segments)]
    if random.choice((False, True)):
        idx = random.randrange(n_segments)
        seg_lengths[idx] = random.randint(k + 1, min(k + 3, 8))

    abstract_path = _random_path(seg_lengths)
    if abstract_path is None:
        return None

    rows = [r for r, _ in abstract_path]
    cols = [c for _, c in abstract_path]
    top_pad = random.randint(2, 4)
    left_pad = random.randint(2, 4)
    translated_path = _translate_path(abstract_path, top_pad - min(rows), left_pad - min(cols))

    path_rows = [r for r, _ in translated_path]
    path_cols = [c for _, c in translated_path]
    gap = random.randint(3, 5)
    right_pad = random.randint(1, 2)
    bottom_pad = random.randint(2, 4)

    h = max(max(path_rows) + 1 + bottom_pad, k + 4) + random.randint(0, 3)
    palette_col = max(path_cols) + gap
    w = palette_col + 1 + right_pad
    if h > 30 or w > 30:
        return None

    palette_top = random.randint(1, h - k - 1)
    palette_entries = [(palette_top + i, palette_col, color) for i, color in enumerate(palette)]
    palette_coords = {(r, c) for r, c, _ in palette_entries}

    path_set = set(translated_path)
    endpoints = [translated_path[0], translated_path[-1]]
    marker_positions: list[tuple[int, int]] = []
    used_marker_cells: set[tuple[int, int]] = set()
    for endpoint in endpoints:
        choices = [pt for pt in _marker_candidates(endpoint, path_set, h, w, palette_coords) if pt not in used_marker_cells]
        if not choices:
            return None
        marker = random.choice(choices)
        marker_positions.append(marker)
        used_marker_cells.add(marker)

    low_rank, high_rank = sorted(random.sample(range(k), 2))
    if random.choice((False, True)):
        marker_colors = [palette[low_rank], palette[high_rank]]
        ordered_path = translated_path
    else:
        marker_colors = [palette[high_rank], palette[low_rank]]
        ordered_path = list(reversed(translated_path))

    segments = _split_segments(ordered_path)
    if len({len(seg) for seg in segments}) < 2:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in translated_path:
        grid[r, c] = GREY
    for r, c, color in palette_entries:
        grid[r, c] = color
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = color

    if not _validate_input(grid):
        return None

    expected = _render_expected((h, w), palette_entries, ordered_path, palette)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
