from __future__ import annotations

import random

import numpy as np


BG = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
GENERATE_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(b[0] - a[0]), int(b[1] - a[1])


def _left_normal(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _right_normal(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _ordered_path(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    path_mask = g != int(bg)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert coords, "expected a non-background path"

    coord_set = set(coords)
    degree: dict[tuple[int, int], int] = {}
    edge_endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        deg = sum((rr, cc) in coord_set for rr, cc in _neighbors4(r, c, h, w))
        assert deg in (1, 2), "path must be a single-width simple chain"
        degree[(r, c)] = deg
        if deg == 1 and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
            edge_endpoints.append((r, c))

    endpoints = [pt for pt, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "expected exactly two endpoints"
    assert len(edge_endpoints) == 2, "both endpoints must touch the grid edge"
    start = min(edge_endpoints)

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        for rr, cc in _neighbors4(cur[0], cur[1], h, w):
            if (rr, cc) not in coord_set or (rr, cc) == prev:
                continue
            nxts.append((rr, cc))
        if not nxts:
            break
        assert len(nxts) == 1, "path branches are not allowed"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "path must be connected"
    return ordered


def _color_segments(
    path: list[tuple[int, int]], grid: np.ndarray
) -> list[tuple[int, int, int]]:
    colors = [int(grid[r, c]) for r, c in path]
    segments: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or colors[i] != colors[start]:
            segments.append((start, i - 1, colors[start]))
            start = i
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a multi-color path into color segments and grow a perpendicular ray from each segment based on its length and turn direction.

    Input structure:
    - The background is orange.
    - A single 4-connected, one-pixel-wide path runs from one grid edge to another.
    - Each color appears in one contiguous path segment and neighboring segments have different colors.

    Rule:
    - Traverse the path from the edge endpoint that is lexicographically smallest.
    - For each non-final color segment, let N be its pixel count and inspect the turn at the segment end.
    - From the segment's middle pixel, cast a ray of length N to the side corresponding to that left/right turn.
    - Stop the ray early at the grid boundary or before any non-background cell. Keep the original path unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    path = _ordered_path(g, bg)
    segments = _color_segments(path, g)

    out = g.copy()
    for start, end, color in segments:
        if end == len(path) - 1:
            continue

        seg_len = end - start + 1
        if end > start:
            in_dir = _direction(path[end - 1], path[end])
        elif start > 0:
            in_dir = _direction(path[start - 1], path[start])
        else:
            in_dir = _direction(path[start], path[start + 1])
        out_dir = _direction(path[end], path[end + 1])

        cross = in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0]
        assert cross != 0, "each non-final segment must end at a 90-degree turn"
        ray_dir = _left_normal(in_dir) if cross > 0 else _right_normal(in_dir)

        mid_r, mid_c = path[start + seg_len // 2]
        rr, cc = mid_r, mid_c
        for _ in range(seg_len):
            rr += ray_dir[0]
            cc += ray_dir[1]
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            if int(out[rr, cc]) != int(bg):
                break
            out[rr, cc] = int(color)

    return out


def _sample_parity_values(lo: int, hi: int, parity: int, count: int) -> list[int] | None:
    vals = [x for x in range(lo, hi + 1) if x % 2 == parity]
    if len(vals) < count:
        return None
    random.shuffle(vals)
    vals = sorted(vals[:count])
    return vals


def _build_snake_path() -> tuple[int, int, list[tuple[int, int]]] | None:
    h = random.randint(13, 22)
    w = random.randint(13, 22)
    horiz_count = random.choice([3, 5] if h >= 19 else [3])
    start_east = random.choice([True, False])
    go_down = random.choice([True, False])

    edge_col = 0 if start_east else w - 1
    parity = edge_col % 2

    margin = 2
    min_gap = 4
    if go_down:
        base_lo = margin
        base_hi = h - 1 - margin - min_gap * (horiz_count - 1)
        if base_hi < base_lo:
            return None
        base = random.randint(base_lo, base_hi)
        rows = [base + min_gap * i for i in range(horiz_count)]
    else:
        base_lo = margin + min_gap * (horiz_count - 1)
        base_hi = h - 1 - margin
        if base_hi < base_lo:
            return None
        base = random.randint(base_lo, base_hi)
        rows = [base - min_gap * i for i in range(horiz_count)]

    if len({r % 2 for r in rows}) != 1:
        return None

    left_cols = [c for c in range(2, max(2, w // 3) + 1) if c % 2 == parity]
    right_cols = [c for c in range(min(w - 3, (2 * w) // 3), w - 1) if c % 2 == parity]
    if not left_cols or not right_cols:
        return None

    connectors: list[int] = []
    for i in range(horiz_count - 1):
        choices = right_cols if (start_east == (i % 2 == 0)) else left_cols
        if not choices:
            return None
        connectors.append(random.choice(choices))

    path: list[tuple[int, int]] = []
    cur_r = rows[0]
    cur_c = edge_col
    path.append((cur_r, cur_c))

    for i in range(horiz_count):
        if i == 0:
            seg_end_c = connectors[0]
        elif i == horiz_count - 1:
            seg_end_c = w - 1 if (start_east == (i % 2 == 0)) else 0
        else:
            seg_end_c = connectors[i]

        step_c = 1 if seg_end_c > cur_c else -1
        for cc in range(cur_c + step_c, seg_end_c + step_c, step_c):
            path.append((cur_r, cc))
        cur_c = seg_end_c

        if i == horiz_count - 1:
            break

        next_r = rows[i + 1]
        step_r = 1 if next_r > cur_r else -1
        for rr in range(cur_r + step_r, next_r + step_r, step_r):
            path.append((rr, cur_c))
        cur_r = next_r

    if len(path) != len(set(path)):
        return None

    row_min = min(r for r, _ in path)
    row_max = max(r for r, _ in path)
    col_min = min(c for _, c in path)
    col_max = max(c for _, c in path)
    if row_min <= 0 and row_max >= h - 1:
        return None
    if col_min <= 0 and col_max >= w - 1 and horiz_count == 5:
        # Long snakes get cramped if they span the full box in both axes.
        return None

    return h, w, path


def _render_path(h: int, w: int, path: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)

    if min(path[0], path[-1]) == path[-1]:
        path = list(reversed(path))

    turns = sum(
        _direction(path[i - 1], path[i]) != _direction(path[i], path[i + 1])
        for i in range(1, len(path) - 1)
    )
    segment_count = turns + 1
    colors = random.sample(NON_BG_COLORS, segment_count)

    color_idx = 0
    grid[path[0]] = colors[color_idx]
    for i in range(1, len(path)):
        if 1 <= i - 1 < len(path) - 1:
            prev_dir = _direction(path[i - 2], path[i - 1]) if i >= 2 else None
            cur_dir = _direction(path[i - 1], path[i])
            if prev_dir is not None and prev_dir != cur_dir:
                color_idx += 1
        grid[path[i]] = colors[color_idx]

    return grid


def _count_turn_signs(grid: np.ndarray) -> tuple[int, int]:
    bg = _most_common_color(grid)
    path = _ordered_path(grid, bg)
    segments = _color_segments(path, grid)
    left = 0
    right = 0
    for start, end, _color in segments:
        if end == len(path) - 1:
            continue
        in_dir = _direction(path[end - 1], path[end]) if end > start else _direction(path[start - 1], path[start])
        out_dir = _direction(path[end], path[end + 1])
        cross = in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0]
        if cross > 0:
            left += 1
        elif cross < 0:
            right += 1
    return left, right


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        built = _build_snake_path()
        if built is None:
            continue
        h, w, path = built
        grid = _render_path(h, w, path)
        out = main(grid)

        extra = int(np.sum(out != grid))
        left, right = _count_turn_signs(grid)
        segments = _color_segments(_ordered_path(grid, BG), grid)
        seg_lengths = [end - start + 1 for start, end, _color in segments]

        if extra < 6:
            continue
        if min(seg_lengths) < 3:
            continue
        if left == 0 or right == 0:
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
