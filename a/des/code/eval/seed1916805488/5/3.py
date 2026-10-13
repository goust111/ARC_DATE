import random
from collections import Counter

import numpy as np


GEN_RETRIES = 300
GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (12, 23)
INTERIOR_H_RANGE = (4, 10)
INTERIOR_W_RANGE = (4, 10)
RUN_COUNT_CHOICES = [3, 4, 5]
RUN_LEN_RANGE = (1, 4)


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _edge_runs(seq, bg: int):
    runs = []
    i = 0
    n = len(seq)
    while i < n:
        if int(seq[i]) == bg:
            i += 1
            continue
        color = int(seq[i])
        j = i
        while j < n and int(seq[j]) == color:
            j += 1
        runs.append((color, j - i, i, j))
        i = j
    return runs


def _extract_palette(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    edges = [
        ("top", [int(x) for x in grid[0, :]]),
        ("bottom", [int(x) for x in grid[h - 1, :]]),
        ("left", [int(x) for x in grid[:, 0]]),
        ("right", [int(x) for x in grid[:, w - 1]]),
    ]
    for name, seq in edges:
        runs = _edge_runs(seq, bg)
        if len(runs) < 2:
            continue
        colors = [c for c, _, _, _ in runs]
        widths = [k for _, k, _, _ in runs]
        if len(set(colors)) != len(colors):
            continue
        if not all(1 <= k <= 4 for k in widths):
            continue
        score = sum(widths)
        candidates.append((score, name, colors, widths))
    assert candidates, "failed to locate edge palette strip"
    _, edge, colors, widths = max(candidates)
    return edge, colors, widths


def _find_frame(grid: np.ndarray, bg: int):
    h, w = grid.shape
    best = None
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if interior.size == 0 or np.any(interior != bg):
                        continue
                    border_pts = []
                    for c in range(c0, c1 + 1):
                        border_pts.append((r0, c))
                        border_pts.append((r1, c))
                    for r in range(r0 + 1, r1):
                        border_pts.append((r, c0))
                        border_pts.append((r, c1))
                    vals = [int(grid[r, c]) for r, c in border_pts]
                    cnt = Counter(vals)
                    if bg in cnt:
                        continue
                    frame_color, frame_count = cnt.most_common(1)[0]
                    if frame_count != len(border_pts) - 1:
                        continue
                    marker_colors = [v for v, k in cnt.items() if v != frame_color and k == 1]
                    if len(marker_colors) != 1:
                        continue
                    marker_color = marker_colors[0]
                    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
                    marker_pts = [p for p in corners if int(grid[p]) == marker_color]
                    if len(marker_pts) != 1:
                        continue
                    area = (r1 - r0 - 1) * (c1 - c0 - 1)
                    cand = (area, r0, r1, c0, c1, frame_color, marker_color, marker_pts[0])
                    if best is None or cand > best:
                        best = cand
    assert best is not None, "failed to locate rectangular frame"
    _, r0, r1, c0, c1, frame_color, marker_color, marker_pt = best
    return r0, r1, c0, c1, frame_color, marker_color, marker_pt


def _spiral_path(height: int, width: int, corner_idx: int):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    starts = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    r, c = starts[corner_idx]
    d = corner_idx
    seen = np.zeros((height, width), dtype=bool)
    out = []
    for _ in range(height * width):
        out.append((r, c))
        seen[r, c] = True
        nr = r + dirs[d][0]
        nc = c + dirs[d][1]
        if not (0 <= nr < height and 0 <= nc < width) or seen[nr, nc]:
            d = (d + 1) % 4
            nr = r + dirs[d][0]
            nc = c + dirs[d][1]
        r, c = nr, nc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    _, colors, widths = _extract_palette(g, bg)
    r0, r1, c0, c1, _, _, marker_pt = _find_frame(g, bg)

    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    corner_idx = corners.index(marker_pt)
    path = _spiral_path(r1 - r0 - 1, c1 - c0 - 1, corner_idx)

    out = g.copy()
    color_idx = 0
    remain = widths[0]
    for rr, cc in path:
        out[r0 + 1 + rr, c0 + 1 + cc] = colors[color_idx]
        remain -= 1
        if remain == 0:
            color_idx = (color_idx + 1) % len(colors)
            remain = widths[color_idx]
    return out


def _choose_colors(rng: random.Random, count: int):
    palette = list(range(10))
    rng.shuffle(palette)
    return palette[:count]


def _paint_strip(grid: np.ndarray, edge: str, start: int, colors, widths):
    if edge == "top":
        pos = start
        for color, width in zip(colors, widths):
            grid[0, pos : pos + width] = color
            pos += width
    elif edge == "bottom":
        pos = start
        for color, width in zip(colors, widths):
            grid[-1, pos : pos + width] = color
            pos += width
    elif edge == "left":
        pos = start
        for color, width in zip(colors, widths):
            grid[pos : pos + width, 0] = color
            pos += width
    else:
        pos = start
        for color, width in zip(colors, widths):
            grid[pos : pos + width, -1] = color
            pos += width


def _gen_attempt(rng: random.Random):
    h = rng.randrange(*GRID_H_RANGE)
    w = rng.randrange(*GRID_W_RANGE)
    bg, frame_color, marker_color = _choose_colors(rng, 3)
    grid = np.full((h, w), bg, dtype=int)

    edge = rng.choice(["top", "bottom", "left", "right"])
    run_count = rng.choice(RUN_COUNT_CHOICES)
    widths = [rng.randint(*RUN_LEN_RANGE) for _ in range(run_count)]
    total_len = sum(widths)
    strip_colors = _choose_colors(rng, 3 + run_count)[3:]

    if edge in ("top", "bottom"):
        if total_len > w - 2:
            return None
        start = rng.randint(1, w - total_len - 1)
        _paint_strip(grid, edge, start, strip_colors, widths)
    else:
        if total_len > h - 2:
            return None
        start = rng.randint(1, h - total_len - 1)
        _paint_strip(grid, edge, start, strip_colors, widths)

    interior_h = rng.randint(*INTERIOR_H_RANGE)
    interior_w = rng.randint(*INTERIOR_W_RANGE)
    if interior_h + 2 >= h or interior_w + 2 >= w:
        return None

    row_min = 1
    row_max = h - interior_h - 2
    col_min = 1
    col_max = w - interior_w - 2
    if edge == "top":
        row_min = 2
    if edge == "bottom":
        row_max = h - interior_h - 3
    if edge == "left":
        col_min = 2
    if edge == "right":
        col_max = w - interior_w - 3
    if row_min > row_max or col_min > col_max:
        return None

    r0 = rng.randint(row_min, row_max)
    c0 = rng.randint(col_min, col_max)
    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1

    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    marker_idx = rng.randrange(4)
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    mr, mc = corners[marker_idx]
    grid[mr, mc] = marker_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(np.unique(out[r0 + 1 : r1, c0 + 1 : c1])) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
