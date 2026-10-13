from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
AZURE = 8
MAROON = 9
LEGEND_COLORS = (1, 2, 3, 4, 6, 7)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _separator_col(grid: np.ndarray) -> int:
    scores = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        scores.append((int(np.sum(col == GRAY)), c))
    best, col = max(scores)
    if best == 0:
        raise ValueError("missing gray divider")
    return int(col)


def _legend_entries(left: np.ndarray) -> list[tuple[int, int]]:
    entries: list[tuple[int, int]] = []
    r = 0
    while r < left.shape[0]:
        vals = [int(x) for x in left[r].tolist() if int(x) not in (BG, GRAY)]
        if not vals:
            r += 1
            continue
        color = Counter(vals).most_common(1)[0][0]
        length = int(sum(1 for x in left[r].tolist() if int(x) == color))
        entries.append((int(color), length))
        r += 1
    if not entries:
        raise ValueError("missing legend bars")
    return entries


def _path_order(canvas: np.ndarray) -> list[tuple[int, int]]:
    mask = (canvas == AZURE) | (canvas == MAROON)
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return []
    h, w = canvas.shape
    path_set = {tuple(map(int, p)) for p in pts}
    endpoints = []
    for r, c in path_set:
        deg = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c, h, w))
        if deg <= 1 and (r in (0, h - 1) or c in (0, w - 1)):
            endpoints.append((r, c))
    if not endpoints:
        for r, c in path_set:
            if r in (0, h - 1) or c in (0, w - 1):
                endpoints.append((r, c))
    start = min(endpoints or list(path_set))
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if (nr, nc) in path_set and (nr, nc) != prev
        ]
        if not nxts:
            break
        unvisited = [p for p in nxts if p not in order]
        if not unvisited:
            break
        prev, cur = cur, sorted(unvisited)[0]
        order.append(cur)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color an azure path according to the stacked bar legend on the left.

    Input structure:
    - A black grid is split by one vertical gray divider.
    - Left of the divider, horizontal colored bars are stacked top to bottom; each bar's color and pixel length are the legend.
    - Right of the divider, a one-pixel-wide connected path is drawn in azure, with maroon marker pixels embedded along it.

    Rule:
    - Read legend bars from top to bottom.
    - Traverse the right-hand path from the endpoint touching the canvas boundary.
    - For each legend bar, recolor that many azure path pixels with the bar color; maroon marker pixels are skipped for counting and become black.
    - Any azure path pixels beyond the total legend length remain azure; the divider and legend panel stay unchanged.

    Key insight:
    - The visible maroon pixels are only markers, while the true segment lengths come from the bar lengths in the left legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep = _separator_col(g)
    entries = _legend_entries(g[:, :sep])
    canvas = g[:, sep + 1 :]
    order = _path_order(canvas)

    targets: list[int] = []
    for color, length in entries:
        targets.extend([int(color)] * int(length))

    idx = 0
    for r, c in order:
        gr, gc = r, sep + 1 + c
        if int(g[gr, gc]) == MAROON:
            out[gr, gc] = BG
            continue
        if int(g[gr, gc]) != AZURE:
            continue
        if idx < len(targets):
            out[gr, gc] = targets[idx]
        else:
            out[gr, gc] = AZURE
        idx += 1
    return out


def _snake_coords(h: int, w: int, orient: str) -> list[tuple[int, int]]:
    if orient in ("top", "bottom"):
        rows = range(h) if orient == "top" else range(h - 1, -1, -1)
        coords: list[tuple[int, int]] = []
        for i, r in enumerate(rows):
            cols = range(w) if i % 2 == 0 else range(w - 1, -1, -1)
            coords.extend((int(r), int(c)) for c in cols)
        return coords
    cols = range(w) if orient == "left" else range(w - 1, -1, -1)
    coords = []
    for i, c in enumerate(cols):
        rows = range(h) if i % 2 == 0 else range(h - 1, -1, -1)
        coords.extend((int(r), int(c)) for r in rows)
    return coords


def _build_case() -> np.ndarray | None:
    height = random.randint(12, 24)
    left_w = random.randint(5, 9)
    right_w = random.randint(9, 17)
    width = left_w + 1 + right_w
    if width > 30:
        return None

    n_bars = random.randint(2, min(5, height - 3))
    lengths = [random.randint(2, min(7, right_w)) for _ in range(n_bars)]
    colors = random.sample(list(LEGEND_COLORS), n_bars)
    total_color = sum(lengths)
    marker_count = random.randint(max(1, n_bars - 1), n_bars + 2)
    extra = random.randint(3, 12)
    needed = total_color + marker_count + extra
    if needed > height * right_w:
        return None

    grid = np.full((height, width), BG, dtype=int)
    sep = left_w
    grid[:, sep] = GRAY

    top = random.randint(0, height - n_bars)
    for i, (color, length) in enumerate(zip(colors, lengths)):
        start = random.randint(0, max(0, left_w - length))
        grid[top + i, start : start + length] = int(color)

    orient = random.choice(("top", "bottom", "left", "right"))
    path = _snake_coords(height, right_w, orient)[:needed]
    if len(path) < needed:
        return None

    marker_positions: set[int] = set()
    azure_seen = 0
    for length in lengths[:-1]:
        azure_seen += length
        pos = min(len(path) - 2, azure_seen + len(marker_positions))
        if pos > 0:
            marker_positions.add(pos)
    while len(marker_positions) < marker_count:
        pos = random.randint(1, len(path) - 2)
        marker_positions.add(pos)

    for i, (r, c) in enumerate(path):
        grid[r, sep + 1 + c] = MAROON if i in marker_positions else AZURE

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    changed = int(np.sum((grid == AZURE) & (out != AZURE)))
    if changed < total_color:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-coloring input")
