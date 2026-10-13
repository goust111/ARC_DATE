from __future__ import annotations

from collections import deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _extract_legend(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, int]], list[int], str]:
    h, w = grid.shape
    corners = [
        ("TL", 0, 0, "left"),
        ("TR", 0, w - 1, "right"),
        ("BL", h - 1, 0, "left"),
        ("BR", h - 1, w - 1, "right"),
    ]
    active = [item for item in corners if int(grid[item[1], item[2]]) != bg]
    assert len(active) == 1, "expected exactly one corner legend"
    corner, row, _col, side = active[0]

    coords: list[tuple[int, int]] = []
    if side == "left":
        c = 0
        while c < w and int(grid[row, c]) != bg:
            coords.append((row, c))
            c += 1
    else:
        c = w - 1
        while c >= 0 and int(grid[row, c]) != bg:
            coords.append((row, c))
            c -= 1

    ordered = sorted(coords, key=lambda rc: rc[1])
    assert len(ordered) >= 2, "legend must contain at least two colors"
    colors = [int(grid[r, c]) for r, c in ordered]
    return ordered, colors, corner


def _distance_map(path_mask: np.ndarray) -> np.ndarray:
    h, w = path_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))

    assert q, "path must be non-empty"

    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        if r > 0 and dist[r - 1, c] == -1:
            dist[r - 1, c] = nd
            q.append((r - 1, c))
        if r + 1 < h and dist[r + 1, c] == -1:
            dist[r + 1, c] = nd
            q.append((r + 1, c))
        if c > 0 and dist[r, c - 1] == -1:
            dist[r, c - 1] = nd
            q.append((r, c - 1))
        if c + 1 < w and dist[r, c + 1] == -1:
            dist[r, c + 1] = nd
            q.append((r, c + 1))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the path map into a distance heatmap guided by the corner legend.

    Input structure:
    - The background is grey and dominant.
    - Blue path pixels form a one-pixel-wide 4-connected path.
    - A horizontal color legend sits flush in exactly one corner.

    Rule:
    - Keep the path and legend unchanged.
    - Recolor each background cell by its 4-neighbor distance to the nearest
      path pixel, using legend index D-1 and clamping to the last legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)

    legend_coords, legend_colors, _corner = _extract_legend(g, bg)
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in legend_coords:
        legend_mask[r, c] = True

    path_mask = (g != bg) & ~legend_mask
    assert np.any(path_mask), "expected a non-empty path"
    path_colors = np.unique(g[path_mask])
    assert path_colors.size == 1, "path must be monochrome"

    dist = _distance_map(path_mask)
    out = g.copy()
    for r, c in np.argwhere(g == bg):
        rr = int(r)
        cc = int(c)
        d = int(dist[rr, cc])
        assert d >= 1, "background pixels must be at positive distance from the path"
        out[rr, cc] = int(legend_colors[min(d - 1, len(legend_colors) - 1)])

    return out


# --------------- generator ---------------

BG_COLOR = 5
PATH_COLOR = 1
LEGEND_POOL = (0, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
MIN_LEGEND_LEN = 2
MAX_PATH_SEGMENTS = 4
MAX_ATTEMPTS = 400


def _paint_vertical_path(mask: np.ndarray, cols: list[int], turn_rows: list[int]) -> None:
    h, _w = mask.shape
    start_r = 0
    for idx, col in enumerate(cols):
        end_r = turn_rows[idx] if idx < len(turn_rows) else h - 1
        mask[start_r : end_r + 1, col] = True
        if idx + 1 < len(cols):
            nxt = cols[idx + 1]
            mask[end_r, min(col, nxt) : max(col, nxt) + 1] = True
            start_r = end_r


def _paint_horizontal_path(mask: np.ndarray, rows: list[int], turn_cols: list[int]) -> None:
    _h, w = mask.shape
    start_c = 0
    for idx, row in enumerate(rows):
        end_c = turn_cols[idx] if idx < len(turn_cols) else w - 1
        mask[row, start_c : end_c + 1] = True
        if idx + 1 < len(rows):
            nxt = rows[idx + 1]
            mask[min(row, nxt) : max(row, nxt) + 1, end_c] = True
            start_c = end_c


def _make_path(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        segments = random.randint(1, min(MAX_PATH_SEGMENTS, w - 2))
        cols = random.sample(list(range(1, w - 1)), segments)
        turn_rows = sorted(random.sample(list(range(1, h - 1)), segments - 1))
        _paint_vertical_path(mask, cols, turn_rows)
    else:
        segments = random.randint(1, min(MAX_PATH_SEGMENTS, h - 2))
        rows = random.sample(list(range(1, h - 1)), segments)
        turn_cols = sorted(random.sample(list(range(1, w - 1)), segments - 1))
        _paint_horizontal_path(mask, rows, turn_cols)
    return mask


def _corner_length_options(
    path_mask: np.ndarray, maxdist: int, want_clamp: bool
) -> list[tuple[str, list[int]]]:
    h, w = path_mask.shape

    def left_limit(row: int) -> int:
        cols = np.flatnonzero(path_mask[row])
        if cols.size == 0:
            return w - 2
        return int(cols.min()) - 1

    def right_limit(row: int) -> int:
        cols = np.flatnonzero(path_mask[row])
        if cols.size == 0:
            return w - 2
        return w - 2 - int(cols.max())

    raw_limits = {
        "TL": left_limit(0),
        "TR": right_limit(0),
        "BL": left_limit(h - 1),
        "BR": right_limit(h - 1),
    }

    options: list[tuple[str, list[int]]] = []
    for corner, limit in raw_limits.items():
        max_len = min(int(limit), len(LEGEND_POOL))
        if max_len < MIN_LEGEND_LEN:
            continue
        lengths = []
        for length in range(MIN_LEGEND_LEN, max_len + 1):
            if want_clamp and maxdist > length:
                lengths.append(length)
            if not want_clamp and maxdist <= length:
                lengths.append(length)
        if lengths:
            options.append((corner, lengths))

    return options


def _place_legend(grid: np.ndarray, corner: str, colors: list[int]) -> None:
    h, w = grid.shape
    row = 0 if corner in ("TL", "TR") else h - 1
    if corner in ("TL", "BL"):
        start_c = 0
    else:
        start_c = w - len(colors)
    grid[row, start_c : start_c + len(colors)] = np.array(colors, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    path_mask = _make_path(h, w)
    dist = _distance_map(path_mask)
    maxdist = int(dist[~path_mask].max())

    want_clamp = random.random() < 0.5
    options = _corner_length_options(path_mask, maxdist, want_clamp)
    if not options:
        return None

    corner, lengths = random.choice(options)
    legend_len = random.choice(lengths)
    legend_colors = random.sample(list(LEGEND_POOL), legend_len)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[path_mask] = PATH_COLOR
    _place_legend(grid, corner, legend_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert np.any(grid == BG_COLOR), "input must contain background"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
