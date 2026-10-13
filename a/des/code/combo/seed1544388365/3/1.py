from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (11, 24)
GRID_W_RANGE = (12, 30)
LEGEND_LEN_RANGE = (3, 7)
RECT_COUNT_RANGE = (1, 5)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 9)
GEN_RETRIES = 300
PLACE_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, row_limit: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((row_limit, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(row_limit):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < row_limit and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, int, int, list[int]]:
    h, w = grid.shape
    for r in range(h - 1, -1, -1):
        best: tuple[int, int, list[int]] | None = None
        c = 0
        while c < w:
            while c < w and int(grid[r, c]) == BG:
                c += 1
            start = c
            colors: list[int] = []
            while c < w and int(grid[r, c]) != BG:
                colors.append(int(grid[r, c]))
                c += 1
            if len(colors) >= 2 and len(set(colors)) == len(colors):
                if best is None or len(colors) > len(best[2]):
                    best = (start, c - 1, colors)
        if best is not None:
            return r, best[0], best[1], best[2]
    raise AssertionError("legend row not found")


def _find_frames(grid: np.ndarray, legend_row: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _color_components(grid, legend_row):
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        boundary = set()
        for c in range(c0, c1 + 1):
            boundary.add((r0, c))
            boundary.add((r1, c))
        for r in range(r0, r1 + 1):
            boundary.add((r, c0))
            boundary.add((r, c1))
        coord_set = set(coords)
        if coord_set != boundary:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_positions = np.argwhere(interior != BG)
        if marker_positions.shape[0] != 1:
            continue
        mr, mc = map(int, marker_positions[0])
        marker_color = int(interior[mr, mc])
        if marker_color == color:
            continue
        frames.append(
            {
                "frame_color": color,
                "bbox": (r0, c0, r1, c1),
                "marker_color": marker_color,
            }
        )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed region's interior with the legend-successor of its marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_row, _c0, _c1, legend_colors = _find_legend(g)
    successor = {legend_colors[i]: legend_colors[(i + 1) % len(legend_colors)] for i in range(len(legend_colors))}

    out = g.copy()
    frames = _find_frames(g, legend_row)
    assert frames, "expected at least one framed region"
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        marker_color = int(frame["marker_color"])
        assert marker_color in successor, "marker color must appear in the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = successor[marker_color]
    return out


def _sample_rect_count(rng: random.Random, h: int, w: int) -> int:
    low, high = RECT_COUNT_RANGE
    capacity = max(1, min(high, (h * w) // 55))
    return rng.randint(low, capacity)


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _place_rectangles(
    rng: random.Random,
    h: int,
    w: int,
    legend_row: int,
    rect_count: int,
) -> list[tuple[int, int, int, int]]:
    occupied = np.zeros((legend_row, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    max_bottom = legend_row - 2
    if max_bottom < 3:
        return []
    for _ in range(rect_count):
        placed = False
        for _ in range(PLACE_RETRIES):
            rh = rng.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], max_bottom + 1))
            rw = rng.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], w))
            max_r0 = max_bottom - rh + 1
            max_c0 = w - rw
            if max_r0 < 0 or max_c0 < 0:
                continue
            r0 = rng.randint(0, max_r0)
            c0 = rng.randint(0, max_c0)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return []
    return rects


def _marker_colors(rng: random.Random, legend_colors: list[int], count: int) -> list[int]:
    wrap_branch = rng.random() < 0.5
    non_wrap_pool = legend_colors[:-1] if len(legend_colors) > 1 else legend_colors
    colors = [rng.choice(non_wrap_pool if not wrap_branch else legend_colors) for _ in range(count)]
    if wrap_branch and count > 0:
        colors[rng.randrange(count)] = legend_colors[-1]
    return colors


def _paint_frame(
    grid: np.ndarray,
    rng: random.Random,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    marker_color: int,
) -> None:
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    mr = rng.randint(r0 + 1, r1 - 1)
    mc = rng.randint(c0 + 1, c1 - 1)
    grid[mr, mc] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    legend_row = h - 1
    legend_len = rng.randint(LEGEND_LEN_RANGE[0], min(LEGEND_LEN_RANGE[1], w))
    legend_start = rng.randint(0, w - legend_len)
    legend_colors = rng.sample(PALETTE_POOL, legend_len)

    rect_count = _sample_rect_count(rng, h, w)
    rects = _place_rectangles(rng, h, w, legend_row, rect_count)
    if len(rects) != rect_count:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[legend_row, legend_start : legend_start + legend_len] = legend_colors

    markers = _marker_colors(rng, legend_colors, rect_count)
    for (r0, c0, r1, c1), marker_color in zip(rects, markers):
        _paint_frame(grid, rng, r0, c0, r1, c1, marker_color)

    frames = _find_frames(grid, legend_row)
    if len(frames) != rect_count:
        return None
    if _find_legend(grid)[0] != legend_row:
        return None
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
