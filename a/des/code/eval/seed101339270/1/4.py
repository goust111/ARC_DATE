from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5


def _bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = set()
            while stack:
                rr, cc = stack.pop()
                pts.add((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    frames = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            for r1 in range(r0 + 3, min(h, r0 + 8)):
                for c1 in range(c0 + 3, min(w, c0 + 8)):
                    border = _rect_border(r0, c0, r1, c1)
                    if all(int(grid[r, c]) == GREY for r, c in border) and np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                        frames.append(((r0, c0, r1, c1), border))
    frames.sort(key=lambda item: ((item[0][2] - item[0][0]) * (item[0][3] - item[0][1])), reverse=True)
    kept = []
    used: set[tuple[int, int]] = set()
    for bbox, border in frames:
        if border & used:
            continue
        kept.append((bbox, border))
        used |= border
    return kept


def _adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in b:
                return True
    return False


def _source_components(grid: np.ndarray, bg: int) -> list[dict]:
    out = []
    for color in map(int, np.unique(grid)):
        if color in (bg, GREY):
            continue
        for comp in _components(grid, color):
            if len(comp) >= 4:
                out.append({"color": color, "coords": comp, "bbox": _bbox(comp)})
    return out


def _turn_count(path: set[tuple[int, int]]) -> int:
    turns = 0
    for r, c in path:
        dirs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in path:
                dirs.append((dr, dc))
        has_vertical = any(dc == 0 for _, dc in dirs)
        has_horizontal = any(dr == 0 for dr, _ in dirs)
        if has_vertical and has_horizontal:
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg(g)
    grey_comps = _components(g, GREY)
    sources = _source_components(g, bg)
    frames = _find_frames(g, bg)
    for (r0, c0, r1, c1), border in frames:
        comp = next((gc for gc in grey_comps if border & gc), None)
        if comp is None:
            continue
        src = next((s for s in sources if _adjacent(s["coords"], comp)), None)
        if src is None:
            continue
        pipe = set(comp) - border
        layers = min(r1 - r0 - 1, _turn_count(pipe))
        for rr in range(r1 - layers, r1):
            for cc in range(c0 + 1, c1):
                out[rr, cc] = int(src["color"])
    return out


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted([c0, c1])
        grid[r0, lo : hi + 1] = GREY
    elif c0 == c1:
        lo, hi = sorted([r0, r1])
        grid[lo : hi + 1, c0] = GREY


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = GREY
    grid[r0 + h - 1, c0 : c0 + w] = GREY
    grid[r0 : r0 + h, c0] = GREY
    grid[r0 : r0 + h, c0 + w - 1] = GREY


def generate_input() -> np.ndarray:
    group_count = random.randint(2, 4)
    h = 8 * group_count + 4
    w = random.randint(28, 34)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], group_count)
    for idx, color in enumerate(colors):
        top = 2 + idx * 8
        sr, sc = top + random.randint(2, 3), 2
        grid[sr : sr + 2, sc : sc + 2] = color
        fh, fw = random.randint(5, 6), random.randint(5, 7)
        fr = top + random.randint(1, 2)
        fc = random.randint(w - 10, w - fw - 2)
        _draw_frame(grid, fr, fc, fh, fw)
        start = (sr + 1, sc + 2)
        end = (fr + fh // 2, fc - 1)
        mid1 = random.randint(7, 11)
        mid2 = random.randint(14, fc - 2)
        bend_r = top + random.randint(1, 6)
        points = [start, (start[0], mid1), (bend_r, mid1), (bend_r, mid2), (end[0], mid2), end]
        for a, b in zip(points, points[1:]):
            _draw_segment(grid, a, b)
    return grid


if __name__ == "__main__":
    for _ in range(20):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
