from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Iterable
import importlib.util

import numpy as np

try:
    from arc_utils import Color, bbox_coords, colors_present, get_bg, neighbors4
except Exception:
    _UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _spec = importlib.util.spec_from_file_location("arc_utils", _UTILS_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot locate arc_utils")
    _arc_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_arc_utils)
    Color = _arc_utils.Color
    bbox_coords = _arc_utils.bbox_coords
    colors_present = _arc_utils.colors_present
    get_bg = _arc_utils.get_bg
    neighbors4 = _arc_utils.neighbors4


def _distances_from(starts: Iterable[tuple[int, int]], passable: np.ndarray) -> np.ndarray:
    """BFS multi-source distances on passable cells."""
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and passable[r, c]:
            dist[r, c] = 0
            q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    """Draw a filled rectangular border (top/bottom/left/right edges)."""
    r0, c0, r1, c1 = map(int, bbox)
    col = int(color)
    grid[r0, c0 : c1 + 1] = col
    grid[r1, c0 : c1 + 1] = col
    grid[r0 : r1 + 1, c0] = col
    grid[r0 : r1 + 1, c1] = col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete invisible marker-defined hollow rectangles and fill all shortest-path cells between two cyan endpoints in green.

    Input structure:
    - A black background with exactly two cyan endpoint pixels.
    - Additional groups of exactly three same-color marker pixels, with each group in its own color.
    - For each group, the three markers lie on the boundary of their shared rectangular bounding box.

    Rule:
    - For each marker group, reconstruct the full rectangular border and draw it with that group color (the rectangle is a wall).
    - Treat all reconstructed borders as blocked cells and find every passable 4-connected shortest-path cell between the two endpoint pixels.
    - Paint the path cells green, but keep both endpoint pixels cyan.

    Key insight:
    - The three visible markers per color define a unique rectangle; only by completing each rectangle first do the shortest-path constraints become well-defined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg == Color.BLACK
    non_bg = colors_present(g, exclude={bg})
    assert non_bg, "expected non-background colors"
    counts = Counter(map(int, g.flatten()))
    counts = {int(c): int(v) for c, v in counts.items() if c in set(non_bg) | {bg}}

    endpoint_candidates = [c for c in non_bg if int(counts.get(c, 0)) == 2]
    assert len(endpoint_candidates) == 1, "expected exactly one 2-pixel color as endpoints"
    endpoint_color = int(endpoint_candidates[0])

    other_colors = [c for c in non_bg if c != endpoint_color]
    assert other_colors, "expected at least one marker color"
    for c in other_colors:
        assert int(counts.get(c, 0)) == 3, "each marker color must have exactly 3 pixels"

    endpoints = np.argwhere(g == endpoint_color)
    assert endpoints.shape == (2, 2), "endpoints must be exactly two pixels"
    (r0, c0), (r1, c1) = endpoints.tolist()
    p0 = (int(r0), int(c0))
    p1 = (int(r1), int(c1))

    out = g.copy()
    for color in other_colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (3, 2), "each marker color must have exactly three pixels"
        box = bbox_coords(coords)
        _r0, _c0, _r1, _c1 = map(int, box)
        on_border = (
            (coords[:, 0] == _r0)
            | (coords[:, 0] == _r1)
            | (coords[:, 1] == _c0)
            | (coords[:, 1] == _c1)
        )
        assert bool(np.all(on_border)), "marker pixels must lie on the marker rectangle border"
        _draw_rect_border(out, box, color)

    wall_mask = (out != bg) & (out != endpoint_color)
    passable = ~wall_mask
    dist0 = _distances_from([p0], passable)
    dist1 = _distances_from([p1], passable)
    shortest = int(dist0[p1[0], p1[1]])
    assert shortest >= 0, "endpoints must be connected after completing walls"

    shortest_mask = (dist0 >= 0) & (dist1 >= 0) & (dist0 + dist1 == shortest)
    out[(out == bg) & shortest_mask] = Color.GREEN
    out[p0[0], p0[1]] = endpoint_color
    out[p1[0], p1[1]] = endpoint_color
    return out


# ---------------------------- generator ----------------------------

BG = Color.BLACK
ENDPOINT_COLOR = Color.TEAL
PATH_COLOR = Color.GREEN
MARKER_COLOR_POOL = (
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.MAROON,
)

_RNG = np.random.default_rng()


def _rand_int(low: int, high: int) -> int:
    return int(_RNG.integers(low, high + 1))


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    a0, b0, a1, b1 = map(int, a)
    c0, d0, c1, d1 = map(int, b)
    return not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0)


def _point_in_bbox(p: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = map(int, p)
    r0, c0, r1, c1 = map(int, bbox)
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells: list[tuple[int, int]] = []
    for col in range(c0, c1 + 1):
        cells.append((r0, col))
        cells.append((r1, col))
    for row in range(r0 + 1, r1):
        cells.append((row, c0))
        cells.append((row, c1))
    uniq = list(dict.fromkeys(cells))
    return uniq


def _complete_rectangles(markers_only: np.ndarray, groups: list[dict]) -> np.ndarray:
    """Rebuild every marked rectangle to wall form."""
    out = markers_only.copy()
    for g in groups:
        _draw_rect_border(out, g["bbox"], int(g["color"]))
    return out


def _shortest_union_mask(grid: np.ndarray, endpoints: list[tuple[int, int]]) -> tuple[int, np.ndarray]:
    (r0, c0), (r1, c1) = endpoints
    wall_mask = (grid != BG) & (grid != ENDPOINT_COLOR) & (grid != PATH_COLOR)
    passable = ~wall_mask
    d0 = _distances_from([(r0, c0)], passable)
    d1 = _distances_from([(r1, c1)], passable)
    if d0[r1, c1] < 0:
        return -1, np.zeros_like(grid, dtype=bool)
    shortest = int(d0[r1, c1])
    mask = (d0 >= 0) & (d1 >= 0) & (d0 + d1 == shortest)
    return shortest, mask


def generate_input() -> np.ndarray:
    """
    随机生成满足三点定义框与最短路规律的新题输入。
    """
    for _ in range(400):
        h = _rand_int(12, 30)
        w = _rand_int(12, 30)
        canvas = np.full((h, w), BG, dtype=int)

        # 2 cyan endpoints (same color, two pixels, not too trivial).
        min_manhattan = 5
        flat_count = h * w
        a = int(_RNG.integers(0, flat_count))
        b = int(_RNG.integers(0, flat_count))
        if a == b:
            continue
        p0 = divmod(a, w)
        p1 = divmod(b, w)
        if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < min_manhattan:
            continue

        marker_count = int(_RNG.integers(1, 6))
        colors = list(MARKER_COLOR_POOL)
        if len(colors) < marker_count:
            continue
        selected_colors = _RNG.choice(colors, size=marker_count, replace=False)

        groups: list[dict] = []
        occupied: list[tuple[int, int, int, int]] = []
        failed = False

        for col in map(int, selected_colors):
            if failed:
                break

            rect_success = False
            for _ in range(250):
                box_h = _rand_int(2, max(2, min(10, h - 1)))
                box_w = _rand_int(2, max(2, min(10, w - 1)))
                if box_h > h or box_w > w:
                    continue

                r0 = _rand_int(0, h - box_h)
                c0 = _rand_int(0, w - box_w)
                r1 = r0 + box_h - 1
                c1 = c0 + box_w - 1
                bbox = (r0, c0, r1, c1)

                if _point_in_bbox(p0, bbox) or _point_in_bbox(p1, bbox):
                    continue
                if any(_bbox_intersects(bbox, other) for other in occupied):
                    continue

                candidates = _rect_border_cells(bbox)
                if len(candidates) < 3:
                    continue
                # ensure three distinct markers, and at least one corner is missing for visual clarity.
                idx = sorted(set(_RNG.choice(len(candidates), size=3, replace=False).tolist()))
                marker_cells = [candidates[i] for i in idx]
                if len(set(marker_cells)) != 3:
                    continue

                groups.append({"color": int(col), "bbox": bbox, "cells": marker_cells})
                occupied.append(bbox)
                rect_success = True
                break

            if not rect_success:
                failed = True
                break

        if failed:
            continue

        # Render markers only.
        marker_only = canvas.copy()
        marker_only[p0[0], p0[1]] = ENDPOINT_COLOR
        marker_only[p1[0], p1[1]] = ENDPOINT_COLOR
        for g in groups:
            rr, cc = zip(*g["cells"])
            marker_only[list(rr), list(cc)] = int(g["color"])

        completed = _complete_rectangles(marker_only, groups)
        shortest, path_mask = _shortest_union_mask(completed, [tuple(p0), tuple(p1)])
        if shortest < 0:
            continue

        manhattan = abs(int(p0[0]) - int(p1[0])) + abs(int(p0[1]) - int(p1[1]))
        force_detour = bool(_RNG.random() < 0.65)
        if force_detour and shortest <= manhattan:
            continue

        # Keep the generated path visually bounded and non-degenerate.
        if int(np.count_nonzero(path_mask)) <= 2:
            continue
        if int(np.count_nonzero(path_mask)) > h * w // 2:
            continue

        # Sanity-check that this input indeed satisfies parser assumptions used in main.
        try:
            out = main(marker_only)
        except Exception:
            continue
        if not np.any(out == PATH_COLOR):
            continue

        return marker_only

    raise RuntimeError("failed to generate a valid 1/3/017 input")
