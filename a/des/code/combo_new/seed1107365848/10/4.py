from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_BG = 0
_ALL_COLORS = tuple(range(1, 10))
_CORNERS = ("tl", "tr", "bl", "br")
_MISSING_CELL = {
    "tl": (0, 0),
    "tr": (0, 2),
    "bl": (2, 0),
    "br": (2, 2),
}
_GRID_RANGES = {
    1: ((8, 13), (10, 16)),
    2: ((8, 15), (11, 18)),
    3: ((9, 17), (13, 21)),
    4: ((10, 18), (15, 24)),
}
_OBJECT_COUNT_RANGE = (1, 4)
_GEN_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return counts.most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            groups.append(pts)
    return groups


def _shape_cells(missing_corner: str) -> list[tuple[int, int]]:
    missing = _MISSING_CELL[missing_corner]
    return [(r, c) for r in range(3) for c in range(3) if (r, c) != missing]


def _axis_col_from_bbox(c0: int, c1: int, missing_corner: str) -> int:
    return c1 if missing_corner in ("tl", "bl") else c0


def _object_info(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> dict[str, object]:
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each object must occupy a 3x3 box"
    assert len(pts) == 8, "each object must contain 8 cells"

    rel = {(r - r0, c - c0) for r, c in pts}
    missing = [corner for corner, cell in _MISSING_CELL.items() if cell not in rel]
    assert len(missing) == 1, "each object must miss exactly one corner"
    missing_corner = missing[0]

    colors = Counter(int(grid[r, c]) for r, c in pts)
    assert bg not in colors
    assert len(colors) == 2, "each object must contain body color plus one marker color"
    body_color, body_count = colors.most_common(1)[0]
    assert body_count == 7, "body color must fill 7 cells"
    marker_items = [(color, count) for color, count in colors.items() if color != body_color]
    assert len(marker_items) == 1 and marker_items[0][1] == 1, "marker color must appear once"
    marker_color = marker_items[0][0]

    marker_pos = next((r, c) for r, c in pts if int(grid[r, c]) == marker_color)
    assert marker_pos != (r0 + 1, c0 + 1), "marker should sit on an arm, not at the center"

    return {
        "cells": pts,
        "cell_set": set(pts),
        "body_color": int(body_color),
        "marker_color": int(marker_color),
        "marker_pos": marker_pos,
        "missing_corner": missing_corner,
        "axis_col": _axis_col_from_bbox(c0, c1, missing_corner),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each missing-corner 3x3 L-shape across the edge opposite its missing corner and color the new half from its internal marker.

    Input structure:
    - The background is black or otherwise the dominant color.
    - Several separated foreground objects appear; each object fills a 3x3 box except for one missing corner, so it has 8 occupied cells.
    - Every object has one body color plus exactly one differently colored marker pixel inside the shape.

    Rule:
    - For each object, inspect which corner of its 3x3 box is missing.
    - If the missing corner is on the left, mirror the object horizontally across its rightmost column; if the missing corner is on the right, mirror it across its leftmost column.
    - Keep the original object unchanged, and paint only the newly exposed mirrored cells: use the marker color for the mirrored copy, except at the mirrored marker position where the original body color is used.

    Key insight:
    - The missing corner determines the geometric mirror axis, while the lone internal marker determines the mirrored copy's color role swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground object"

    out = g.copy()
    for pts in comps:
        info = _object_info(g, pts, bg)
        axis_col = int(info["axis_col"])
        marker_pos = info["marker_pos"]
        cell_set = info["cell_set"]
        body_color = int(info["body_color"])
        marker_color = int(info["marker_color"])

        for r, c in info["cells"]:
            rr, cc = r, 2 * axis_col - c
            if (rr, cc) in cell_set:
                continue
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "mirrored copy must stay in bounds"
            assert out[rr, cc] == bg, "objects should be spaced so mirrored copies do not overlap"
            out[rr, cc] = body_color if (r, c) == marker_pos else marker_color

    return out


def _marker_choices(missing_corner: str) -> list[tuple[int, int]]:
    axis_col = 2 if missing_corner in ("tl", "bl") else 0
    return [
        (r, c)
        for r, c in _shape_cells(missing_corner)
        if (r, c) != (1, 1) and c != axis_col
    ]


def _footprint_bbox(r0: int, c0: int, missing_corner: str) -> tuple[int, int, int, int]:
    if missing_corner in ("tl", "bl"):
        return r0, c0, r0 + 2, c0 + 4
    return r0, c0 - 2, r0 + 2, c0 + 2


def _candidate_positions(
    blocked: np.ndarray, missing_corner: str, h: int, w: int
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    c_start = 0 if missing_corner in ("tl", "bl") else 2
    c_stop = (w - 5) if missing_corner in ("tl", "bl") else (w - 3)

    for r0 in range(h - 2):
        for c0 in range(c_start, c_stop + 1):
            fr0, fc0, fr1, fc1 = _footprint_bbox(r0, c0, missing_corner)
            er0 = max(0, fr0 - 1)
            ec0 = max(0, fc0 - 1)
            er1 = min(h - 1, fr1 + 1)
            ec1 = min(w - 1, fc1 + 1)
            if not blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                positions.append((r0, c0))
    return positions


def _reserve(blocked: np.ndarray, r0: int, c0: int, missing_corner: str) -> None:
    h, w = blocked.shape
    fr0, fc0, fr1, fc1 = _footprint_bbox(r0, c0, missing_corner)
    er0 = max(0, fr0 - 1)
    ec0 = max(0, fc0 - 1)
    er1 = min(h - 1, fr1 + 1)
    ec1 = min(w - 1, fc1 + 1)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _draw_object(
    grid: np.ndarray,
    r0: int,
    c0: int,
    missing_corner: str,
    body_color: int,
    marker_color: int,
    marker_local: tuple[int, int],
) -> None:
    for dr, dc in _shape_cells(missing_corner):
        grid[r0 + dr, c0 + dc] = body_color
    mr, mc = marker_local
    grid[r0 + mr, c0 + mc] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*_OBJECT_COUNT_RANGE)
    (h0, h1), (w0, w1) = _GRID_RANGES[count]
    h = rng.randint(h0, h1)
    w = rng.randint(w0, w1)

    grid = np.full((h, w), _BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = list(_ALL_COLORS)
    rng.shuffle(colors)
    body_colors = colors[:count]
    marker_colors = colors[count : 2 * count]
    missing_corners = rng.sample(_CORNERS, count)
    rng.shuffle(missing_corners)

    placed = 0
    for i, missing_corner in enumerate(missing_corners):
        candidates = _candidate_positions(blocked, missing_corner, h, w)
        rng.shuffle(candidates)
        if not candidates:
            return None

        r0, c0 = candidates[0]
        marker_local = rng.choice(_marker_choices(missing_corner))
        _draw_object(
            grid,
            r0,
            c0,
            missing_corner,
            int(body_colors[i]),
            int(marker_colors[i]),
            marker_local,
        )
        _reserve(blocked, r0, c0, missing_corner)
        placed += 1

    if placed != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    in_fg = int(np.count_nonzero(grid != _BG))
    out_fg = int(np.count_nonzero(out != _BG))
    if out_fg != in_fg + 5 * count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
