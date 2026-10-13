from __future__ import annotations

from typing import Iterable

import numpy as np


BACKGROUND = 0
LEGEND_HEIGHT = 3
LEGEND_RECT_W = 3
LEGEND_CROSS_W = 3
LEGEND_L_W = 2
GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _find_legend_start(grid: np.ndarray) -> int:
    rows_with_fg = np.any(grid != BACKGROUND, axis=1)
    r = grid.shape[0] - 1
    while r >= 0 and not rows_with_fg[r]:
        r -= 1
    if r < 0:
        return grid.shape[0]
    while r >= 0 and rows_with_fg[r]:
        r -= 1
    return r + 1


def _classify_bbox(height: int, width: int) -> str:
    if abs(height - width) <= 1:
        return "cross"
    if width >= height + 2:
        return "rect"
    return "l"


def _shape_cells(kind: str, n: int) -> tuple[int, int, list[tuple[int, int]]]:
    if kind == "rect":
        cells = [(r, c) for r in range(2) for c in range(n)]
        return 2, n, cells

    if kind == "cross":
        radius = n - 2
        size = 2 * radius + 1
        center = radius
        cells = [(center, c) for c in range(size)]
        cells.extend((r, center) for r in range(size) if r != center)
        return size, size, cells

    if kind == "l":
        # Height N and width 2 preserves the 4-pixel exemplar when N == 3.
        cells = [(r, 0) for r in range(n)]
        cells.append((n - 1, 1))
        return n, 2, cells

    raise ValueError(f"unknown kind: {kind}")


def _centered_top_left(center_r2: int, center_c2: int, height: int, width: int) -> tuple[int, int]:
    top = (center_r2 - (height - 1)) // 2
    left = (center_c2 - (width - 1)) // 2
    return int(top), int(left)


def _paint_cells(grid: np.ndarray, top: int, left: int, cells: Iterable[tuple[int, int]], color: int, row_limit: int) -> None:
    h, w = grid.shape
    for dr, dc in cells:
        rr = top + int(dr)
        cc = left + int(dc)
        if 0 <= rr < min(h, row_limit) and 0 <= cc < w:
            grid[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_start = _find_legend_start(g)
    upper = g[:legend_start]

    out = g.copy()
    out[:legend_start] = BACKGROUND

    colors = [int(color) for color in np.unique(upper) if int(color) != BACKGROUND]
    for color in colors:
        coords = np.argwhere(upper == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        count = int(coords.shape[0])

        kind = _classify_bbox(height, width)
        shape_h, shape_w, cells = _shape_cells(kind, count)
        top, left = _centered_top_left(r0 + r1, c0 + c1, shape_h, shape_w)
        _paint_cells(out, top, left, cells, color, legend_start)

    return out


def _rect_scatter_cells(n: int) -> tuple[int, int, list[tuple[int, int]]]:
    width = 2 * n - 1 if n % 2 == 1 else 2 * n
    cols = [2 * i for i in range(n)]
    cols[-1] = width - 1
    cells = [(i % 2, cols[i]) for i in range(n)]
    return 2, width, cells


def _cross_scatter_cells(n: int, rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int]]]:
    size = 2 * n - 3
    corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
    corner_ids = list(range(4))
    rng.shuffle(corner_ids)
    if n == 3:
        chosen = [corners[i] for i in corner_ids[:3]]
        return size, size, chosen

    chosen = list(corners)
    lattice = [(r, c) for r in range(0, size, 2) for c in range(0, size, 2)]
    extra = [cell for cell in lattice if cell not in corners]
    rng.shuffle(extra)
    chosen.extend(extra[: n - 4])
    return size, size, chosen


def _l_scatter_cells(n: int) -> tuple[int, int, list[tuple[int, int]]]:
    height = 2 * n - 1 if n % 2 == 1 else 2 * n
    rows = [2 * i for i in range(n)]
    rows[-1] = height - 1
    cells = [(rows[i], i % 2) for i in range(n)]
    return height, 2, cells


def _scatter_cells(kind: str, n: int, rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int]]]:
    if kind == "rect":
        return _rect_scatter_cells(n)
    if kind == "cross":
        return _cross_scatter_cells(n, rng)
    if kind == "l":
        return _l_scatter_cells(n)
    raise ValueError(f"unknown kind: {kind}")


def _place_safe_boxes(specs: list[dict], upper_h: int, width: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((upper_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["safe_h"] * specs[i]["safe_w"], reverse=True)

    for idx in order:
        safe_h = int(specs[idx]["safe_h"])
        safe_w = int(specs[idx]["safe_w"])
        pad = int(specs[idx]["pad"])
        placed = None
        for _ in range(300):
            top = int(rng.integers(0, upper_h - safe_h + 1))
            left = int(rng.integers(0, width - safe_w + 1))
            r0 = max(0, top - pad)
            r1 = min(upper_h, top + safe_h + pad)
            c0 = max(0, left - pad)
            c1 = min(width, left + safe_w + pad)
            if occupancy[r0:r1, c0:c1].any():
                continue
            occupancy[top : top + safe_h, left : left + safe_w] = True
            placed = (top, left)
            break
        if placed is None:
            return None
        placements[idx] = placed

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _legend_min_width(gap1: int, gap2: int) -> int:
    return LEGEND_RECT_W + gap1 + LEGEND_CROSS_W + gap2 + LEGEND_L_W


def _draw_legend(grid: np.ndarray, legend_start: int, legend_colors: tuple[int, int, int], left: int, gap1: int, gap2: int) -> None:
    rect_color, cross_color, l_color = map(int, legend_colors)

    rect_top = legend_start
    rect_left = left
    grid[rect_top : rect_top + 2, rect_left : rect_left + 3] = rect_color

    cross_top = legend_start
    cross_left = rect_left + LEGEND_RECT_W + gap1
    cross_cells = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    for dr, dc in cross_cells:
        grid[cross_top + dr, cross_left + dc] = cross_color

    l_top = legend_start
    l_left = cross_left + LEGEND_CROSS_W + gap2
    l_cells = [(0, 0), (1, 0), (2, 0), (2, 1)]
    for dr, dc in l_cells:
        grid[l_top + dr, l_left + dc] = l_color


def _spec_signature(spec: dict) -> tuple[int, int, str]:
    return int(spec["safe_h"]), int(spec["safe_w"]), str(spec["kind"])


def _validate_input(grid: np.ndarray, specs: list[dict]) -> bool:
    legend_start = _find_legend_start(grid)
    if legend_start >= grid.shape[0]:
        return False
    if legend_start == 0:
        return False
    if np.any(grid[legend_start - 1] != BACKGROUND):
        return False

    upper = grid[:legend_start]
    used_colors = {int(spec["color"]) for spec in specs}
    if used_colors != {int(color) for color in np.unique(upper) if int(color) != BACKGROUND}:
        return False

    for spec in specs:
        color = int(spec["color"])
        coords = np.argwhere(upper == color)
        if int(coords.shape[0]) != int(spec["n"]):
            return False
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if _classify_bbox(height, width) != str(spec["kind"]):
            return False
        if any(
            max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1
            for i, a in enumerate(coords)
            for b in coords[i + 1 :]
        ):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG

    object_count = int(rng.integers(3, 6))
    kinds = ["rect", "cross", "l"]
    while len(kinds) < object_count:
        kinds.append(str(rng.choice(["rect", "cross", "l"])))
    rng.shuffle(kinds)

    counts = [int(rng.integers(3, 8)) for _ in kinds]

    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    object_colors = all_colors[:object_count]
    legend_colors = tuple(all_colors[object_count : object_count + 3])

    specs: list[dict] = []
    total_area = 0
    max_safe_h = 0
    max_safe_w = 0
    for kind, n, color in zip(kinds, counts, object_colors):
        out_h, out_w, _out_cells = _shape_cells(kind, n)
        in_h, in_w, in_cells = _scatter_cells(kind, n, rng)
        safe_h = max(out_h, in_h)
        safe_w = max(out_w, in_w)
        spec = {
            "kind": kind,
            "n": int(n),
            "color": int(color),
            "out_h": int(out_h),
            "out_w": int(out_w),
            "in_h": int(in_h),
            "in_w": int(in_w),
            "in_cells": in_cells,
            "safe_h": int(safe_h),
            "safe_w": int(safe_w),
            "pad": 2,
        }
        specs.append(spec)
        total_area += (safe_h + 4) * (safe_w + 4)
        max_safe_h = max(max_safe_h, safe_h)
        max_safe_w = max(max_safe_w, safe_w)

    gap1 = int(rng.integers(2, 5))
    gap2 = int(rng.integers(2, 5))
    legend_core_w = _legend_min_width(gap1, gap2)

    gap_rows = int(rng.integers(1, 4))
    upper_h_low = max(max_safe_h + 2, int(np.ceil(np.sqrt(total_area) * 0.8)))
    upper_h = int(rng.integers(upper_h_low, upper_h_low + 8))
    width_low = max(
        legend_core_w + 4,
        max_safe_w + 4,
        int(np.ceil(total_area / max(upper_h - 1, 1))) + 4,
    )
    width = int(rng.integers(width_low, width_low + 11))
    height = upper_h + gap_rows + LEGEND_HEIGHT

    placements = _place_safe_boxes(specs, upper_h, width, rng)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)

    for spec, (safe_top, safe_left) in zip(specs, placements):
        center_r2 = 2 * safe_top + int(spec["safe_h"]) - 1
        center_c2 = 2 * safe_left + int(spec["safe_w"]) - 1
        in_top, in_left = _centered_top_left(center_r2, center_c2, int(spec["in_h"]), int(spec["in_w"]))
        for dr, dc in spec["in_cells"]:
            grid[in_top + dr, in_left + dc] = int(spec["color"])

    legend_start = upper_h + gap_rows
    legend_left_max = width - legend_core_w
    legend_left = int(rng.integers(1, max(2, legend_left_max)))
    _draw_legend(grid, legend_start, legend_colors, legend_left, gap1, gap2)

    if not _validate_input(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
