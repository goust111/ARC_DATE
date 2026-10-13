from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

LEGEND_BLOCK = np.array([[RED, BLUE], [YELLOW, ORANGE]], dtype=int)
MARKER_COLORS = [BLUE, RED, YELLOW]
SIDES = ("top", "bottom", "left", "right")


def _legend_bbox(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            if np.all(block != BG) and len(set(map(int, block.ravel()))) == 4:
                candidates.append((r, c))
    assert candidates, "expected one 2x2 legend block"
    candidates.sort(key=lambda rc: (-rc[0], rc[1]))
    return candidates[0]


def _swap_map(legend: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in range(2):
        left = int(legend[row, 0])
        right = int(legend[row, 1])
        mapping[left] = right
        mapping[right] = left
    return mapping


def _diamond_params(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(grid == GRAY)
    assert coords.size > 0, "expected a gray diamond frame"
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    assert r1 - r0 == c1 - c0, "diamond bbox must be square"
    assert (r1 - r0) % 2 == 0, "diamond bbox side must be odd"
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    radius = (r1 - r0) // 2
    manhattan = np.abs(rows - cr) + np.abs(cols - cc)
    assert np.all(manhattan == radius), "gray cells must form a single-pixel diamond frame"
    return cr, cc, radius, r0, c0


def _ray_specs(cr: int, cc: int, radius: int) -> dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    r0 = cr - radius
    r1 = cr + radius
    c0 = cc - radius
    c1 = cc + radius
    return {
        "top": ((r0 - 1, cc), (r0 + 1, cc), (1, 0)),
        "bottom": ((r1 + 1, cc), (r1 - 1, cc), (-1, 0)),
        "left": ((cr, c0 - 1), (cr, c0 + 1), (0, 1)),
        "right": ((cr, c1 + 1), (cr, c1 - 1), (0, -1)),
    }


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def main(grid: np.ndarray) -> np.ndarray:
    """Fire side rays into a gray diamond interior, then swap every ray color through the paired colors in the 2x2 legend.

    Input structure:
    - Black background contains one gray single-pixel-thick diamond frame.
    - Outside the diamond, 2-3 colored side markers sit at the top, bottom, left, or right launch positions.
    - Inside the diamond are 1-2 green blocker pixels, and a separate 2x2 legend block sits in the lower-left area.

    Rule:
    - Read the legend row-wise as bidirectional color pairs.
    - Each side marker launches a straight ray inward along the central row or column for that side.
    - Along that line, recolor only strict interior black cells until the ray reaches a green blocker or the gray frame boundary.
    - After all ray cells are identified, replace each filled ray color by its legend row-mate; blockers, frame, legend, markers, and untouched black cells stay unchanged.

    Key insight:
    - The visible interior colors are not the marker colors directly: the marker first determines where the ray fills, then the legend swaps that intermediate fill color to its paired output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_r, legend_c = _legend_bbox(g)
    swap = _swap_map(g[legend_r : legend_r + 2, legend_c : legend_c + 2])

    cr, cc, radius, _r0, _c0 = _diamond_params(g)
    rows, cols = np.indices(g.shape)
    interior = np.abs(rows - cr) + np.abs(cols - cc) < radius

    out = g.copy()
    for marker_pos, start_pos, step in _ray_specs(cr, cc, radius).values():
        mr, mc = marker_pos
        if not _inside(g.shape, mr, mc):
            continue
        marker_color = int(g[mr, mc])
        if marker_color not in swap:
            continue

        r, c = start_pos
        dr, dc = step
        while _inside(g.shape, r, c) and bool(interior[r, c]):
            cell = int(g[r, c])
            if cell == GREEN:
                break
            if cell == BG:
                out[r, c] = swap[marker_color]
            r += dr
            c += dc

    return out


def _diamond_frame(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.abs(rows - cr) + np.abs(cols - cc) == radius


def _marker_positions(cr: int, cc: int, radius: int) -> dict[str, tuple[int, int]]:
    return {side: spec[0] for side, spec in _ray_specs(cr, cc, radius).items()}


def _second_blocker_candidates(cr: int, cc: int, radius: int, active_sides: list[str]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for side in active_sides:
        if side == "top":
            candidates.extend((r, cc) for r in range(cr - radius + 2, cr))
        elif side == "bottom":
            candidates.extend((r, cc) for r in range(cr + 1, cr + radius - 1))
        elif side == "left":
            candidates.extend((cr, c) for c in range(cc - radius + 2, cc))
        else:
            candidates.extend((cr, c) for c in range(cc + 1, cc + radius - 1))
    return sorted(set(candidates))


def _valid_geometry_options() -> list[tuple[int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int]] = []
    for h in range(18, 23):
        for w in range(18, 23):
            for radius in (4, 5):
                row_lo = radius + 2
                row_hi = h - radius - 5
                col_lo = radius + 5
                col_hi = w - radius - 3
                if row_lo > row_hi or col_lo > col_hi:
                    continue
                for cr in range(row_lo, row_hi + 1):
                    for cc in range(col_lo, col_hi + 1):
                        options.append((h, w, radius, cr, cc))
    return options


GEOMETRY_OPTIONS = _valid_geometry_options()
GEN_RETRIES = 400


def _attempt() -> np.ndarray | None:
    h, w, radius, cr, cc = random.choice(GEOMETRY_OPTIONS)
    grid = np.zeros((h, w), dtype=int)
    grid[_diamond_frame(grid.shape, cr, cc, radius)] = GRAY
    grid[h - 2 : h, 0:2] = LEGEND_BLOCK

    marker_count = random.choice([2, 3])
    sides = random.sample(list(SIDES), marker_count)
    colors = random.sample(MARKER_COLORS, marker_count)
    for side, color in zip(sides, colors):
        r, c = _marker_positions(cr, cc, radius)[side]
        grid[r, c] = color

    blockers = {(cr, cc)}
    if random.random() < 0.5:
        candidates = [cell for cell in _second_blocker_candidates(cr, cc, radius, sides) if cell not in blockers]
        if candidates:
            blockers.add(random.choice(candidates))
    for r, c in blockers:
        grid[r, c] = GREEN

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
