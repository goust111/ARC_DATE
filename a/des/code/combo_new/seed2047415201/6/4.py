import math
import random
from itertools import combinations

import numpy as np


AZURE = 8
MAROON = 9
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    cells: list[tuple[int, int]] = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _foreground_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == AZURE:
                continue
            color = int(grid[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))
            arr = np.array(coords, dtype=int)
            center = (
                _round_half_up(float(arr[:, 0].mean())),
                _round_half_up(float(arr[:, 1].mean())),
            )
            components.append({"color": color, "coords": arr, "center": center})
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the rounded visual centers of the separated colored regions.

    Input structure:
    - The background is Azure.
    - There are three to five separated compact 4-connected foreground regions.
    - Each region is monochromatic; multiple separated regions may share a color.

    Rule:
    - For every foreground region, average its row coordinates and column
      coordinates, then round each average to the nearest integer center cell.
    - For every pair of region centers, trace the Bresenham one-pixel-wide line
      between them.
    - If the two endpoint regions have the same color, draw that line in that
      color; otherwise draw it in Maroon.
    - Draw only on cells that were Azure in the input, leaving all original
      foreground-region pixels unchanged.

    Key insight:
    The visible objects define geometric center anchors, and the connector color
    is inferred from whether the two endpoint region colors match or differ.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == AZURE), "expected an Azure background"

    regions = _foreground_components(g)
    assert 3 <= len(regions) <= 5, "expected three to five foreground regions"

    original_foreground = g != AZURE
    out = g.copy()

    for i, j in combinations(range(len(regions)), 2):
        a = regions[i]
        b = regions[j]
        line_color = a["color"] if a["color"] == b["color"] else MAROON
        for r, c in _bresenham(a["center"], b["center"]):
            if not original_foreground[r, c]:
                out[r, c] = line_color

    return out


# ---------------- generator ----------------

MIN_GRID_SIZE = 15
MAX_GRID_SIZE = 27
MIN_CENTER_SPAN = 8
CENTER_MARGIN = 3
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 7]
REGION_COUNTS = (3, 4, 5)
SHAPE_SIZE_MIN = 5
SHAPE_SIZE_MAX = 11
SHAPE_RADIUS = 2
SHAPE_RETRIES = 80
GENERATION_RETRIES = 600


def _offset_center_is_origin(offsets: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return _round_half_up(sum(rows) / len(rows)) == 0 and _round_half_up(sum(cols) / len(cols)) == 0


def _random_shape_offsets() -> list[tuple[int, int]]:
    for _ in range(SHAPE_RETRIES):
        target = random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        cells = {(0, 0)}
        tries = 0
        while len(cells) < target and tries < 200:
            tries += 1
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(NEIGHBORS4)
            nr, nc = r + dr, c + dc
            if max(abs(nr), abs(nc)) <= SHAPE_RADIUS:
                cells.add((nr, nc))
        if len(cells) != target:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue
        bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if len(cells) == bbox_area:
            continue
        if _offset_center_is_origin(cells):
            return sorted(cells)

    return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)]


def _sample_bounds(h: int, w: int) -> tuple[int, int, int, int] | None:
    if h < 2 * CENTER_MARGIN + MIN_CENTER_SPAN + 1:
        return None
    if w < 2 * CENTER_MARGIN + MIN_CENTER_SPAN + 1:
        return None
    r0 = random.randint(CENTER_MARGIN, h - CENTER_MARGIN - MIN_CENTER_SPAN - 1)
    r1 = random.randint(r0 + MIN_CENTER_SPAN, h - CENTER_MARGIN - 1)
    c0 = random.randint(CENTER_MARGIN, w - CENTER_MARGIN - MIN_CENTER_SPAN - 1)
    c1 = random.randint(c0 + MIN_CENTER_SPAN, w - CENTER_MARGIN - 1)
    return r0, c0, r1, c1


def _top_layout(n: int, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    rm = random.randint(r0 + 3, r1 - 3)
    cm = random.randint(c0 + 3, c1 - 3)
    if n == 3:
        return [(r0, c0), (r0, c1), (r1, cm)]
    if n == 4:
        return [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    return [(r0, c0), (r0, c1), (rm, c1), (r1, cm), (rm, c0)]


def _left_layout(n: int, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    rm = random.randint(r0 + 3, r1 - 3)
    cm = random.randint(c0 + 3, c1 - 3)
    if n == 3:
        return [(r0, c0), (r1, c0), (rm, c1)]
    if n == 4:
        return [(r0, c0), (r1, c0), (r1, c1), (r0, c1)]
    return [(r0, c0), (r1, c0), (r1, cm), (rm, c1), (r0, cm)]


def _sample_centers(h: int, w: int, n: int) -> list[tuple[int, int]] | None:
    box = _sample_bounds(h, w)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        centers = _top_layout(n, box)
        if side == "bottom":
            centers = [(r0 + r1 - r, c) for r, c in centers]
        return centers

    centers = _left_layout(n, box)
    if side == "right":
        centers = [(r, c0 + c1 - c) for r, c in centers]
    return centers


def _sample_colors(n: int) -> list[int]:
    colors = FOREGROUND_COLORS[:]
    random.shuffle(colors)
    same_color = colors.pop()
    return [same_color, same_color] + colors[: n - 2]


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _place_regions(
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    colors: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), AZURE, dtype=int)
    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for center, color in zip(centers, colors):
        offsets = _random_shape_offsets()
        cells = [(center[0] + dr, center[1] + dc) for dr, dc in offsets]
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in cells):
            return None
        if _touches_existing(cells, occupied):
            return None
        for r, c in cells:
            grid[r, c] = color
            occupied.add((r, c))
        placed.append({"center": center, "color": color, "cells": set(cells)})

    return grid, placed


def _line_demands_are_clear(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    owner = np.full((h, w), -1, dtype=int)
    for idx, region in enumerate(regions):
        for r, c in region["cells"]:
            owner[r, c] = idx

    demands: dict[tuple[int, int], set[int]] = {}
    pair_has_same = False
    pair_has_mixed = False

    for i, j in combinations(range(len(regions)), 2):
        color_i = int(regions[i]["color"])
        color_j = int(regions[j]["color"])
        line_color = color_i if color_i == color_j else MAROON
        drew_cell = False
        for r, c in _bresenham(regions[i]["center"], regions[j]["center"]):
            own = int(owner[r, c])
            if own not in (-1, i, j):
                return False
            if own == -1:
                demands.setdefault((r, c), set()).add(line_color)
                drew_cell = True
        if not drew_cell:
            return False
        if color_i == color_j:
            pair_has_same = True
        else:
            pair_has_mixed = True

    if not pair_has_same or not pair_has_mixed:
        return False
    return all(len(colors) == 1 for colors in demands.values())


def _centers_match_generated_regions(grid: np.ndarray, planned: list[dict]) -> bool:
    found = _foreground_components(grid)
    if len(found) != len(planned):
        return False
    planned_centers = sorted((region["center"], int(region["color"])) for region in planned)
    found_centers = sorted((region["center"], int(region["color"])) for region in found)
    return planned_centers == found_centers


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    n_regions = random.choice(REGION_COUNTS)
    centers = _sample_centers(h, w, n_regions)
    if centers is None:
        return None

    colors = _sample_colors(n_regions)
    candidate = _place_regions(h, w, centers, colors)
    if candidate is None:
        return None
    grid, regions = candidate

    if not _centers_match_generated_regions(grid, regions):
        return None
    if not _line_demands_are_clear(grid, regions):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    added = out != grid
    if not np.any(added & (out == MAROON)):
        return None
    if not np.any(added & (out == colors[0])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid center-connector input")
