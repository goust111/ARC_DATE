from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _draw_connections(grid: np.ndarray) -> np.ndarray:
    markers = np.array(grid, dtype=int, copy=False)
    out = np.array(markers, dtype=int, copy=True)
    height, width = markers.shape

    for r in range(height):
        cols = sorted(int(c) for c in np.flatnonzero(markers[r] == RED))
        for c0, c1 in zip(cols, cols[1:]):
            if c1 > c0 + 1:
                out[r, c0 + 1 : c1] = RED

    for c in range(width):
        rows = sorted(int(r) for r in np.flatnonzero(markers[:, c] == RED))
        for r0, r1 in zip(rows, rows[1:]):
            if r1 > r0 + 1:
                out[r0 + 1 : r1, c] = RED

    return out


def _components(mask: np.ndarray) -> list[dict]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == height - 1 or cc == width - 1:
                    touches_border = True
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "touches_border": touches_border,
                }
            )

    return comps


def _transform_coords(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    transpose: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    height, width = map(int, shape)
    pts = [(int(r), int(c)) for r, c in coords]

    if transpose:
        pts = [(c, r) for r, c in pts]
        height, width = width, height

    if flip_rows:
        pts = [(height - 1 - r, c) for r, c in pts]
    if flip_cols:
        pts = [(r, width - 1 - c) for r, c in pts]

    pts = sorted(set(pts))
    return pts, (height, width)


def _render_markers(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in coords:
        grid[r, c] = RED
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned red markers, then recolor the unique smallest background region green.

    Input structure:
    - A black grid contains only sparse red marker pixels.
    - Some rows and columns contain multiple red markers, and these alignments define straight connections.
    - After drawing all row-wise and column-wise connections, the red network partitions the background into several 4-connected regions.

    Rule:
    - Keep every original red marker.
    - In each row, connect consecutive red markers with red cells; do the same in each column.
    - Find all 4-connected black background regions after the red network is drawn.
    - Recolor every cell of the unique smallest background region green and leave all other cells unchanged.

    Key insight:
    The red segments are not the final target; they act as walls whose global topology creates enclosed background regions, and only the smallest enclosed region is selected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED}), "input must contain only black background and red markers"

    connected = _draw_connections(g)
    bg_comps = _components(connected == BLACK)
    assert bg_comps, "expected at least one background region"

    min_area = min(comp["area"] for comp in bg_comps)
    smallest = [comp for comp in bg_comps if comp["area"] == min_area]
    assert len(smallest) == 1, "expected a unique smallest background region"

    out = connected.copy()
    for r, c in smallest[0]["coords"]:
        out[r, c] = GREEN
    return out


def _layout_single_partial() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(120):
        left_width = random.randint(1, 4)
        right_width = random.randint(2, 5)
        top_height = random.randint(1, 4)
        bottom_height = random.randint(1, 4)
        side = random.choice(["left", "right"])

        if side == "left":
            areas = [
                left_width * top_height,
                left_width * bottom_height,
                right_width * (top_height + bottom_height + 1),
            ]
        else:
            areas = [
                left_width * (top_height + bottom_height + 1),
                right_width * top_height,
                right_width * bottom_height,
            ]
        if areas.count(min(areas)) != 1:
            continue

        margin_top = random.randint(1, 3)
        margin_bottom = random.randint(1, 3)
        margin_left = random.randint(1, 3)
        margin_right = random.randint(1, 3)

        r0 = margin_top
        r1 = r0 + top_height + 1
        r2 = r1 + bottom_height + 1
        c0 = margin_left
        c1 = c0 + left_width + 1
        c2 = c1 + right_width + 1

        coords = [
            (r0, c0),
            (r0, c1),
            (r0, c2),
            (r2, c0),
            (r2, c1),
            (r2, c2),
        ]
        if side == "left":
            coords.extend([(r1, c0), (r1, c1)])
        else:
            coords.extend([(r1, c1), (r1, c2)])

        shape = (r2 + margin_bottom + 1, c2 + margin_right + 1)
        return sorted(coords), shape

    raise RuntimeError("failed to build single-partial layout")


def _layout_double_divider() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(120):
        widths = random.sample(range(1, 6), 3)
        height = random.randint(2, 5)

        if widths.count(min(widths)) != 1:
            continue

        margin_top = random.randint(1, 3)
        margin_bottom = random.randint(1, 3)
        margin_left = random.randint(1, 3)
        margin_right = random.randint(1, 3)

        r0 = margin_top
        r1 = r0 + height + 1
        c0 = margin_left
        c1 = c0 + widths[0] + 1
        c2 = c1 + widths[1] + 1
        c3 = c2 + widths[2] + 1

        coords = [
            (r0, c0),
            (r0, c1),
            (r0, c2),
            (r0, c3),
            (r1, c0),
            (r1, c1),
            (r1, c2),
            (r1, c3),
        ]
        shape = (r1 + margin_bottom + 1, c3 + margin_right + 1)
        return sorted(coords), shape

    raise RuntimeError("failed to build double-divider layout")


def _layout_double_partial() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(160):
        left_width = random.randint(1, 4)
        right_width = random.randint(1, 4)
        top_height = random.randint(1, 3)
        middle_height = random.randint(1, 3)
        bottom_height = random.randint(1, 3)

        areas = [
            left_width * top_height,
            left_width * (middle_height + bottom_height + 1),
            right_width * (top_height + middle_height + 1),
            right_width * bottom_height,
        ]
        if areas.count(min(areas)) != 1:
            continue

        margin_top = random.randint(1, 3)
        margin_bottom = random.randint(1, 3)
        margin_left = random.randint(1, 3)
        margin_right = random.randint(1, 3)

        r0 = margin_top
        r1 = r0 + top_height + 1
        r2 = r1 + middle_height + 1
        r3 = r2 + bottom_height + 1
        c0 = margin_left
        c1 = c0 + left_width + 1
        c2 = c1 + right_width + 1

        coords = [
            (r0, c0),
            (r0, c1),
            (r0, c2),
            (r3, c0),
            (r3, c1),
            (r3, c2),
            (r1, c0),
            (r1, c1),
            (r2, c1),
            (r2, c2),
        ]
        shape = (r3 + margin_bottom + 1, c2 + margin_right + 1)
        return sorted(coords), shape

    raise RuntimeError("failed to build double-partial layout")


def _generate_layout() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    layout_fn = random.choice([_layout_single_partial, _layout_double_divider, _layout_double_partial])
    coords, shape = layout_fn()
    return _transform_coords(
        coords,
        shape,
        transpose=random.choice([False, True]),
        flip_rows=random.choice([False, True]),
        flip_cols=random.choice([False, True]),
    )


def _input_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    marker_count = int(np.count_nonzero(grid == RED))
    if not 6 <= marker_count <= 10:
        return False

    if set(map(int, np.unique(grid))) != {BLACK, RED}:
        return False

    connected = _draw_connections(grid)
    red_comps = _components(connected == RED)
    if len(red_comps) != 1:
        return False

    bg_comps = _components(connected == BLACK)
    enclosed = [comp for comp in bg_comps if not comp["touches_border"]]
    if len(enclosed) < 3:
        return False

    min_area = min(comp["area"] for comp in bg_comps)
    smallest = [comp for comp in bg_comps if comp["area"] == min_area]
    if len(smallest) != 1 or smallest[0]["touches_border"]:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        coords, shape = _generate_layout()
        grid = _render_markers(coords, shape)
        if _input_is_valid(grid):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
