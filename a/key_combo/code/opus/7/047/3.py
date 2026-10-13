from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
OBJECT_COLORS = (1, 3, 4, 5)
GEN_RETRY_LIMIT = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(
    grid: np.ndarray,
    *,
    include: callable,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not include(int(grid[r, c])):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "top": min(rows),
                    "left": min(cols),
                    "bottom": max(rows),
                    "right": max(cols),
                }
            )
    return components


def _connect_markers(grid: np.ndarray, marker_color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    markers = grid == marker_color
    h, w = grid.shape

    for r in range(h):
        cols = np.flatnonzero(markers[r])
        if len(cols) >= 2:
            out[r, int(cols[0]) : int(cols[-1]) + 1] = marker_color

    for c in range(w):
        rows = np.flatnonzero(markers[:, c])
        if len(rows) >= 2:
            out[int(rows[0]) : int(rows[-1]) + 1, c] = marker_color

    return out


def _find_containers(grid: np.ndarray, bg: int, marker_color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    containers: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if touches_border:
                continue

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if r0 == 0 or r1 == h - 1 or c0 == 0 or c1 == w - 1:
                continue
            if not np.all(grid[r0 - 1, c0 : c1 + 1] == marker_color):
                continue
            if not np.all(grid[r1 + 1, c0 : c1 + 1] == marker_color):
                continue
            if not np.all(grid[r0 : r1 + 1, c0 - 1] == marker_color):
                continue
            if not np.all(grid[r0 : r1 + 1, c1 + 1] == marker_color):
                continue

            containers.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )

    containers.sort(key=lambda item: (item["r0"], item["c0"]))
    return containers


def _container_for_left_col(containers: list[dict], left_col: int) -> dict | None:
    matches = [container for container in containers if container["c0"] <= left_col <= container["c1"]]
    if not matches:
        return None
    return min(matches, key=lambda item: item["r0"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect red marker spans into rectangular containers, then drop each colored object into a matching container and fill by object area.

    Input structure:
    - The grid has a black background, red marker pixels in the lower region, and 3-5 small monochrome colored objects above them.
    - Red markers share some rows and columns so their row-wise and column-wise spans form a rectilinear skeleton.
    - Valid containers are black rectangular cavities whose four surrounding sides become fully red after the span connections.

    Rule:
    - Connect every row span between the leftmost and rightmost original red markers on that row, and every column span between the topmost and bottommost original red markers on that column.
    - Detect every enclosed rectangular black cavity bounded on all four sides by the connected red skeleton.
    - Erase the original colored objects, assign each object to the highest container whose interior column range contains the object's leftmost column, and fill full-width rows from each container bottom upward by that object's area; multiple objects stack in top-row order.
    - If an object is not caught by any container, it falls through and colors the bottom row.

    Key insight:
    - The red markers are only seeds: their spans create the real geometry, and the objects keep only two attributes after falling, namely their target column and their pixel area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    marker_color = RED
    assert np.any(g == marker_color), "expected red marker pixels"

    objects = _find_components(g, include=lambda value: value not in {bg, marker_color})
    skeleton = _connect_markers(g, marker_color)
    containers = _find_containers(skeleton, bg, marker_color)

    out = skeleton.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = bg

    grouped: dict[int, list[dict]] = {idx: [] for idx in range(len(containers))}
    uncaught: list[dict] = []
    for obj in objects:
        container = _container_for_left_col(containers, int(obj["left"]))
        if container is None:
            uncaught.append(obj)
            continue
        grouped[containers.index(container)].append(obj)

    for idx, container in enumerate(containers):
        used_rows = 0
        for obj in sorted(grouped[idx], key=lambda item: (item["top"], item["left"])):
            fill_rows = min(int(obj["area"]), container["height"] - used_rows)
            if fill_rows <= 0:
                break
            start = container["r1"] - used_rows - fill_rows + 1
            stop = container["r1"] - used_rows + 1
            out[start:stop, container["c0"] : container["c1"] + 1] = int(obj["color"])
            used_rows += fill_rows

    for obj in sorted(uncaught, key=lambda item: (item["top"], item["left"])):
        out[-1, :] = int(obj["color"])

    return out


def _random_blob(area: int, color: int, max_width: int) -> np.ndarray:
    dims = [
        (height, width)
        for height in range(1, min(4, area) + 1)
        for width in range(1, max_width + 1)
        if height * width >= area
    ]
    if not dims:
        raise ValueError("no blob dimensions available")

    for _ in range(200):
        height, width = random.choice(dims)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        coords = np.array(sorted(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
        if patch.shape[1] <= max_width:
            return patch

    raise RuntimeError("failed to build a valid blob")


def _can_place(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    ph, pw = patch.shape
    h, w = occupied.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + ph + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + pw + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top_band_end: int, left_min: int, left_max: int) -> bool:
    ph, pw = patch.shape
    if left_min > left_max or top_band_end < ph:
        return False

    candidates = [
        (top, left)
        for top in range(0, top_band_end - ph + 1)
        for left in range(left_min, left_max + 1)
    ]
    random.shuffle(candidates)
    for top, left in candidates:
        if not _can_place(occupied, top, left, patch):
            continue
        mask = patch != 0
        grid_view = grid[top : top + ph, left : left + pw]
        occ_view = occupied[top : top + ph, left : left + pw]
        grid_view[mask] = patch[mask]
        occ_view[mask] = True
        return True
    return False


def _choose_colors(count: int) -> list[int]:
    base = list(OBJECT_COLORS)
    random.shuffle(base)
    if count <= len(base):
        return base[:count]
    return base + [random.choice(base) for _ in range(count - len(base))]


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.choice((3, 4))
    widths = [random.randint(2, 4) for _ in range(n_containers)]
    total_span = sum(widths) + n_containers + 1

    min_width = max(14, total_span + 2)
    max_width = min(22, total_span + 6)
    if min_width > max_width:
        return None

    h = random.randint(14, 18)
    interior_h = random.randint(2, 4)
    top_wall_min = h // 2
    top_wall_max = h - interior_h - 3
    if top_wall_min > top_wall_max:
        return None
    top_wall = random.randint(top_wall_min, top_wall_max)
    bottom_wall = top_wall + interior_h + 1

    w = random.randint(min_width, max_width)
    left_margin = random.randint(1, w - total_span - 1)
    xs = [left_margin]
    for width in widths:
        xs.append(xs[-1] + width + 1)

    grid = np.zeros((h, w), dtype=int)
    for r in (top_wall, bottom_wall):
        for c in xs:
            grid[r, c] = RED

    skeleton = _connect_markers(grid, RED)
    containers = _find_containers(skeleton, BG, RED)
    if len(containers) != n_containers:
        return None
    if any(container["height"] < 2 for container in containers):
        return None

    top_band_end = min(max(4, h // 3 + 1), top_wall - 2)
    if top_band_end < 2:
        return None

    n_objects = random.randint(3, 5)
    assignments = [random.randrange(n_containers) for _ in range(n_objects)]
    if len(set(assignments)) == n_objects and n_objects > 1:
        assignments[-1] = assignments[0]

    colors = _choose_colors(n_objects)
    occupied = np.zeros_like(grid, dtype=bool)

    object_order = list(range(n_objects))
    random.shuffle(object_order)
    for idx in object_order:
        container = containers[assignments[idx]]
        area = random.randint(2, 6)
        patch = _random_blob(area, colors[idx], min(3, container["width"]))
        left_min = container["c0"]
        left_max = container["c1"] - patch.shape[1] + 1
        if not _place_patch(grid, occupied, patch, top_band_end, left_min, left_max):
            return None

    if np.count_nonzero(grid == RED) != 2 * (n_containers + 1):
        return None

    objects = _find_components(grid, include=lambda value: value not in {BG, RED})
    if not (3 <= len(objects) <= 5):
        return None
    if any(obj["top"] >= (h + 2) // 3 for obj in objects):
        return None

    for obj in objects:
        if _container_for_left_col(containers, int(obj["left"])) is None:
            return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
