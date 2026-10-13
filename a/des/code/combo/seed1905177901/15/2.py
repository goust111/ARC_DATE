from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
QUADRANTS = ("TL", "TR", "BL", "BR")

MIN_GRID = 14
MAX_GRID = 22
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _corner_positions(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = shape
    return ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))


def _foreground_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append({"color": color, "cells": cells})

    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_mask = patch == BG
    if not bool(np.any(bg_mask)):
        return []

    ph, pw = bg_mask.shape
    outside = np.zeros((ph, pw), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def enqueue(pr: int, pc: int) -> None:
        if bg_mask[pr, pc] and not outside[pr, pc]:
            outside[pr, pc] = True
            q.append((pr, pc))

    for pr in range(ph):
        enqueue(pr, 0)
        enqueue(pr, pw - 1)
    for pc in range(pw):
        enqueue(0, pc)
        enqueue(ph - 1, pc)

    while q:
        pr, pc = q.popleft()
        for nr, nc in _neighbors4(pr, pc):
            if 0 <= nr < ph and 0 <= nc < pw and bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = np.argwhere(bg_mask & ~outside)
    return [(r0 + int(pr), c0 + int(pc)) for pr, pc in holes]


def _nearest_corner_index(shape: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[int, bool]:
    r0, c0, r1, c1 = box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    dists = []
    for cr, cc in _corner_positions(shape):
        dists.append((center_r - cr) ** 2 + (center_c - cc) ** 2)

    best = min(dists)
    winners = [idx for idx, dist in enumerate(dists) if abs(dist - best) < 1e-9]
    return winners[0], len(winners) == 1


def _nearest_corner_color(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    idx, _ = _nearest_corner_index(grid.shape, box)
    r, c = _corner_positions(grid.shape)[idx]
    return int(grid[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's enclosed bbox holes with the color from the nearest corner marker.

    Objects with no enclosed background region inside their bounding box are left unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _foreground_components(g):
        cells = comp["cells"]
        assert isinstance(cells, list)
        holes = _hole_cells(g, cells)
        if not holes:
            continue
        fill_color = _nearest_corner_color(g, _bbox(cells))
        for r, c in holes:
            out[r, c] = fill_color

    return out


def _zone_bounds(shape: tuple[int, int], quadrant: str) -> tuple[int, int, int, int]:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2

    if quadrant[0] == "T":
        r0, r1 = 1, mid_r - 2
    else:
        r0, r1 = mid_r + 1, h - 2

    if quadrant[1] == "L":
        c0, c1 = 1, mid_c - 2
    else:
        c0, c1 = mid_c + 1, w - 2

    return r0, r1, c0, c1


def _solid_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h = random.randint(2, min(max_h, 6))
    w = random.randint(2, min(max_w, 6))
    return np.ones((h, w), dtype=bool)


def _hollow_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None

    h = random.randint(5, min(max_h, 8))
    w = random.randint(5, min(max_w, 8))
    thickness = random.randint(1, min(2, (h - 1) // 2, (w - 1) // 2))

    mask = np.ones((h, w), dtype=bool)
    mask[thickness : h - thickness, thickness : w - thickness] = False

    inner_h = h - 2 * thickness
    inner_w = w - 2 * thickness
    divider_options = ["none"]
    if inner_w >= 3:
        divider_options.append("vertical")
    if inner_h >= 3:
        divider_options.append("horizontal")

    divider = random.choice(divider_options)
    if divider == "vertical":
        col = random.randint(thickness + 1, w - thickness - 2)
        mask[:, col] = True
    elif divider == "horizontal":
        row = random.randint(thickness + 1, h - thickness - 2)
        mask[row, :] = True

    return mask


def _place_mask(grid: np.ndarray, mask: np.ndarray, quadrant: str, color: int) -> bool:
    r_lo, r_hi, c_lo, c_hi = _zone_bounds(grid.shape, quadrant)
    mh, mw = mask.shape
    max_r0 = r_hi - mh + 1
    max_c0 = c_hi - mw + 1
    if max_r0 < r_lo or max_c0 < c_lo:
        return False

    starts = [(r0, c0) for r0 in range(r_lo, max_r0 + 1) for c0 in range(c_lo, max_c0 + 1)]
    random.shuffle(starts)

    for r0, c0 in starts:
        patch = grid[r0 : r0 + mh, c0 : c0 + mw]
        if bool(np.any(patch[mask] != BG)):
            continue
        patch[mask] = color
        return True

    return False


def _object_infos(grid: np.ndarray) -> list[dict[str, object]]:
    corners = set(_corner_positions(grid.shape))
    infos: list[dict[str, object]] = []

    for comp in _foreground_components(grid):
        cells = comp["cells"]
        assert isinstance(cells, list)
        if len(cells) == 1 and cells[0] in corners:
            continue

        box = _bbox(cells)
        corner_index, unique_corner = _nearest_corner_index(grid.shape, box)
        infos.append(
            {
                "cells": cells,
                "bbox": box,
                "holes": _hole_cells(grid, cells),
                "corner_index": corner_index,
                "unique_corner": unique_corner,
            }
        )

    return infos


def _expected_output(grid: np.ndarray, infos: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    corners = _corner_positions(grid.shape)
    for info in infos:
        fill_color = int(grid[corners[int(info["corner_index"])]])
        for r, c in info["holes"]:
            out[r, c] = fill_color
    return out


def _generate_candidate() -> tuple[np.ndarray, int] | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    grid = np.full((h, w), BG, dtype=int)

    colors = list(PALETTE)
    random.shuffle(colors)
    corner_colors = colors[:4]
    object_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    object_colors = colors[4 : 4 + object_count]

    for (r, c), color in zip(_corner_positions(grid.shape), corner_colors):
        grid[r, c] = color

    quadrants = random.sample(list(QUADRANTS), object_count)
    hollow_count = object_count // 2
    if object_count % 2 == 1:
        hollow_count += random.randint(0, 1)

    kinds = ["hollow"] * hollow_count + ["solid"] * (object_count - hollow_count)
    random.shuffle(kinds)

    for color, quadrant, kind in zip(object_colors, quadrants, kinds):
        r0, r1, c0, c1 = _zone_bounds(grid.shape, quadrant)
        max_h = r1 - r0 + 1
        max_w = c1 - c0 + 1
        mask = _hollow_mask(max_h, max_w) if kind == "hollow" else _solid_mask(max_h, max_w)
        if mask is None or not _place_mask(grid, mask, quadrant, color):
            return None

    return grid, object_count


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is None:
            continue

        grid, object_count = candidate
        infos = _object_infos(grid)
        if len(infos) != object_count:
            continue
        if not all(bool(info["unique_corner"]) for info in infos):
            continue

        hollow = [info for info in infos if info["holes"]]
        solid = [info for info in infos if not info["holes"]]
        if not hollow or not solid:
            continue

        expected = _expected_output(grid, infos)
        actual = main(grid)
        if np.array_equal(actual, grid):
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
