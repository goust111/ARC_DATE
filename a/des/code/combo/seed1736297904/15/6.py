from __future__ import annotations

import random

import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
NON_GREY_COLORS = tuple(color for color in ALL_COLORS if color != GREY)
MISSING_CORNERS = ("TL", "TR", "BL", "BR")

# Generator configuration.
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 24
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 6
OBJECT_RETRIES = 80
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _missing_corner(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    assert h >= 2 and w >= 2, "objects must be at least 2x2"
    assert len(local) == h * w - 1, "object must be a rectangle missing one pixel"

    corners = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    missing = [name for name, cell in corners.items() if cell not in local]
    assert len(missing) == 1, "object must miss exactly one corner pixel"
    return missing[0]


def _shadow_delta(missing_corner: str, distance: int) -> tuple[int, int]:
    mapping = {
        "TL": (distance, distance),
        "TR": (distance, -distance),
        "BL": (-distance, distance),
        "BR": (-distance, -distance),
    }
    return mapping[missing_corner]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a grey shadow for each foreground object.

    Every object is a monochrome 4-connected rectangle missing exactly one corner
    pixel. The missing corner determines the opposite diagonal direction of the
    shadow, and the shadow offset is min(object height, object width).
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()
    h, w = g.shape

    for coords in _components4(g != bg):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each foreground object must be monochrome"

        missing = _missing_corner(coords)
        r0, c0, r1, c1 = _bbox(coords)
        distance = min(r1 - r0 + 1, c1 - c0 + 1)
        dr, dc = _shadow_delta(missing, distance)

        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                out[rr, cc] = GREY

    return out


def _object_cells(r0: int, c0: int, height: int, width: int, missing_corner: str) -> list[tuple[int, int]]:
    missing_local = {
        "TL": (0, 0),
        "TR": (0, width - 1),
        "BL": (height - 1, 0),
        "BR": (height - 1, width - 1),
    }[missing_corner]
    cells = []
    for r in range(height):
        for c in range(width):
            if (r, c) != missing_local:
                cells.append((r0 + r, c0 + c))
    return cells


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _placement_ranges(
    grid_h: int, grid_w: int, height: int, width: int, dr: int, dc: int
) -> tuple[range, range] | None:
    row_lo = max(0, -dr)
    row_hi = min(grid_h - height, grid_h - height - dr)
    col_lo = max(0, -dc)
    col_hi = min(grid_w - width, grid_w - width - dc)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _try_place_object(
    grid: np.ndarray, reserved: np.ndarray, color: int, missing_corner: str
) -> bool:
    grid_h, grid_w = grid.shape

    for _ in range(OBJECT_RETRIES):
        height = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        width = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        distance = min(height, width)
        dr, dc = _shadow_delta(missing_corner, distance)
        ranges = _placement_ranges(grid_h, grid_w, height, width, dr, dc)
        if ranges is None:
            continue
        row_range, col_range = ranges
        candidates = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(candidates)

        for r0, c0 in candidates:
            obj_cells = _object_cells(r0, c0, height, width, missing_corner)
            shadow_cells = [(r + dr, c + dc) for r, c in obj_cells]
            union = set(obj_cells) | set(shadow_cells)

            if any(reserved[r, c] for r, c in union):
                continue

            for r, c in obj_cells:
                grid[r, c] = int(color)

            for r, c in _dilate_cells(union, grid.shape):
                reserved[r, c] = True
            return True

    return False


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid_w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    bg = random.choice(NON_GREY_COLORS)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample([color for color in NON_GREY_COLORS if color != bg], n_objects)

    directions = list(MISSING_CORNERS)
    random.shuffle(directions)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for index in range(n_objects):
        if not _try_place_object(grid, reserved, colors[index], directions[index]):
            return None

    fg_count = int(np.sum(grid != bg))
    if fg_count * 2 >= grid.size:
        return None

    out = main(grid)
    if int(np.sum(out == GREY)) != fg_count:
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
