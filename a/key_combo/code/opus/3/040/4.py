from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_ROWS = (14, 22)
GRID_COLS = (14, 22)
RECT_COUNT = (3, 5)
RECT_SIDE = (3, 6)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _rectangle_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_hollow_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for coords in _components_mask(grid == FRAME):
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border = _rectangle_border_mask(height, width)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if coords.shape[0] != int(border.sum()):
            continue
        if not np.all(sub[border] == FRAME):
            continue
        if not np.all(sub[~border] == BG):
            continue
        rectangles.append((r0, c0, r1, c1))
    rectangles.sort()
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill or erase each hollow gray rectangle according to whether its four diagonal corner markers agree.

    Input structure:
    - The grid has a black background with several gray 1-pixel-thick hollow rectangles.
    - Each rectangle has one colored marker pixel at each outside diagonal corner position.
    - Rectangle interiors are black and rectangles are separated so their markers do not interfere.

    Rule:
    - Detect every gray hollow rectangle and read the four marker colors just outside its corners.
    - If all four markers have the same color, fill only the rectangle interior with that shared color.
    - Otherwise erase the whole rectangle border and its four markers to black, leaving the interior black.

    Key insight:
    - The corner markers are not decoration; they jointly classify each rectangle, and unanimous color agreement both triggers filling and supplies the fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_hollow_rectangles(g):
        markers = [
            (r0 - 1, c0 - 1),
            (r0 - 1, c1 + 1),
            (r1 + 1, c0 - 1),
            (r1 + 1, c1 + 1),
        ]
        if not all(_in_bounds(g.shape, r, c) for r, c in markers):
            continue

        colors = [int(g[r, c]) for r, c in markers]
        if any(color in (BG, FRAME) for color in colors):
            continue

        if len(set(colors)) == 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = colors[0]
        else:
            border = _rectangle_border_mask(r1 - r0 + 1, c1 - c0 + 1)
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[border] = BG
            for r, c in markers:
                out[r, c] = BG

    return out


def _reserve_bounds(r0: int, c0: int, height: int, width: int) -> tuple[int, int, int, int]:
    return r0 - 1, c0 - 1, r0 + height, c0 + width


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    rr0, cc0, rr1, cc1 = _reserve_bounds(r0, c0, height, width)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    rr0, cc0, rr1, cc1 = _reserve_bounds(r0, c0, height, width)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_rectangle(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _marker_pattern(fill_branch: bool) -> list[int]:
    if fill_branch:
        color = random.choice(MARKER_COLORS)
        return [color, color, color, color]

    mode = random.choice(("one_odd", "two_pairs", "mostly_unique"))
    if mode == "one_odd":
        common, odd = random.sample(MARKER_COLORS, 2)
        markers = [common, common, common, odd]
        random.shuffle(markers)
        return markers
    if mode == "two_pairs":
        a, b = random.sample(MARKER_COLORS, 2)
        markers = [a, a, b, b]
        random.shuffle(markers)
        return markers

    color_count = random.choice((3, 4))
    palette = random.sample(MARKER_COLORS, color_count)
    while True:
        markers = [random.choice(palette) for _ in range(4)]
        if len(set(markers)) >= 3:
            return markers


def _place_one(
    grid: np.ndarray,
    occupied: np.ndarray,
    height: int,
    width: int,
    fill_branch: bool,
) -> bool:
    h, w = grid.shape
    row_max = h - height - 1
    col_max = w - width - 1
    if row_max < 1 or col_max < 1:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(1, row_max)
        c0 = random.randint(1, col_max)
        if not _can_place(occupied, r0, c0, height, width):
            continue

        _draw_rectangle(grid, r0, c0, height, width)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        markers = _marker_pattern(fill_branch)
        positions = [
            (r0 - 1, c0 - 1),
            (r0 - 1, c1 + 1),
            (r1 + 1, c0 - 1),
            (r1 + 1, c1 + 1),
        ]
        for color, (r, c) in zip(markers, positions):
            grid[r, c] = color

        _mark_occupied(occupied, r0, c0, height, width)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    rect_count = random.randint(*RECT_COUNT)
    branches = [True, False]
    while len(branches) < rect_count:
        branches.append(random.choice((True, False)))
    random.shuffle(branches)

    sizes = [
        (
            random.randint(*RECT_SIDE),
            random.randint(*RECT_SIDE),
            branch,
        )
        for branch in branches
    ]
    sizes.sort(key=lambda item: item[0] * item[1], reverse=True)

    for height, width, branch in sizes:
        if not _place_one(grid, occupied, height, width, branch):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any((grid == FRAME) & (out == BG)):
        return None
    if not np.any((grid == BG) & (out != BG)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
