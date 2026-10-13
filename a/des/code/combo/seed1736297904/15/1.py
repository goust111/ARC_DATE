from __future__ import annotations

import random

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SEED_TO_DIRECTION = {
    RED: (-1, 0),
    GREEN: (1, 0),
    YELLOW: (0, -1),
    ORANGE: (0, 1),
}
SEED_COLORS = tuple(SEED_TO_DIRECTION)
NON_SEED_COLORS = tuple(color for color in range(10) if color not in SEED_TO_DIRECTION)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 8)
MAX_GENERATION_ATTEMPTS = 400
MAX_RECT_PLACEMENT_ATTEMPTS = 250


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False

    coord_set = {tuple(map(int, cell)) for cell in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in coord_set):
                return False
    return True


def _find_frame_components(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        comps = _components(grid == color)
        if comps and all(_is_rect_ring(component) for component in comps):
            candidates.append((color, comps))

    assert len(candidates) == 1, "expected exactly one frame color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _background_color(g)
    frame_color, frame_components = _find_frame_components(g, bg)

    out = g.copy()
    for component in frame_components:
        r0, c0, r1, c1 = _bbox(component)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_offsets = np.argwhere((interior != bg) & (interior != frame_color))
        assert len(seed_offsets) == 1, "each rectangle must contain exactly one seed"

        seed_dr, seed_dc = map(int, seed_offsets[0])
        sr = r0 + 1 + seed_dr
        sc = c0 + 1 + seed_dc
        seed_color = int(g[sr, sc])
        assert seed_color in SEED_TO_DIRECTION, "seed color must encode a direction"

        expected = np.full_like(interior, bg)
        expected[seed_dr, seed_dc] = seed_color
        assert np.array_equal(interior, expected), "rectangle interior must be empty except for the seed"

        dr, dc = SEED_TO_DIRECTION[seed_color]
        if dr == -1:
            opening = (r0, sc)
        elif dr == 1:
            opening = (r1, sc)
        elif dc == -1:
            opening = (sr, c0)
        else:
            opening = (sr, c1)

        out[opening] = bg
        rr, cc = opening[0] + dr, opening[1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = frame_color
            rr += dr
            cc += dc

    return out


def _cell_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _corridor_cells(
    height: int,
    width: int,
    opening: tuple[int, int],
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = direction
    rr, cc = opening[0] + dr, opening[1] + dc
    cells: set[tuple[int, int]] = set()
    while 0 <= rr < height and 0 <= cc < width:
        cells.add((rr, cc))
        rr += dr
        cc += dc
    return cells


def _draw_rectangle_border(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _apply_known_transform(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    seed_r: int,
    seed_c: int,
    seed_color: int,
    bg: int,
    frame_color: int,
) -> None:
    dr, dc = SEED_TO_DIRECTION[seed_color]
    if dr == -1:
        opening = (r0, seed_c)
    elif dr == 1:
        opening = (r1, seed_c)
    elif dc == -1:
        opening = (seed_r, c0)
    else:
        opening = (seed_r, c1)

    grid[opening] = bg
    rr, cc = opening[0] + dr, opening[1] + dc
    while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
        grid[rr, cc] = frame_color
        rr += dr
        cc += dc


def _max_rectangles(height: int, width: int) -> int:
    area = height * width
    if area < 190:
        return 2
    if area < 300:
        return 3
    return 4


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg, frame_color = random.sample(NON_SEED_COLORS, 2)
    rect_count = random.randint(1, _max_rectangles(height, width))

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()

    occupied_boxes: set[tuple[int, int]] = set()
    buffered_boxes: set[tuple[int, int]] = set()
    corridors: set[tuple[int, int]] = set()

    placements: list[tuple[int, int, int, int, int, int, int]] = []
    for seed_color in random.sample(SEED_COLORS, k=rect_count):
        placed = False
        for _ in range(MAX_RECT_PLACEMENT_ATTEMPTS):
            rect_h = random.randint(*RECT_H_RANGE)
            rect_w = random.randint(*RECT_W_RANGE)
            if rect_h > height - 2 or rect_w > width - 2:
                continue

            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            seed_r = random.randint(r0 + 1, r1 - 1)
            seed_c = random.randint(c0 + 1, c1 - 1)

            dr, dc = SEED_TO_DIRECTION[seed_color]
            if dr == -1:
                opening = (r0, seed_c)
            elif dr == 1:
                opening = (r1, seed_c)
            elif dc == -1:
                opening = (seed_r, c0)
            else:
                opening = (seed_r, c1)

            area_cells = _cell_set(r0, c0, r1, c1)
            buffer_cells = _cell_set(
                max(0, r0 - 1),
                max(0, c0 - 1),
                min(height - 1, r1 + 1),
                min(width - 1, c1 + 1),
            )
            corridor_cells = _corridor_cells(height, width, opening, (dr, dc))

            if area_cells & (buffered_boxes | corridors):
                continue
            if corridor_cells & (buffered_boxes | corridors):
                continue

            placements.append((r0, c0, r1, c1, seed_r, seed_c, seed_color))
            occupied_boxes |= area_cells
            buffered_boxes |= buffer_cells
            corridors |= corridor_cells
            placed = True
            break

        if not placed:
            return None

    for r0, c0, r1, c1, seed_r, seed_c, seed_color in placements:
        _draw_rectangle_border(grid, r0, c0, r1, c1, frame_color)
        grid[seed_r, seed_c] = seed_color
        _draw_rectangle_border(expected, r0, c0, r1, c1, frame_color)
        expected[seed_r, seed_c] = seed_color
        _apply_known_transform(expected, r0, c0, r1, c1, seed_r, seed_c, seed_color, bg, frame_color)

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[bg] <= max(counts[color] for color in range(10) if color != bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
