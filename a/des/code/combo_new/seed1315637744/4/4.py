from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, WALL))
GEN_ATTEMPTS = 400


def _palette_from_corner(
    grid: np.ndarray,
    bg: int,
    wall: int,
    row_start: int,
    row_step: int,
    cols: tuple[int, int],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = map(int, grid.shape)
    mask = np.zeros((h, w), dtype=bool)
    pairs: list[tuple[int, int]] = []

    r = int(row_start)
    while 0 <= r < h:
        left = int(grid[r, cols[0]])
        right = int(grid[r, cols[1]])
        if left in (bg, wall) or right in (bg, wall):
            break
        pairs.append((left, right))
        mask[r, cols[0]] = True
        mask[r, cols[1]] = True

        next_row = int(r + 2 * row_step)
        if not (0 <= next_row < h):
            break
        sep_row = int(r + row_step)
        if int(grid[sep_row, cols[0]]) != bg or int(grid[sep_row, cols[1]]) != bg:
            break
        r = next_row

    return pairs, mask


def _detect_palette(grid: np.ndarray, bg: int, wall: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = map(int, grid.shape)
    specs = [
        (0, 1, (0, 1)),
        (0, 1, (w - 2, w - 1)),
        (h - 1, -1, (0, 1)),
        (h - 1, -1, (w - 2, w - 1)),
    ]

    best_pairs: list[tuple[int, int]] = []
    best_mask: np.ndarray | None = None
    for row_start, row_step, cols in specs:
        pairs, mask = _palette_from_corner(grid, bg, wall, row_start, row_step, cols)
        if len(pairs) >= 2 and len(pairs) > len(best_pairs) and len({k for k, _v in pairs}) == len(pairs):
            best_pairs = pairs
            best_mask = mask

    assert best_pairs and best_mask is not None, "expected a corner palette with at least two rows"
    return {int(k): int(v) for k, v in best_pairs}, best_mask


def _row_span(grid: np.ndarray, row: int, col: int, wall: int) -> tuple[int, int]:
    c0 = int(col)
    c1 = int(col)
    while c0 - 1 >= 0 and int(grid[row, c0 - 1]) != wall:
        c0 -= 1
    while c1 + 1 < grid.shape[1] and int(grid[row, c1 + 1]) != wall:
        c1 += 1
    return c0, c1


def _col_span(grid: np.ndarray, row: int, col: int, wall: int) -> tuple[int, int]:
    r0 = int(row)
    r1 = int(row)
    while r0 - 1 >= 0 and int(grid[r0 - 1, col]) != wall:
        r0 -= 1
    while r1 + 1 < grid.shape[0] and int(grid[r1 + 1, col]) != wall:
        r1 += 1
    return r0, r1


def _is_horizontal_corridor(grid: np.ndarray, row: int, c0: int, c1: int, wall: int) -> bool:
    h, w = map(int, grid.shape)
    if not (0 < row < h - 1 and 0 < c0 <= c1 < w - 1):
        return False
    if int(grid[row, c0 - 1]) != wall or int(grid[row, c1 + 1]) != wall:
        return False
    return bool(np.all(grid[row - 1, c0 : c1 + 1] == wall) and np.all(grid[row + 1, c0 : c1 + 1] == wall))


def _is_vertical_corridor(grid: np.ndarray, r0: int, r1: int, col: int, wall: int) -> bool:
    h, w = map(int, grid.shape)
    if not (0 < col < w - 1 and 0 < r0 <= r1 < h - 1):
        return False
    if int(grid[r0 - 1, col]) != wall or int(grid[r1 + 1, col]) != wall:
        return False
    return bool(np.all(grid[r0 : r1 + 1, col - 1] == wall) and np.all(grid[r0 : r1 + 1, col + 1] == wall))


def _corridor_orientation(grid: np.ndarray, row: int, col: int, wall: int) -> str:
    h, w = map(int, grid.shape)
    up_wall = row > 0 and int(grid[row - 1, col]) == wall
    down_wall = row + 1 < h and int(grid[row + 1, col]) == wall
    left_wall = col > 0 and int(grid[row, col - 1]) == wall
    right_wall = col + 1 < w and int(grid[row, col + 1]) == wall

    if up_wall and down_wall and not (left_wall and right_wall):
        return "h"
    if left_wall and right_wall and not (up_wall and down_wall):
        return "v"

    c0, c1 = _row_span(grid, row, col, wall)
    r0, r1 = _col_span(grid, row, col, wall)
    row_len = int(c1 - c0 + 1)
    col_len = int(r1 - r0 + 1)

    if row_len > col_len and _is_horizontal_corridor(grid, row, c0, c1, wall):
        return "h"
    if col_len > row_len and _is_vertical_corridor(grid, r0, r1, col, wall):
        return "v"
    if _is_horizontal_corridor(grid, row, c0, c1, wall):
        return "h"
    if _is_vertical_corridor(grid, r0, r1, col, wall):
        return "v"
    raise AssertionError("seed pixel does not sit in a valid one-cell corridor")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lookup each corridor seed in the corner palette and recolor the whole corridor with the mapped growth color.

    Input structure:
    - The background is black.
    - Several one-cell-wide horizontal or vertical corridors are enclosed by gray walls on both sides and blocked by gray wall pixels at both ends.
    - Each corridor interior is black except for exactly one non-black, non-gray seed pixel somewhere inside the corridor.
    - One corner contains a small palette made of separated two-pixel rows: the left pixel is a seed color key and the right pixel is its growth color.

    Rule:
    - Locate the corner palette and read it as a seed-color to growth-color mapping.
    - For every non-palette colored seed pixel, determine whether its corridor is horizontal or vertical from the surrounding gray walls.
    - Replace the seed by its mapped growth color and extend that color along the corridor in both directions until the gray end walls are reached.
    - Keep the gray walls and the palette unchanged.

    Key insight:
    - The seed color is only a lookup key; the final fill color must first be decoded from the corner palette and only then propagated along the enclosing corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    bg = BG
    wall = WALL
    mapping, palette_mask = _detect_palette(g, bg, wall)

    out = g.copy()
    seed_mask = (g != bg) & (g != wall) & (~palette_mask)
    seeds = np.argwhere(seed_mask)
    assert len(seeds) >= 1, "expected at least one corridor seed"

    for row, col in seeds:
        row = int(row)
        col = int(col)
        seed_color = int(g[row, col])
        assert seed_color in mapping, "every seed color must appear in the palette"
        growth = int(mapping[seed_color])
        orient = _corridor_orientation(g, row, col, wall)

        if orient == "h":
            c0, c1 = _row_span(g, row, col, wall)
            assert _is_horizontal_corridor(g, row, c0, c1, wall)
            out[row, c0 : c1 + 1] = growth
        else:
            r0, r1 = _col_span(g, row, col, wall)
            assert _is_vertical_corridor(g, r0, r1, col, wall)
            out[r0 : r1 + 1, col] = growth

    return out


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _fits(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = map(int, blocked.shape)
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(h - 1, int(r1 + pad))
    cc1 = min(w - 1, int(c1 + pad))
    return not bool(np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_used(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = map(int, blocked.shape)
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(h - 1, int(r1 + pad))
    cc1 = min(w - 1, int(c1 + pad))
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_palette(
    grid: np.ndarray,
    blocked: np.ndarray,
    pairs: list[tuple[int, int]],
    corner: str,
) -> None:
    h, w = map(int, grid.shape)
    rows = [2 * i for i in range(len(pairs))]

    if corner == "tl":
        for r, (seed, growth) in zip(rows, pairs, strict=False):
            grid[r, 0] = int(seed)
            grid[r, 1] = int(growth)
        _reserve(blocked, 0, 0, min(h - 1, 2 * len(pairs) - 1), min(w - 1, 3))
    elif corner == "tr":
        for r, (seed, growth) in zip(rows, pairs, strict=False):
            grid[r, w - 2] = int(seed)
            grid[r, w - 1] = int(growth)
        _reserve(blocked, 0, max(0, w - 4), min(h - 1, 2 * len(pairs) - 1), w - 1)
    elif corner == "bl":
        start = h - 1
        for i, (seed, growth) in enumerate(pairs):
            r = int(start - 2 * i)
            grid[r, 0] = int(seed)
            grid[r, 1] = int(growth)
        _reserve(blocked, max(0, h - 2 * len(pairs)), 0, h - 1, min(w - 1, 3))
    else:
        start = h - 1
        for i, (seed, growth) in enumerate(pairs):
            r = int(start - 2 * i)
            grid[r, w - 2] = int(seed)
            grid[r, w - 1] = int(growth)
        _reserve(blocked, max(0, h - 2 * len(pairs)), max(0, w - 4), h - 1, w - 1)


def _orientation_plan(count: int) -> list[str]:
    if count == 2:
        plan = ["h", "v"]
    elif count == 3:
        plan = ["h", "h", "v"] if random.random() < 0.5 else ["h", "v", "v"]
    else:
        plan = ["h", "h", "v", "v"]
    random.shuffle(plan)
    return plan


def _draw_horizontal_corridor(
    grid: np.ndarray,
    row: int,
    col0: int,
    length: int,
    seed_color: int,
) -> dict[str, int]:
    grid[row - 1, col0 - 1 : col0 + length + 1] = WALL
    grid[row + 1, col0 - 1 : col0 + length + 1] = WALL
    grid[row, col0 - 1] = WALL
    grid[row, col0 + length] = WALL
    seed_col = int(col0 + random.randrange(length))
    grid[row, seed_col] = int(seed_color)
    return {"orientation": 0, "row": int(row), "col0": int(col0), "col1": int(col0 + length - 1), "seed": int(seed_color)}


def _draw_vertical_corridor(
    grid: np.ndarray,
    row0: int,
    col: int,
    length: int,
    seed_color: int,
) -> dict[str, int]:
    grid[row0 - 1, col] = WALL
    grid[row0 + length, col] = WALL
    grid[row0 - 1 : row0 + length + 1, col - 1] = WALL
    grid[row0 - 1 : row0 + length + 1, col + 1] = WALL
    seed_row = int(row0 + random.randrange(length))
    grid[seed_row, col] = int(seed_color)
    return {"orientation": 1, "col": int(col), "row0": int(row0), "row1": int(row0 + length - 1), "seed": int(seed_color)}


def _expected_output(
    grid: np.ndarray,
    corridors: list[dict[str, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for corridor in corridors:
        growth = int(mapping[int(corridor["seed"])])
        if corridor["orientation"] == 0:
            out[int(corridor["row"]), int(corridor["col0"]) : int(corridor["col1"]) + 1] = growth
        else:
            out[int(corridor["row0"]) : int(corridor["row1"]) + 1, int(corridor["col"])] = growth
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    corridor_count = random.randint(2, 4)
    palette_rows = random.randint(corridor_count, 4)
    if 2 * palette_rows > len(COLOR_POOL):
        return None

    palette_colors = random.sample(COLOR_POOL, 2 * palette_rows)
    seed_keys = palette_colors[:palette_rows]
    growth_values = palette_colors[palette_rows:]
    mapping = {int(seed): int(growth) for seed, growth in zip(seed_keys, growth_values, strict=False)}

    pairs = list(mapping.items())
    random.shuffle(pairs)
    corner = random.choice(("tl", "tr", "bl", "br"))
    _place_palette(grid, blocked, pairs, corner)

    used_seed_colors = random.sample(seed_keys, corridor_count)
    orientations = _orientation_plan(corridor_count)
    corridors: list[dict[str, int]] = []

    for seed_color, orient in zip(used_seed_colors, orientations, strict=False):
        placed = False
        for _ in range(120):
            if orient == "h":
                max_len = min(12, width - 4)
                if max_len < 3:
                    return None
                length = random.randint(3, max_len)
                row = random.randint(1, height - 2)
                col0 = random.randint(1, width - length - 1)
                r0, c0 = row - 1, col0 - 1
                r1, c1 = row + 1, col0 + length
                if not _fits(blocked, r0, c0, r1, c1, pad=1):
                    continue
                info = _draw_horizontal_corridor(grid, row, col0, length, int(seed_color))
            else:
                max_len = min(12, height - 4)
                if max_len < 3:
                    return None
                length = random.randint(3, max_len)
                row0 = random.randint(1, height - length - 1)
                col = random.randint(1, width - 2)
                r0, c0 = row0 - 1, col - 1
                r1, c1 = row0 + length, col + 1
                if not _fits(blocked, r0, c0, r1, c1, pad=1):
                    continue
                info = _draw_vertical_corridor(grid, row0, col, length, int(seed_color))

            corridors.append(info)
            _mark_used(blocked, r0, c0, r1, c1, pad=1)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, corridors, mapping)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
