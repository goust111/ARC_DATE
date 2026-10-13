from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SPECIAL_INPUT_COLORS = {BLACK, GREY, MAROON}
TOWER_COLORS = [BLUE, RED, GREEN, MAGENTA, ORANGE, CYAN]

GEN_HEIGHT_RANGE = (13, 23)
GEN_WIDTH_RANGE = (18, 30)
GEN_TOWER_COUNT_RANGE = (2, 4)
GEN_INTERNAL_GAP = 2
GEN_EXTRA_WALLS_MAX = 2
GEN_ATTEMPTS = 500
GEN_WALL_MODES = ("none", "one", "many")


def _tower_body_cells(baseline: int, col: int, height: int) -> list[tuple[int, int]]:
    return [(baseline - delta, col) for delta in range(1, height + 1)]


def _sector_cells(top_r: int, col: int, height: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for distance in range(height + 1):
        r = top_r - distance
        for c in range(col - distance, col + distance + 1):
            cells.append((r, c))
    return cells


def _ray_cell(top_r: int, col: int, direction: int, distance: int) -> tuple[int, int]:
    return top_r - distance, col + direction * distance


def _find_towers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    baseline = h - 1

    assert h >= 4 and w >= 5
    assert bool(np.all(g[baseline] == GREY)), "expected a grey baseline on the last row"

    tower_mask = np.zeros((h, w), dtype=bool)
    towers: list[dict[str, int]] = []

    for col in range(w):
        color = int(g[baseline - 1, col])
        if color in SPECIAL_INPUT_COLORS:
            continue

        height = 0
        r = baseline - 1
        while r >= 0 and int(g[r, col]) == color:
            tower_mask[r, col] = True
            height += 1
            r -= 1

        assert height >= 2, "tower height must be at least 2"
        top_r = baseline - height
        towers.append(
            {
                "col": int(col),
                "color": int(color),
                "height": int(height),
                "top_r": int(top_r),
            }
        )

    assert towers, "expected at least one tower"

    non_special_above = ~np.isin(g[:baseline], list(SPECIAL_INPUT_COLORS))
    assert bool(np.array_equal(tower_mask[:baseline], non_special_above)), "all non-special pixels above baseline must belong to towers"

    return sorted(towers, key=lambda tower: (tower["col"], tower["top_r"]))


def _paint_sectors(out: np.ndarray, towers: list[dict[str, int]]) -> None:
    h, w = out.shape
    baseline = h - 1
    for tower in towers:
        top_r = int(tower["top_r"])
        col = int(tower["col"])
        height = int(tower["height"])
        color = int(tower["color"])
        for distance in range(height + 1):
            r = top_r - distance
            if not (0 <= r < baseline):
                continue
            left = max(0, col - distance)
            right = min(w - 1, col + distance)
            row = out[r, left : right + 1]
            row[row == BLACK] = color
            out[r, left : right + 1] = row


def _apply_shadow(inp: np.ndarray, out: np.ndarray, tower: dict[str, int], direction: int) -> None:
    h, w = inp.shape
    baseline = h - 1
    top_r = int(tower["top_r"])
    col = int(tower["col"])
    height = int(tower["height"])

    for distance in range(1, height + 1):
        r, c = _ray_cell(top_r, col, direction, distance)
        if not (0 <= r < baseline and 0 <= c < w):
            break
        if int(inp[r, c]) != MAROON:
            continue
        for shadow_distance in range(distance + 1, height + 1):
            rr, cc = _ray_cell(top_r, col, direction, shadow_distance)
            if 0 <= rr < baseline and 0 <= cc < w and int(out[rr, cc]) != MAROON:
                out[rr, cc] = BLACK
        break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint broadcast fans from colored towers, carve diagonal wall shadows, then mark blind zones yellow.
    """
    inp = np.asarray(grid, dtype=int)
    assert inp.ndim == 2

    towers = _find_towers(inp)
    out = inp.copy()
    baseline = inp.shape[0] - 1

    _paint_sectors(out, towers)
    for tower in towers:
        _apply_shadow(inp, out, tower, -1)
        _apply_shadow(inp, out, tower, 1)

    top = out[:baseline]
    top[top == BLACK] = YELLOW
    out[:baseline] = top
    return out


def _distribute(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _isolated(cell: tuple[int, int], blockers: set[tuple[int, int]]) -> bool:
    return cell not in blockers and all(neighbor not in blockers for neighbor in _neighbors8(cell))


def _sample_heights(height_limit: int, width: int, tower_count: int) -> list[int] | None:
    budget = width - GEN_INTERNAL_GAP * (tower_count - 1)
    heights: list[int] = []
    for idx in range(tower_count):
        towers_left = tower_count - idx - 1
        min_future_width = towers_left * 5
        max_height = min(height_limit, (budget - min_future_width - 1) // 2)
        if max_height < 2:
            return None
        height = random.randint(2, max_height)
        heights.append(int(height))
        budget -= 2 * int(height) + 1
    random.shuffle(heights)
    return heights


def _build_towers(height: int, width: int, heights: list[int], colors: list[int]) -> list[dict[str, int]]:
    widths = [2 * tower_height + 1 for tower_height in heights]
    required = sum(widths) + GEN_INTERNAL_GAP * (len(heights) - 1)
    extra = width - required
    gap_extras = _distribute(extra, len(heights) + 1)

    towers: list[dict[str, int]] = []
    cursor = gap_extras[0]
    baseline = height - 1
    for idx, (tower_height, color) in enumerate(zip(heights, colors)):
        left = cursor
        col = left + tower_height
        top_r = baseline - tower_height
        towers.append(
            {
                "col": int(col),
                "color": int(color),
                "height": int(tower_height),
                "top_r": int(top_r),
            }
        )
        cursor = left + widths[idx]
        if idx < len(heights) - 1:
            cursor += GEN_INTERNAL_GAP + gap_extras[idx + 1]
    return towers


def _sector_union(towers: list[dict[str, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tower in towers:
        cells.update(_sector_cells(int(tower["top_r"]), int(tower["col"]), int(tower["height"])))
    return cells


def _choose_blocked_rays(towers: list[dict[str, int]], wall_mode: str) -> list[tuple[int, int]]:
    candidates = [
        (idx, direction)
        for idx, tower in enumerate(towers)
        if int(tower["height"]) >= 3
        for direction in (-1, 1)
    ]
    if wall_mode == "none":
        return []
    if wall_mode == "one":
        if not candidates:
            return []
        return [random.choice(candidates)]
    if len(candidates) < 2:
        return []
    count = random.randint(2, len(candidates))
    return random.sample(candidates, count)


def _wall_distance(height: int) -> int:
    return random.randint(2, height - 1)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    height_limit = min(6, (height - 1) // 2)
    if height_limit < 2:
        return None

    max_towers = min(GEN_TOWER_COUNT_RANGE[1], (width + 2) // 7)
    if max_towers < GEN_TOWER_COUNT_RANGE[0]:
        return None

    tower_count = random.randint(GEN_TOWER_COUNT_RANGE[0], max_towers)
    heights = _sample_heights(height_limit, width, tower_count)
    if heights is None:
        return None

    wall_mode = random.choice(GEN_WALL_MODES)
    blocked_cap = sum(tower_height >= 3 for tower_height in heights) * 2
    if wall_mode == "one" and blocked_cap < 1:
        return None
    if wall_mode == "many" and blocked_cap < 2:
        return None

    colors = random.sample(TOWER_COLORS, tower_count)
    specs = list(zip(heights, colors))
    random.shuffle(specs)
    heights = [int(height_value) for height_value, _color in specs]
    colors = [int(color_value) for _height_value, color_value in specs]

    towers = _build_towers(height, width, heights, colors)
    sector_cells = _sector_union(towers)
    baseline = height - 1

    grid = np.full((height, width), BLACK, dtype=int)
    grid[baseline, :] = GREY

    tower_cells: set[tuple[int, int]] = set()
    for tower in towers:
        for cell in _tower_body_cells(baseline, int(tower["col"]), int(tower["height"])):
            tower_cells.add(cell)
            grid[cell] = int(tower["color"])

    wall_cells: set[tuple[int, int]] = set()
    blocked_rays = _choose_blocked_rays(towers, wall_mode)
    if wall_mode != "none" and not blocked_rays:
        return None

    for tower_idx, direction in blocked_rays:
        tower = towers[tower_idx]
        height_value = int(tower["height"])
        if height_value < 3:
            return None
        distance = _wall_distance(height_value)
        cell = _ray_cell(int(tower["top_r"]), int(tower["col"]), direction, distance)
        if cell in wall_cells or cell in tower_cells:
            return None
        if not _isolated(cell, tower_cells | wall_cells):
            return None
        wall_cells.add(cell)
        grid[cell] = MAROON

    extra_candidates = [
        (r, c)
        for r in range(baseline)
        for c in range(width)
        if (r, c) not in tower_cells and (r, c) not in wall_cells and (r, c) not in sector_cells
    ]
    random.shuffle(extra_candidates)
    extra_target = random.randint(0, GEN_EXTRA_WALLS_MAX)
    for cell in extra_candidates:
        if len(wall_cells) >= len(blocked_rays) + extra_target:
            break
        if _isolated(cell, tower_cells | wall_cells):
            wall_cells.add(cell)
            grid[cell] = MAROON

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not bool((out[:baseline] == YELLOW).any()):
        return None
    if not bool((out[:baseline] != YELLOW).any()):
        return None
    if int((grid == MAROON).sum()) != int((out == MAROON).sum()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
