import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
DOT_COLORS = [BLUE, RED, GREEN, YELLOW, 7, 8, 9]

UP, RIGHT, DOWN, LEFT = range(4)
DIR_STEPS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

BLOCK_OPTIONS = (2, 3)
SIZE_OPTIONS = (5, 7, 9)
COUNT_OPTIONS_BY_MOD = {
    0: (4, 8),
    1: (5,),
    2: (6,),
    3: (7,),
}
GEN_RETRIES = 300


def _divider_indices(grid: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return np.flatnonzero(np.all(grid == MAGENTA, axis=1))
    return np.flatnonzero(np.all(grid == MAGENTA, axis=0))


def _region_spans(length: int, dividers: np.ndarray) -> list[tuple[int, int]]:
    cuts = [-1, *map(int, dividers.tolist()), length]
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        spans.append((a + 1, b))
    return spans


def _parse_pointer(region: np.ndarray) -> tuple[int, int]:
    coords = {(int(r), int(c)) for r, c in np.argwhere(region == GREY)}
    assert coords, "expected one gray pointer per region"

    center_r = region.shape[0] // 2
    center_c = region.shape[1] // 2
    assert (center_r, center_c) in coords, "pointer must start at the region center"

    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if rows == {center_r}:
        lo = min(cols)
        hi = max(cols)
        if lo == center_c and hi > center_c:
            length = hi - center_c
            expected = {(center_r, c) for c in range(center_c, hi + 1)}
            direction = RIGHT
        else:
            assert hi == center_c and lo < center_c, "horizontal pointer must extend on one side only"
            length = center_c - lo
            expected = {(center_r, c) for c in range(lo, center_c + 1)}
            direction = LEFT
    else:
        assert cols == {center_c}, "pointer must be a straight center-aligned segment"
        lo = min(rows)
        hi = max(rows)
        if lo == center_r and hi > center_r:
            length = hi - center_r
            expected = {(r, center_c) for r in range(center_r, hi + 1)}
            direction = DOWN
        else:
            assert hi == center_r and lo < center_r, "vertical pointer must extend on one side only"
            length = center_r - lo
            expected = {(r, center_c) for r in range(lo, center_r + 1)}
            direction = UP

    assert length >= 1
    assert coords == expected, "pointer cells must form one contiguous segment"
    return direction, length


def _parse_region(region: np.ndarray) -> dict:
    direction, length = _parse_pointer(region)
    dots = []
    mask = ~np.isin(region, [BLACK, GREY, MAGENTA])
    for r, c in np.argwhere(mask):
        dots.append((int(r), int(c), int(region[r, c])))
    return {"direction": direction, "length": length, "dots": dots}


def _parse_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[dict]]]:
    row_dividers = _divider_indices(grid, axis=0)
    col_dividers = _divider_indices(grid, axis=1)
    assert len(row_dividers) == len(col_dividers), "expected a square region layout"
    assert len(row_dividers) in (1, 2), "expected a 2x2 or 3x3 partition"

    row_spans = _region_spans(grid.shape[0], row_dividers)
    col_spans = _region_spans(grid.shape[1], col_dividers)

    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "subregions must be uniform"

    parsed = []
    for r0, r1 in row_spans:
        row_info = []
        for c0, c1 in col_spans:
            row_info.append(_parse_region(grid[r0:r1, c0:c1]))
        parsed.append(row_info)
    return row_spans, col_spans, parsed


def _angle_cw_from_up(dr: int, dc: int) -> float:
    return (math.degrees(math.atan2(dc, -dr)) + 360.0) % 360.0


def _point_in_sweep(local_r: int, local_c: int, shape: tuple[int, int], direction: int, turns: int) -> bool:
    if turns % 4 == 0:
        return True

    center_r = shape[0] // 2
    center_c = shape[1] // 2
    dr = local_r - center_r
    dc = local_c - center_c
    assert dr != 0 or dc != 0, "target dots may not occupy the center"

    theta = _angle_cw_from_up(dr, dc)
    start = direction * 90.0
    delta = (theta - start) % 360.0
    return delta <= turns * 90.0 + 1e-9


def _draw_pointer(block: np.ndarray, direction: int, length: int) -> None:
    center_r = block.shape[0] // 2
    center_c = block.shape[1] // 2
    dr, dc = DIR_STEPS[direction]
    for step in range(length + 1):
        rr = center_r + dr * step
        cc = center_c + dc * step
        block[rr, cc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each region's gray pointer from the right neighbor's target count, then keep only dots inside the swept fan.

    Input structure:
    - The background is black.
    - Full magenta rows and columns partition the canvas into a uniform 2x2 or 3x3 grid of rectangular subregions.
    - Each subregion contains one gray center-anchored pointer pointing up, right, down, or left, plus several colored single-pixel target dots.

    Rule:
    - For each subregion, count the colored target dots in the subregion immediately to its right, wrapping around within the same row.
    - Rotate the current gray pointer clockwise by that count modulo 4 quarter-turns, and redraw the rotated pointer with the same length.
    - Keep only the target dots lying inside the clockwise swept fan from the original direction to the rotated direction; when the rotation count is a multiple of 4, the swept fan is the full circle and all dots remain.
    - Erase dots outside the swept fan to black, and keep the magenta partition lines unchanged.

    Key insight:
    - The rotation amount is not local: each region reads its right neighbor's dot count, and that count controls both the pointer's new direction and which directional quadrants survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, parsed = _parse_grid(g)
    blocks = len(row_spans)
    counts = [[len(parsed[r][c]["dots"]) for c in range(blocks)] for r in range(blocks)]

    out = np.where(g == MAGENTA, MAGENTA, BLACK)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            info = parsed[r][c]
            turns = counts[r][(c + 1) % blocks] % 4
            block = out[r0:r1, c0:c1]

            for local_r, local_c, color in info["dots"]:
                if _point_in_sweep(local_r, local_c, block.shape, info["direction"], turns):
                    block[local_r, local_c] = color

            new_direction = (info["direction"] + turns) % 4
            _draw_pointer(block, new_direction, info["length"])

    return out


def _quadrant_cells(height: int, width: int) -> dict[int, list[tuple[int, int]]]:
    center_r = height // 2
    center_c = width // 2
    cells = {0: [], 1: [], 2: [], 3: []}
    for r in range(height):
        for c in range(width):
            if r == center_r or c == center_c:
                continue
            if r < center_r and c > center_c:
                cells[0].append((r, c))
            elif r > center_r and c > center_c:
                cells[1].append((r, c))
            elif r > center_r and c < center_c:
                cells[2].append((r, c))
            else:
                assert r < center_r and c < center_c
                cells[3].append((r, c))
    return cells


def _take_spread(quadrants: dict[int, list[tuple[int, int]]], count: int) -> list[tuple[int, int]]:
    pools = {q: cells[:] for q, cells in quadrants.items() if cells}
    for cells in pools.values():
        random.shuffle(cells)

    total = sum(len(cells) for cells in pools.values())
    assert count <= total

    chosen = []
    order = list(pools)
    while len(chosen) < count:
        random.shuffle(order)
        progressed = False
        for q in order:
            if pools[q] and len(chosen) < count:
                chosen.append(pools[q].pop())
                progressed = True
        assert progressed, "not enough candidate cells to satisfy the dot count"
    return chosen


def _choose_dot_positions(height: int, width: int, count: int, direction: int, turns: int) -> list[tuple[int, int]]:
    quadrants = _quadrant_cells(height, width)
    if turns == 0:
        return _take_spread(quadrants, count)

    kept_quadrants = {(direction + step) % 4 for step in range(turns)}
    kept = {q: quadrants[q] for q in kept_quadrants}
    removed = {q: quadrants[q] for q in quadrants if q not in kept_quadrants}
    kept_total = sum(len(cells) for cells in kept.values())
    removed_total = sum(len(cells) for cells in removed.values())

    min_keep = min(2, kept_total)
    min_remove = min(2, removed_total)
    low = max(min_keep, count - removed_total)
    high = min(kept_total, count - min_remove)
    assert low <= high, "cannot split dots across kept and removed sectors"

    n_keep = random.randint(low, high)
    n_remove = count - n_keep
    positions = _take_spread(kept, n_keep) + _take_spread(removed, n_remove)
    random.shuffle(positions)
    return positions


def _balanced_values(size: int) -> list[int]:
    values = [v for _ in range(size // 4) for v in range(4)]
    if size % 4:
        values.extend(random.sample(range(4), size % 4))
    random.shuffle(values)
    return values


def _pointer_length(height: int, width: int, direction: int) -> int:
    center_r = height // 2
    center_c = width // 2
    if direction in (UP, DOWN):
        maximum = center_r
    else:
        maximum = center_c
    minimum = max(1, maximum - 1)
    return random.randint(minimum, maximum)

def _gen_attempt() -> np.ndarray | None:
    blocks = random.choice(BLOCK_OPTIONS)
    cell_h = random.choice([size for size in SIZE_OPTIONS if blocks * size + (blocks - 1) <= 30])
    cell_w = random.choice([size for size in SIZE_OPTIONS if blocks * size + (blocks - 1) <= 30])
    height = blocks * cell_h + (blocks - 1)
    width = blocks * cell_w + (blocks - 1)

    grid = np.full((height, width), BLACK, dtype=int)
    divider_rows = [row * (cell_h + 1) - 1 for row in range(1, blocks)]
    divider_cols = [col * (cell_w + 1) - 1 for col in range(1, blocks)]
    for rr in divider_rows:
        grid[rr, :] = MAGENTA
    for cc in divider_cols:
        grid[:, cc] = MAGENTA

    n_regions = blocks * blocks
    count_mods = _balanced_values(n_regions)
    directions = _balanced_values(n_regions)
    palette = random.sample(DOT_COLORS, random.randint(3, 5))

    count_grid = np.empty((blocks, blocks), dtype=int)
    direction_grid = np.empty((blocks, blocks), dtype=int)
    length_grid = np.empty((blocks, blocks), dtype=int)
    for idx in range(n_regions):
        r, c = divmod(idx, blocks)
        mod = count_mods[idx]
        count_grid[r, c] = random.choice(COUNT_OPTIONS_BY_MOD[mod])
        direction_grid[r, c] = directions[idx]
        length_grid[r, c] = _pointer_length(cell_h, cell_w, direction_grid[r, c])

    turns = np.zeros_like(count_grid)
    for r in range(blocks):
        for c in range(blocks):
            turns[r, c] = count_grid[r, (c + 1) % blocks] % 4
    assert set(turns.ravel().tolist()) == {0, 1, 2, 3}

    for r in range(blocks):
        for c in range(blocks):
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            dot_positions = _choose_dot_positions(
                cell_h,
                cell_w,
                int(count_grid[r, c]),
                int(direction_grid[r, c]),
                int(turns[r, c]),
            )
            dots = [(rr, cc, random.choice(palette)) for rr, cc in dot_positions]
            block = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            _draw_pointer(block, int(direction_grid[r, c]), int(length_grid[r, c]))
            for rr, cc, color in dots:
                block[rr, cc] = color

    try:
        row_spans, col_spans, parsed = _parse_grid(grid)
        assert len(row_spans) == blocks and len(col_spans) == blocks
        for r in range(blocks):
            for c in range(blocks):
                info = parsed[r][c]
                assert info["direction"] == int(direction_grid[r, c])
                assert info["length"] == int(length_grid[r, c])
                assert len(info["dots"]) == int(count_grid[r, c])
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
