from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

REGION_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


_RIGHT_ARROW = np.array(
    [
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ],
    dtype=bool,
)

ARROW_PATTERNS = {
    "right": _RIGHT_ARROW,
    "up": np.rot90(_RIGHT_ARROW, 1),
    "left": np.rot90(_RIGHT_ARROW, 2),
    "down": np.rot90(_RIGHT_ARROW, 3),
}

DIR_OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ADJACENT_REGION_PAIRS = [(0, 1), (0, 2), (1, 3), (2, 3)]


@dataclass(frozen=True)
class Arrow:
    region: int
    top: int
    left: int
    direction: str
    cells: tuple[tuple[int, int], ...]


def _region_bounds(shape: tuple[int, int], sep_r: int, sep_c: int) -> dict[int, tuple[int, int, int, int]]:
    h, w = shape
    return {
        0: (0, sep_r, 0, sep_c),
        1: (0, sep_r, sep_c + 1, w),
        2: (sep_r + 1, h, 0, sep_c),
        3: (sep_r + 1, h, sep_c + 1, w),
    }


def _region_pos(region: int) -> tuple[int, int]:
    return divmod(region, 2)


def _target_region(region: int, direction: str) -> int:
    rr, cc = _region_pos(region)
    dr, dc = DIR_OFFSETS[direction]
    nr = rr + dr
    nc = cc + dc
    assert 0 <= nr < 2 and 0 <= nc < 2, "arrow must point to an adjacent region"
    return nr * 2 + nc


def _valid_directions(region: int) -> list[str]:
    rr, cc = _region_pos(region)
    choices = []
    for direction, (dr, dc) in DIR_OFFSETS.items():
        nr = rr + dr
        nc = cc + dc
        if 0 <= nr < 2 and 0 <= nc < 2:
            choices.append(direction)
    return choices


def _direction_between(src: int, dst: int) -> str:
    sr, sc = _region_pos(src)
    tr, tc = _region_pos(dst)
    dr = tr - sr
    dc = tc - sc
    for direction, delta in DIR_OFFSETS.items():
        if delta == (dr, dc):
            return direction
    raise AssertionError("regions are not adjacent")


def _separator_indices(g: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected exactly one full grey row and one full grey column"
    sep_r = int(grey_rows[0])
    sep_c = int(grey_cols[0])
    assert 0 < sep_r < g.shape[0] - 1 and 0 < sep_c < g.shape[1] - 1
    return sep_r, sep_c


def _arrow_cells(top: int, left: int, direction: str) -> tuple[tuple[int, int], ...]:
    pattern = ARROW_PATTERNS[direction]
    cells = []
    for dr, dc in np.argwhere(pattern):
        cells.append((top + int(dr), left + int(dc)))
    return tuple(sorted(cells))


def _match_arrow_pattern(local_cells: np.ndarray) -> str:
    for direction, pattern in ARROW_PATTERNS.items():
        if np.array_equal(local_cells, pattern):
            return direction
    raise AssertionError("maroon pixels do not form a valid 3x3 arrow")


def _parse_region(region_grid: np.ndarray, region_id: int, abs_r0: int, abs_c0: int) -> tuple[int, Arrow | None]:
    maroon = np.argwhere(region_grid == MAROON)
    non_maroon = region_grid[region_grid != MAROON]
    assert non_maroon.size > 0
    colors = np.unique(non_maroon)
    assert len(colors) == 1, "each region must have a single fill color"
    fill_color = int(colors[0])
    assert fill_color not in (BLACK, GREY, MAROON)

    if maroon.size == 0:
        return fill_color, None

    assert len(maroon) == 5, "an arrow region must contain exactly five maroon pixels"
    r0, c0 = maroon.min(axis=0)
    r1, c1 = maroon.max(axis=0)
    assert int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3, "arrow must fit in a 3x3 box"
    local = np.zeros((3, 3), dtype=bool)
    for r, c in maroon:
        local[int(r - r0), int(c - c0)] = True
    direction = _match_arrow_pattern(local)
    top = abs_r0 + int(r0)
    left = abs_c0 + int(c0)
    return fill_color, Arrow(region_id, top, left, direction, _arrow_cells(top, left, direction))


def _current_region_color(g: np.ndarray, bounds: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = bounds
    sub = g[r0:r1, c0:c1]
    colors = np.unique(sub[sub != MAROON])
    assert len(colors) == 1
    return int(colors[0])


def _recolor_region_fill(g: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bounds
    sub = g[r0:r1, c0:c1]
    sub[sub != MAROON] = color


def _erase_arrow(g: np.ndarray, arrow: Arrow, color: int) -> None:
    for r, c in arrow.cells:
        g[r, c] = color


def _sorted_arrows(arrows: list[Arrow]) -> list[Arrow]:
    return sorted(arrows, key=lambda arrow: (arrow.top, arrow.left))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow region's color into the adjacent region it points to, in row-major arrow order.
    """
    g = np.asarray(grid, dtype=int)
    sep_r, sep_c = _separator_indices(g)
    bounds = _region_bounds(g.shape, sep_r, sep_c)

    region_colors: list[int] = []
    arrows: list[Arrow] = []
    for region_id in range(4):
        r0, r1, c0, c1 = bounds[region_id]
        color, arrow = _parse_region(g[r0:r1, c0:c1], region_id, r0, c0)
        region_colors.append(color)
        if arrow is not None:
            _ = _target_region(region_id, arrow.direction)
            arrows.append(arrow)

    assert len(set(region_colors)) == 4, "the four region colors must be distinct"
    assert len(arrows) == 2, "expected exactly two arrows"
    assert len({arrow.region for arrow in arrows}) == 2, "arrows must occupy different regions"

    out = g.copy()
    for arrow in _sorted_arrows(arrows):
        src_bounds = bounds[arrow.region]
        color = _current_region_color(out, src_bounds)
        target = _target_region(arrow.region, arrow.direction)
        _recolor_region_fill(out, bounds[target], color)
        _erase_arrow(out, arrow, color)

    assert np.all(out[sep_r, :] == GREY) and np.all(out[:, sep_c] == GREY)
    assert not np.any(out == MAROON), "all arrows must be erased in the final output"
    return out


def _paint_region_grid(shape: tuple[int, int], sep_r: int, sep_c: int, region_colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    bounds = _region_bounds(shape, sep_r, sep_c)
    for region_id, color in enumerate(region_colors):
        r0, r1, c0, c1 = bounds[region_id]
        grid[r0:r1, c0:c1] = int(color)
    grid[sep_r, :] = GREY
    grid[:, sep_c] = GREY
    return grid


def _arrow_anchor(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = bounds
    height = r1 - r0
    width = c1 - c0
    assert height >= 5 and width >= 5
    top = random.randint(r0 + 1, r1 - 4)
    left = random.randint(c0 + 1, c1 - 4)
    return top, left


def _build_input(shape: tuple[int, int], sep_r: int, sep_c: int, region_colors: list[int], arrows: list[Arrow]) -> np.ndarray:
    grid = _paint_region_grid(shape, sep_r, sep_c, region_colors)
    for arrow in arrows:
        for r, c in arrow.cells:
            grid[r, c] = MAROON
    return grid


def _oracle_output(shape: tuple[int, int], sep_r: int, sep_c: int, region_colors: list[int], arrows: list[Arrow]) -> np.ndarray:
    colors = list(region_colors)
    for arrow in _sorted_arrows(arrows):
        target = _target_region(arrow.region, arrow.direction)
        colors[target] = colors[arrow.region]
    return _paint_region_grid(shape, sep_r, sep_c, colors)


def _random_arrow_pair(cascade: bool) -> tuple[int, int]:
    if cascade:
        return random.choice(ADJACENT_REGION_PAIRS)
    a, b = random.sample(range(4), 2)
    return int(a), int(b)


def _choose_arrows(bounds: dict[int, tuple[int, int, int, int]], cascade: bool) -> list[Arrow]:
    region_a, region_b = _random_arrow_pair(cascade)
    top_a, left_a = _arrow_anchor(bounds[region_a])
    top_b, left_b = _arrow_anchor(bounds[region_b])
    provisional = [
        Arrow(region_a, top_a, left_a, "up", _arrow_cells(top_a, left_a, "up")),
        Arrow(region_b, top_b, left_b, "up", _arrow_cells(top_b, left_b, "up")),
    ]
    ordered = _sorted_arrows(provisional)
    first_region = ordered[0].region
    second_region = ordered[1].region

    directions: dict[int, str] = {}
    if cascade:
        directions[first_region] = _direction_between(first_region, second_region)
        directions[second_region] = random.choice(_valid_directions(second_region))
    else:
        first_choices = _valid_directions(first_region)
        first_choices = [
            direction
            for direction in first_choices
            if _target_region(first_region, direction) != second_region
        ]
        assert first_choices
        directions[first_region] = random.choice(first_choices)
        directions[second_region] = random.choice(_valid_directions(second_region))

    result = []
    for region, top, left in [(region_a, top_a, left_a), (region_b, top_b, left_b)]:
        direction = directions[region]
        result.append(Arrow(region, top, left, direction, _arrow_cells(top, left, direction)))
    return result


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(5, 10)
    bottom_h = random.randint(5, 10)
    left_w = random.randint(5, 10)
    right_w = random.randint(5, 10)
    sep_r = top_h
    sep_c = left_w
    shape = (top_h + 1 + bottom_h, left_w + 1 + right_w)
    bounds = _region_bounds(shape, sep_r, sep_c)

    region_colors = random.sample(REGION_COLOR_POOL, 4)
    cascade = random.choice([False, True])
    arrows = _choose_arrows(bounds, cascade)

    grid = _build_input(shape, sep_r, sep_c, region_colors, arrows)
    expected = _oracle_output(shape, sep_r, sep_c, region_colors, arrows)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
