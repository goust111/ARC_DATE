from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


Region = tuple[int, int, int, int]  # (r0, c0, h, w)

RED = Color.RED
BLUE = Color.BLUE
BG_OPTIONS = (Color.BLACK, Color.GREY, Color.YELLOW)

DEPTH_OPTIONS = (0, 1, 1, 2, 2, 3, 3)
LEAF_RANGE_BY_DEPTH = {
    0: (6, 18),
    1: (4, 12),
    2: (2, 7),
    3: (1, 3),
}
MAX_GENERATION_ATTEMPTS = 300


def _region_slices(region: Region) -> tuple[slice, slice]:
    r0, c0, h, w = region
    return slice(r0, r0 + h), slice(c0, c0 + w)


def _region_contains_color(grid: np.ndarray, region: Region, color: int) -> bool:
    rs, cs = _region_slices(region)
    return bool(np.any(grid[rs, cs] == int(color)))


def _child_region(region: Region, point: tuple[int, int]) -> Region:
    r0, c0, h, w = region
    assert h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2, "active region must split evenly"
    half_h = h // 2
    half_w = w // 2
    r, c = point
    nr0 = r0 if r < r0 + half_h else r0 + half_h
    nc0 = c0 if c < c0 + half_w else c0 + half_w
    return nr0, nc0, half_h, half_w


def _iter_region_cells(region: Region):
    r0, c0, h, w = region
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            yield r, c


def _resolve_final_region(grid: np.ndarray) -> tuple[Region, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, RED, BLUE}, "input may only contain background, one red marker, and blue fuel"

    red_coords = np.argwhere(g == RED)
    assert len(red_coords) == 1, "expected exactly one red marker pixel"
    red = tuple(map(int, red_coords[0]))

    region = (0, 0, int(g.shape[0]), int(g.shape[1]))
    while _region_contains_color(g, region, BLUE):
        region = _child_region(region, red)
    return region, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly quarter the red-selected region while it still contains blue fuel, then fill the terminal region red.

    Input structure:
    - The grid has a single red control pixel, zero or more blue pixels, and one background color.
    - Whenever the currently selected region still contains blue, that region's height and width are both even.
    - The red control pixel acts as an address pointer that always lies inside exactly one quadrant of every required split.

    Rule:
    - Start with the whole grid as the current region.
    - If the current region contains at least one blue pixel, split it into equal top-left, top-right, bottom-left, and bottom-right quadrants.
    - Replace the current region by the quadrant containing the red pixel, and repeat until the current region contains no blue.
    - Clear the output to the background color and fill the final selected rectangular region entirely red.

    Key insight:
    - Blue pixels are recursive fuel, not a global counter: only blue pixels that remain inside the repeatedly selected red-address path can force another split, while off-path blues disappear from consideration as soon as the path branches away from them.
    """
    g = np.asarray(grid, dtype=int)
    final_region, bg = _resolve_final_region(g)
    out = np.full_like(g, bg)
    rs, cs = _region_slices(final_region)
    out[rs, cs] = RED
    return out


def _sample_shape(depth: int) -> tuple[int, int]:
    scale = 2 ** depth
    leaf_lo, leaf_hi = LEAF_RANGE_BY_DEPTH[depth]
    max_leaf_h = min(leaf_hi, 30 // scale)
    max_leaf_w = min(leaf_hi, 30 // scale)
    leaf_h = random.randint(leaf_lo, max_leaf_h)
    leaf_w = random.randint(leaf_lo, max_leaf_w)
    return leaf_h * scale, leaf_w * scale


def _sample_target_path(shape: tuple[int, int], depth: int) -> list[Region]:
    region = (0, 0, int(shape[0]), int(shape[1]))
    path = [region]
    for _ in range(depth):
        r0, c0, h, w = region
        half_h = h // 2
        half_w = w // 2
        nr0 = r0 + random.randrange(2) * half_h
        nc0 = c0 + random.randrange(2) * half_w
        region = (nr0, nc0, half_h, half_w)
        path.append(region)
    return path


def _sample_cell(region: Region) -> tuple[int, int]:
    r0, c0, h, w = region
    return random.randrange(r0, r0 + h), random.randrange(c0, c0 + w)


def _difference_cells(outer: Region, inner: Region) -> list[tuple[int, int]]:
    inner_r0, inner_c0, inner_h, inner_w = inner
    inner_r1 = inner_r0 + inner_h
    inner_c1 = inner_c0 + inner_w
    cells = []
    for r, c in _iter_region_cells(outer):
        if inner_r0 <= r < inner_r1 and inner_c0 <= c < inner_c1:
            continue
        cells.append((r, c))
    return cells


def _off_terminal_cells(shape: tuple[int, int], terminal: Region) -> list[tuple[int, int]]:
    r0, c0, h, w = terminal
    r1 = r0 + h
    c1 = c0 + w
    cells = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if r0 <= r < r1 and c0 <= c < c1:
                continue
            cells.append((r, c))
    return cells


def _semantic_checks(grid: np.ndarray, depth: int, path: list[Region], bg: int) -> bool:
    reds = np.argwhere(grid == RED)
    blues = np.argwhere(grid == BLUE)
    if len(reds) != 1:
        return False

    terminal = path[-1]
    for region in path[:-1]:
        if not _region_contains_color(grid, region, BLUE):
            return False
    if _region_contains_color(grid, terminal, BLUE):
        return False

    if depth > 0:
        total_blue = int(len(blues))
        if total_blue <= depth:
            return False

    out = main(grid)
    expected = np.full_like(grid, int(bg))
    rs, cs = _region_slices(terminal)
    expected[rs, cs] = RED
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    depth = random.choice(DEPTH_OPTIONS)
    shape = _sample_shape(depth)
    path = _sample_target_path(shape, depth)
    terminal = path[-1]

    bg = int(random.choice(BG_OPTIONS))
    grid = np.full(shape, bg, dtype=int)
    red = _sample_cell(terminal)
    grid[red] = RED

    used = {red}
    for level in range(depth):
        fuel_candidates = [cell for cell in _difference_cells(path[level], path[level + 1]) if cell not in used]
        if not fuel_candidates:
            return None
        fuel_cell = random.choice(fuel_candidates)
        grid[fuel_cell] = BLUE
        used.add(fuel_cell)

    if depth > 0:
        distractor_candidates = [cell for cell in _off_terminal_cells(shape, terminal) if cell not in used]
        if not distractor_candidates:
            return None
        max_extra = min(len(distractor_candidates), max(1, (shape[0] * shape[1]) // 10))
        extra_count = random.randint(1, max_extra)
        for cell in random.sample(distractor_candidates, k=extra_count):
            grid[cell] = BLUE
            used.add(cell)

    if not _semantic_checks(grid, depth, path, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
