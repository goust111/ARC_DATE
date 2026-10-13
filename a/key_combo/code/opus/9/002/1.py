from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BLUE = Color.BLUE
BLACK = Color.BLACK
RANK_COLORS = (Color.RED, Color.GREEN, Color.YELLOW, Color.GREY)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    lab, n = label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(lab == k) for k in range(1, n + 1)]


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _region_sort_key(coords: np.ndarray) -> tuple[int, float, float]:
    center = coords.mean(axis=0)
    return (int(coords.shape[0]), float(center[0]), float(center[1]))


def _completed_boundary(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid so 90-degree rotation keeps the same center pixel"

    blue = g == BLUE
    restored = blue.copy()
    cur = blue
    for _ in range(3):
        cur = np.rot90(cur, -1)
        restored |= cur

    out = np.zeros_like(g)
    out[restored] = BLUE
    return out


def _enclosed_regions(boundary_grid: np.ndarray) -> list[np.ndarray]:
    regions = []
    for coords in _components(boundary_grid == BLACK):
        if not _touches_border(coords, boundary_grid.shape):
            regions.append(coords)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a quarter-missing rotational blue boundary, then color enclosed black regions by increasing area.

    Input structure:
    - The grid is an odd square with black background and blue single-cell-wide
      boundary pixels arranged around the center.
    - The intended full blue pattern is 4-fold rotationally symmetric, but one
      quadrant has some blue boundary cells erased to black.
    - After restoration, the blue pattern encloses several black regions that do
      not touch the grid border.

    Rule:
    - Rotate the observed blue pixels around the center by 90, 180, and 270
      degrees, and add any missing rotated blue cells back into the grid.
    - In the restored grid, find every 4-connected black region that does not
      touch any border cell.
    - Sort those enclosed regions by area from small to large, breaking ties by
      centroid row then centroid column.
    - Fill the sorted regions with red, green, yellow, then grey while leaving
      blue boundary pixels and the exterior black background unchanged.

    Key insight:
    - The missing blue cells matter because they determine which black areas are
      actually closed; only after rotational completion can the enclosed-region
      areas be measured and mapped to output colors.
    """
    restored = _completed_boundary(grid)
    out = restored.copy()

    regions = sorted(_enclosed_regions(restored), key=_region_sort_key)
    assert len(regions) <= len(RANK_COLORS), "too many enclosed regions for the fixed color ranking"
    for color, coords in zip(RANK_COLORS, regions):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _draw_square_frame(grid: np.ndarray, radius: int) -> None:
    n = grid.shape[0]
    c = n // 2
    r0 = c - radius
    r1 = c + radius
    grid[r0, r0 : r1 + 1] = BLUE
    grid[r1, r0 : r1 + 1] = BLUE
    grid[r0 : r1 + 1, r0] = BLUE
    grid[r0 : r1 + 1, r1] = BLUE


def _sample_radii(size: int, n_loops: int) -> list[int] | None:
    center = size // 2
    candidates = list(range(2, center - 1))
    if len(candidates) < n_loops:
        return None

    for _ in range(80):
        radii = sorted(random.sample(candidates, n_loops), reverse=True)
        if any(radii[i] - radii[i + 1] < 2 for i in range(len(radii) - 1)):
            continue
        if radii[0] > center - 2:
            continue
        return radii
    return None


def _full_grid(size: int, radii: list[int]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for radius in radii:
        _draw_square_frame(grid, radius)
    return grid


def _rotated_cells(cells: set[tuple[int, int]], size: int, quadrant: int) -> set[tuple[int, int]]:
    if quadrant == 0:
        return set(cells)
    cur = set(cells)
    for _ in range(quadrant):
        cur = {(c, size - 1 - r) for r, c in cur}
    return cur


def _corner_arc(size: int, radius: int, span_h: int, span_v: int) -> set[tuple[int, int]]:
    c = size // 2
    top = c - radius
    right = c + radius
    cells = {(top, col) for col in range(right - span_h + 1, right + 1)}
    cells |= {(row, right) for row in range(top, top + span_v)}
    return cells


def _erase_quadrant_arcs(full: np.ndarray, radii: list[int]) -> tuple[np.ndarray, set[tuple[int, int]]]:
    size = full.shape[0]
    quadrant = random.randrange(4)
    erase = set()

    broken_count = random.randint(1, min(2, len(radii)))
    broken_radii = [radii[0]]
    if broken_count == 2 and len(radii) >= 2:
        broken_radii.append(random.choice(radii[1:]))

    for radius in broken_radii:
        max_span = max(2, radius - 1)
        span_h = random.randint(2, max_span)
        span_v = random.randint(2, max_span)
        erase |= _rotated_cells(_corner_arc(size, radius, span_h, span_v), size, quadrant)

    inp = full.copy()
    for r, c in erase:
        inp[r, c] = BLACK
    return inp, erase


def _blue_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int) == BLUE


def _validate_sample(inp: np.ndarray, full: np.ndarray, erase: set[tuple[int, int]], n_loops: int) -> bool:
    if not erase:
        return False
    if not np.array_equal(_completed_boundary(inp), full):
        return False

    before_regions = _enclosed_regions(inp)
    after_regions = _enclosed_regions(full)
    if len(after_regions) != n_loops or not (2 <= len(after_regions) <= 4):
        return False
    if len(before_regions) >= len(after_regions):
        return False

    areas = [int(coords.shape[0]) for coords in after_regions]
    if len(set(areas)) != len(areas):
        return False

    out = main(inp)
    expected_blue = _blue_mask(full)
    if not np.array_equal(_blue_mask(out), expected_blue):
        return False

    sorted_regions = sorted(after_regions, key=_region_sort_key)
    for idx, coords in enumerate(sorted_regions):
        want = int(RANK_COLORS[idx])
        vals = out[coords[:, 0], coords[:, 1]]
        if not np.all(vals == want):
            return False

    colored = np.isin(out, RANK_COLORS)
    if int(np.sum(colored)) != sum(int(coords.shape[0]) for coords in after_regions):
        return False
    if np.any(out[(full == BLACK) & (~colored)] != BLACK):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice((15, 17, 19, 21, 23, 25))
    n_loops = random.randint(2, 4)
    radii = _sample_radii(size, n_loops)
    if radii is None:
        return None

    full = _full_grid(size, radii)
    areas = [coords.shape[0] for coords in _enclosed_regions(full)]
    if len(areas) != n_loops or len(set(map(int, areas))) != n_loops:
        return None

    inp, erase = _erase_quadrant_arcs(full, radii)
    if np.array_equal(inp, full):
        return None
    if not _validate_sample(inp, full, erase, n_loops):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-completion input")
