from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


CHECKER_COLORS = (0, 5)
DIAMOND_COLOR = 6
BLOCK_COLORS = (1, 2, 3, 4)
RAY_ORDER = (
    ((0, 0), (0, 0), (1, 1)),
    ((0, 1), (0, -1), (1, -1)),
    ((1, 0), (-1, 0), (-1, 1)),
    ((1, 1), (-1, -1), (-1, -1)),
)
GRID_SIZES = (13, 15, 17, 19, 21, 23, 25)
MAX_GENERATION_ATTEMPTS = 200


def _top_two_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.ravel()))
    top = [int(color) for color, _ in counts.most_common(2)]
    assert len(top) == 2
    return top[0], top[1]


def _checker_value(r: int, c: int, top_left: int) -> int:
    return int(top_left if (r + c) % 2 == 0 else CHECKER_COLORS[0] + CHECKER_COLORS[1] - top_left)


def _make_checkerboard(n: int, top_left: int) -> np.ndarray:
    rows, cols = np.indices((n, n))
    other = CHECKER_COLORS[0] + CHECKER_COLORS[1] - int(top_left)
    return np.where((rows + cols) % 2 == 0, int(top_left), other).astype(int)


def _diamond_distance_mask(n: int) -> np.ndarray:
    center = n // 2
    rows, cols = np.indices((n, n))
    return np.abs(rows - center) + np.abs(cols - center)


def _diamond_outline(n: int, radius: int) -> np.ndarray:
    return _diamond_distance_mask(n) == int(radius)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_block(grid: np.ndarray, bg_colors: set[int]) -> tuple[int, int, np.ndarray]:
    mask = ~np.isin(grid, list(bg_colors) + [DIAMOND_COLOR])
    coords = np.argwhere(mask)
    assert coords.shape == (4, 2), "expected exactly four non-background non-diamond cells"
    r0, c0, r1, c1 = _bbox_of(mask)
    assert (r1 - r0, c1 - c0) == (1, 1), "foreground cells must form a 2x2 block"
    block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert block.shape == (2, 2)
    assert not np.isin(block, list(bg_colors) + [DIAMOND_COLOR]).any()
    assert len({int(v) for v in block.ravel()}) == 4
    return r0, c0, block


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1

    n = int(g.shape[0])
    bg_colors = set(_top_two_colors(g))
    assert bg_colors == set(CHECKER_COLORS), "background must be the black/grey checkerboard"
    top_left = int(g[0, 0])
    assert top_left in bg_colors

    r0, c0, block = _find_block(g, bg_colors)
    out = g.copy()

    for dr in range(2):
        for dc in range(2):
            out[r0 + dr, c0 + dc] = _checker_value(r0 + dr, c0 + dc, top_left)

    corners = [
        ((0, 0), int(block[0, 0]), (1, 1)),
        ((0, n - 1), int(block[0, 1]), (1, -1)),
        ((n - 1, 0), int(block[1, 0]), (-1, 1)),
        ((n - 1, n - 1), int(block[1, 1]), (-1, -1)),
    ]
    for (r, c), color, _ in corners:
        out[r, c] = int(color)

    for (r, c), color, (dr, dc) in corners:
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < n and int(out[nr, nc]) in bg_colors:
            out[nr, nc] = int(color)
            nr += dr
            nc += dc

    return out


def _ray_length(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int], color: int) -> int:
    n = grid.shape[0]
    r, c = start
    dr, dc = direction
    length = 0
    r += dr
    c += dc
    while 0 <= r < n and 0 <= c < n and int(grid[r, c]) == int(color):
        length += 1
        r += dr
        c += dc
    return length


def _valid_block_positions(n: int, radius: int) -> list[tuple[int, int]]:
    dist = _diamond_distance_mask(n)
    outline = dist == int(radius)
    forbidden = outline.copy()
    for r, c in np.argwhere(outline):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    forbidden[nr, nc] = True

    positions: list[tuple[int, int]] = []
    for r0 in range(1, n - 2):
        for c0 in range(1, n - 2):
            cells = [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
            if any(forbidden[r, c] for r, c in cells):
                continue
            if any(int(dist[r, c]) < int(radius) for r, c in cells):
                continue
            positions.append((r0, c0))
    return positions


def _permutation_index(block: np.ndarray) -> int:
    perms = list(itertools.permutations(BLOCK_COLORS))
    flat = tuple(int(v) for v in block.ravel())
    return perms.index(flat)


def _validate_instance(grid: np.ndarray, radius: int, top_left: int, block: np.ndarray) -> None:
    assert grid.shape[0] in GRID_SIZES and grid.shape[0] == grid.shape[1]
    assert int(grid[0, 0]) == int(top_left)
    bg_mask = np.isin(grid, CHECKER_COLORS)
    assert np.count_nonzero(grid == DIAMOND_COLOR) > 0
    assert np.count_nonzero(~bg_mask & (grid != DIAMOND_COLOR)) == 4

    out = main(grid)
    n = grid.shape[0]
    assert out.shape == grid.shape
    assert np.array_equal(out[_diamond_outline(n, radius)], np.full(np.count_nonzero(_diamond_outline(n, radius)), DIAMOND_COLOR))
    assert int(out[0, 0]) == int(block[0, 0])
    assert int(out[0, -1]) == int(block[0, 1])
    assert int(out[-1, 0]) == int(block[1, 0])
    assert int(out[-1, -1]) == int(block[1, 1])

    ray_lengths = [
        _ray_length(out, (0, 0), (1, 1), int(block[0, 0])),
        _ray_length(out, (0, n - 1), (1, -1), int(block[0, 1])),
        _ray_length(out, (n - 1, 0), (-1, 1), int(block[1, 0])),
        _ray_length(out, (n - 1, n - 1), (-1, -1), int(block[1, 1])),
    ]
    if radius % 2 == 0:
        expected = (n - radius - 3) // 2
        assert expected >= 3
        assert ray_lengths == [expected, expected, expected, expected]
    else:
        assert ray_lengths == [n - 2, n - 2, 0, 0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 2x2 color block to the four corners, then fire corner rays across a checkerboard until they hit the diamond or an occupied diagonal cell.

    Input structure:
    - The grid is an odd square filled with a black/grey checkerboard background.
    - A hollow magenta diamond outline sits around the center.
    - One separate 2x2 foreground block contains four distinct colors and lies outside the diamond.

    Rule:
    - Move the 2x2 block's top-left, top-right, bottom-left, and bottom-right colors to the matching grid corners, and restore the old block location to checkerboard background.
    - From those four corner pixels, cast diagonal rays toward the center in the fixed order top-left, top-right, bottom-left, bottom-right.
    - Each ray paints only checkerboard cells; it stops as soon as the next diagonal cell is already occupied by the diamond, a previously drawn ray, or an already occupied corner.

    Key insight:
    - The small 2x2 block is a corner-position code, and after the expansion its corner colors become the ray sources whose diagonal growth is clipped by the central diamond or by earlier rays on the same diagonal.
    """
    return _apply_rule(grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = int(np.random.choice(GRID_SIZES))
        top_left = int(np.random.choice(CHECKER_COLORS))
        odd_radii = [r for r in range(3, n // 2, 2)]
        even_radii = [r for r in range(4, n // 2, 2)]
        use_odd = bool(np.random.randint(0, 2))
        radii = odd_radii if use_odd else even_radii
        if not radii:
            radii = odd_radii or even_radii
        radius = int(np.random.choice(radii))

        block = np.array(np.random.permutation(BLOCK_COLORS), dtype=int).reshape(2, 2)
        positions = _valid_block_positions(n, radius)
        if not positions:
            continue

        positions.sort()
        idx_seed = 97 * n + 31 * radius + 11 * top_left + 7 * _permutation_index(block)
        r0, c0 = positions[idx_seed % len(positions)]

        grid = _make_checkerboard(n, top_left)
        grid[_diamond_outline(n, radius)] = DIAMOND_COLOR
        grid[r0 : r0 + 2, c0 : c0 + 2] = block

        try:
            _validate_instance(grid, radius, top_left, block)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
