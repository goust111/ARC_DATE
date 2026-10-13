"""ARC task: rearrange separated one-hole tiles by their coordinate pointers."""

from __future__ import annotations

import random

import numpy as np


def _order_from_side(side: int) -> int:
    """Return the unique n with side == n*n+n-1."""
    candidates = [n for n in range(2, 6) if n * n + n - 1 == side]
    assert len(candidates) == 1, "grid side must have the form n*n+n-1"
    return candidates[0]


def _tile_start(index: int, n: int) -> int:
    return index * (n + 1)


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely most frequent"
    return int(winners[0])


def _read_tiles(
    grid: np.ndarray, n: int, bg: int
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Map each occupied array coordinate to (fill color, local hole)."""
    tiles: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for tr in range(n):
        for tc in range(n):
            r0, c0 = _tile_start(tr, n), _tile_start(tc, n)
            tile = grid[r0 : r0 + n, c0 : c0 + n]
            non_bg = tile[tile != bg]
            if non_bg.size == 0:
                continue
            colors = np.unique(non_bg)
            assert len(colors) == 1, "each tile must be monochromatic"
            holes = np.argwhere(tile == bg)
            assert len(holes) == 1, "each occupied tile must have exactly one hole"
            assert non_bg.size == n * n - 1, "tile must otherwise be solid"
            hole = tuple(map(int, holes[0]))
            tiles[(tr, tc)] = (int(colors[0]), hole)
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """Rearrange one-hole tiles by their hole pointers and relabel them at destination.

    Input structure:
    A square background grid of side n*n+n-1 is split by single background
    rows and columns into an n by n array of n by n cells.  Each occupied cell
    contains a monochrome solid tile with exactly one background hole; empty
    array cells and all separators are background.

    Rule:
    Read a tile's local hole coordinate (r,c) as its destination array
    coordinate.  The pointers must be a bijection of the occupied positions.
    Fill the old pointer hole, move the solid tile to (r,c), then dig its one
    output hole at local coordinate (r,c).  Leave separators and empty cells
    background.

    Key insight:
    The same visible coordinate code is read as a destination pointer in the
    input and written as a self-location label in the output; tile colors make
    the pointer-directed permutation visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h <= 30, "expected an ARC-sized square grid"
    n = _order_from_side(h)
    bg = _background(g)

    # Validate the stated global reference frame before reading local tiles.
    for i in range(1, n):
        sep = i * (n + 1) - 1
        assert np.all(g[sep, :] == bg), "separator row must be background"
        assert np.all(g[:, sep] == bg), "separator column must be background"

    tiles = _read_tiles(g, n, bg)
    assert len(tiles) >= 2, "expected at least two occupied tiles"
    occupied = set(tiles)
    destinations = [hole for _color, hole in tiles.values()]
    assert set(destinations) == occupied, "holes must point exactly to occupied positions"
    assert len(destinations) == len(set(destinations)), "tile pointers must be one-to-one"

    out = np.full_like(g, bg)
    for _source, (color, destination) in tiles.items():
        tr, tc = destination
        r0, c0 = _tile_start(tr, n), _tile_start(tc, n)

        # These two assignments deliberately spell out the read/write steps:
        # first restore a solid moved tile, then write its destination label.
        out[r0 : r0 + n, c0 : c0 + n] = color
        out[r0 + tr, c0 + tc] = bg
    return out


def _deranged_cycle(items: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    """Create one directed cycle, hence a bijection with no fixed points."""
    order = items.copy()
    random.shuffle(order)
    step = random.randrange(1, len(order))
    while np.gcd(step, len(order)) != 1:
        step = random.randrange(1, len(order))
    return {pos: order[(i + step) % len(order)] for i, pos in enumerate(order)}


def _independent_expected(
    shape: tuple[int, int], n: int, bg: int,
    colors: dict[tuple[int, int], int],
    pointers: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Generator oracle built directly from the declarative destination rule."""
    expected = np.full(shape, bg, dtype=int)
    for source, destination in pointers.items():
        tr, tc = destination
        r0, c0 = _tile_start(tr, n), _tile_start(tc, n)
        expected[r0 : r0 + n, c0 : c0 + n] = colors[source]
        expected[r0 + tr, c0 + tc] = bg
    return expected


def _generate_attempt() -> np.ndarray | None:
    n = random.choice((2, 3, 4, 5))
    side = n * n + n - 1
    all_positions = [(r, c) for r in range(n) for c in range(n)]

    # Include sparse and dense arrays, while retaining distinct visible colors.
    max_tiles = min(9, n * n)
    if random.random() < 0.35:
        count = max_tiles
    else:
        count = random.randint(2, max_tiles)
    occupied = random.sample(all_positions, count)
    pointers = _deranged_cycle(occupied)

    palette = list(range(10))
    bg = random.choice(palette)
    foreground = [c for c in palette if c != bg]
    chosen = random.sample(foreground, count)
    colors = dict(zip(occupied, chosen))

    grid = np.full((side, side), bg, dtype=int)
    for source in occupied:
        sr, sc = source
        r0, c0 = _tile_start(sr, n), _tile_start(sc, n)
        grid[r0 : r0 + n, c0 : c0 + n] = colors[source]
        hr, hc = pointers[source]
        grid[r0 + hr, c0 + hc] = bg

    expected = _independent_expected(grid.shape, n, bg, colors, pointers)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None

    # Semantic/non-degeneracy checks, not merely generator/solver closure.
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if any(source == destination for source, destination in pointers.items()):
        return None
    # Every occupied location changes visible tile color, defeating relabel-only.
    input_colors = {pos: colors[pos] for pos in occupied}
    output_colors = {destination: colors[source] for source, destination in pointers.items()}
    if any(input_colors[pos] == output_colors[pos] for pos in occupied):
        return None
    # Independently re-read the result: every output hole must self-label.
    out_tiles = _read_tiles(actual, n, bg)
    if set(out_tiles) != set(occupied):
        return None
    if any(hole != pos for pos, (_color, hole) in out_tiles.items()):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-identity pointer permutation with a checked oracle."""
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
