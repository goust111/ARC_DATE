from __future__ import annotations

import numpy as np
import itertools
import random


MISSING_COLOR = 6


def _consistent_tile(
    g: np.ndarray, row_period: int, col_period: int
) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r, c in np.argwhere(g != MISSING_COLOR):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(g[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None

    if np.any(tile == -1):
        return None
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a corrupted periodic two-color pattern by replacing each pink marker with the tile value at that position.

    Input structure:
    - The grid contains exactly two visible pattern colors plus pink marker pixels.
    - Ignoring the pink cells, all visible pixels are consistent with one smallest rectangular repeating tile.

    Rule:
    - Treat pink pixels as missing values.
    - Infer the unique minimum-area row/column period whose residues all have a single visible color.
    - Repaint each pink pixel with the inferred tile color at the same grid coordinate.

    Key insight:
    The pink pixels are occlusions in a global periodic tile, not independent objects.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert MISSING_COLOR in colors, "expected pink missing-value pixels"
    visible_colors = colors - {MISSING_COLOR}
    assert len(visible_colors) == 2, "expected exactly two visible pattern colors"

    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _consistent_tile(g, row_period, col_period)
            if tile is not None:
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "visible pixels must define a periodic tile"
    min_area = min(area for area, _row_period, _col_period, _tile in candidates)
    minimal = [
        (row_period, col_period, tile)
        for area, row_period, col_period, tile in candidates
        if area == min_area
    ]
    assert len(minimal) == 1, "minimum-area period must be unique"

    row_period, col_period, tile = minimal[0]
    assert set(map(int, tile.ravel())) == visible_colors, "tile must use both pattern colors"

    out = g.copy()
    for r, c in np.argwhere(out == MISSING_COLOR):
        out[int(r), int(c)] = tile[int(r) % row_period, int(c) % col_period]
    return out

# --------------- generator (o5x) ---------------

# Color and marker choices.
_GEN_MARKER_COLOR = 6
_GEN_PATTERN_COLORS = tuple(c for c in range(10) if c != _GEN_MARKER_COLOR)
_GEN_COLOR_SWAP_PROBABILITY = 0.5

# Tile and canvas bounds.
_GEN_TILE_SIZES = (2, 3)
_GEN_MINORITY_COUNT_BY_SIZE = {2: (1, 1), 3: (1, 4)}
_GEN_MIN_GRID_SIZE = 12
_GEN_MAX_GRID_SIZE = 19

# Corruption rates and retry budget.
_GEN_CORRUPT_MIN_PERCENT = 15
_GEN_CORRUPT_MAX_PERCENT = 25
_GEN_PERCENT_DENOMINATOR = 100
_GEN_MAX_ATTEMPTS = 300


def _gen_min_period_area(tile: np.ndarray) -> int:
    h, w = tile.shape
    areas = []
    for rh in range(1, h + 1):
        for cw in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if tile[r, c] != tile[r % rh, c % cw]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                areas.append(rh * cw)
    return min(areas)


def _gen_tile_from_coords(
    majority: int,
    minority: int,
    size: int,
    minority_coords: tuple[tuple[int, int], ...],
) -> np.ndarray:
    tile = np.full((size, size), majority, dtype=int)
    for r, c in minority_coords:
        tile[r, c] = minority
    return tile


def _gen_make_tile(colors: tuple[int, int], size: int) -> np.ndarray:
    majority, minority = colors
    area = size * size
    minority_low, minority_high = _GEN_MINORITY_COUNT_BY_SIZE[size]
    minority_count = minority_low
    if minority_low != minority_high:
        minority_count = random.randint(minority_low, minority_high)
    cell_coords = [(r, c) for r in range(size) for c in range(size)]

    valid_sets: list[tuple[tuple[int, int], ...]] = []
    for coords in itertools.combinations(cell_coords, minority_count):
        tile = _gen_tile_from_coords(majority, minority, size, coords)
        if _gen_min_period_area(tile) == area:
            valid_sets.append(coords)

    coords = random.choice(valid_sets)
    return _gen_tile_from_coords(majority, minority, size, coords)


def _gen_tile_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _gen_corrupt_output(out: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray | None:
    height, width = out.shape
    total = height * width
    low = (
        _GEN_CORRUPT_MIN_PERCENT * total + _GEN_PERCENT_DENOMINATOR - 1
    ) // _GEN_PERCENT_DENOMINATOR
    high = _GEN_CORRUPT_MAX_PERCENT * total // _GEN_PERCENT_DENOMINATOR
    if low > high:
        return None
    corrupt_count = random.randint(low, high)

    all_coords = [(r, c) for r in range(height) for c in range(width)]
    preserve: set[tuple[int, int]] = set()
    tile_h, tile_w = tile_shape
    for rr in range(tile_h):
        for cc in range(tile_w):
            residue_coords = [
                (r, c)
                for r in range(rr, height, tile_h)
                for c in range(cc, width, tile_w)
            ]
            preserve.add(random.choice(residue_coords))

    corrupt: set[tuple[int, int]] = set()
    colors = sorted(int(v) for v in np.unique(out))
    for color in colors:
        choices = [
            (r, c)
            for r, c in all_coords
            if int(out[r, c]) == color and (r, c) not in preserve
        ]
        if not choices:
            return None
        corrupt.add(random.choice(choices))

    remaining = [
        coord for coord in all_coords if coord not in preserve and coord not in corrupt
    ]
    if len(remaining) < corrupt_count - len(corrupt):
        return None
    corrupt.update(random.sample(remaining, corrupt_count - len(corrupt)))

    inp = out.copy()
    rows, cols = zip(*corrupt)
    inp[np.array(rows), np.array(cols)] = _GEN_MARKER_COLOR
    return inp


def _gen_attempt() -> np.ndarray | None:
    c1, c2 = random.sample(_GEN_PATTERN_COLORS, 2)
    if random.random() < _GEN_COLOR_SWAP_PROBABILITY:
        c1, c2 = c2, c1

    size = random.choice(_GEN_TILE_SIZES)
    tile = _gen_make_tile((c1, c2), size)
    height = random.randint(_GEN_MIN_GRID_SIZE, _GEN_MAX_GRID_SIZE)
    width = random.randint(_GEN_MIN_GRID_SIZE, _GEN_MAX_GRID_SIZE)
    out = _gen_tile_grid(tile, height, width)
    inp = _gen_corrupt_output(out, tile.shape)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f823c43c input")
