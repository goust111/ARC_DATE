from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg, in_bounds, neighbors4


def _adjacent_cells_with_color(g: np.ndarray, cell: tuple[int, int], color: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r, c in neighbors4(*cell)
        if in_bounds(g.shape, r, c) and int(g[r, c]) == int(color)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every red-green domino to a teal pixel at the green cell and erase the red cell.

    Input structure
    - The grid is a black-background scene containing red pixels, green pixels, and grey marker pixels.
    - Some red and green pixels form isolated 4-connected two-cell dominoes; other red/green pixels are unpaired distractors.
    Rule
    - For each 4-connected red-green domino, set the green pixel to teal and set the red pixel to the background color.
    - Preserve grey pixels and all unpaired red/green pixels.
    Key insight
    - Only orthogonal red-green contact matters: the green side survives as the new teal marker while the red side is deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black to be the background"
    present = set(map(int, np.unique(g)))
    assert present <= {Color.BLACK, Color.RED, Color.GREEN, Color.GREY}, "unexpected input color"
    assert {Color.RED, Color.GREEN, Color.GREY} <= present, "expected red, green, and grey pixels"

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    paired_reds: set[tuple[int, int]] = set()
    paired_greens: set[tuple[int, int]] = set()
    for rr, cc in zip(*np.where(g == Color.GREEN)):
        green = (int(rr), int(cc))
        red_neighbors = _adjacent_cells_with_color(g, green, Color.RED)
        if not red_neighbors:
            continue
        assert len(red_neighbors) == 1, "each paired green must touch exactly one red"

        red = red_neighbors[0]
        green_neighbors = _adjacent_cells_with_color(g, red, Color.GREEN)
        assert green_neighbors == [green], "each paired red must touch exactly one green"
        assert green not in paired_greens and red not in paired_reds, "dominoes must not overlap"
        paired_greens.add(green)
        paired_reds.add(red)
        pairs.append((green, red))

    assert pairs, "expected at least one red-green domino"

    out = g.copy()
    for green, red in pairs:
        out[green] = Color.TEAL
        out[red] = bg
    return out

# --------------- generator (o5x) ---------------
# Canvas bounds; max values are exclusive for numpy randint/range use.
GEN_HEIGHT_BOUNDS = (3, 8)
GEN_WIDTH_BOUNDS = (3, 10)

# Object count bounds; max values are exclusive.
GEN_DOMINO_COUNT_BOUNDS = (1, 4)
GEN_UNPAIRED_RED_BOUNDS = (0, 4)
GEN_UNPAIRED_GREEN_BOUNDS = (0, 3)
GEN_GREY_BOUNDS = (1, 6)
GEN_DOMINO_CELL_COUNT = 2

# Sparse-scene acceptance and retry limits.
GEN_DENSITY_BOUNDS = (0.15, 0.35)
GEN_MAX_ATTEMPTS = 300


def _gen_neighbors(cell, shape):
    for rr, cc in neighbors4(*cell):
        rr, cc = int(rr), int(cc)
        if in_bounds(shape, rr, cc):
            yield rr, cc


def _gen_touching_nonblack(grid, cell, allowed=()):
    allowed = set(allowed)
    for rr, cc in _gen_neighbors(cell, grid.shape):
        if (rr, cc) not in allowed and int(grid[rr, cc]) != Color.BLACK:
            return True
    return False


def _gen_domino_candidates(grid):
    candidates = []
    height, width = grid.shape
    for gr in range(height):
        for gc in range(width):
            if int(grid[gr, gc]) != Color.BLACK:
                continue
            green = (gr, gc)
            for red in _gen_neighbors(green, grid.shape):
                rr, rc = red
                if int(grid[rr, rc]) != Color.BLACK:
                    continue
                if _gen_touching_nonblack(grid, green, allowed=[red]):
                    continue
                if _gen_touching_nonblack(grid, red, allowed=[green]):
                    continue
                candidates.append((green, red))
    return candidates


def _gen_single_candidates(grid):
    candidates = []
    height, width = grid.shape
    for rr in range(height):
        for cc in range(width):
            if int(grid[rr, cc]) == Color.BLACK and not _gen_touching_nonblack(grid, (rr, cc)):
                candidates.append((rr, cc))
    return candidates


def _gen_count_options(area):
    min_density, max_density = GEN_DENSITY_BOUNDS
    lo = int(np.ceil(min_density * area))
    hi = int(np.floor(max_density * area))
    options = []
    for domino_count in range(*GEN_DOMINO_COUNT_BOUNDS):
        for red_count in range(*GEN_UNPAIRED_RED_BOUNDS):
            for green_count in range(*GEN_UNPAIRED_GREEN_BOUNDS):
                for grey_count in range(*GEN_GREY_BOUNDS):
                    total = GEN_DOMINO_CELL_COUNT * domino_count + red_count + green_count + grey_count
                    if lo <= total <= hi:
                        options.append((domino_count, red_count, green_count, grey_count))
    return options


def _gen_pick_counts(area):
    options = _gen_count_options(area)
    if not options:
        return None

    chosen = []
    for index in range(len(options[0])):
        values = sorted({opt[index] for opt in options})
        value = int(np.random.choice(values))
        chosen.append(value)
        options = [opt for opt in options if opt[index] == value]
    return tuple(chosen)


def _gen_place_dominoes(grid, expected, domino_count):
    """Mutate both grids with isolated green-red dominoes; return pairs or None."""
    pairs = []
    for _ in range(domino_count):
        candidates = _gen_domino_candidates(grid)
        if not candidates:
            return None
        green, red = candidates[int(np.random.randint(len(candidates)))]
        grid[green] = Color.GREEN
        grid[red] = Color.RED
        expected[green] = Color.TEAL
        pairs.append((green, red))
    return pairs


def _gen_place_singles(grid, expected, red_count, green_count, grey_count):
    """Mutate both grids with isolated red/green distractors and grey markers."""
    singles = [Color.RED] * red_count + [Color.GREEN] * green_count + [Color.GREY] * grey_count
    singles = list(np.random.permutation(singles))
    for color in singles:
        candidates = _gen_single_candidates(grid)
        if not candidates:
            return False
        cell = candidates[int(np.random.randint(len(candidates)))]
        grid[cell] = int(color)
        expected[cell] = int(color)
    return True


def _gen_matches_oracle(grid, expected):
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    height = int(np.random.randint(*GEN_HEIGHT_BOUNDS))
    width = int(np.random.randint(*GEN_WIDTH_BOUNDS))
    counts = _gen_pick_counts(height * width)
    if counts is None:
        return None
    domino_count, red_count, green_count, grey_count = counts

    grid = np.zeros((height, width), dtype=int)
    expected = grid.copy()

    pairs = _gen_place_dominoes(grid, expected, domino_count)
    if pairs is None:
        return None
    if not _gen_place_singles(grid, expected, red_count, green_count, grey_count):
        return None

    for _, red in pairs:
        expected[red] = Color.BLACK

    nonblack = int(np.count_nonzero(grid))
    density = nonblack / float(grid.size)
    min_density, max_density = GEN_DENSITY_BOUNDS
    if not (min_density <= density <= max_density):
        return None

    if not _gen_matches_oracle(grid, expected):
        return None
    return grid


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d90796e8 input")
