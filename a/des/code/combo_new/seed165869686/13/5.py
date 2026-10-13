import random
from typing import Iterable

import numpy as np


BG = 0
SPLIT = 5

# Visible additive color pairs in the standard ARC palette.
MIX_RULES = {
    frozenset((1, 2)): 6,  # blue + red -> magenta
    frozenset((3, 4)): 7,  # green + yellow -> orange
}
SAME_COLOR_CHOICES = (1, 2, 3, 4, 6, 7)
DIFF_COLOR_CHOICES = ((1, 2), (3, 4))

REGION_COUNT_CHOICES = (2, 3)
REGION_SIZE_MIN = 3
REGION_SIZE_MAX = 8
GEN_ATTEMPTS = 300


def _divider_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = set()
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            candidates.add(int(grid[r, 0]))
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            candidates.add(int(grid[0, c]))
    candidates.discard(BG)
    assert len(candidates) == 1, "expected one divider color spanning full rows/cols"
    return next(iter(candidates))


def _regions_from_dividers(grid: np.ndarray, divider: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    split_rows = [-1] + [r for r in range(h) if np.all(grid[r, :] == divider)] + [h]
    split_cols = [-1] + [c for c in range(w) if np.all(grid[:, c] == divider)] + [w]

    regions = []
    for r0, r1 in zip(split_rows, split_rows[1:]):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in zip(split_cols, split_cols[1:]):
            if c1 - c0 <= 1:
                continue
            regions.append((r0 + 1, r1, c0 + 1, c1))
    assert regions, "expected at least one non-divider region"
    return regions


def _mix_color(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return color_a
    key = frozenset((int(color_a), int(color_b)))
    assert key in MIX_RULES, f"unsupported visible color mix: {sorted(key)}"
    return MIX_RULES[key]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed pair into a wall-to-wall stripe inside its divider-bounded region.

    Input structure:
    - The background is black.
    - Full grey rows and full grey columns partition the grid into rectangular regions.
    - Each region contains exactly two colored seed pixels, and the two seeds are
      aligned either in one row or in one column.

    Rule:
    - Process each rectangular region independently.
    - If the two seeds share a row, fill that entire region row from the left inner
      wall to the right inner wall.
    - If the two seeds share a column, fill that entire region column from the top
      inner wall to the bottom inner wall.
    - Same-color seeds keep their color; distinct seeds follow the fixed visible
      mixing pairs blue+red->magenta and green+yellow->orange.
    - The stripe overwrites the seed pixels, while the grey divider lines stay unchanged.

    Key insight:
    - The grey through-lines define independent subproblems, and each subproblem is
      solved by combining one geometric cue (row-vs-column alignment) with one color
      cue (same color or one of the recurring additive color pairs).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _divider_color(g)
    out = g.copy()

    for r0, r1, c0, c1 in _regions_from_dividers(g, divider):
        region = g[r0:r1, c0:c1]
        seeds = np.argwhere((region != BG) & (region != divider))
        assert len(seeds) == 2, "each region must contain exactly two seeds"

        (sr1, sc1), (sr2, sc2) = (tuple(map(int, seed)) for seed in seeds)
        color = _mix_color(int(region[sr1, sc1]), int(region[sr2, sc2]))

        if sr1 == sr2:
            out[r0 + sr1, c0:c1] = color
        else:
            assert sc1 == sc2, "seed pair must be horizontally or vertically aligned"
            out[r0:r1, c0 + sc1] = color

    return out


# --------------- generator (gpt-5) ---------------

def _pick_two(rng: random.Random, values: Iterable[int]) -> tuple[int, int]:
    chosen = rng.sample(list(values), 2)
    return int(chosen[0]), int(chosen[1])


def _interior_positions(start: int, length: int) -> list[int]:
    if length >= 5:
        return list(range(start + 1, start + length - 1))
    return list(range(start, start + length))


def _specs_for_regions(rng: random.Random, n_regions: int) -> list[tuple[str, str, tuple[int, ...] | None]]:
    specs: list[tuple[str, str, tuple[int, ...] | None]] = [
        ("H", "same", None),
        ("V", "same", None),
        ("H", "diff", DIFF_COLOR_CHOICES[0]),
        ("V", "diff", DIFF_COLOR_CHOICES[1]),
    ]
    while len(specs) < n_regions:
        orientation = rng.choice(("H", "V"))
        mode = rng.choice(("same", "diff"))
        pair = rng.choice(DIFF_COLOR_CHOICES) if mode == "diff" else None
        specs.append((orientation, mode, pair))
    rng.shuffle(specs)
    return specs


def _paint_region(
    grid: np.ndarray,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    spec: tuple[str, str, tuple[int, ...] | None],
) -> None:
    r0, r1, c0, c1 = bounds
    height = r1 - r0
    width = c1 - c0
    orientation, mode, pair = spec

    if mode == "same":
        colors = (rng.choice(SAME_COLOR_CHOICES),) * 2
    else:
        assert pair is not None
        colors = tuple(rng.sample(list(pair), 2))

    if orientation == "H":
        row = rng.choice(_interior_positions(r0, height))
        cols = _interior_positions(c0, width)
        c_a, c_b = _pick_two(rng, cols)
        grid[row, c_a] = colors[0]
        grid[row, c_b] = colors[1]
    else:
        col = rng.choice(_interior_positions(c0, width))
        rows = _interior_positions(r0, height)
        r_a, r_b = _pick_two(rng, rows)
        grid[r_a, col] = colors[0]
        grid[r_b, col] = colors[1]


def _validate_input(grid: np.ndarray) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    divider = _divider_color(grid)
    regions = _regions_from_dividers(grid, divider)

    same_count = 0
    diff_count = 0
    horizontal_count = 0
    vertical_count = 0
    diff_pairs = set()

    for r0, r1, c0, c1 in regions:
        region = grid[r0:r1, c0:c1]
        seeds = np.argwhere((region != BG) & (region != divider))
        assert len(seeds) == 2
        assert np.count_nonzero((region != BG) & (region != divider)) == 2

        (sr1, sc1), (sr2, sc2) = (tuple(map(int, seed)) for seed in seeds)
        colors = (int(region[sr1, sc1]), int(region[sr2, sc2]))
        if sr1 == sr2:
            horizontal_count += 1
        else:
            assert sc1 == sc2
            vertical_count += 1

        if colors[0] == colors[1]:
            same_count += 1
        else:
            diff_count += 1
            diff_pairs.add(frozenset(colors))
            assert frozenset(colors) in MIX_RULES

    assert same_count >= 1 and diff_count >= 1
    assert horizontal_count >= 1 and vertical_count >= 1
    assert diff_pairs == set(MIX_RULES), "every visible mixing pair should appear"


def _gen_attempt(rng: random.Random) -> np.ndarray:
    region_rows = rng.choice(REGION_COUNT_CHOICES)
    region_cols = rng.choice(REGION_COUNT_CHOICES)
    heights = [rng.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(region_rows)]
    widths = [rng.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(region_cols)]

    h = sum(heights) + (region_rows - 1)
    w = sum(widths) + (region_cols - 1)
    assert h <= 30 and w <= 30

    grid = np.full((h, w), BG, dtype=int)

    row_bounds = []
    cursor = 0
    for i, height in enumerate(heights):
        row_bounds.append((cursor, cursor + height))
        cursor += height
        if i != region_rows - 1:
            grid[cursor, :] = SPLIT
            cursor += 1

    col_bounds = []
    cursor = 0
    for i, width in enumerate(widths):
        col_bounds.append((cursor, cursor + width))
        cursor += width
        if i != region_cols - 1:
            grid[:, cursor] = SPLIT
            cursor += 1

    bounds = [(r0, r1, c0, c1) for (r0, r1) in row_bounds for (c0, c1) in col_bounds]
    specs = _specs_for_regions(rng, len(bounds))
    for region_bounds, spec in zip(bounds, specs):
        _paint_region(grid, rng, region_bounds, spec)

    _validate_input(grid)
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-pair stripe puzzle")
