import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8


SEPARATOR = Color.GREY
FOREGROUND = Color.GREEN
RANK_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
)
OUTER_POSITIONS = (
    ("TL", (0, 0)),
    ("TC", (0, 1)),
    ("TR", (0, 2)),
    ("ML", (1, 0)),
    ("MR", (1, 2)),
    ("BL", (2, 0)),
    ("BC", (2, 1)),
    ("BR", (2, 2)),
)
GEN_ATTEMPTS = 400
OUTER_BAND_RANGE = (4, 8)
CENTER_BAND_RANGE = (3, 7)
COUNT_POOL = tuple(range(1, 11))


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(int(x) for x in separators)
    assert len(separators) == 2
    start = 0
    bands = []
    for cut in separators:
        assert start < cut, "separator lines must leave non-empty regions"
        bands.append((start, cut))
        start = cut + 1
    assert start < size, "separator lines must leave a trailing region"
    bands.append((start, size))
    return bands


def _parse_input(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert set(map(int, np.unique(g))) <= {Color.BLACK, FOREGROUND, SEPARATOR}

    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_rows) == 2, "expected exactly two full separator rows"
    assert len(sep_cols) == 2, "expected exactly two full separator columns"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3
    assert row_bands[1][1] - row_bands[1][0] >= 3, "center region must fit a 3x3 thumbnail"
    assert col_bands[1][1] - col_bands[1][0] >= 3, "center region must fit a 3x3 thumbnail"

    counts: dict[str, int] = {}
    for name, (ri, ci) in OUTER_POSITIONS:
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        region = g[r0:r1, c0:c1]
        assert set(map(int, np.unique(region))) <= {Color.BLACK, FOREGROUND}
        counts[name] = int(np.count_nonzero(region == FOREGROUND))

    cr0, cr1 = row_bands[1]
    cc0, cc1 = col_bands[1]
    center = g[cr0:cr1, cc0:cc1]
    assert np.all(center == Color.BLACK), "center region must start black"

    values = list(counts.values())
    assert min(values) >= 1, "every outer region must contain foreground"
    assert len(set(values)) == 8, "outer foreground counts must be pairwise distinct"
    return row_bands, col_bands, counts


def _color_by_position(counts: dict[str, int]) -> dict[str, int]:
    ranked = sorted(counts.items(), key=lambda item: item[1])
    return {name: RANK_COLORS[rank] for rank, (name, _count) in enumerate(ranked)}


def _thumbnail(color_map: dict[str, int]) -> np.ndarray:
    thumb = np.zeros((3, 3), dtype=int)
    for name, (ri, ci) in OUTER_POSITIONS:
        thumb[ri, ci] = int(color_map[name])
    thumb[1, 1] = Color.BLACK
    return thumb


def _expected_output(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    counts: dict[str, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    color_map = _color_by_position(counts)
    thumb = _thumbnail(color_map)
    cr0, _cr1 = row_bands[1]
    cc0, _cc1 = col_bands[1]
    out[cr0 : cr0 + 3, cc0 : cc0 + 3] = thumb
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the center region with a 3x3 thumbnail whose colors encode the sorted foreground counts of the surrounding regions.
    
    Input structure:
    - Two full grey rows and two full grey columns partition the grid into a 3x3 lattice of rectangular regions.
    - The center region is all black, while each of the eight outer regions contains a distinct number of green pixels on black background.
    
    Rule:
    - Count the green pixels in each outer region and rank the regions from fewest to most.
    - Map rank 1..8 to the fixed color sequence blue, red, green, yellow, grey, pink, orange, teal.
    - Keep the separators and outer regions unchanged, and write a 3x3 thumbnail into the top-left of the center region so each thumbnail cell shows the rank color of the outer region in the same directional position; the thumbnail center stays black.
    
    Key insight:
    - The center colors are not copied from local pixels; they are determined indirectly by comparing green-pixel counts across all eight surrounding regions and then mapping the resulting order back to spatial positions.
    """
    row_bands, col_bands, counts = _parse_input(grid)
    return _expected_output(grid, row_bands, col_bands, counts)


def _sample_band_sizes(rng: random.Random) -> tuple[int, int, int]:
    outer_min, outer_max = OUTER_BAND_RANGE
    center_min, center_max = CENTER_BAND_RANGE
    center = 3 if rng.random() < 0.35 else rng.randint(center_min, center_max)
    return (
        rng.randint(outer_min, outer_max),
        center,
        rng.randint(outer_min, outer_max),
    )


def _separator_indices(bands: tuple[int, int, int]) -> tuple[int, int]:
    return bands[0], bands[0] + 1 + bands[1]


def _random_counts(rng: random.Random) -> list[int]:
    while True:
        counts = rng.sample(COUNT_POOL, 8)
        ordered = sorted(counts)
        if ordered == list(range(1, 9)):
            continue
        if any(value != rank + 1 for rank, value in enumerate(ordered)):
            return counts


def _priority_score(h: int, w: int, ri: int, ci: int, r: int, c: int, mode: str) -> tuple[int, int, int, int]:
    top = r
    bottom = h - 1 - r
    left = c
    right = w - 1 - c
    inward_vertical = bottom if ri == 0 else top if ri == 2 else 0
    inward_horizontal = right if ci == 0 else left if ci == 2 else 0
    outward_vertical = top if ri == 0 else bottom if ri == 2 else max(top, bottom)
    outward_horizontal = left if ci == 0 else right if ci == 2 else max(left, right)
    diag_a = -(abs(r - c))
    diag_b = -(abs((r + c) - (h - 1)))
    parity = -((r + c) % 2)
    if mode == "inward":
        return (inward_vertical + inward_horizontal, parity, -top - left, -bottom - right)
    if mode == "outward":
        return (outward_vertical + outward_horizontal, parity, -inward_vertical, -inward_horizontal)
    if mode == "diag_a":
        return (diag_a, inward_vertical + outward_horizontal, parity, -top)
    if mode == "diag_b":
        return (diag_b, inward_horizontal + outward_vertical, parity, -left)
    return (0, 0, 0, 0)


def _ordered_cells(
    h: int,
    w: int,
    ri: int,
    ci: int,
    count: int,
    mode: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    if mode == "random":
        rng.shuffle(cells)
        return cells[:count]
    if mode == "spread":
        chosen = [rng.choice(cells)]
        pool = [cell for cell in cells if cell != chosen[0]]
        while len(chosen) < count:
            best = None
            best_score = None
            rng.shuffle(pool)
            for cell in pool:
                dist = min(abs(cell[0] - rr) + abs(cell[1] - cc) for rr, cc in chosen)
                score = (dist, _priority_score(h, w, ri, ci, cell[0], cell[1], "outward"))
                if best_score is None or score > best_score:
                    best_score = score
                    best = cell
            chosen.append(best)
            pool.remove(best)
        return chosen

    keyed = []
    for cell in cells:
        keyed.append((_priority_score(h, w, ri, ci, cell[0], cell[1], mode), rng.random(), cell))
    keyed.sort(reverse=True)
    return [cell for _score, _jitter, cell in keyed[:count]]


def _place_region(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    count: int,
    mode: str,
    rng: random.Random,
) -> None:
    r0, r1, c0, c1 = bounds
    h = r1 - r0
    w = c1 - c0
    assert count <= h * w
    ri = 0 if r0 == 0 else 2 if r1 == grid.shape[0] else 1
    ci = 0 if c0 == 0 else 2 if c1 == grid.shape[1] else 1
    for rr, cc in _ordered_cells(h, w, ri, ci, count, mode, rng):
        grid[r0 + rr, c0 + cc] = FOREGROUND


def _center_thumbnail(out: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> np.ndarray:
    cr0, _cr1 = row_bands[1]
    cc0, _cc1 = col_bands[1]
    return out[cr0 : cr0 + 3, cc0 : cc0 + 3].copy()


def _check_attempt(
    grid: np.ndarray,
    planted_counts: dict[str, int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> bool:
    try:
        parsed_rows, parsed_cols, parsed_counts = _parse_input(grid)
        out = main(grid)
    except AssertionError:
        return False

    if parsed_rows != row_bands or parsed_cols != col_bands or parsed_counts != planted_counts:
        return False

    expected = _expected_output(grid, row_bands, col_bands, planted_counts)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    thumb = _center_thumbnail(out, row_bands, col_bands)
    if set(map(int, thumb.ravel())) != set(RANK_COLORS) | {Color.BLACK}:
        return False

    cr0, cr1 = row_bands[1]
    cc0, cc1 = col_bands[1]
    if not np.all(out[cr0:cr1, cc0:cc1][3:, :] == Color.BLACK):
        return False
    if not np.all(out[cr0:cr1, cc0:cc1][:, 3:] == Color.BLACK):
        return False

    ranks = {name: rank + 1 for rank, (name, _count) in enumerate(sorted(planted_counts.items(), key=lambda item: item[1]))}
    if not any(planted_counts[name] != ranks[name] for name in planted_counts):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_sizes = _sample_band_sizes(rng)
    col_sizes = _sample_band_sizes(rng)
    h = sum(row_sizes) + 2
    w = sum(col_sizes) + 2
    if h > 30 or w > 30:
        return None

    sep_rows = _separator_indices(row_sizes)
    sep_cols = _separator_indices(col_sizes)
    row_bands = _bands(h, list(sep_rows))
    col_bands = _bands(w, list(sep_cols))

    grid = np.zeros((h, w), dtype=int)
    grid[list(sep_rows), :] = SEPARATOR
    grid[:, list(sep_cols)] = SEPARATOR

    counts = _random_counts(rng)
    rng.shuffle(counts)
    position_counts = {name: count for (name, _slot), count in zip(OUTER_POSITIONS, counts)}
    rank_order = [name for name, _count in sorted(position_counts.items(), key=lambda item: item[1])]
    if rank_order == [name for name, _slot in OUTER_POSITIONS]:
        return None
    if rank_order == [name for name, _slot in reversed(OUTER_POSITIONS)]:
        return None

    modes = ["spread", "random", "inward", "outward", "diag_a", "diag_b", "spread", "random"]
    rng.shuffle(modes)
    for (name, (ri, ci)), mode in zip(OUTER_POSITIONS, modes):
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        _place_region(grid, (r0, r1, c0, c1), position_counts[name], mode, rng)

    if not _check_attempt(grid, position_counts, row_bands, col_bands):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
