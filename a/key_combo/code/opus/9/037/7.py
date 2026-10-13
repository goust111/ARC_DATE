from __future__ import annotations

import random
from collections import Counter

import numpy as np


_START_OFFSETS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
_CORNER_SPECS = (
    {"name": "tl", "corner": (0, 0), "row_dir": 1, "col_dir": 1},
    {"name": "tr", "corner": (0, -1), "row_dir": 1, "col_dir": -1},
    {"name": "bl", "corner": (-1, 0), "row_dir": -1, "col_dir": 1},
    {"name": "br", "corner": (-1, -1), "row_dir": -1, "col_dir": -1},
)
_EVEN_SIZES = (10, 12, 14, 16)
_GEN_RETRIES = 400


def _corner_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _corner_colors(grid: np.ndarray) -> list[int]:
    return [int(grid[r, c]) for r, c in _corner_positions(grid.shape)]


def _quadrant_bounds(n: int, corner_index: int) -> tuple[int, int, int, int]:
    half = n // 2
    if corner_index == 0:
        return 0, half, 0, half
    if corner_index == 1:
        return 0, half, half, n
    if corner_index == 2:
        return half, n, 0, half
    return half, n, half, n


def _local_seed(half: int, corner_index: int) -> tuple[int, int]:
    if corner_index == 0:
        return 0, 0
    if corner_index == 1:
        return 0, half - 1
    if corner_index == 2:
        return half - 1, 0
    return half - 1, half - 1


def _axis_positions(start: int, size: int, direction: int) -> range:
    if not 0 <= start < size:
        return range(0)
    if direction == 1:
        return range(start, size, 2)
    return range(start, -1, -2)


def _grow_quadrant(half: int, corner_index: int, color: int) -> np.ndarray:
    spec = _CORNER_SPECS[corner_index]
    seed_r, seed_c = _local_seed(half, corner_index)
    out = np.zeros((half, half), dtype=int)
    for dr, dc in _START_OFFSETS:
        start_r = seed_r + dr
        start_c = seed_c + dc
        for r in _axis_positions(start_r, half, spec["row_dir"]):
            for c in _axis_positions(start_c, half, spec["col_dir"]):
                out[r, c] = int(color)
    return out


def _corner_frequencies(grid: np.ndarray) -> list[int]:
    counts = Counter(map(int, grid.ravel()))
    return [int(counts[color]) for color in _corner_colors(grid)]


def _rarest_corner_index(grid: np.ndarray) -> int:
    freqs = _corner_frequencies(grid)
    minimum = min(freqs)
    rarest = [idx for idx, freq in enumerate(freqs) if freq == minimum]
    assert len(rarest) == 1, "expected exactly one rarest corner color"
    return rarest[0]


def _candidate_output(grid: np.ndarray, corner_index: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    half = n // 2
    r0, r1, c0, c1 = _quadrant_bounds(n, corner_index)
    color = int(_corner_colors(g)[corner_index])
    expected = _grow_quadrant(half, corner_index, color)
    crop = np.zeros((half, half), dtype=int)
    crop[:, :] = expected
    assert crop.shape == (r1 - r0, c1 - c0)
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the quadrant whose corner color is globally rarest, then grow that corner's seed inward on a period-2 diagonal lattice and crop to that quadrant.

    Input structure:
    - The input is an even-sized square grid on a black background.
    - The four grid corners hold four distinct clue colors from the corner palette.
    - Additional filler pixels of those corner colors may appear elsewhere, but the chosen seed quadrant contains only its corner seed on black.

    Rule:
    - Count how many times each corner color appears in the whole grid.
    - The unique rarest corner color identifies the seed corner and therefore the active quadrant.
    - Inside that quadrant only, start from the seed and its four diagonal-neighbor offsets, then repeat every 2 rows away from the seed edge and every 2 columns inward from the seed side, clipping to the quadrant bounds.
    - Return the active quadrant as a half-size crop.

    Key insight:
    - The corner is not chosen by position or color name; it is chosen by comparing whole-grid frequencies of the four corner colors, and that same corner also fixes the growth orientation and crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0 and h > 0, "expected an even square grid"
    corner_colors = _corner_colors(g)
    assert len(set(corner_colors)) == 4, "corner colors must be distinct"

    seed_corner = _rarest_corner_index(g)
    return _candidate_output(g, seed_corner)


def _quadrant_mask(n: int, corner_index: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    r0, r1, c0, c1 = _quadrant_bounds(n, corner_index)
    mask[r0:r1, c0:c1] = True
    return mask


def _pick_extra_empty_quadrant(rng: random.Random, seed_corner: int) -> int | None:
    if rng.random() < 0.5:
        choices = [idx for idx in range(4) if idx != seed_corner]
        return rng.choice(choices)
    return None


def _target_totals(rng: random.Random, seed_color: int, colors: list[int]) -> dict[int, int]:
    seed_total = 1 + rng.randint(0, 3)
    increments = sorted(rng.sample(range(2, 9), 3))
    other_colors = [color for color in colors if color != seed_color]
    rng.shuffle(other_colors)
    totals = {seed_color: seed_total}
    for color, inc in zip(other_colors, increments):
        totals[color] = seed_total + inc
    return totals


def _allowed_cells(n: int, blocked_quadrants: set[int]) -> list[tuple[int, int]]:
    blocked = np.zeros((n, n), dtype=bool)
    for quadrant in blocked_quadrants:
        blocked |= _quadrant_mask(n, quadrant)
    cells: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            if blocked[r, c]:
                continue
            if (r, c) in _corner_positions((n, n)):
                continue
            cells.append((r, c))
    return cells


def _shape_options(area: int) -> list[list[tuple[int, int]]]:
    options = [[(0, 0)]]
    if area >= 2:
        options.extend([[(0, 0), (0, 1)], [(0, 0), (1, 0)]])
    if area >= 3:
        options.extend(
            [
                [(0, 0), (0, 1), (0, 2)],
                [(0, 0), (1, 0), (2, 0)],
                [(0, 0), (0, 1), (1, 0)],
                [(0, 0), (0, 1), (1, 1)],
                [(0, 0), (1, 0), (1, 1)],
                [(0, 1), (1, 0), (1, 1)],
            ]
        )
    if area >= 4:
        options.extend(
            [
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (0, 2), (0, 3)],
                [(0, 0), (1, 0), (2, 0), (3, 0)],
            ]
        )
    return [shape for shape in options if len(shape) <= area]


def _place_cluster(
    rng: random.Random,
    grid: np.ndarray,
    allowed: np.ndarray,
    color: int,
    area: int,
) -> int:
    shapes = _shape_options(min(area, 4))
    rng.shuffle(shapes)
    h, w = grid.shape
    for shape in shapes:
        candidates = []
        max_r = max(dr for dr, _dc in shape)
        max_c = max(dc for _dr, dc in shape)
        for r0 in range(h - max_r):
            for c0 in range(w - max_c):
                coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if all(allowed[r, c] and grid[r, c] == 0 for r, c in coords):
                    candidates.append(coords)
        if candidates:
            coords = rng.choice(candidates)
            for r, c in coords:
                grid[r, c] = color
            return len(coords)
    return 0


def _paint_fillers(
    rng: random.Random,
    grid: np.ndarray,
    totals: dict[int, int],
    seed_corner: int,
    extra_empty_quadrant: int | None,
) -> bool:
    blocked_quadrants = {seed_corner}
    if extra_empty_quadrant is not None:
        blocked_quadrants.add(extra_empty_quadrant)
    allowed = np.zeros_like(grid, dtype=bool)
    for r, c in _allowed_cells(grid.shape[0], blocked_quadrants):
        allowed[r, c] = True

    for color, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        remaining = total - 1
        while remaining > 0:
            max_piece = min(4, remaining)
            piece = rng.randint(1, max_piece)
            painted = _place_cluster(rng, grid, allowed, color, piece)
            if painted == 0:
                return False
            remaining -= painted
    return True


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    half = n // 2
    seed_corner = _rarest_corner_index(g)
    spec = _CORNER_SPECS[seed_corner]
    r0, r1, c0, c1 = _quadrant_bounds(n, seed_corner)
    seed_r = r0 if spec["corner"][0] == 0 else r1 - 1
    seed_c = c0 if spec["corner"][1] == 0 else c1 - 1
    color = int(g[seed_r, seed_c])

    full = np.zeros_like(g)
    for dr, dc in _START_OFFSETS:
        start_r = seed_r + dr
        start_c = seed_c + dc
        if not (r0 <= start_r < r1 and c0 <= start_c < c1):
            continue
        row_stop = r1 if spec["row_dir"] == 1 else r0 - 1
        col_stop = c1 if spec["col_dir"] == 1 else c0 - 1
        for rr in range(start_r, row_stop, 2 * spec["row_dir"]):
            for cc in range(start_c, col_stop, 2 * spec["col_dir"]):
                full[rr, cc] = color
    return full[r0:r1, c0:c1].copy()


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 != 0:
        return False
    n = g.shape[0]
    if n > 30:
        return False
    corners = _corner_colors(g)
    if sorted(corners) != [1, 2, 3, 4]:
        return False

    try:
        seed_corner = _rarest_corner_index(g)
    except AssertionError:
        return False

    r0, r1, c0, c1 = _quadrant_bounds(n, seed_corner)
    seed_crop = g[r0:r1, c0:c1]
    if int(np.count_nonzero(seed_crop)) != 1:
        return False

    freqs = _corner_frequencies(g)
    if freqs[seed_corner] + 1 > sorted(freqs)[1]:
        return False

    oracle = _oracle_output(g)
    try:
        actual = main(g)
    except AssertionError:
        return False
    if not np.array_equal(actual, oracle):
        return False
    if int(np.count_nonzero(actual)) < 6:
        return False

    candidate_sigs = set()
    half = n // 2
    for idx in range(4):
        candidate = _grow_quadrant(half, idx, corners[idx])
        candidate_sigs.add((candidate.shape, candidate.tobytes()))
    if len(candidate_sigs) != 4:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(_EVEN_SIZES)
    colors = [1, 2, 3, 4]
    rng.shuffle(colors)
    seed_corner = rng.randrange(4)
    extra_empty_quadrant = _pick_extra_empty_quadrant(rng, seed_corner)

    grid = np.zeros((n, n), dtype=int)
    for (r, c), color in zip(_corner_positions((n, n)), colors):
        grid[r, c] = color

    seed_color = colors[seed_corner]
    totals = _target_totals(rng, seed_color, colors)
    if not _paint_fillers(rng, grid, totals, seed_corner, extra_empty_quadrant):
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
