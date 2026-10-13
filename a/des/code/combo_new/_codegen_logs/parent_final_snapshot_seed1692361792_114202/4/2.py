from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SPECIAL_MIX = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, GREEN)): YELLOW,
    frozenset((BLUE, GREEN)): AZURE,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

SPECIAL_ONLY_PLANS = (
    (RED, BLUE, GREEN),
    (RED, BLUE, YELLOW),
)

DEFAULT_MIX_PLANS = (
    (RED, GREEN, MAGENTA),
    (BLUE, GREEN, YELLOW),
    (RED, YELLOW, MAGENTA),
    (BLUE, YELLOW, MAGENTA),
)

WIDE_PLANS = (
    (RED, BLUE, GREEN, MAGENTA),
    (RED, BLUE, YELLOW, MAGENTA),
    (RED, GREEN, YELLOW, MAGENTA),
    (BLUE, GREEN, YELLOW, MAGENTA),
    (RED, BLUE, GREEN, YELLOW, MAGENTA),
)

MIN_SIZE = 11
MAX_SIZE = 24
MIN_TOWER_GAP = 2
MAX_ATTEMPTS = 300


def _mix_colors(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return MAROON
    return SPECIAL_MIX.get(frozenset((int(color_a), int(color_b))), MAROON)


def _extract_towers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        towers.append((int(r), int(c), int(grid[r, c])))
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """Keep the signal towers and paint only the orthogonal ray-crossing mix colors.

    Input structure:
    - The background is black.
    - Several single-pixel colored signal towers sit in the interior; grey pixels may
      appear scattered along the outer border as irrelevant wall markers.
    - Towers use visible tower colors such as red, blue, green, yellow, or magenta.

    Rule:
    - Ignore the grey border markers completely.
    - Each tower emits invisible rays along its full row and full column up to the grid edge.
    - A non-tower cell is painted only when at least one horizontal tower ray and one
      vertical tower ray meet there; if three or more rays meet, use the two nearest
      source towers. Mix the two tower colors with the fixed table
      red+blue->magenta, red+green->yellow, blue+green->azure,
      red+yellow->orange, blue+yellow->green, and map every other pair to maroon.
    - Preserve the original tower pixels and leave every other cell black.

    Key insight:
    - The visible output is not the rays themselves but the Cartesian crossing points
      formed by tower rows and columns, with color decided by a fixed pairwise mix table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    towers = _extract_towers(g)
    out = np.full((h, w), BLACK, dtype=int)
    tower_points = {(r, c) for r, c, _color in towers}
    row_towers: dict[int, list[tuple[int, int]]] = {}
    col_towers: dict[int, list[tuple[int, int]]] = {}

    for r, c, color in towers:
        out[r, c] = color
        row_towers.setdefault(r, []).append((c, color))
        col_towers.setdefault(c, []).append((r, color))

    for r in range(h):
        horizontals = row_towers.get(r)
        if not horizontals:
            continue
        for c in range(w):
            if (r, c) in tower_points:
                continue
            verticals = col_towers.get(c)
            if not verticals:
                continue

            arrivals: list[tuple[int, int, int, int]] = []
            for tc, color in horizontals:
                arrivals.append((abs(c - tc), 0, tc, color))
            for tr, color in verticals:
                arrivals.append((abs(r - tr), 1, tr, color))

            arrivals.sort()
            if len(arrivals) < 2:
                continue
            out[r, c] = _mix_colors(arrivals[0][3], arrivals[1][3])

    return out


def _sample_spaced_positions(
    rng: np.random.Generator,
    start: int,
    end: int,
    count: int,
    min_delta: int,
) -> list[int] | None:
    span = end - start + 1
    required = count + (count - 1) * (min_delta - 1)
    if required > span:
        return None
    compressed = span - (count - 1) * (min_delta - 1)
    base = np.sort(rng.choice(compressed, size=count, replace=False))
    return [int(start + value + i * (min_delta - 1)) for i, value in enumerate(base)]


def _choose_color_plan(rng: np.random.Generator) -> tuple[int, ...]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return SPECIAL_ONLY_PLANS[int(rng.integers(len(SPECIAL_ONLY_PLANS)))]
    if mode == 1:
        return DEFAULT_MIX_PLANS[int(rng.integers(len(DEFAULT_MIX_PLANS)))]
    return WIDE_PLANS[int(rng.integers(len(WIDE_PLANS)))]


def _expected_output(shape: tuple[int, int], towers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r, c, color in towers:
        out[r, c] = color

    n = len(towers)
    for i in range(n):
        r1, c1, color1 = towers[i]
        for j in range(i + 1, n):
            r2, c2, color2 = towers[j]
            if r1 == r2 or c1 == c2:
                continue
            mix = _mix_colors(color1, color2)
            out[r1, c2] = mix
            out[r2, c1] = mix
    return out


def _edge_positions(
    rng: np.random.Generator,
    length: int,
    min_count: int = 1,
    max_count: int = 3,
) -> list[int]:
    available = list(range(1, length - 1))
    valid_counts = [
        count
        for count in range(min_count, min(max_count, len(available)) + 1)
        if _sample_spaced_positions(rng, 1, length - 2, count, MIN_TOWER_GAP) is not None
    ]
    assert valid_counts
    count = valid_counts[int(rng.integers(len(valid_counts)))]
    positions = _sample_spaced_positions(rng, 1, length - 2, count, MIN_TOWER_GAP)
    assert positions is not None
    return positions


def _paint_border_markers(rng: np.random.Generator, grid: np.ndarray) -> None:
    h, w = grid.shape
    for c in _edge_positions(rng, w):
        grid[0, c] = GREY
    for c in _edge_positions(rng, w):
        grid[h - 1, c] = GREY
    for r in _edge_positions(rng, h):
        grid[r, 0] = GREY
    for r in _edge_positions(rng, h):
        grid[r, w - 1] = GREY


def _plan_towers(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    colors = list(_choose_color_plan(rng))
    rng.shuffle(colors)
    tower_count = len(colors)
    min_dim = max(MIN_SIZE, 2 * tower_count + 1)
    if min_dim > MAX_SIZE:
        return None

    h = int(rng.integers(min_dim, MAX_SIZE + 1))
    w = int(rng.integers(min_dim, MAX_SIZE + 1))
    gap = 3 if rng.random() < 0.35 else MIN_TOWER_GAP
    rows = _sample_spaced_positions(rng, 1, h - 2, tower_count, gap)
    cols = _sample_spaced_positions(rng, 1, w - 2, tower_count, gap)
    if rows is None or cols is None:
        rows = _sample_spaced_positions(rng, 1, h - 2, tower_count, MIN_TOWER_GAP)
        cols = _sample_spaced_positions(rng, 1, w - 2, tower_count, MIN_TOWER_GAP)
    if rows is None or cols is None:
        return None

    rng.shuffle(cols)
    towers = [(int(r), int(c), int(color)) for r, c, color in zip(rows, cols, colors)]
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c, color in towers:
        grid[r, c] = color
    _paint_border_markers(rng, grid)

    expected = _expected_output((h, w), towers)
    mixed = expected[(expected != BLACK) & (expected != grid)]
    if mixed.size < 4:
        return None
    if len(set(map(int, mixed.tolist()))) < 2:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        planned = _plan_towers(rng)
        if planned is None:
            continue
        return planned[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
