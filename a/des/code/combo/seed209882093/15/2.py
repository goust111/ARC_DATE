from __future__ import annotations

import numpy as np


BACKGROUND = 0
AZURE = 8
PAIR_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)

CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
X_OFFSETS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
RELATIVE_PATTERNS = {
    "cross": frozenset((r + 1, c + 1) for r, c in CROSS_OFFSETS),
    "x": frozenset((r + 1, c + 1) for r, c in X_OFFSETS),
}

AXIS_DIRECTIONS = ("h", "v")
DIAGONAL_DIRECTIONS = ("d1", "d2")
ALL_DIRECTIONS = AXIS_DIRECTIONS + DIAGONAL_DIRECTIONS
DIR_TO_STEP = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}

GRID_MIN = 11
GRID_MAX = 19
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
HALF_DISTANCE_MIN = 2
PLACEMENT_RETRIES = 240
GENERATION_RETRIES = 400
HALO_RADIUS = 1
LEGEND_CORNERS = ("tl", "tr", "bl", "br")


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _corner_origin(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    height, width = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 3
    if corner == "bl":
        return height - 3, 0
    if corner == "br":
        return height - 3, width - 3
    raise ValueError(f"unknown corner {corner}")


def _offset_cells(center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    r0, c0 = center
    return {(r0 + dr, c0 + dc) for dr, dc in offsets}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cells_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _expanded_cells(
    cells: set[tuple[int, int]], radius: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    expanded.add((rr, cc))
    return expanded


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], str]:
    matches: list[tuple[set[tuple[int, int]], str]] = []
    for corner in LEGEND_CORNERS:
        r0, c0 = _corner_origin(grid.shape, corner)
        block = grid[r0 : r0 + 3, c0 : c0 + 3]
        if block.shape != (3, 3):
            continue
        if np.any((block != bg) & (block != AZURE)):
            continue

        azure_rel = frozenset((int(r), int(c)) for r, c in np.argwhere(block == AZURE))
        for kind, pattern in RELATIVE_PATTERNS.items():
            if azure_rel == pattern:
                matches.append(({(r0 + r, c0 + c) for r, c in pattern}, kind))
                break

    assert len(matches) == 1, "expected exactly one azure 3x3 legend in a corner"
    return matches[0]


def _pattern_offsets_for_direction(direction: str) -> tuple[tuple[int, int], ...]:
    return CROSS_OFFSETS if direction in AXIS_DIRECTIONS else X_OFFSETS


def _render_pairs(
    grid: np.ndarray, pairs: list[tuple[int, tuple[int, int], tuple[int, int]]], bg: int
) -> np.ndarray:
    out = grid.copy()
    for color, a, b in pairs:
        (r0, c0), (r1, c1) = a, b
        dr = r1 - r0
        dc = c1 - c0
        assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "pair midpoint must land on a cell"
        assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "pair must lie on an axis or 45-degree diagonal"

        mid = ((r0 + r1) // 2, (c0 + c1) // 2)
        offsets = CROSS_OFFSETS if dr == 0 or dc == 0 else X_OFFSETS
        out[r0, c0] = bg
        out[r1, c1] = bg
        for rr, cc in _offset_cells(mid, offsets):
            assert _in_bounds(out.shape, rr, cc)
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each same-color pixel pair onto its midpoint and replace it with a 3x3 direction symbol.

    Horizontal and vertical pairs become plus signs, while 45-degree diagonal
    pairs become X signs. The azure 3x3 corner legend is preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3

    bg = _background_color(g)
    legend_cells, _legend_kind = _detect_legend(g, bg)

    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != bg):
        cell = (int(r), int(c))
        color = int(g[cell])
        if cell in legend_cells and color == AZURE:
            continue
        coords_by_color.setdefault(color, []).append(cell)

    assert coords_by_color, "expected at least one colored pair outside the legend"

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, coords in coords_by_color.items():
        assert len(coords) == 2, "each non-legend color must appear exactly twice"
        pairs.append((int(color), coords[0], coords[1]))

    return _render_pairs(g, pairs, bg)


def _place_legend(grid: np.ndarray, corner: str, kind: str) -> set[tuple[int, int]]:
    r0, c0 = _corner_origin(grid.shape, corner)
    offsets = CROSS_OFFSETS if kind == "cross" else X_OFFSETS
    cells = {(r0 + 1 + dr, c0 + 1 + dc) for dr, dc in offsets}
    for r, c in cells:
        grid[r, c] = AZURE
    return cells


def _pair_count_for_shape(shape: tuple[int, int]) -> int:
    area = shape[0] * shape[1]
    max_pairs = min(
        PAIR_COUNT_MAX,
        len(PAIR_COLOR_POOL),
        2 + int(area >= 180) + int(area >= 280),
    )
    return max(PAIR_COUNT_MIN, max_pairs)


def _sample_directions(rng: np.random.Generator, pair_count: int) -> list[str]:
    directions = [str(rng.choice(AXIS_DIRECTIONS)), str(rng.choice(DIAGONAL_DIRECTIONS))]
    while len(directions) < pair_count:
        directions.append(str(rng.choice(ALL_DIRECTIONS)))
    rng.shuffle(directions)
    return directions


def _max_half_distance(
    center: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]
) -> int:
    r, c = center
    sr, sc = step
    row_limit = shape[0]
    col_limit = shape[1]
    if sr != 0:
        row_limit = min(r, shape[0] - 1 - r)
    if sc != 0:
        col_limit = min(c, shape[1] - 1 - c)
    return min(row_limit, col_limit)


def _try_place_pair(
    grid: np.ndarray,
    color: int,
    direction: str,
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    shape = grid.shape
    step = DIR_TO_STEP[direction]
    offsets = _pattern_offsets_for_direction(direction)

    mids = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)]
    rng.shuffle(mids)

    for mid in mids[:PLACEMENT_RETRIES]:
        symbol_cells = _offset_cells(mid, offsets)
        if not _cells_in_bounds(shape, symbol_cells):
            continue
        max_half = _max_half_distance(mid, step, shape)
        if max_half < HALF_DISTANCE_MIN:
            continue

        distances = list(range(HALF_DISTANCE_MIN, max_half + 1))
        rng.shuffle(distances)
        for half in distances:
            a = (mid[0] - step[0] * half, mid[1] - step[1] * half)
            b = (mid[0] + step[0] * half, mid[1] + step[1] * half)
            footprint = symbol_cells | {a, b}
            if not _cells_in_bounds(shape, footprint):
                continue
            if footprint & reserved:
                continue

            for r, c in (a, b):
                grid[r, c] = color
            reserved.update(_expanded_cells(footprint, HALO_RADIUS, shape))
            return a, b

    return None


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((height, width), BACKGROUND, dtype=int)

    legend_corner = str(rng.choice(LEGEND_CORNERS))
    legend_kind = str(rng.choice(("cross", "x")))
    legend_cells = _place_legend(grid, legend_corner, legend_kind)

    r0, c0 = _corner_origin(grid.shape, legend_corner)
    legend_block = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}
    reserved = _expanded_cells(legend_block | legend_cells, HALO_RADIUS, grid.shape)

    pair_count = int(rng.integers(PAIR_COUNT_MIN, _pair_count_for_shape(grid.shape) + 1))
    colors = rng.choice(PAIR_COLOR_POOL, size=pair_count, replace=False).tolist()
    directions = _sample_directions(rng, pair_count)

    placed_pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, direction in zip(colors, directions):
        placed = _try_place_pair(grid, int(color), direction, reserved, rng)
        if placed is None:
            return None
        placed_pairs.append((int(color), placed[0], placed[1]))

    expected = _render_pairs(grid, placed_pairs, BACKGROUND)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = np.random.default_rng()
        built = _build_attempt(rng)
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
