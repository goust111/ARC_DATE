from __future__ import annotations

import random

import numpy as np


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
DIAGONAL = "diagonal"
DIRECTIONS = (HORIZONTAL, VERTICAL, DIAGONAL)

ARC_COLORS = tuple(range(10))
GRID_HEIGHT_MIN = 12
GRID_HEIGHT_MAX = 24
GRID_WIDTH_MIN = 12
GRID_WIDTH_MAX = 24
PAIR_COUNT_MIN = 3
PAIR_COUNT_MAX = 5
PAIR_LENGTH_MIN = 2
PAIR_LENGTH_MAX = 10
MAIN_MIN_COL = 3
LEGEND_HEIGHT = {
    HORIZONTAL: 1,
    VERTICAL: 2,
    DIAGONAL: 2,
}
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _direction_from_pair(coords: np.ndarray, *, adjacent_only: bool) -> str:
    coords = np.asarray(coords, dtype=int)
    assert coords.shape == (2, 2), "expected exactly two pixels"
    (r0, c0), (r1, c1) = coords
    dr = abs(int(r1) - int(r0))
    dc = abs(int(c1) - int(c0))
    if adjacent_only:
        assert max(dr, dc) == 1, "legend pixels must be adjacent"
    if dr == 0 and dc >= 1:
        return HORIZONTAL
    if dc == 0 and dr >= 1:
        return VERTICAL
    if dr == dc and dr >= 1:
        return DIAGONAL
    raise AssertionError("pixel pair must align horizontally, vertically, or diagonally")


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    _direction_from_pair(np.array([a, b], dtype=int), adjacent_only=False)
    steps = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)))
    sr = _sign(int(r1) - int(r0))
    sc = _sign(int(c1) - int(c0))
    return [(int(r0 + i * sr), int(c0 + i * sc)) for i in range(steps + 1)]


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[str, int]:
    legend = np.asarray(grid[:, :2], dtype=int)
    mapping: dict[str, int] = {}
    for color in (int(v) for v in np.unique(legend) if int(v) != bg):
        coords = np.argwhere(legend == color)
        kind = _direction_from_pair(coords, adjacent_only=True)
        assert kind not in mapping, "each direction must appear exactly once in the legend"
        mapping[kind] = int(color)
    assert set(mapping) == set(DIRECTIONS), "legend must define horizontal, vertical, and diagonal colors"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each scattered pixel pair by drawing its connecting segment in the legend color for that segment direction.

    Input structure:
    - The background is the dominant color.
    - The leftmost two columns contain a direction legend made of three same-color
      two-pixel entries, separated by background rows.
    - A horizontal adjacent pair encodes horizontal segments, a vertical adjacent
      pair encodes vertical segments, and a diagonal adjacent pair encodes 45-degree
      diagonal segments.
    - To the right of the legend, each non-background color appears as exactly one
      pair of pixels whose endpoints lie on a horizontal, vertical, or 45-degree
      diagonal line.

    Rule:
    - Decode the legend to build a mapping from segment direction type to legend color.
    - For every same-color endpoint pair in the main area, determine whether the two
      pixels align horizontally, vertically, or diagonally.
    - Draw the full straight segment between the two endpoints, including both ends,
      using the legend color for that direction.
    - Leave the left legend area unchanged.

    Key insight:
    - The legend colors are not labeled directly; their meaning is encoded by the
      relative placement of the two legend pixels, and that decoded direction must be
      reused to recolor the matching endpoint pairs on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 3, "expected two legend columns plus a main area"

    bg = _most_common_color(g)
    legend = _legend_mapping(g, bg)
    out = g.copy()

    for color in (int(v) for v in np.unique(g[:, 2:]) if int(v) != bg):
        coords = np.argwhere(g[:, 2:] == color)
        assert coords.shape == (2, 2), "each main-area color must mark exactly one pixel pair"
        coords = coords.copy()
        coords[:, 1] += 2
        kind = _direction_from_pair(coords, adjacent_only=False)
        for r, c in _line_cells(tuple(map(int, coords[0])), tuple(map(int, coords[1]))):
            out[r, c] = legend[kind]

    return out


def _sample_legend_rows(height: int, order: list[str]) -> dict[str, int] | None:
    rows: dict[str, int] = {}
    next_row = 0
    for i, kind in enumerate(order):
        remaining_height = sum(LEGEND_HEIGHT[k] for k in order[i + 1 :])
        remaining_gaps = max(0, len(order) - i - 1)
        max_start = height - LEGEND_HEIGHT[kind] - remaining_height - remaining_gaps
        if max_start < next_row:
            return None
        start = random.randint(next_row, max_start)
        rows[kind] = start
        next_row = start + LEGEND_HEIGHT[kind] + 1
    return rows


def _legend_cells(kind: str, top_row: int) -> list[tuple[int, int]]:
    if kind == HORIZONTAL:
        return [(top_row, 0), (top_row, 1)]
    if kind == VERTICAL:
        col = random.randint(0, 1)
        return [(top_row, col), (top_row + 1, col)]
    if random.random() < 0.5:
        return [(top_row, 0), (top_row + 1, 1)]
    return [(top_row, 1), (top_row + 1, 0)]


def _sample_segment(kind: str, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if kind == HORIZONTAL:
        max_len = min(PAIR_LENGTH_MAX, width - MAIN_MIN_COL)
        if max_len < PAIR_LENGTH_MIN:
            return None
        length = random.randint(PAIR_LENGTH_MIN, max_len)
        row = random.randint(0, height - 1)
        col0 = random.randint(MAIN_MIN_COL, width - length)
        return (row, col0), (row, col0 + length - 1)

    if kind == VERTICAL:
        max_len = min(PAIR_LENGTH_MAX, height)
        if max_len < PAIR_LENGTH_MIN:
            return None
        length = random.randint(PAIR_LENGTH_MIN, max_len)
        row0 = random.randint(0, height - length)
        col = random.randint(MAIN_MIN_COL, width - 1)
        return (row0, col), (row0 + length - 1, col)

    max_len = min(PAIR_LENGTH_MAX, height, width - MAIN_MIN_COL)
    if max_len < PAIR_LENGTH_MIN:
        return None
    length = random.randint(PAIR_LENGTH_MIN, max_len)
    row0 = random.randint(0, height - length)
    if random.random() < 0.5:
        col0 = random.randint(MAIN_MIN_COL, width - length)
        return (row0, col0), (row0 + length - 1, col0 + length - 1)
    col0 = random.randint(MAIN_MIN_COL + length - 1, width - 1)
    return (row0, col0), (row0 + length - 1, col0 - length + 1)


def _halo(cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    around: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < height and 0 <= nc < width:
                    around.add((int(nr), int(nc)))
    return around


def _place_segments(height: int, width: int, kinds: list[str]) -> list[tuple[str, tuple[int, int], tuple[int, int]]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    blocked[:, :MAIN_MIN_COL] = True
    segments: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    for kind in kinds:
        for _ in range(PLACEMENT_RETRIES):
            endpoints = _sample_segment(kind, height, width)
            if endpoints is None:
                return None
            start, end = endpoints
            cells = _line_cells(start, end)
            halo = _halo(cells, height, width)
            if any(blocked[r, c] for r, c in halo):
                continue
            for r, c in halo:
                blocked[r, c] = True
            segments.append((kind, start, end))
            break
        else:
            return None

    return segments


def _build_sample() -> np.ndarray | None:
    height = random.randint(GRID_HEIGHT_MIN, GRID_HEIGHT_MAX)
    width = random.randint(GRID_WIDTH_MIN, GRID_WIDTH_MAX)
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)

    palette = list(ARC_COLORS)
    random.shuffle(palette)
    bg = int(palette[0])
    legend_colors = [int(v) for v in palette[1:4]]
    pair_colors = [int(v) for v in palette[4 : 4 + pair_count]]

    legend_color_by_kind = dict(zip(DIRECTIONS, legend_colors))
    legend_order = list(DIRECTIONS)
    random.shuffle(legend_order)
    legend_rows = _sample_legend_rows(height, legend_order)
    if legend_rows is None:
        return None

    extra_kinds = [random.choice(DIRECTIONS) for _ in range(pair_count - len(DIRECTIONS))]
    kinds = list(DIRECTIONS) + extra_kinds
    random.shuffle(kinds)
    segments = _place_segments(height, width, kinds)
    if segments is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for kind in DIRECTIONS:
        for r, c in _legend_cells(kind, legend_rows[kind]):
            grid[r, c] = legend_color_by_kind[kind]

    for color, (_kind, start, end) in zip(pair_colors, segments):
        grid[start] = color
        grid[end] = color

    counts = {kind: 0 for kind in DIRECTIONS}
    for _kind, _start, _end in segments:
        counts[_kind] += 1
    assert all(counts[kind] >= 1 for kind in DIRECTIONS), "generator must cover all direction types"
    assert _most_common_color(grid) == bg, "background must remain dominant"

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_sample()
        if grid is None:
            continue
        output = main(grid)
        assert isinstance(output, np.ndarray)
        assert grid.ndim == 2 and max(grid.shape) <= 30
        assert output.shape == grid.shape
        return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")
