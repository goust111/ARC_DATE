from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

CORNER_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
GRID_CHOICES = ((4, 5), (5, 4), (5, 5))
CELL_SIZE_CHOICES = (4, 5)
BIG_TEMPLATE_COUNTS = (4, 5)
GEN_RETRIES = 200

TEMPLATE_BANK: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((1, 1),),
    ),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    ),
    3: (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
    ),
    5: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ),
}


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((start, prev + 1))
        start = idx
        prev = idx
    spans.append((start, prev + 1))
    return spans


def _cell_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _bands(np.where(~np.all(grid == GREY, axis=1))[0])
    col_spans = _bands(np.where(~np.all(grid == GREY, axis=0))[0])
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a partitioned grid"
    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "expected equal-sized regions"
    return row_spans, col_spans


def _get_cell(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], ri: int, ci: int) -> np.ndarray:
    r0, r1 = row_spans[ri]
    c0, c1 = col_spans[ci]
    return grid[r0:r1, c0:c1]


def _quadrant_index(r: float, c: float, h: int, w: int) -> int:
    top = r < h / 2.0
    left = c < w / 2.0
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _dominant_quadrant(markers: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    counts = [0, 0, 0, 0]
    for r, c in markers:
        counts[_quadrant_index(float(r), float(c), h, w)] += 1
    best = max(counts)
    winners = [idx for idx, count in enumerate(counts) if count == best]
    if len(winners) == 1:
        return winners[0]
    mean_r = float(markers[:, 0].mean())
    mean_c = float(markers[:, 1].mean())
    return _quadrant_index(mean_r, mean_c, h, w)


def _rotation_for_quadrant(quadrant: int) -> int:
    return {0: 0, 1: 3, 2: 2, 3: 1}[int(quadrant)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner cells as a count-to-template legend, then stamp the matching
    template into each non-corner cell after rotating it toward the dominant red quadrant.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _cell_spans(g)
    row_count = len(row_spans)
    col_count = len(col_spans)

    corner_indices = {
        (0, 0),
        (0, col_count - 1),
        (row_count - 1, 0),
        (row_count - 1, col_count - 1),
    }

    templates: dict[int, np.ndarray] = {}
    big_template_seen = False
    for ri, ci in corner_indices:
        cell = _get_cell(g, row_spans, col_spans, ri, ci)
        assert int(np.count_nonzero(cell == RED)) == 0, "corner legends must not contain markers"
        count = int(np.count_nonzero(cell != BLACK))
        assert count > 0, "corner legends must be non-empty"
        assert count not in templates, "corner legend counts must be unique"
        templates[count] = cell.copy()
        if count > 3:
            big_template_seen = True

    assert {1, 2, 3}.issubset(templates), "missing one of the 1/2/3 marker templates"
    assert big_template_seen, "expected one distractor template with more than 3 cells"

    out = g.copy()
    for ri in range(row_count):
        for ci in range(col_count):
            if (ri, ci) in corner_indices:
                continue
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]
            cell = g[r0:r1, c0:c1]
            markers = np.argwhere(cell == RED)
            marker_count = int(markers.shape[0])
            if marker_count == 0:
                out[r0:r1, c0:c1] = BLACK
                continue
            assert marker_count in (1, 2, 3), "non-corner regions must have 0..3 markers"
            quadrant = _dominant_quadrant(markers, cell.shape)
            rotation = _rotation_for_quadrant(quadrant)
            out[r0:r1, c0:c1] = np.rot90(templates[marker_count], rotation)
    return out


def _section_spans(count: int, cell_size: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for _ in range(count):
        spans.append((pos, pos + cell_size))
        pos += cell_size + 1
    return spans


def _paint_template(cell_size: int, coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    cell = np.full((cell_size, cell_size), BLACK, dtype=int)
    for r, c in coords:
        cell[r, c] = color
    return cell


def _choose_template_coords(cell_size: int, count: int) -> tuple[tuple[int, int], ...]:
    candidates = [coords for coords in TEMPLATE_BANK[count] if max(r for r, _ in coords) < cell_size and max(c for _, c in coords) < cell_size]
    assert candidates, "template bank does not fit the chosen cell size"
    return random.choice(candidates)


def _quadrant_pool(cell_size: int, quadrant: int) -> list[tuple[int, int]]:
    half = cell_size // 2
    top_rows = list(range(half))
    bottom_rows = list(range(cell_size - half, cell_size))
    left_cols = list(range(half))
    right_cols = list(range(cell_size - half, cell_size))
    row_choices = top_rows if quadrant in (0, 1) else bottom_rows
    col_choices = left_cols if quadrant in (0, 3) else right_cols
    return [(r, c) for r in row_choices for c in col_choices]


def _all_non_corner_cells(row_count: int, col_count: int) -> list[tuple[int, int]]:
    corners = {
        (0, 0),
        (0, col_count - 1),
        (row_count - 1, 0),
        (row_count - 1, col_count - 1),
    }
    return [(ri, ci) for ri in range(row_count) for ci in range(col_count) if (ri, ci) not in corners]


def _base_states() -> list[tuple[int, int | None]]:
    states: list[tuple[int, int | None]] = [(0, None)]
    for count in (1, 2, 3):
        for quadrant in range(4):
            states.append((count, quadrant))
    return states


def _fill_separators(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> None:
    for r0, r1 in row_spans[:-1]:
        grid[r1, :] = GREY
    for c0, c1 in col_spans[:-1]:
        grid[:, c1] = GREY


def _gen_attempt() -> np.ndarray:
    row_count, col_count = random.choice(GRID_CHOICES)
    cell_size = random.choice(CELL_SIZE_CHOICES)
    row_spans = _section_spans(row_count, cell_size)
    col_spans = _section_spans(col_count, cell_size)
    height = row_spans[-1][1]
    width = col_spans[-1][1]
    grid = np.full((height, width), BLACK, dtype=int)
    _fill_separators(grid, row_spans, col_spans)

    big_count = random.choice(BIG_TEMPLATE_COUNTS)
    legend_counts = [1, 2, 3, big_count]
    random.shuffle(legend_counts)
    corner_colors = list(CORNER_COLORS)
    random.shuffle(corner_colors)
    corner_positions = [
        (0, 0),
        (0, col_count - 1),
        (row_count - 1, 0),
        (row_count - 1, col_count - 1),
    ]

    for (ri, ci), count, color in zip(corner_positions, legend_counts, corner_colors, strict=True):
        coords = _choose_template_coords(cell_size, count)
        cell = _paint_template(cell_size, coords, color)
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        grid[r0:r1, c0:c1] = cell

    states = _base_states()
    cells = _all_non_corner_cells(row_count, col_count)
    assert len(cells) >= len(states), "generator requires enough non-corner cells to cover all branches"

    extra = len(cells) - len(states)
    for _ in range(extra):
        count = random.randint(0, 3)
        quadrant = None if count == 0 else random.randrange(4)
        states.append((count, quadrant))

    random.shuffle(states)
    random.shuffle(cells)

    for (ri, ci), (count, quadrant) in zip(cells, states, strict=True):
        cell = np.full((cell_size, cell_size), BLACK, dtype=int)
        if count > 0:
            assert quadrant is not None
            markers = random.sample(_quadrant_pool(cell_size, quadrant), count)
            for r, c in markers:
                cell[r, c] = RED
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        grid[r0:r1, c0:c1] = cell

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
