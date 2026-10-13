from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")

# Generator bounds.
GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
GROUP_COUNT_RANGE = (2, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 5
RECT_AREA_MAX = 16
PLACEMENT_RETRIES = 300
MAX_GENERATION_ATTEMPTS = 300
SEPARATION_GAP = 1

Cell = tuple[int, int]


def _edge_direction(cell: Cell, height: int, width: int) -> str:
    r, c = cell
    hits = []
    if r == 0:
        hits.append("up")
    if r == height - 1:
        hits.append("down")
    if c == 0:
        hits.append("left")
    if c == width - 1:
        hits.append("right")
    assert len(hits) == 1, "leader pixel must lie on exactly one non-corner edge"
    return hits[0]


def _contiguous(values: list[int]) -> bool:
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def _target_region(coords: np.ndarray, direction: str, height: int, width: int) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]

    if direction in ("up", "down"):
        band_rows: dict[int, list[int]] = defaultdict(list)
        for r, c in coords:
            band_rows[int(c)].append(int(r))
        span = sorted(band_rows)
        assert _contiguous(span), "vertical groups must occupy a contiguous column span"
        counts = {len(v) for v in band_rows.values()}
        assert len(counts) == 1, "every occupied column must contain the same number of pixels"
        rect_h = counts.pop()
        r0 = 0 if direction == "up" else height - rect_h
        return r0, span[0], r0 + rect_h - 1, span[-1]

    band_cols: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        band_cols[int(r)].append(int(c))
    span = sorted(band_cols)
    assert _contiguous(span), "horizontal groups must occupy a contiguous row span"
    counts = {len(v) for v in band_cols.values()}
    assert len(counts) == 1, "every occupied row must contain the same number of pixels"
    rect_w = counts.pop()
    c0 = 0 if direction == "left" else width - rect_w
    return span[0], c0, span[-1], c0 + rect_w - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored pixels, read each color's edge leader, and compress the group
    toward that edge until its occupied rows or columns pack into a solid rectangle.

    Vertical groups keep their columns and collapse upward or downward; horizontal
    groups keep their rows and collapse leftward or rightward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert np.all((0 <= g) & (g <= 9))

    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        coords = np.argwhere(g == color)
        assert coords.size > 0

        edge_cells = []
        for r, c in coords:
            rr, cc = int(r), int(c)
            if rr in (0, height - 1) or cc in (0, width - 1):
                edge_cells.append((rr, cc))
        assert len(edge_cells) == 1, "each color must have exactly one leader pixel on an edge"
        direction = _edge_direction(edge_cells[0], height, width)
        r0, c0, r1, c1 = _target_region(coords, direction, height, width)

        assert not occupied[r0 : r1 + 1, c0 : c1 + 1].any(), "output rectangles must be disjoint"
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    return out


def _cells_clear(mask: np.ndarray, cells: set[Cell], gap: int) -> bool:
    height, width = mask.shape
    for r, c in cells:
        r0 = max(0, r - gap)
        r1 = min(height, r + gap + 1)
        c0 = max(0, c - gap)
        c1 = min(width, c + gap + 1)
        if mask[r0:r1, c0:c1].any():
            return False
    return True


def _mark(mask: np.ndarray, cells: set[Cell]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _paint(grid: np.ndarray, cells: set[Cell], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_dims(max_h: int, max_w: int) -> tuple[int, int] | None:
    dims = [
        (h, w)
        for h in range(RECT_SIDE_MIN, min(RECT_SIDE_MAX, max_h) + 1)
        for w in range(RECT_SIDE_MIN, min(RECT_SIDE_MAX, max_w) + 1)
        if h * w <= RECT_AREA_MAX
    ]
    if not dims:
        return None
    return random.choice(dims)


def _sample_vertical_group(height: int, width: int, direction: str) -> tuple[set[Cell], set[Cell]] | None:
    dims = _sample_dims(height - 2, width - 2)
    if dims is None:
        return None
    rect_h, rect_w = dims
    if width - rect_w - 1 < 1:
        return None

    c0 = random.randint(1, width - rect_w - 1)
    cols = list(range(c0, c0 + rect_w))
    leader_col = random.choice(cols)
    leader_row = 0 if direction == "up" else height - 1

    input_cells: set[Cell] = set()
    interior_rows = list(range(1, height - 1))
    for col in cols:
        if col == leader_col:
            rows = random.sample(interior_rows, rect_h - 1) + [leader_row]
        else:
            rows = random.sample(interior_rows, rect_h)
        for row in rows:
            input_cells.add((row, col))

    if len(input_cells) != rect_h * rect_w:
        return None

    if direction == "up":
        output_cells = {(row, col) for row in range(rect_h) for col in cols}
    else:
        output_cells = {(row, col) for row in range(height - rect_h, height) for col in cols}
    return input_cells, output_cells


def _sample_horizontal_group(height: int, width: int, direction: str) -> tuple[set[Cell], set[Cell]] | None:
    dims = _sample_dims(height - 2, width - 2)
    if dims is None:
        return None
    rect_h, rect_w = dims
    if height - rect_h - 1 < 1:
        return None

    r0 = random.randint(1, height - rect_h - 1)
    rows = list(range(r0, r0 + rect_h))
    leader_row = random.choice(rows)
    leader_col = 0 if direction == "left" else width - 1

    input_cells: set[Cell] = set()
    interior_cols = list(range(1, width - 1))
    for row in rows:
        if row == leader_row:
            cols = random.sample(interior_cols, rect_w - 1) + [leader_col]
        else:
            cols = random.sample(interior_cols, rect_w)
        for col in cols:
            input_cells.add((row, col))

    if len(input_cells) != rect_h * rect_w:
        return None

    if direction == "left":
        output_cells = {(row, col) for row in rows for col in range(rect_w)}
    else:
        output_cells = {(row, col) for row in rows for col in range(width - rect_w, width)}
    return input_cells, output_cells


def _sample_group(height: int, width: int, direction: str) -> tuple[set[Cell], set[Cell]] | None:
    if direction in ("up", "down"):
        return _sample_vertical_group(height, width, direction)
    return _sample_horizontal_group(height, width, direction)


def _sample_directions(count: int) -> list[str]:
    if random.random() < 0.7:
        return random.sample(DIRECTIONS, count)
    return [random.choice(DIRECTIONS) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    colors = random.sample(COLORS, group_count)
    directions = _sample_directions(group_count)

    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    in_mask = np.zeros((height, width), dtype=bool)
    out_mask = np.zeros((height, width), dtype=bool)

    for color, direction in zip(colors, directions):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            sampled = _sample_group(height, width, direction)
            if sampled is None:
                continue
            input_cells, output_cells = sampled
            if not _cells_clear(in_mask, input_cells, SEPARATION_GAP):
                continue
            if not _cells_clear(out_mask, output_cells, SEPARATION_GAP):
                continue
            _paint(inp, input_cells, color)
            _paint(expected, output_cells, color)
            _mark(in_mask, input_cells)
            _mark(out_mask, output_cells)
            placed = True
            break
        if not placed:
            return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
