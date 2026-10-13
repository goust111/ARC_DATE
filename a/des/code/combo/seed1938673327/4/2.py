from __future__ import annotations

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segment_cells(a: Coord, b: Coord) -> list[Coord] | None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    dr = r1 - r0
    dc = c1 - c0
    if abs(dr) != abs(dc):
        return None
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    return [(r0 + i * step_r, c0 + i * step_c) for i in range(abs(dr) + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a filled segment matching its alignment.

    Horizontal pairs fill the cells between them on the row, vertical pairs fill the
    cells between them on the column, and 45-degree diagonal pairs fill the diagonal.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for color in sorted(int(x) for x in np.unique(g) if int(x) != bg):
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each foreground color must mark exactly one pair"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        cells = _segment_cells(a, b)
        assert cells is not None, "pair must be horizontal, vertical, or 45-degree diagonal"
        for r, c in cells:
            out[r, c] = color

    return out


BG_COLOR = 5
COLOR_POOL = tuple(color for color in range(10) if color != BG_COLOR)
GRID_HEIGHT_RANGE = (10, 19)
GRID_WIDTH_RANGE = (10, 19)
PAIR_COUNT_RANGE = (3, 6)
MIN_SPAN = 3
MAX_SPAN = 8
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250


def _endpoints_are_isolated(endpoint_mask: np.ndarray, pair: tuple[Coord, Coord]) -> bool:
    height, width = endpoint_mask.shape
    for r, c in pair:
        r0 = max(0, r - 1)
        r1 = min(height - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(width - 1, c + 1)
        if np.any(endpoint_mask[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _sample_pair(
    rng: np.random.Generator,
    height: int,
    width: int,
    orientation: str,
) -> tuple[Coord, Coord] | None:
    if orientation == "h":
        max_span = min(MAX_SPAN, width - 1)
        if max_span < MIN_SPAN:
            return None
        span = int(rng.integers(MIN_SPAN, max_span + 1))
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width - span))
        return (r, c), (r, c + span)

    if orientation == "v":
        max_span = min(MAX_SPAN, height - 1)
        if max_span < MIN_SPAN:
            return None
        span = int(rng.integers(MIN_SPAN, max_span + 1))
        r = int(rng.integers(0, height - span))
        c = int(rng.integers(0, width))
        return (r, c), (r + span, c)

    if orientation == "d":
        max_span = min(MAX_SPAN, height - 1, width - 1)
        if max_span < MIN_SPAN:
            return None
        span = int(rng.integers(MIN_SPAN, max_span + 1))
        if bool(rng.integers(0, 2)):
            r = int(rng.integers(0, height - span))
            c = int(rng.integers(0, width - span))
            return (r, c), (r + span, c + span)
        r = int(rng.integers(0, height - span))
        c = int(rng.integers(span, width))
        return (r, c), (r + span, c - span)

    raise ValueError(f"unknown orientation: {orientation}")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_HEIGHT_RANGE))
    width = int(rng.integers(*GRID_WIDTH_RANGE))
    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))

    orientations = ["h", "v", "d"]
    if pair_count > 3:
        extras = rng.choice(np.array(orientations), size=pair_count - 3, replace=True)
        orientations.extend(str(x) for x in extras.tolist())
    rng.shuffle(orientations)

    colors = [int(x) for x in rng.choice(np.array(COLOR_POOL), size=pair_count, replace=False)]
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    endpoint_mask = np.zeros((height, width), dtype=bool)

    for orientation, color in zip(orientations, colors):
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            pair = _sample_pair(rng, height, width, orientation)
            if pair is None or not _endpoints_are_isolated(endpoint_mask, pair):
                continue
            cells = _segment_cells(*pair)
            assert cells is not None
            if any(reserved[r, c] for r, c in cells):
                continue
            for r, c in cells:
                reserved[r, c] = True
            for r, c in pair:
                grid[r, c] = color
                endpoint_mask[r, c] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
