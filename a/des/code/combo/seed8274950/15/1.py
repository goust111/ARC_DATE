from __future__ import annotations

import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _marker_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        assert len(pts) == 2, "each non-background color must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, pt)) for pt in pts]
        assert r0 != r1 and c0 != c1, "paired markers must be diagonal corners"
        pairs.append((color, (r0, c0), (r1, c1)))
    assert pairs, "expected at least one marker pair"
    return pairs


def _paint_pair(out: np.ndarray, color: int, a: tuple[int, int], b: tuple[int, int]) -> None:
    (r0, c0), (r1, c1) = a, b
    rmin, rmax = sorted((int(r0), int(r1)))
    cmin, cmax = sorted((int(c0), int(c1)))
    step = min(rmax - rmin, cmax - cmin)
    assert step > 0

    for r in range(rmin, rmax + 1, step):
        out[r, cmin : cmax + 1] = color
    for c in range(cmin, cmax + 1, step):
        out[rmin : rmax + 1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each same-color diagonal marker pair and draw a rectangular grid pattern.

    Each pair defines the opposite corners of an axis-aligned rectangle. Let S be the
    shorter axis distance between the two markers. Inside that rectangle, draw
    horizontal and vertical lines of the marker color every S cells, starting from
    the rectangle's top-left corner. The original marker pixels remain as part of the
    drawn grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    out = g.copy()
    for color, a, b in _marker_pairs(g, bg):
        _paint_pair(out, color, a, b)
    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
MAX_DISTANCE = 8
MIN_DISTANCE = 2
OUTER_MARGIN = 1
BBOX_PAD = 1
PAIR_RETRIES = 120
GEN_RETRIES = 400
SHAPE_KINDS = ("square", "tall", "wide")


def _pair_count_for_size(height: int, width: int) -> int:
    area = height * width
    if area < 180:
        options = [1, 2]
    elif area < 280:
        options = [2, 3]
    else:
        options = [2, 3, 4]
    return random.choice(options)


def _dimension_options(height: int, width: int, kind: str) -> list[tuple[int, int]]:
    max_dr = min(MAX_DISTANCE, height - 1 - 2 * OUTER_MARGIN)
    max_dc = min(MAX_DISTANCE, width - 1 - 2 * OUTER_MARGIN)
    options: list[tuple[int, int]] = []
    for dr in range(MIN_DISTANCE, max_dr + 1):
        for dc in range(MIN_DISTANCE, max_dc + 1):
            if kind == "square" and dr == dc:
                options.append((dr, dc))
            elif kind == "tall" and dr > dc:
                options.append((dr, dc))
            elif kind == "wide" and dc > dr:
                options.append((dr, dc))
    return options


def _place_pair(
    height: int,
    width: int,
    blocked: np.ndarray,
    kind: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    options = _dimension_options(height, width, kind)
    if not options:
        return None

    for _ in range(PAIR_RETRIES):
        dr, dc = random.choice(options)
        max_r0 = height - OUTER_MARGIN - 1 - dr
        max_c0 = width - OUTER_MARGIN - 1 - dc
        if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
            continue

        r0 = random.randint(OUTER_MARGIN, max_r0)
        c0 = random.randint(OUTER_MARGIN, max_c0)
        r1 = r0 + dr
        c1 = c0 + dc

        rr0 = max(0, r0 - BBOX_PAD)
        rr1 = min(height - 1, r1 + BBOX_PAD)
        cc0 = max(0, c0 - BBOX_PAD)
        cc1 = min(width - 1, c1 + BBOX_PAD)
        if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue

        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return (r0, c0), (r1, c1)
    return None


def _render_input(
    height: int,
    width: int,
    bg: int,
    specs: list[tuple[int, tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, a, b in specs:
        grid[a] = color
        grid[b] = color
    return grid


def _expected_output(inp: np.ndarray, specs: list[tuple[int, tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for color, a, b in specs:
        _paint_pair(out, color, a, b)
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)
    colors = [color for color in PALETTE if color != bg]
    pair_count = min(_pair_count_for_size(height, width), len(colors))

    blocked = np.zeros((height, width), dtype=bool)
    specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    chosen_colors = random.sample(colors, pair_count)

    first_kind = random.choice(SHAPE_KINDS)
    kind_sequence = [first_kind]
    while len(kind_sequence) < pair_count:
        kind_sequence.append(random.choice(SHAPE_KINDS))
    random.shuffle(kind_sequence)

    for color, kind in zip(chosen_colors, kind_sequence):
        placement = _place_pair(height, width, blocked, kind)
        if placement is None:
            return None
        a, b = placement
        specs.append((color, a, b))

    inp = _render_input(height, width, bg, specs)
    expected = _expected_output(inp, specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
