from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
OBJECT_COUNT_CAP = 5
RECT_SIZE_MIN = 2
RECT_SIZE_MAX = 8
RECT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _anchor_specs(grid: np.ndarray) -> list[tuple[int, int, int, int, int, bool]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert colors, "expected at least one anchor pair"

    specs: list[tuple[int, int, int, int, int, bool]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each foreground color must occur exactly twice"

        (r_a, c_a), (r_b, c_b) = coords.tolist()
        r_a, c_a, r_b, c_b = map(int, (r_a, c_a, r_b, c_b))
        assert r_a != r_b and c_a != c_b, "anchors must define opposite rectangle corners"

        top, bottom = sorted((r_a, r_b))
        left, right = sorted((c_a, c_b))
        diagonal_span = (bottom - top) + (right - left)
        assert diagonal_span % 2 == 0, "opposite stripe anchors must share checkerboard parity"

        points = {(r_a, c_a), (r_b, c_b)}
        backslash = points == {(top, left), (bottom, right)}
        slash = points == {(bottom, left), (top, right)}
        assert backslash != slash, "anchor pair must form exactly one diagonal orientation"

        specs.append((color, top, left, bottom, right, backslash))
    return specs


def _paint_stripes(
    out: np.ndarray,
    color: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    backslash: bool,
) -> None:
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if backslash:
                paint = ((r - top) + (c - left)) % 2 == 0
            else:
                paint = ((r - top) + (right - c)) % 2 == 0
            if not paint:
                continue
            assert int(out[r, c]) in (BG, color), "stripe rectangles must not overlap"
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal stripes inside the rectangle implied by each same-color anchor pair.

    Each foreground color appears exactly twice on a black background. The two cells are
    opposite corners of an axis-aligned rectangle and determine whether the rectangle
    should receive "\" or "/" diagonal stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color, top, left, bottom, right, backslash in _anchor_specs(g):
        _paint_stripes(out, color, top, left, bottom, right, backslash)
    return out


def _shape_choices(height: int, width: int) -> list[tuple[int, int]]:
    max_h = min(RECT_SIZE_MAX, height)
    max_w = min(RECT_SIZE_MAX, width)
    choices = []
    for rect_h in range(RECT_SIZE_MIN, max_h + 1):
        for rect_w in range(RECT_SIZE_MIN, max_w + 1):
            if rect_h % 2 != rect_w % 2:
                continue
            if rect_h == 2 and rect_w == 2:
                continue
            choices.append((rect_h, rect_w))
    return choices


def _mark_occupied(mask: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    r0 = max(0, top - RECT_PADDING)
    c0 = max(0, left - RECT_PADDING)
    r1 = min(mask.shape[0] - 1, bottom + RECT_PADDING)
    c1 = min(mask.shape[1] - 1, right + RECT_PADDING)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _orientations(count: int) -> list[bool]:
    if count == 1:
        return [bool(random.getrandbits(1))]
    if random.random() < 0.7:
        values = [True, False] + [bool(random.getrandbits(1)) for _ in range(count - 2)]
        random.shuffle(values)
        return values
    return [bool(random.getrandbits(1)) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    shape_choices = _shape_choices(height, width)
    if not shape_choices:
        return None

    max_objects = min(OBJECT_COUNT_CAP, len(PALETTE), max(1, (height * width) // 35))
    object_count = random.randint(1, max_objects)
    colors = random.sample(PALETTE, object_count)
    orientations = _orientations(object_count)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for color, backslash in zip(colors, orientations):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rect_h, rect_w = random.choice(shape_choices)
            top = random.randint(0, height - rect_h)
            left = random.randint(0, width - rect_w)
            bottom = top + rect_h - 1
            right = left + rect_w - 1

            r0 = max(0, top - RECT_PADDING)
            c0 = max(0, left - RECT_PADDING)
            r1 = min(height - 1, bottom + RECT_PADDING)
            c1 = min(width - 1, right + RECT_PADDING)
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            if backslash:
                anchors = ((top, left), (bottom, right))
            else:
                anchors = ((bottom, left), (top, right))
            for r, c in anchors:
                grid[r, c] = color

            _mark_occupied(occupied, top, left, bottom, right)
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.count_nonzero(out) <= np.count_nonzero(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-anchor ARC input")
