import itertools
import random
from dataclasses import dataclass

import numpy as np


MAGENTA = 6
GENERATION_RETRIES = 400
RECT_PLACEMENT_RETRIES = 200
GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
RECT_COUNT_RANGE = (2, 5)
MIN_RECT_SIDE = 2
MAX_RECT_SIDE = 7
MIN_RECT_AREA = 4
MAX_RECT_AREA = 30
SEPARATION_PAD = 1
ARC_COLORS = tuple(range(10))


@dataclass(frozen=True)
class Rect:
    r0: int
    r1: int
    c0: int
    c1: int
    color: int

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _diagonal_options(rect: Rect) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        ((rect.r0, rect.c0), (rect.r1, rect.c1)),
        ((rect.r0, rect.c1), (rect.r1, rect.c0)),
    )


def _overlap_area(a: Rect, b: Rect) -> int:
    row_overlap = min(a.r1, b.r1) - max(a.r0, b.r0) + 1
    col_overlap = min(a.c1, b.c1) - max(a.c0, b.c0) + 1
    if row_overlap <= 0 or col_overlap <= 0:
        return 0
    return row_overlap * col_overlap


def _expanded_overlap(a: Rect, b: Rect, pad: int) -> bool:
    return not (
        a.r1 + pad < b.r0 - pad
        or b.r1 + pad < a.r0 - pad
        or a.c1 + pad < b.c0 - pad
        or b.c1 + pad < a.c0 - pad
    )


def _render_output(shape: tuple[int, int], background: int, rects: list[Rect]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    out = np.full(shape, background, dtype=int)
    for rect in rects:
        region = (slice(rect.r0, rect.r1 + 1), slice(rect.c0, rect.c1 + 1))
        cover[region] += 1
        out[region] = rect.color
    out[cover >= 2] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each color that appears exactly twice defines a filled axis-aligned rectangle.
    Cells covered by two or more such rectangles are recolored to magenta.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == MAGENTA), "input must not already contain magenta"

    background = _background_color(g)
    rects: list[Rect] = []

    for color in np.unique(g):
        color = int(color)
        if color == background:
            continue
        pts = np.argwhere(g == color)
        assert pts.shape == (2, 2), "each non-background color must appear exactly twice"
        (ra, ca), (rb, cb) = pts
        ra, ca, rb, cb = map(int, (ra, ca, rb, cb))
        assert ra != rb and ca != cb, "same-color endpoints must be on different rows and columns"
        rects.append(Rect(min(ra, rb), max(ra, rb), min(ca, cb), max(ca, cb), color))

    assert rects, "expected at least one rectangle marker pair"
    return _render_output(g.shape, background, rects)


def _sample_rect_geometry(rng: random.Random, height: int, width: int) -> Rect | None:
    max_h = min(MAX_RECT_SIDE, height)
    max_w = min(MAX_RECT_SIDE, width)
    if max_h < MIN_RECT_SIDE or max_w < MIN_RECT_SIDE:
        return None

    for _ in range(RECT_PLACEMENT_RETRIES):
        rect_h = rng.randint(MIN_RECT_SIDE, max_h)
        rect_w = rng.randint(MIN_RECT_SIDE, max_w)
        area = rect_h * rect_w
        if not (MIN_RECT_AREA <= area <= min(MAX_RECT_AREA, height * width // 2)):
            continue
        r0 = rng.randint(0, height - rect_h)
        c0 = rng.randint(0, width - rect_w)
        return Rect(r0, r0 + rect_h - 1, c0, c0 + rect_w - 1, 0)
    return None


def _sample_disjoint_rect(
    rng: random.Random,
    height: int,
    width: int,
    existing: list[Rect],
) -> Rect | None:
    for _ in range(RECT_PLACEMENT_RETRIES):
        rect = _sample_rect_geometry(rng, height, width)
        if rect is None:
            continue
        if any(_expanded_overlap(rect, other, SEPARATION_PAD) for other in existing):
            continue
        return rect
    return None


def _sample_overlapping_pair(rng: random.Random, height: int, width: int) -> tuple[Rect, Rect] | None:
    for _ in range(RECT_PLACEMENT_RETRIES * 2):
        first = _sample_rect_geometry(rng, height, width)
        second = _sample_rect_geometry(rng, height, width)
        if first is None or second is None:
            continue

        overlap = _overlap_area(first, second)
        if overlap <= 0:
            continue
        if first == second:
            continue
        if first.area - overlap < 2 or second.area - overlap < 2:
            continue
        return first, second
    return None


def _assign_endpoints(rects: list[Rect]) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    options = [_diagonal_options(rect) for rect in rects]
    for picks in itertools.product((0, 1), repeat=len(rects)):
        used: set[tuple[int, int]] = set()
        chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
        valid = True
        for rect_options, pick in zip(options, picks):
            a, b = rect_options[pick]
            if a in used or b in used:
                valid = False
                break
            used.add(a)
            used.add(b)
            chosen.append((a, b))
        if valid:
            return chosen
    return None


def _build_input(
    shape: tuple[int, int],
    background: int,
    rects: list[Rect],
) -> np.ndarray | None:
    endpoint_pairs = _assign_endpoints(rects)
    if endpoint_pairs is None:
        return None

    grid = np.full(shape, background, dtype=int)
    for rect, (a, b) in zip(rects, endpoint_pairs):
        grid[a] = rect.color
        grid[b] = rect.color
    return grid


def _validate_input(grid: np.ndarray, rects: list[Rect], background: int, force_overlap: bool) -> bool:
    if grid is None:
        return False
    if grid.ndim != 2 or np.any(grid == MAGENTA):
        return False

    unique_colors = [int(color) for color in np.unique(grid) if int(color) != background]
    if len(unique_colors) != len(rects):
        return False
    for color in unique_colors:
        pts = np.argwhere(grid == color)
        if pts.shape != (2, 2):
            return False
        if int(pts[0, 0]) == int(pts[1, 0]) or int(pts[0, 1]) == int(pts[1, 1]):
            return False

    expected = _render_output(grid.shape, background, rects)
    has_overlap = bool(np.any(expected == MAGENTA))
    if has_overlap != force_overlap:
        return False
    if background != _background_color(grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    background = rng.choice([color for color in ARC_COLORS if color != MAGENTA])
    usable_colors = [color for color in ARC_COLORS if color not in (background, MAGENTA)]
    rect_count = rng.randint(*RECT_COUNT_RANGE)
    colors = rng.sample(usable_colors, rect_count)
    force_overlap = rng.random() < 0.5

    rects: list[Rect] = []
    if force_overlap:
        pair = _sample_overlapping_pair(rng, height, width)
        if pair is None:
            return None
        first, second = pair
        rects.append(Rect(first.r0, first.r1, first.c0, first.c1, colors[0]))
        rects.append(Rect(second.r0, second.r1, second.c0, second.c1, colors[1]))
        start_index = 2
    else:
        start_index = 0

    for color in colors[start_index:]:
        rect = _sample_disjoint_rect(rng, height, width, rects)
        if rect is None:
            return None
        rects.append(Rect(rect.r0, rect.r1, rect.c0, rect.c1, color))

    grid = _build_input((height, width), background, rects)
    if not _validate_input(grid, rects, background, force_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
