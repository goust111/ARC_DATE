from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BACKGROUND = GREY
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
NOISE_COLORS = (PINK, TEAL, MAROON)

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
RECT_COUNT_CHOICES = (2, 3, 4)
LARGEST_SIDE_RANGE = (6, 10)
SMALL_RECT_SIDE_RANGE = (2, 6)
NOISE_COUNT_RANGE = (5, 10)
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _solid_black_prefix(grid: np.ndarray) -> np.ndarray:
    mask = (grid == BLACK).astype(int)
    return mask.cumsum(axis=0).cumsum(axis=1)


def _rect_black_count(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _largest_solid_black_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    heights = np.zeros(grid.shape[1], dtype=int)
    best_area = -1
    best_rects: set[tuple[int, int, int, int]] = set()

    for r in range(grid.shape[0]):
        heights = np.where(grid[r] == BLACK, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(grid.shape[1] + 1):
            cur = int(heights[c]) if c < grid.shape[1] else 0
            start = c
            while stack and stack[-1][1] > cur:
                c0, h = stack.pop()
                area = h * (c - c0)
                rect = (r - h + 1, c0, r, c - 1)
                if area > best_area:
                    best_area = area
                    best_rects = {rect}
                elif area == best_area:
                    best_rects.add(rect)
                start = c0
            if not stack or stack[-1][1] < cur:
                stack.append((start, cur))

    assert best_area > 0 and len(best_rects) == 1
    return next(iter(best_rects))


def _marker_position(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.isin(grid, MARKER_COLORS))
    assert len(coords) == 1, "expected exactly one marker-color pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _apply_rule(grid: np.ndarray, rect: tuple[int, int, int, int], marker: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mr, mc = marker
    marker_color = int(grid[mr, mc])

    out = np.array(grid, copy=True)
    out[mr, mc] = BACKGROUND
    out[r0 + 1 : r1, c0 + 1 : c1] = BACKGROUND

    if r1 - r0 >= 2 and c1 - c0 >= 2:
        rows, cols = np.indices((r1 - r0 - 1, c1 - c0 - 1))
        rows += r0 + 1
        cols += c0 + 1
        lattice = (rows + cols) % 4 == mc % 4
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[lattice] = marker_color
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve the border of the unique largest black rectangle and refill its interior with a marker-controlled diagonal lattice.

    Input structure:
    - The grid has a grey background with 2-4 solid black axis-aligned rectangles of different sizes plus a few scattered non-black noise pixels.
    - Exactly one marker pixel lies outside every rectangle; its color is one of {blue, red, green, yellow, orange}.
    - The largest solid black rectangle is uniquely largest, and its interior is at least 4x4 after removing the border.

    Rule:
    - Find the unique largest solid black rectangle anywhere in the grid.
    - Keep that rectangle's 1-pixel black border unchanged.
    - Replace every strict interior cell by grey unless it lies on the NW-SE period-4 lattice whose phase is set by the marker column: color interior cell (r, c) iff (r + c) mod 4 == marker_col mod 4.
    - The lattice color is the marker color, then erase the marker back to grey.
    - Leave all other rectangles and all noise pixels unchanged.

    Key insight:
    - A single control pixel encodes two independent parameters at once: its color chooses the lattice color, and its column chooses the modulo-4 lattice phase inside the largest rectangle.
    """
    g = np.asarray(grid, dtype=int)
    rect = _largest_solid_black_rectangle(g)
    marker = _marker_position(g)
    return _apply_rule(g, rect, marker)


def _boxes_separated(
    box: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _paint_rect(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK


def _sample_largest_dims(h: int, w: int, rng: random.Random) -> tuple[int, int]:
    candidates = []
    for rh in range(LARGEST_SIDE_RANGE[0], min(LARGEST_SIDE_RANGE[1], h - 2) + 1):
        for rw in range(LARGEST_SIDE_RANGE[0], min(LARGEST_SIDE_RANGE[1], w - 2) + 1):
            frac = (rh * rw) / float(h * w)
            if 0.12 <= frac <= 0.30:
                candidates.append((rh, rw))
    if not candidates:
        raise ValueError("no largest rectangle sizes fit the grid")
    return rng.choice(candidates)


def _sample_smaller_dims(
    largest_area: int,
    count: int,
    h: int,
    w: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    candidates = [
        (rh, rw)
        for rh in range(SMALL_RECT_SIDE_RANGE[0], min(SMALL_RECT_SIDE_RANGE[1], h - 2) + 1)
        for rw in range(SMALL_RECT_SIDE_RANGE[0], min(SMALL_RECT_SIDE_RANGE[1], w - 2) + 1)
        if rh * rw < largest_area
    ]
    if not candidates:
        raise ValueError("no smaller rectangle sizes fit the grid")

    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for _ in range(count):
        area_limit = largest_area - 1 - len(dims)
        viable = [dim for dim in candidates if dim[0] * dim[1] <= area_limit]
        if not viable:
            viable = candidates
        rng.shuffle(viable)
        chosen = None
        for dim in viable:
            area = dim[0] * dim[1]
            if area not in used_areas or len(used_areas) + (count - len(dims)) > len(candidates):
                chosen = dim
                used_areas.add(area)
                break
        if chosen is None:
            chosen = viable[0]
        dims.append(chosen)
    return dims


def _place_rectangles(
    h: int,
    w: int,
    rect_dims: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for idx, (rh, rw) in enumerate(rect_dims):
        positions = [
            (r0, c0, r0 + rh - 1, c0 + rw - 1)
            for r0 in range(1, h - rh)
            for c0 in range(1, w - rw)
        ]
        rng.shuffle(positions)
        gap = 2 if idx == 0 else 1
        placed = None
        for box in positions:
            if _boxes_separated(box, boxes, gap=gap):
                placed = box
                break
        if placed is None:
            return None
        boxes.append(placed)
    return boxes


def _cell_touches_box(cell: tuple[int, int], box: tuple[int, int, int, int], gap: int) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 - gap <= r <= r1 + gap and c0 - gap <= c <= c1 + gap


def _sample_marker(
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int] | None:
    target_phase = rng.randrange(4)
    candidates = []
    for r in range(h):
        for c in range(w):
            if c % 4 != target_phase:
                continue
            if any(_cell_touches_box((r, c), box, gap=1) for box in boxes):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    mr, mc = rng.choice(candidates)
    return mr, mc, rng.choice(MARKER_COLORS)


def _sample_noise_cells(
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
    marker: tuple[int, int],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) == marker:
                continue
            if any(_cell_touches_box((r, c), box, gap=1) for box in boxes):
                continue
            if max(abs(r - marker[0]), abs(c - marker[1])) <= 1:
                continue
            candidates.append((r, c))
    if len(candidates) < count:
        raise ValueError("not enough room for noise")
    return rng.sample(candidates, count)


def _bruteforce_unique_largest_black_rectangle(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    prefix = _solid_black_prefix(grid)
    best_area = -1
    best_rects: set[tuple[int, int, int, int]] = set()
    for r0 in range(grid.shape[0]):
        for c0 in range(grid.shape[1]):
            for r1 in range(r0, grid.shape[0]):
                for c1 in range(c0, grid.shape[1]):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if _rect_black_count(prefix, r0, c0, r1, c1) != area:
                        continue
                    rect = (r0, c0, r1, c1)
                    if area > best_area:
                        best_area = area
                        best_rects = {rect}
                    else:
                        best_rects.add(rect)
    if best_area <= 0 or len(best_rects) != 1:
        return None
    return next(iter(best_rects))


def _expected_output(
    grid: np.ndarray,
    largest_box: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> np.ndarray:
    return _apply_rule(grid, largest_box, marker)


def _validate_scene(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    largest_box: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> bool:
    brute = _bruteforce_unique_largest_black_rectangle(grid)
    if brute != largest_box:
        return False

    expected_black = 0
    for box in boxes:
        r0, c0, r1, c1 = box
        expected_black += (r1 - r0 + 1) * (c1 - c0 + 1)
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
            return False
    if int(np.count_nonzero(grid == BLACK)) != expected_black:
        return False

    mr, mc = marker
    if int(grid[mr, mc]) not in MARKER_COLORS:
        return False

    if any(_cell_touches_box(marker, box, gap=1) for box in boxes):
        return False

    r0, c0, r1, c1 = largest_box
    if r1 - r0 < 5 or c1 - c0 < 5:
        return False

    interior_rows, interior_cols = np.indices((r1 - r0 - 1, c1 - c0 - 1))
    interior_rows += r0 + 1
    interior_cols += c0 + 1
    hits = int(np.count_nonzero((interior_rows + interior_cols) % 4 == mc % 4))
    interior_area = int((r1 - r0 - 1) * (c1 - c0 - 1))
    if not (0 < hits < interior_area):
        return False

    expected = _expected_output(grid, largest_box, marker)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if actual[mr, mc] != BACKGROUND:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    rect_count = rng.choice(RECT_COUNT_CHOICES)

    largest_h, largest_w = _sample_largest_dims(h, w, rng)
    smaller_dims = _sample_smaller_dims(largest_h * largest_w, rect_count - 1, h, w, rng)
    rect_dims = [(largest_h, largest_w), *smaller_dims]
    boxes = _place_rectangles(h, w, rect_dims, rng)
    if boxes is None:
        return None

    largest_box = boxes[0]
    marker_info = _sample_marker(h, w, boxes, rng)
    if marker_info is None:
        return None
    mr, mc, marker_color = marker_info

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for box in boxes:
        _paint_rect(grid, box)
    grid[mr, mc] = marker_color

    noise_count = rng.randint(*NOISE_COUNT_RANGE)
    try:
        noise_cells = _sample_noise_cells(h, w, boxes, (mr, mc), noise_count, rng)
    except ValueError:
        return None
    for nr, nc in noise_cells:
        grid[nr, nc] = rng.choice(NOISE_COLORS)

    if not _validate_scene(grid, boxes, largest_box, (mr, mc)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
