from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
WALL_PALETTE = (1, 4, 6, 7, 8, 9)


def _is_wall(color: int) -> bool:
    return int(color) not in (BLACK, RED, GREEN)


def _is_horizontal_corridor_cell(g: np.ndarray, r: int, c: int) -> bool:
    h, w = g.shape
    if not (0 < r < h - 1 and 0 <= c < w):
        return False
    return (
        int(g[r, c]) in (BLACK, RED, GREEN)
        and _is_wall(int(g[r - 1, c]))
        and _is_wall(int(g[r + 1, c]))
    )


def _is_vertical_corridor_cell(g: np.ndarray, r: int, c: int) -> bool:
    h, w = g.shape
    if not (0 <= r < h and 0 < c < w - 1):
        return False
    return (
        int(g[r, c]) in (BLACK, RED, GREEN)
        and _is_wall(int(g[r, c - 1]))
        and _is_wall(int(g[r, c + 1]))
    )


def _horizontal_span(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    left = c
    while left - 1 >= 0 and _is_horizontal_corridor_cell(g, r, left - 1):
        left -= 1

    right = c
    while right + 1 < g.shape[1] and _is_horizontal_corridor_cell(g, r, right + 1):
        right += 1
    return left, right


def _vertical_span(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    top = r
    while top - 1 >= 0 and _is_vertical_corridor_cell(g, top - 1, c):
        top -= 1

    bottom = r
    while bottom + 1 < g.shape[0] and _is_vertical_corridor_cell(g, bottom + 1, c):
        bottom += 1
    return top, bottom


def _grow(
    inp: np.ndarray,
    out: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    length: int,
    color: int,
) -> None:
    if length <= 0 or int(color) == GREY:
        return

    h, w = inp.shape
    r, c = start
    dr, dc = step
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(inp[r, c]) != BLACK or int(out[r, c]) != BLACK:
            break
        out[r, c] = int(color)
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each red seed outward from the two open ends of its 1-cell-wide channel.

    Horizontal channels are bounded by a top wall and a bottom wall; the top
    wall colors the leftward branch and the bottom wall colors the rightward
    branch. Vertical channels are bounded by a left wall and a right wall; the
    left wall colors the upward branch and the right wall colors the downward
    branch. Grey walls suppress growth on their side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[(g == RED) | (g == GREEN)] = BLACK

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    for r, c in seeds:
        horizontal = _is_horizontal_corridor_cell(g, r, c)
        vertical = _is_vertical_corridor_cell(g, r, c)
        assert horizontal ^ vertical, "each seed must sit in exactly one straight channel"

        if horizontal:
            left, right = _horizontal_span(g, r, c)
            corridor = g[r, left : right + 1]
            assert int(np.count_nonzero(corridor == RED)) == 1
            length = int(np.count_nonzero(corridor == GREEN))
            left_color = int(g[r - 1, c])
            right_color = int(g[r + 1, c])
            _grow(g, out, (r, left - 1), (0, -1), length, left_color)
            _grow(g, out, (r, right + 1), (0, 1), length, right_color)
        else:
            top, bottom = _vertical_span(g, r, c)
            corridor = g[top : bottom + 1, c]
            assert int(np.count_nonzero(corridor == RED)) == 1
            length = int(np.count_nonzero(corridor == GREEN))
            up_color = int(g[r, c - 1])
            down_color = int(g[r, c + 1])
            _grow(g, out, (top - 1, c), (-1, 0), length, up_color)
            _grow(g, out, (bottom + 1, c), (1, 0), length, down_color)

    return out


GEN_RETRIES = 400
PLACEMENT_TRIES = 120
GRID_SIZE_RANGE = (15, 24)
CHANNEL_LENGTH_RANGE = (4, 8)
MAX_MARKERS = 4
GREY_SIDE_PROB = 0.35
STOPPER_PROB = 0.4


def _sample_side_color() -> int:
    if random.random() < GREY_SIDE_PROB:
        return GREY
    return random.choice(WALL_PALETTE)


def _sample_stopper_color() -> int:
    if random.random() < 0.7:
        return GREY
    return random.choice(WALL_PALETTE)


def _branch_plan(marker_count: int, side_color: int, available: int) -> tuple[int, int | None]:
    if marker_count <= 0 or side_color == GREY or available <= 0:
        return 0, None

    if available >= 2 and marker_count >= 2 and random.random() < STOPPER_PROB:
        max_painted = min(marker_count - 1, available - 1)
        if max_painted >= 1:
            painted = random.randint(1, max_painted)
            return painted, painted + 1

    return min(marker_count, available), None


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _fits(candidate_cells: set[tuple[int, int]], reserved: np.ndarray) -> bool:
    h, w = reserved.shape
    for r, c in candidate_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    return all(not bool(reserved[r, c]) for r, c in candidate_cells)


def _place(grid: np.ndarray, reserved: np.ndarray, paint: dict[tuple[int, int], int], used: set[tuple[int, int]]) -> None:
    for (r, c), color in paint.items():
        grid[r, c] = int(color)
    for r, c in _expanded_cells(used, grid.shape):
        reserved[r, c] = True


def _horizontal_candidate(h: int, w: int) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], dict] | None:
    length = random.randint(*CHANNEL_LENGTH_RANGE)
    if w < length + 2 or h < 3:
        return None

    row = random.randint(1, h - 2)
    left = random.randint(1, w - length - 1)
    seed_idx = random.randrange(length)
    max_markers = min(MAX_MARKERS, length - 1)
    marker_count = random.randint(0, max_markers)
    others = [idx for idx in range(length) if idx != seed_idx]
    green_idxs = set(random.sample(others, marker_count))

    top_color = _sample_side_color()
    bottom_color = _sample_side_color()
    if marker_count > 0 and top_color == GREY and bottom_color == GREY:
        if random.random() < 0.5:
            top_color = random.choice(WALL_PALETTE)
        else:
            bottom_color = random.choice(WALL_PALETTE)

    available_left = left
    available_right = w - (left + length)
    left_growth, left_stopper = _branch_plan(marker_count, top_color, available_left)
    right_growth, right_stopper = _branch_plan(marker_count, bottom_color, available_right)

    paint: dict[tuple[int, int], int] = {}
    used: set[tuple[int, int]] = set()

    for idx in range(length):
        col = left + idx
        used.add((row, col))
        used.add((row - 1, col))
        used.add((row + 1, col))
        paint[(row - 1, col)] = top_color
        paint[(row + 1, col)] = bottom_color
        if idx == seed_idx:
            paint[(row, col)] = RED
        elif idx in green_idxs:
            paint[(row, col)] = GREEN

    for step in range(1, left_growth + 1):
        used.add((row, left - step))
    for step in range(1, right_growth + 1):
        used.add((row, left + length - 1 + step))

    if left_stopper is not None:
        stopper = (row, left - left_stopper)
        used.add(stopper)
        paint[stopper] = _sample_stopper_color()
    if right_stopper is not None:
        stopper = (row, left + length - 1 + right_stopper)
        used.add(stopper)
        paint[stopper] = _sample_stopper_color()

    meta = {
        "orientation": "h",
        "marker_count": marker_count,
        "growth": left_growth + right_growth,
    }
    return paint, used, meta


def _vertical_candidate(h: int, w: int) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], dict] | None:
    length = random.randint(*CHANNEL_LENGTH_RANGE)
    if h < length + 2 or w < 3:
        return None

    top = random.randint(1, h - length - 1)
    col = random.randint(1, w - 2)
    seed_idx = random.randrange(length)
    max_markers = min(MAX_MARKERS, length - 1)
    marker_count = random.randint(0, max_markers)
    others = [idx for idx in range(length) if idx != seed_idx]
    green_idxs = set(random.sample(others, marker_count))

    left_color = _sample_side_color()
    right_color = _sample_side_color()
    if marker_count > 0 and left_color == GREY and right_color == GREY:
        if random.random() < 0.5:
            left_color = random.choice(WALL_PALETTE)
        else:
            right_color = random.choice(WALL_PALETTE)

    available_up = top
    available_down = h - (top + length)
    up_growth, up_stopper = _branch_plan(marker_count, left_color, available_up)
    down_growth, down_stopper = _branch_plan(marker_count, right_color, available_down)

    paint: dict[tuple[int, int], int] = {}
    used: set[tuple[int, int]] = set()

    for idx in range(length):
        row = top + idx
        used.add((row, col))
        used.add((row, col - 1))
        used.add((row, col + 1))
        paint[(row, col - 1)] = left_color
        paint[(row, col + 1)] = right_color
        if idx == seed_idx:
            paint[(row, col)] = RED
        elif idx in green_idxs:
            paint[(row, col)] = GREEN

    for step in range(1, up_growth + 1):
        used.add((top - step, col))
    for step in range(1, down_growth + 1):
        used.add((top + length - 1 + step, col))

    if up_stopper is not None:
        stopper = (top - up_stopper, col)
        used.add(stopper)
        paint[stopper] = _sample_stopper_color()
    if down_stopper is not None:
        stopper = (top + length - 1 + down_stopper, col)
        used.add(stopper)
        paint[stopper] = _sample_stopper_color()

    meta = {
        "orientation": "v",
        "marker_count": marker_count,
        "growth": up_growth + down_growth,
    }
    return paint, used, meta


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_channels = 2 if h * w < 320 else 3
    channel_count = random.randint(1, max_channels)
    metas = []

    for _ in range(channel_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            if random.random() < 0.5:
                candidate = _horizontal_candidate(h, w)
            else:
                candidate = _vertical_candidate(h, w)
            if candidate is None:
                continue

            paint, used, meta = candidate
            if not _fits(_expanded_cells(used, (h, w)), reserved):
                continue

            _place(grid, reserved, paint, used)
            metas.append(meta)
            placed = True
            break
        if not placed:
            return None

    if not any(meta["growth"] > 0 for meta in metas):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    grown_pixels = int(np.count_nonzero((grid == BLACK) & (out != BLACK)))
    if grown_pixels == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-growth ARC input")
