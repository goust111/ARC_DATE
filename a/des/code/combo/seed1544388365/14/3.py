from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PALETTE_FILL_COLORS = (MAGENTA, ORANGE, AZURE)

GRID_H_RANGE = (10, 22)
LEFT_W_RANGE = (4, 8)
RIGHT_W_RANGE = (8, 18)
TURN_COUNT_CHOICES = (1, 2, 2, 3, 3, 4, 4, 5)
MIN_SEG_LEN = 2
MAX_SEG_LEN = 6
PATH_ATTEMPTS = 150
GEN_ATTEMPTS = 300


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    return int(cols[0])


def _read_palette(left: np.ndarray) -> list[tuple[int, int]]:
    entries: list[tuple[int, int]] = []
    for r in range(left.shape[0]):
        cols = np.flatnonzero(left[r] != BLACK)
        if cols.size == 0:
            continue
        assert cols.size == 2 and int(cols[1]) == int(cols[0]) + 1, "palette rows must contain one adjacent pair"
        c0, c1 = int(cols[0]), int(cols[1])
        direction_color = int(left[r, c0])
        paint_color = int(left[r, c1])
        assert direction_color in (RED, BLUE), "palette direction must be red or blue"
        assert paint_color not in (BLACK, GREY), "palette paint color must be visible and non-grey"
        entries.append((direction_color, paint_color))
    return entries


def _path_cells(region: np.ndarray) -> set[tuple[int, int]]:
    mask = (region == GREY) | (region == GREEN) | (region == YELLOW)
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _trace_path(region: np.ndarray) -> list[tuple[int, int]]:
    starts = np.argwhere(region == GREEN)
    ends = np.argwhere(region == YELLOW)
    assert starts.shape == (1, 2), "expected a unique green start"
    assert ends.shape == (1, 2), "expected a unique yellow end"

    start = (int(starts[0, 0]), int(starts[0, 1]))
    end = (int(ends[0, 0]), int(ends[0, 1]))
    cells = _path_cells(region)
    assert start in cells and end in cells

    for cell in cells:
        degree = sum((nr, nc) in cells for nr, nc in neighbors4(*cell))
        if cell in (start, end):
            assert degree == 1, "path endpoints must have degree 1"
        else:
            assert degree == 2, "internal path cells must have degree 2"

    path = [start]
    prev: tuple[int, int] | None = None
    curr = start
    while curr != end:
        nxts = [(nr, nc) for nr, nc in neighbors4(*curr) if (nr, nc) in cells and (nr, nc) != prev]
        assert len(nxts) == 1, "path must be uniquely traceable"
        nxt = nxts[0]
        path.append(nxt)
        prev, curr = curr, nxt

    assert len(path) == len(cells), "path must visit every path cell exactly once"
    return path


def _turn_indicator(v_in: tuple[int, int], v_out: tuple[int, int]) -> int:
    dot = int(v_in[0] * v_out[0] + v_in[1] * v_out[1])
    cross = int(v_in[0] * v_out[1] - v_in[1] * v_out[0])
    assert dot == 0 and abs(cross) == 1, "turns must be 90 degrees"
    return RED if cross < 0 else BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the right-side path from green to yellow and recolor each segment
    using the left-side palette, matching red=right turn and blue=left turn.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    palette = _read_palette(left)
    path = _trace_path(right)

    turn_indices: list[int] = []
    actual_indicators: list[int] = []
    for i in range(1, len(path) - 1):
        v_in = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        v_out = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if v_in != v_out:
            turn_indices.append(i)
            actual_indicators.append(_turn_indicator(v_in, v_out))

    assert len(palette) >= len(turn_indices), "palette must cover every turn"

    out = np.zeros((h, w), dtype=int)
    colors_by_index: list[int | None] = [None] * len(path)

    first_end = turn_indices[0] if turn_indices else len(path) - 1
    for idx in range(0, first_end + 1):
        colors_by_index[idx] = GREEN

    for k, turn_idx in enumerate(turn_indices):
        pred_indicator, paint_color = palette[k]
        seg_color = paint_color if pred_indicator == actual_indicators[k] else MAROON
        seg_start = turn_idx + 1
        seg_end = turn_indices[k + 1] if k + 1 < len(turn_indices) else len(path) - 1
        limit = seg_end if seg_end < len(path) - 1 else seg_end - 1
        for idx in range(seg_start, limit + 1):
            colors_by_index[idx] = seg_color

    colors_by_index[-1] = YELLOW
    assert all(color is not None for color in colors_by_index)

    col_offset = sep + 1
    for (r, c), color in zip(path, colors_by_index):
        out[r, c + col_offset] = int(color)
    return out


def _segment_cells(start: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS4[dir_idx]
    return [(start[0] + dr * step, start[1] + dc * step) for step in range(1, length + 1)]


def _segment_valid(
    occupied: set[tuple[int, int]],
    start: tuple[int, int],
    dir_idx: int,
    length: int,
    h: int,
    w: int,
) -> bool:
    new_cells = _segment_cells(start, dir_idx, length)
    prev = start
    seen: set[tuple[int, int]] = set()
    for cell in new_cells:
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return False
        if cell in occupied or cell in seen:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied and (nr, nc) != prev:
                return False
        seen.add(cell)
        prev = cell
    return True


def _search_path(
    h: int,
    w: int,
    dirs: list[int],
    seg_idx: int,
    current: tuple[int, int],
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if seg_idx == len(dirs):
        return path

    lengths = list(range(MIN_SEG_LEN, MAX_SEG_LEN + 1))
    rng.shuffle(lengths)
    for length in lengths:
        if not _segment_valid(occupied, current, dirs[seg_idx], length, h, w):
            continue
        segment = _segment_cells(current, dirs[seg_idx], length)
        next_current = segment[-1]
        result = _search_path(
            h,
            w,
            dirs,
            seg_idx + 1,
            next_current,
            occupied | set(segment),
            path + segment,
            rng,
        )
        if result is not None:
            return result
    return None


def _path_degree_ok(path: list[tuple[int, int]]) -> bool:
    cells = set(path)
    if len(cells) != len(path):
        return False
    for i, cell in enumerate(path):
        degree = sum((nr, nc) in cells for nr, nc in neighbors4(*cell))
        if i in (0, len(path) - 1):
            if degree != 1:
                return False
        elif degree != 2:
            return False
    return True


def _build_random_polyline(rng: random.Random, h: int, w: int, turn_count: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    usable_h = h - 2
    usable_w = w - 2
    if usable_h < 3 or usable_w < 3:
        return None

    for _ in range(PATH_ATTEMPTS):
        sides = [rng.choice((-1, 1)) for _ in range(turn_count)]
        dirs = [rng.randrange(4)]
        for side in sides:
            dirs.append((dirs[-1] + side) % 4)

        starts = [(r, c) for r in range(usable_h) for c in range(usable_w)]
        rng.shuffle(starts)
        for start in starts[: min(len(starts), 40)]:
            path = _search_path(usable_h, usable_w, dirs, 0, start, {start}, [start], rng)
            if path is None:
                continue
            if not _path_degree_ok(path):
                continue
            shifted = [(r + 1, c + 1) for r, c in path]
            indicators = [RED if side == 1 else BLUE for side in sides]
            return shifted, indicators
    return None


def _place_palette(
    grid: np.ndarray,
    sep: int,
    turn_indicators: list[int],
    rng: random.Random,
) -> None:
    h = grid.shape[0]
    left_w = sep
    pair_c = rng.randint(1, left_w - 2)
    rows = sorted(rng.sample(list(range(1, h - 1)), len(turn_indicators)))
    for row, actual in zip(rows, turn_indicators):
        matched = bool(rng.randrange(2))
        direction_color = actual if matched else (BLUE if actual == RED else RED)
        paint_color = rng.choice(PALETTE_FILL_COLORS)
        grid[row, pair_c] = direction_color
        grid[row, pair_c + 1] = paint_color


def _validate_generated_grid(grid: np.ndarray) -> None:
    sep = _find_separator_col(grid)
    palette = _read_palette(grid[:, :sep])
    path = _trace_path(grid[:, sep + 1 :])
    turns = 0
    for i in range(1, len(path) - 1):
        v_in = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        v_out = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if v_in != v_out:
            turns += 1
    assert len(palette) == turns


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    left_w = rng.randint(*LEFT_W_RANGE)
    right_w = rng.randint(*RIGHT_W_RANGE)
    sep = left_w
    w = left_w + 1 + right_w

    turn_count = rng.choice(TURN_COUNT_CHOICES)
    max_pairs = h - 2
    if turn_count > max_pairs:
        turn_count = max_pairs
    if turn_count <= 0:
        return None

    built = _build_random_polyline(rng, h, right_w, turn_count)
    if built is None:
        return None
    path, actual_indicators = built

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep] = GREY
    _place_palette(grid, sep, actual_indicators, rng)

    for r, c in path[1:-1]:
        grid[r, sep + c] = GREY
    sr, sc = path[0]
    er, ec = path[-1]
    grid[sr, sep + sc] = GREEN
    grid[er, sep + ec] = YELLOW

    _validate_generated_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
