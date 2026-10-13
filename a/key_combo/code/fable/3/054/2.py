from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
START = 3
HEAD = 5
PALETTE = [1, 2, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
MIN_BAR_LEN = 2
MAX_BAR_LEN = 8
MIN_SEG_LEN = 2
MAX_SEG_LEN = 6
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _expand_cells(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _scan_bottom_bars(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    h, w = map(int, grid.shape)
    bar_lengths: dict[int, int] = {}
    bar_cols: dict[int, int] = {}
    for c in range(w):
        color = int(grid[h - 1, c])
        if color in (bg, START, HEAD):
            continue
        if color in bar_lengths:
            continue
        length = 0
        r = h - 1
        while r >= 0 and int(grid[r, c]) == color:
            length += 1
            r -= 1
        bar_lengths[color] = int(length)
        bar_cols[color] = int(c)
    return bar_lengths, bar_cols


def _decode_segment(
    grid: np.ndarray,
    color: int,
    bar_len: int,
    bar_col: int,
) -> tuple[tuple[int, int], int]:
    h, _w = map(int, grid.shape)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    bar_cells = {(r, int(bar_col)) for r in range(h - int(bar_len), h)}
    seg_cells = [cell for cell in coords if cell not in bar_cells]
    assert seg_cells, f"missing segment for color {color}"

    if len(seg_cells) == 1:
        r, c = seg_cells[0]
        dirs = []
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == HEAD:
                dirs.append((int(dr), int(dc)))
        assert len(dirs) == 1
        return dirs[0], 1

    rows = sorted({r for r, _c in seg_cells})
    cols = sorted({c for _r, c in seg_cells})
    length = len(seg_cells)

    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[0] + length))
        left_head = _in_bounds(grid.shape, r, cols[0] - 1) and int(grid[r, cols[0] - 1]) == HEAD
        right_head = _in_bounds(grid.shape, r, cols[-1] + 1) and int(grid[r, cols[-1] + 1]) == HEAD
        assert left_head ^ right_head
        direction = (0, -1) if left_head else (0, 1)
        return direction, length

    if len(cols) == 1:
        c = cols[0]
        assert rows == list(range(rows[0], rows[0] + length))
        up_head = _in_bounds(grid.shape, rows[0] - 1, c) and int(grid[rows[0] - 1, c]) == HEAD
        down_head = _in_bounds(grid.shape, rows[-1] + 1, c) and int(grid[rows[-1] + 1, c]) == HEAD
        assert up_head ^ down_head
        direction = (-1, 0) if up_head else (1, 0)
        return direction, length

    raise AssertionError(f"segment for color {color} is not straight")


def _render_trace(
    shape: tuple[int, int],
    bg: int,
    start: tuple[int, int],
    instructions: list[tuple[int, tuple[int, int], int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    cur_r, cur_c = map(int, start)
    out[cur_r, cur_c] = START
    for color, (dr, dc), length in instructions:
        for _ in range(int(length)):
            cur_r += int(dr)
            cur_c += int(dc)
            assert _in_bounds(shape, cur_r, cur_c)
            out[cur_r, cur_c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the colored direction segments by bottom-bar length and concatenate them into one trace.

    Input structure:
    - The grid contains one green start pixel, several colored straight segments, and bottom-edge vertical bars.
    - Each segment is one cell thick, uses a unique color, and has a single gray head pixel attached to one end.
    - Each bottom bar is one cell wide, has a unique length, and shares its color with exactly one segment.

    Rule:
    - Sort the bottom bars by length from longest to shortest to obtain the execution order of colors.
    - For each color in that order, read the matching scattered segment: its movement direction is from the far tail toward the gray head, and its step count is the colored segment length.
    - Starting from the green pixel, draw these colored runs end to end. Remove the original scattered segments and bars, keeping only the background, the green start, and the traced polyline.

    Key insight:
    - The bottom bars are not geometry to copy; they are a schedule. The scattered segment placement is irrelevant except for its length and the gray-head direction cue.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    starts = np.argwhere(g == START)
    assert starts.shape == (1, 2)
    start = tuple(map(int, starts[0]))

    bar_lengths, bar_cols = _scan_bottom_bars(g, bg)
    assert bar_lengths
    order = sorted(bar_lengths, key=lambda color: int(bar_lengths[color]), reverse=True)

    instructions = []
    for color in order:
        direction, length = _decode_segment(g, int(color), int(bar_lengths[color]), int(bar_cols[color]))
        instructions.append((int(color), direction, int(length)))

    return _render_trace(g.shape, bg, start, instructions)


def _sample_trace(
    h: int,
    w: int,
    trace_bottom: int,
    n_segments: int,
) -> tuple[tuple[int, int], list[tuple[tuple[int, int], int]]] | None:
    if trace_bottom < 6 or w < 8:
        return None

    for _ in range(160):
        start = (random.randint(1, trace_bottom), random.randint(1, w - 2))

        def dfs(
            cur: tuple[int, int],
            step: int,
            occupied: set[tuple[int, int]],
            prev_dir: tuple[int, int] | None,
        ) -> list[tuple[tuple[int, int], int]] | None:
            if step == n_segments:
                return []

            dirs = list(DIR4)
            random.shuffle(dirs)
            if prev_dir is not None:
                dirs.sort(key=lambda direction: direction == prev_dir)

            for dr, dc in dirs:
                rr, cc = cur
                cells: list[tuple[int, int]] = []
                feasible: list[tuple[int, tuple[tuple[int, int], ...]]] = []
                for length in range(1, MAX_SEG_LEN + 1):
                    rr += dr
                    cc += dc
                    if not (1 <= rr <= trace_bottom and 1 <= cc < w - 1):
                        break
                    if (rr, cc) in occupied:
                        break
                    cells.append((int(rr), int(cc)))
                    if length >= MIN_SEG_LEN:
                        feasible.append((int(length), tuple(cells)))
                random.shuffle(feasible)
                for length, new_cells in feasible:
                    result = dfs(new_cells[-1], step + 1, occupied | set(new_cells), (dr, dc))
                    if result is not None:
                        return [((int(dr), int(dc)), int(length))] + result
            return None

        trace = dfs(start, 0, {start}, None)
        if trace is None:
            continue

        axes = {"v" if dr else "h" for (dr, _dc), _length in trace}
        if len(axes) < 2:
            continue
        return start, trace

    return None


def _trace_cells(
    start: tuple[int, int],
    trace: list[tuple[tuple[int, int], int]],
) -> list[tuple[int, int]]:
    cells = [tuple(map(int, start))]
    cur_r, cur_c = map(int, start)
    for (dr, dc), length in trace:
        for _ in range(int(length)):
            cur_r += int(dr)
            cur_c += int(dc)
            cells.append((int(cur_r), int(cur_c)))
    return cells


def _segment_object_cells(
    head: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    hr, hc = map(int, head)
    cells = [(hr, hc)]
    for step in range(1, int(length) + 1):
        cells.append((int(hr - dr * step), int(hc - dc * step)))
    return cells


def _place_segment_object(
    grid: np.ndarray,
    color: int,
    direction: tuple[int, int],
    length: int,
    top_limit: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid.shape)
    candidates = [(r, c) for r in range(1, top_limit + 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for head in candidates:
        cells = _segment_object_cells(head, direction, length)
        if not all(1 <= r <= top_limit and 1 <= c < w - 1 for r, c in cells):
            continue
        if any((r, c) in forbidden for r, c in cells):
            continue
        hr, hc = cells[0]
        grid[hr, hc] = HEAD
        for r, c in cells[1:]:
            grid[r, c] = int(color)
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    colors = random.sample(PALETTE, n_segments)
    bar_lengths = sorted(random.sample(range(MIN_BAR_LEN, MAX_BAR_LEN + 1), n_segments), reverse=True)

    max_bar = max(bar_lengths)
    min_h = max(18, max_bar + 12)
    min_w = max(16, 2 * n_segments + 6)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    trace_bottom = h - max_bar - 4
    sampled = _sample_trace(h, w, trace_bottom, n_segments)
    if sampled is None:
        return None
    start, trace = sampled

    instructions = [
        (int(color), (int(direction[0]), int(direction[1])), int(length))
        for color, (direction, length) in zip(colors, trace)
    ]
    expected = _render_trace((h, w), BG, start, instructions)

    grid = np.full((h, w), BG, dtype=int)
    grid[start] = START

    bar_cols = list(range(1, w - 1, 2))
    if len(bar_cols) < n_segments:
        return None
    chosen_bar_cols = random.sample(bar_cols, n_segments)
    random.shuffle(chosen_bar_cols)
    for color, bar_len, col in zip(colors, bar_lengths, chosen_bar_cols):
        grid[h - bar_len : h, col] = int(color)

    reserved = _expand_cells(_trace_cells(start, trace), grid.shape)
    reserved |= {(r, c) for r in range(trace_bottom + 1, h) for c in range(w)}

    place_order = sorted(instructions, key=lambda item: item[2], reverse=True)
    for color, direction, length in place_order:
        cells = _place_segment_object(grid, color, direction, length, trace_bottom, reserved)
        if cells is None:
            return None
        reserved |= _expand_cells(cells, grid.shape)

    if np.count_nonzero(grid == START) != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
