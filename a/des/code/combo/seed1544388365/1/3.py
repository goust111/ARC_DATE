from __future__ import annotations

import random
from collections import deque

import numpy as np


COLOR_IDS = tuple(range(10))
GEN_RETRIES = 400
ROW_COL_OPTIONS = (1, 2, 3)
CELL_SIZE_RANGE = (3, 6)
MARGIN_RANGE = (1, 4)
BRANCHES = ("solid", "stripe_h", "stripe_v", "checker")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _fence_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    pairs = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != bg]
    assert pairs, "expected fence and seed colors"
    pairs.sort(key=lambda item: item[1], reverse=True)
    if len(pairs) >= 2:
        assert pairs[0][1] > pairs[1][1], "fence color must be the unique largest non-background color"
    return pairs[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _fill_solid(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = int(color)


def _fill_stripes(
    out: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    bg: int,
    horizontal: bool,
    top: int,
    left: int,
) -> None:
    for r, c in cells:
        offset = (r - top) if horizontal else (c - left)
        out[r, c] = int(color) if offset % 2 == 0 else int(bg)


def _fill_checkerboard(
    out: np.ndarray,
    cells: list[tuple[int, int]],
    color_a: int,
    color_b: int,
    top: int,
    left: int,
) -> None:
    for r, c in cells:
        parity = (r - top + c - left) % 2
        out[r, c] = int(color_a) if parity == 0 else int(color_b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced-in region according to the seeds it contains.

    - A single seed triggers a solid fill with that color.
    - Same-color aligned seeds trigger stripes parallel to the line between them.
    - Different-color seeds trigger a checkerboard using reading order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    fence = _fence_color(g, bg)

    out = g.copy()
    passable = g != fence
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        seeds = sorted(
            (r, c, int(g[r, c]))
            for r, c in comp
            if int(g[r, c]) != bg and int(g[r, c]) != fence
        )
        if not seeds:
            continue

        top = min(r for r, _c in comp)
        left = min(c for _r, c in comp)
        colors = [color for _r, _c, color in seeds]
        unique_colors = list(dict.fromkeys(colors))

        if len(seeds) == 1:
            _fill_solid(out, comp, seeds[0][2])
            continue

        same_row = all(r == seeds[0][0] for r, _c, _color in seeds)
        same_col = all(c == seeds[0][1] for _r, c, _color in seeds)

        if len(unique_colors) == 1:
            if same_row:
                _fill_stripes(out, comp, unique_colors[0], bg, True, top, left)
            elif same_col:
                _fill_stripes(out, comp, unique_colors[0], bg, False, top, left)
            else:
                _fill_solid(out, comp, unique_colors[0])
            continue

        checker_colors = []
        for _r, _c, color in seeds:
            if color not in checker_colors:
                checker_colors.append(color)
            if len(checker_colors) == 2:
                break
        assert len(checker_colors) == 2
        _fill_checkerboard(out, comp, checker_colors[0], checker_colors[1], top, left)

    return out


def _sample_partition(count: int) -> list[int]:
    lo, hi = CELL_SIZE_RANGE
    return [random.randint(lo, hi) for _ in range(count)]


def _line_positions(start: int, spans: list[int]) -> list[int]:
    positions = [start]
    cur = start
    for span in spans:
        cur += span + 1
        positions.append(cur)
    return positions


def _choose_color(exclude: set[int]) -> int:
    choices = [color for color in COLOR_IDS if color not in exclude]
    assert choices
    return int(random.choice(choices))


def _random_cell_coord(top: int, left: int, height: int, width: int) -> tuple[int, int]:
    return top + random.randrange(height), left + random.randrange(width)


def _place_cell_seeds(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    bg: int,
    fence: int,
) -> None:
    branch = random.choice(BRANCHES)
    if branch == "solid":
        color = _choose_color({bg, fence})
        r, c = _random_cell_coord(top, left, height, width)
        grid[r, c] = color
        return

    if branch == "stripe_h":
        color = _choose_color({bg, fence})
        row = top + random.randrange(height)
        cols = random.sample(range(left, left + width), 2)
        for c in cols:
            grid[row, c] = color
        return

    if branch == "stripe_v":
        color = _choose_color({bg, fence})
        col = left + random.randrange(width)
        rows = random.sample(range(top, top + height), 2)
        for r in rows:
            grid[r, col] = color
        return

    color_a = _choose_color({bg, fence})
    color_b = _choose_color({bg, fence, color_a})
    cells = [(r, c) for r in range(top, top + height) for c in range(left, left + width)]
    (r0, c0), (r1, c1) = random.sample(cells, 2)
    grid[r0, c0] = color_a
    grid[r1, c1] = color_b


def _draw_partition_fence(
    grid: np.ndarray,
    top: int,
    left: int,
    row_heights: list[int],
    col_widths: list[int],
    fence: int,
) -> list[tuple[int, int, int, int]]:
    line_rows = _line_positions(top, row_heights)
    line_cols = _line_positions(left, col_widths)
    bottom = line_rows[-1]
    right = line_cols[-1]

    for r in line_rows:
        grid[r, left : right + 1] = fence
    for c in line_cols:
        grid[top : bottom + 1, c] = fence

    cells: list[tuple[int, int, int, int]] = []
    for i, height in enumerate(row_heights):
        for j, width in enumerate(col_widths):
            r0 = line_rows[i] + 1
            c0 = line_cols[j] + 1
            cells.append((r0, c0, height, width))
    return cells


def _generate_candidate() -> np.ndarray | None:
    row_count = random.choice(ROW_COL_OPTIONS)
    col_count = random.choice(ROW_COL_OPTIONS)
    row_heights = _sample_partition(row_count)
    col_widths = _sample_partition(col_count)

    frame_h = sum(row_heights) + row_count + 1
    frame_w = sum(col_widths) + col_count + 1
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    grid_h = top + frame_h + bottom_margin
    grid_w = left + frame_w + right_margin
    if not (10 <= grid_h <= 30 and 10 <= grid_w <= 30):
        return None

    bg = random.choice(COLOR_IDS)
    fence = _choose_color({bg})
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    cells = _draw_partition_fence(grid, top, left, row_heights, col_widths, fence)
    for r0, c0, height, width in cells:
        _place_cell_seeds(grid, r0, c0, height, width, bg, fence)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-region puzzle")
