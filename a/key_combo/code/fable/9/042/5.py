from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_runs(line: np.ndarray, bg: int, *, offset: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    idx = 0
    n = int(line.size)
    while idx < n:
        color = int(line[idx])
        if color == bg:
            idx += 1
            continue
        start = idx
        while idx + 1 < n and int(line[idx + 1]) == color:
            idx += 1
        end = idx
        runs.append((color, offset + start, offset + end))
        idx += 1
    return runs


def _parse_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid must leave an interior canvas"

    bg = _get_bg(g)
    assert np.all(g[1:, 1:] == bg), "input interior must be empty background"

    top_runs = _extract_runs(g[0, 1:], bg, offset=1)
    left_runs = _extract_runs(g[1:, 0], bg, offset=1)
    assert top_runs, "expected at least one top guide segment"
    assert len(top_runs) == len(left_runs), "top/left guide counts must match"

    top_by_color: dict[int, tuple[int, int]] = {}
    left_by_color: dict[int, tuple[int, int]] = {}
    for color, c0, c1 in top_runs:
        assert color not in top_by_color, "each color must appear once on the top edge"
        top_by_color[int(color)] = (int(c0), int(c1))
    for color, r0, r1 in left_runs:
        assert color not in left_by_color, "each color must appear once on the left edge"
        left_by_color[int(color)] = (int(r0), int(r1))
    assert set(top_by_color) == set(left_by_color), "top and left colors must pair one-to-one"

    rectangles: list[tuple[int, int, int, int, int, int]] = []
    areas: list[int] = []
    for color in sorted(top_by_color):
        c0, c1 = top_by_color[color]
        r0, r1 = left_by_color[color]
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        rectangles.append((area, color, r0, r1, c0, c1))
        areas.append(area)

    assert len(set(areas)) == len(areas), "rectangle areas must be distinct for an unambiguous layer order"
    return bg, rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unpainted rectangles from matching top/left projections and erase the guides.

    Input structure:
    - A background grid has several colored horizontal guide segments on row 0 and
      several colored vertical guide segments on column 0.
    - Every non-background color appears exactly once on the top edge and exactly once
      on the left edge, and the interior canvas is otherwise empty background.

    Rule:
    - Match top and left guide segments by color.
    - For each color, use the top segment as the rectangle's column span and the left
      segment as the rectangle's row span, then draw the filled rectangle in that color.
    - If valid rectangles overlap, larger rectangles are painted first so smaller-area
      rectangles remain visible on top; finally row 0 and column 0 are cleared to background.

    Key insight:
    - The rectangles themselves never appear in the input; each one is encoded only by
      a same-colored horizontal and vertical projection segment on two orthogonal borders.
    """
    bg, rectangles = _parse_rectangles(grid)
    out = np.full(np.asarray(grid).shape, bg, dtype=int)
    for _area, color, r0, r1, c0, c1 in sorted(rectangles, reverse=True):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _partition_sum(total: int, slots: int) -> list[int]:
    if slots == 1:
        return [total]
    cuts = sorted(random.sample(range(total + slots - 1), slots - 1))
    values = []
    prev = -1
    for cut in cuts + [total + slots - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_intervals(length: int, count: int, *, min_size: int, max_size: int) -> list[tuple[int, int]]:
    segment_budget = length - (count - 1)
    if segment_budget < count * min_size:
        raise RuntimeError("interval request is infeasible")

    sizes = [min_size] * count
    remaining = segment_budget - count * min_size
    caps = [max(0, max_size - min_size) for _ in range(count)]
    while remaining > 0 and any(cap > 0 for cap in caps):
        choices = [idx for idx, cap in enumerate(caps) if cap > 0]
        idx = random.choice(choices)
        sizes[idx] += 1
        caps[idx] -= 1
        remaining -= 1

    extras = _partition_sum(length - (sum(sizes) + (count - 1)), count + 1)
    starts = []
    pos = 1 + extras[0]
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx != count - 1:
            pos += 1 + extras[idx + 1]
    return [(start, start + size - 1) for start, size in zip(starts, sizes)]


def _component_count(grid: np.ndarray, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    count = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            count += 1
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _vertical_stripes(inp: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(inp.shape, bg, dtype=int)
    for color, c0, c1 in _extract_runs(inp[0, 1:], bg, offset=1):
        out[1:, c0 : c1 + 1] = int(color)
    return out


def _horizontal_stripes(inp: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(inp.shape, bg, dtype=int)
    for color, r0, r1 in _extract_runs(inp[1:, 0], bg, offset=1):
        out[r0 : r1 + 1, 1:] = int(color)
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(10, 24)
    width = random.randint(10, 24)
    max_count = min(5, (height - 1) // 3, (width - 1) // 3)
    if max_count < 2:
        return None
    rect_count = random.randint(2, max_count)

    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    colors = palette[:rect_count]

    row_intervals = _sample_intervals(height - 1, rect_count, min_size=2, max_size=min(7, height - 1))
    col_intervals = _sample_intervals(width - 1, rect_count, min_size=2, max_size=min(7, width - 1))
    random.shuffle(row_intervals)
    random.shuffle(col_intervals)

    pairings = list(zip(colors, row_intervals, col_intervals))
    areas = [int((r1 - r0 + 1) * (c1 - c0 + 1)) for _, (r0, r1), (c0, c1) in pairings]
    if len(set(areas)) != len(areas):
        return None
    if len({r1 - r0 + 1 for _, (r0, r1), _ in pairings}) == 1 and len({c1 - c0 + 1 for _, _, (c0, c1) in pairings}) == 1:
        return None

    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)
    for color, (r0, r1), (c0, c1) in sorted(pairings, key=lambda item: (item[1][1] - item[1][0] + 1) * (item[2][1] - item[2][0] + 1), reverse=True):
        inp[0, c0 : c1 + 1] = int(color)
        inp[r0 : r1 + 1, 0] = int(color)
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    if not np.all(inp[1:, 1:] == bg):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, _vertical_stripes(inp, bg)):
        return None
    if np.array_equal(solved, _horizontal_stripes(inp, bg)):
        return None
    if _component_count(solved, bg) != rect_count:
        return None

    colors_out = {int(v) for v in np.unique(solved) if int(v) != bg}
    if colors_out != set(colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
