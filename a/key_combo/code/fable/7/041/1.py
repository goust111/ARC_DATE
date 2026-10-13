import random

import numpy as np


BG = 0
BAR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATE_ATTEMPTS = 400


def _find_runs(line: np.ndarray, bar_color: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        color = int(line[i])
        if color == bar_color:
            i += 1
            continue
        j = i
        while j + 1 < n and int(line[j + 1]) == color:
            j += 1
        runs.append((i, j, color))
        i = j + 1
    return runs


def _rectangles_from_grid(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bar_color = int(g[h - 1, w - 1])
    bottom_runs = _find_runs(g[h - 1, : w - 1], bar_color)
    right_runs = _find_runs(g[: h - 1, w - 1], bar_color)

    assert bottom_runs and right_runs
    assert abs(len(bottom_runs) - len(right_runs)) <= 1

    bottom_order = list(reversed(bottom_runs))
    right_order = list(reversed(right_runs))
    pair_count = min(len(bottom_order), len(right_order))

    rectangles: list[tuple[int, int, int, int, int, int]] = []
    for idx in range(pair_count):
        c0, c1, fill_color = bottom_order[idx]
        r0, r1, border_color = right_order[idx]
        rectangles.append((r0, c0, r1, c1, fill_color, border_color))
    return rectangles


def _paint_rectangle(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    fill_color: int,
    border_color: int,
) -> None:
    out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    out[r0, c0 : c1 + 1] = border_color
    out[r1, c0 : c1 + 1] = border_color
    out[r0 : r1 + 1, c0] = border_color
    out[r0 : r1 + 1, c1] = border_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden rectangles from the bottom and right shadow runs, keeping smaller rectangles on top.

    Input structure:
    - The bottom row and right column form a solid gray edge bar, with a black interior elsewhere.
    - Each bar contains several colored contiguous runs separated by gray cells.
    - Bottom-bar colors and right-bar colors are disjoint; one bar may contain one extra farthest run.

    Rule:
    - Sort bottom runs from right to left and right runs from bottom to top, then pair them by order.
    - Each pair defines one filled rectangle over the row span from the right run and the column span from the bottom run.
    - Paint each rectangle with the bottom-run color inside and the right-run color on its outer border; if one bar has an extra farthest run, ignore it.
    - Preserve the two edge bars unchanged, and when rectangles overlap, the smaller-area rectangle is visible on top.

    Key insight:
    - The edge runs are two one-dimensional projections of unseen rectangles, so the true 2D objects are recovered by pairing shadows by distance from the shared corner rather than by color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rectangles = _rectangles_from_grid(g)

    rectangles.sort(key=lambda rect: (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1), reverse=True)
    for r0, c0, r1, c1, fill_color, border_color in rectangles:
        _paint_rectangle(out, r0, c0, r1, c1, fill_color, border_color)
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_runs(axis_len: int, count: int, min_len: int = 1, max_len: int = 6) -> list[tuple[int, int]] | None:
    if count <= 0:
        return []
    compressed_len = axis_len - (count - 1)
    if compressed_len < count * min_len:
        return None

    lengths: list[int] = []
    remaining = compressed_len
    for idx in range(count):
        min_after = (count - idx - 1) * min_len
        hi = min(max_len, remaining - min_after)
        if hi < min_len:
            return None
        length = random.randint(min_len, hi)
        lengths.append(length)
        remaining -= length

    gap_extras = _random_composition(remaining, count + 1)
    runs: list[tuple[int, int]] = []
    pos = gap_extras[0]
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        runs.append((start, end))
        pos = end + 1
        if idx < count - 1:
            pos += 1 + gap_extras[idx + 1]
    return runs


def _make_grid(bottom_runs: list[tuple[int, int]], right_runs: list[tuple[int, int]]) -> np.ndarray:
    interior_h = max(end for _start, end in right_runs) + 1
    interior_w = max(end for _start, end in bottom_runs) + 1
    grid = np.full((interior_h + 1, interior_w + 1), BG, dtype=int)
    grid[-1, :] = BAR
    grid[:, -1] = BAR
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        extra_side = random.choice(["none", "bottom", "right"])
        pair_count = random.randint(2, 4 if extra_side == "none" else 3)
        bottom_count = pair_count + (1 if extra_side == "bottom" else 0)
        right_count = pair_count + (1 if extra_side == "right" else 0)

        interior_h = random.randint(10, 23)
        interior_w = random.randint(10, 23)
        bottom_spans = _sample_runs(interior_w, bottom_count, min_len=1, max_len=6)
        right_spans = _sample_runs(interior_h, right_count, min_len=1, max_len=6)
        if bottom_spans is None or right_spans is None:
            continue

        colors = random.sample(PALETTE, bottom_count + right_count)
        bottom_colors = colors[:bottom_count]
        right_colors = colors[bottom_count:]

        grid = _make_grid(bottom_spans, right_spans)
        for (c0, c1), color in zip(bottom_spans, bottom_colors):
            grid[-1, c0 : c1 + 1] = color
        for (r0, r1), color in zip(right_spans, right_colors):
            grid[r0 : r1 + 1, -1] = color

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
