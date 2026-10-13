from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MIN_SQUARES = 2
MAX_SQUARES = 5
MIN_SIDE = 2
MAX_SIDE_BY_COUNT = {
    2: 11,
    3: 9,
    4: 7,
    5: 6,
}
MAX_GEN_ATTEMPTS = 400


def _extract_segments(line: np.ndarray, start_index: int) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    i = 0
    while i < len(line):
        color = int(line[i])
        if color == BG:
            i += 1
            continue
        j = i
        while j < len(line) and int(line[j]) != BG:
            assert int(line[j]) == color, "each edge segment must be monochrome"
            j += 1
        segments.append(
            {
                "color": color,
                "start": start_index + i,
                "end": start_index + j - 1,
                "length": j - i,
            }
        )
        i = j
    return segments


def _paint_squares(shape: tuple[int, int], squares: list[dict[str, int]], base: np.ndarray) -> np.ndarray:
    out = np.array(base, copy=True)
    for square in sorted(squares, key=lambda item: item["size"], reverse=True):
        r0 = square["r0"]
        c0 = square["c0"]
        size = square["size"]
        out[r0 : r0 + size, c0 : c0 + size] = square["color"]
    out[base != BG] = base[base != BG]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct filled squares from their left-edge and bottom-edge projections.

    Input structure:
    - Black background with only two occupied borders: colored vertical segments on the left edge
      and colored horizontal segments on the bottom edge.
    - Each left segment gives the row span of one hidden solid square, and each bottom segment
      gives the column span of one hidden square.
    - Matching projections have the same color and the same length, while square sizes are all distinct.

    Rule:
    - Detect all left-edge and bottom-edge segments and pair them by equal length.
    - For each paired row span and column span, draw the corresponding solid square in that color.
    - If reconstructed squares overlap, the smaller square stays visible on top, and the border projections
      remain exactly as they were in the input.

    Key insight:
    - The real objects never appear in the input; only their two orthogonal shadows do.
      Length uniquely identifies which row span belongs with which column span, even when colors repeat.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3
    assert np.all(g[:-1, 1:] == BG), "only the left edge and bottom edge may contain input marks"

    left_segments = _extract_segments(g[:-1, 0], 0)
    bottom_segments = _extract_segments(g[-1, 1:], 1)

    assert left_segments and len(left_segments) == len(bottom_segments)

    left_by_len = {seg["length"]: seg for seg in left_segments}
    bottom_by_len = {seg["length"]: seg for seg in bottom_segments}
    assert len(left_by_len) == len(left_segments), "left projection lengths must be unique"
    assert len(bottom_by_len) == len(bottom_segments), "bottom projection lengths must be unique"
    assert set(left_by_len) == set(bottom_by_len), "every projection length must appear on both edges"

    squares: list[dict[str, int]] = []
    for size in sorted(left_by_len):
        left = left_by_len[size]
        bottom = bottom_by_len[size]
        assert left["color"] == bottom["color"], "paired projections must share a color"
        squares.append(
            {
                "size": size,
                "color": left["color"],
                "r0": left["start"],
                "c0": bottom["start"],
            }
        )

    return _paint_squares(g.shape, squares, g)


def _random_composition(total: int, parts: int) -> list[int]:
    slots = [0] * parts
    for _ in range(total):
        slots[random.randrange(parts)] += 1
    return slots


def _sample_sizes(count: int) -> list[int]:
    max_side = MAX_SIDE_BY_COUNT[count]
    choices = list(range(MIN_SIDE, max_side + 1))
    for _ in range(200):
        sizes = random.sample(choices, count)
        if sum(sizes) + (count - 1) <= 29:
            return sizes
    raise RuntimeError("failed to sample square sizes")


def _sample_axis_intervals(sizes: list[int], available: int) -> dict[int, tuple[int, int]]:
    required = sum(sizes) + (len(sizes) - 1)
    assert required <= available
    free = available - required
    extras = _random_composition(free, len(sizes) + 1)
    order = sizes[:]
    random.shuffle(order)

    intervals: dict[int, tuple[int, int]] = {}
    pos = extras[0]
    for i, size in enumerate(order):
        start = pos
        end = start + size - 1
        intervals[size] = (start, end)
        pos = end + 1
        if i != len(order) - 1:
            pos += 1 + extras[i + 1]
    return intervals


def _assign_colors(count: int) -> list[int]:
    if count >= 3 and random.random() < 0.75:
        distinct = random.randint(2, count - 1)
    else:
        distinct = count
    palette = random.sample(PALETTE, distinct)
    colors = palette[:]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _build_input(squares: list[dict[str, int]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for square in squares:
        size = square["size"]
        color = square["color"]
        r0 = square["r0"]
        c0 = square["c0"]
        grid[r0 : r0 + size, 0] = color
        grid[h - 1, c0 : c0 + size] = color
    return grid


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(MIN_SQUARES, MAX_SQUARES)
    sizes = _sample_sizes(count)
    required = sum(sizes) + (count - 1)

    h = required + 1 + random.randint(0, 29 - required)
    w = required + 1 + random.randint(0, 29 - required)

    row_intervals = _sample_axis_intervals(sizes, h - 1)
    col_intervals = _sample_axis_intervals(sizes, w - 1)
    colors = _assign_colors(count)

    squares: list[dict[str, int]] = []
    for size, color in zip(sizes, colors):
        r0, _r1 = row_intervals[size]
        c0, _c1 = col_intervals[size]
        squares.append({"size": size, "color": color, "r0": r0, "c0": c0})

    inp = _build_input(squares, h, w)
    expected = _paint_squares(inp.shape, squares, inp)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-shadow input")
