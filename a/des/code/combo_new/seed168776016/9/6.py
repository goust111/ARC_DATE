import itertools
import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
PLUS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 400
LEGEND_WIDTH = 23


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, int(mask.size) - 1))
    return runs


def _parse_legend_row(row: np.ndarray) -> dict[tuple[int, int], int]:
    top = np.asarray(row, dtype=int)
    width = int(top.size)
    lookup: dict[tuple[int, int], int] = {}
    c = 0
    while c < width:
        if int(top[c]) == BG:
            c += 1
            continue
        assert c + 2 < width, "legend groups must have three cells"
        a = int(top[c])
        result = int(top[c + 1])
        b = int(top[c + 2])
        assert a != BG and result != BG and b != BG, "legend groups must be non-black triplets"
        key = _pair_key(a, b)
        if key in lookup:
            assert lookup[key] == result, "duplicate legend entry must agree"
        lookup[key] = result
        c += 3
        while c < width and int(top[c]) == BG:
            c += 1
    assert lookup, "top row must contain at least one legend group"
    return lookup


def _infer_line_color(values: np.ndarray, orth_support: np.ndarray) -> int | None:
    values = np.asarray(values, dtype=int)
    orth_support = np.asarray(orth_support, dtype=bool)
    exclusive = values[~orth_support]
    exclusive = exclusive[exclusive != BG]
    if exclusive.size:
        unique = np.unique(exclusive)
        if unique.size == 1:
            return int(unique[0])

    counts = Counter(map(int, values.tolist()))
    for color, count in counts.most_common():
        if color == BG or count < 2:
            continue
        mismatch = values != color
        if bool(mismatch.any()) and not bool(np.all(orth_support[mismatch])):
            continue
        if not bool((values == color).any()):
            continue
        return int(color)
    return None


def _find_horizontal_segments(scene: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = scene.shape
    segments: list[tuple[int, int, int, int]] = []
    for r in range(height):
        row_mask = scene[r] != BG
        for c0, c1 in _runs(row_mask):
            if c1 - c0 + 1 < 2:
                continue
            values = scene[r, c0 : c1 + 1]
            support = np.zeros(c1 - c0 + 1, dtype=bool)
            if r > 0:
                support |= scene[r - 1, c0 : c1 + 1] != BG
            if r + 1 < height:
                support |= scene[r + 1, c0 : c1 + 1] != BG
            color = _infer_line_color(values, support)
            if color is None:
                continue
            segments.append((r, c0, c1, color))
    return segments


def _find_vertical_segments(scene: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = scene.shape
    segments: list[tuple[int, int, int, int]] = []
    for c in range(width):
        col_mask = scene[:, c] != BG
        for r0, r1 in _runs(col_mask):
            if r1 - r0 + 1 < 2:
                continue
            values = scene[r0 : r1 + 1, c]
            support = np.zeros(r1 - r0 + 1, dtype=bool)
            if c > 0:
                support |= scene[r0 : r1 + 1, c - 1] != BG
            if c + 1 < width:
                support |= scene[r0 : r1 + 1, c + 1] != BG
            color = _infer_line_color(values, support)
            if color is None:
                continue
            segments.append((c, r0, r1, color))
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace intersecting colored line segments with mixed-color cross markers.

    Input structure:
    - Background is black.
    - The top row is a legend made of three-cell groups [color A, result color, color B] separated by black gaps.
    - Below the legend are one-pixel-wide horizontal and vertical monochrome line segments, and an overlap pixel may show either segment's color.

    Rule:
    - Parse the top row into a symmetric lookup table from an unordered pair of segment colors to the mixed result color.
    - Reconstruct each horizontal row segment and vertical column segment from the foreground runs below the legend, allowing crossing pixels to be overwritten by the orthogonal line color.
    - For every row/column segment overlap, draw a five-pixel "+" centered at that intersection in the mixed result color, clipped to the grid, and erase all original line pixels and the entire legend row.

    Key insight:
    - The visible crossing pixel is not reliable by itself; the true output color comes from combining the underlying horizontal and vertical segment colors via the legend, then converting every continuous line into isolated cross markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    height, width = g.shape
    assert height >= 2 and width >= 3, "grid must leave room for a legend row and scene"

    mix_lookup = _parse_legend_row(g[0])
    scene = g[1:, :]

    horizontals = _find_horizontal_segments(scene)
    verticals = _find_vertical_segments(scene)

    out = np.full_like(g, BG)
    for sr, c0, c1, h_color in horizontals:
        for sc, r0, r1, v_color in verticals:
            if not (c0 <= sc <= c1 and r0 <= sr <= r1):
                continue
            if int(scene[sr, sc]) == BG:
                continue
            result = mix_lookup[_pair_key(h_color, v_color)]
            r = sr + 1
            c = sc
            for dr, dc in PLUS:
                rr = r + dr
                cc = c + dc
                if 1 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = result
    return out


def _build_legend(
    rng: random.Random,
    width: int,
    groups: list[tuple[int, int, int]],
) -> np.ndarray:
    legend = np.zeros(width, dtype=int)
    order = groups[:]
    rng.shuffle(order)
    start = rng.randint(0, width - LEGEND_WIDTH)
    c = start
    for idx, (a, result, b) in enumerate(order):
        if rng.random() < 0.5:
            legend[c : c + 3] = (a, result, b)
        else:
            legend[c : c + 3] = (b, result, a)
        c += 3
        if idx + 1 < len(order):
            c += 1
    return legend


def _choose_axis_positions(rng: random.Random, start: int, stop: int) -> list[int]:
    offset = rng.randrange(3)
    return list(range(start + offset, stop, 3))


def _make_mix_map(
    rng: random.Random, allow_same_color_crossing: bool
) -> tuple[list[int], dict[tuple[int, int], int], list[tuple[int, int, int]]]:
    if allow_same_color_crossing:
        line_colors = rng.sample(list(PALETTE), 3)
        pairs = list(itertools.combinations_with_replacement(sorted(line_colors), 2))
        result_colors = rng.sample([c for c in PALETTE if c not in line_colors], len(pairs))
    else:
        line_colors = rng.sample(list(PALETTE), 4)
        pairs = list(itertools.combinations(sorted(line_colors), 2))
        result_pool = [c for c in PALETTE if c not in line_colors]
        shuffled = result_pool[:]
        rng.shuffle(shuffled)
        result_colors = [shuffled[idx % len(shuffled)] for idx in range(len(pairs))]

    mix_lookup = {_pair_key(a, b): result for (a, b), result in zip(pairs, result_colors, strict=True)}
    legend_groups = [(a, mix_lookup[_pair_key(a, b)], b) for a, b in pairs]
    return line_colors, mix_lookup, legend_groups


def _assign_line_colors(
    rng: random.Random,
    line_colors: list[int],
    num_h: int,
    num_v: int,
    allow_same_color_crossing: bool,
) -> tuple[list[int], list[int]]:
    if allow_same_color_crossing:
        shared = rng.choice(line_colors)
        different = rng.choice([c for c in line_colors if c != shared])
        h_colors = [shared]
        v_colors = [shared]
        while len(h_colors) < num_h:
            h_colors.append(rng.choice(line_colors))
        while len(v_colors) < num_v:
            v_colors.append(rng.choice(line_colors))
        if all(color == shared for color in h_colors) and num_h >= 2:
            h_colors[-1] = different
        if all(color == shared for color in v_colors) and num_v >= 2:
            v_colors[-1] = different
        if not any(h != v for h in h_colors for v in v_colors):
            h_colors[-1] = different
    else:
        shuffled = line_colors[:]
        rng.shuffle(shuffled)
        h_pool = shuffled[:2]
        v_pool = shuffled[2:]
        h_colors = [h_pool[0], h_pool[1]]
        v_colors = [v_pool[0], v_pool[1]]
        while len(h_colors) < num_h:
            h_colors.append(rng.choice(h_pool))
        while len(v_colors) < num_v:
            v_colors.append(rng.choice(v_pool))
        rng.shuffle(h_colors)
        rng.shuffle(v_colors)
    return h_colors, v_colors


def _draw_input(
    shape: tuple[int, int],
    legend: np.ndarray,
    horizontals: list[tuple[int, int, int, int]],
    verticals: list[tuple[int, int, int, int]],
    visible_at_cross: dict[tuple[int, int], int],
) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width), dtype=int)
    grid[0] = legend

    for row, c0, c1, color in horizontals:
        grid[row, c0 : c1 + 1] = color

    intersections = set(visible_at_cross)
    for col, r0, r1, color in verticals:
        for row in range(r0, r1 + 1):
            if (row, col) not in intersections:
                grid[row, col] = color

    for (row, col), color in visible_at_cross.items():
        grid[row, col] = color
    return grid


def _expected_output(
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    mix_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    height, width = shape
    out = np.zeros((height, width), dtype=int)
    for row, h_color in zip(rows, h_colors, strict=True):
        for col, v_color in zip(cols, v_colors, strict=True):
            result = mix_lookup[_pair_key(h_color, v_color)]
            for dr, dc in PLUS:
                rr = row + dr
                cc = col + dc
                if 1 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = result
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    allow_same_color_crossing = rng.random() < 0.5

    height = rng.randint(8, 20)
    width = rng.randint(LEGEND_WIDTH, 30)

    row_candidates = _choose_axis_positions(rng, 1, height)
    col_candidates = _choose_axis_positions(rng, 0, width)
    if len(row_candidates) < 2 or len(col_candidates) < 2:
        return None

    num_h = rng.randint(2, min(4, len(row_candidates)))
    num_v = rng.randint(2, min(4, len(col_candidates)))
    rows = sorted(rng.sample(row_candidates, num_h))
    cols = sorted(rng.sample(col_candidates, num_v))

    line_colors, mix_lookup, legend_groups = _make_mix_map(rng, allow_same_color_crossing)
    h_colors, v_colors = _assign_line_colors(rng, line_colors, num_h, num_v, allow_same_color_crossing)

    min_col, max_col = cols[0], cols[-1]
    min_row, max_row = rows[0], rows[-1]
    horizontals: list[tuple[int, int, int, int]] = []
    verticals: list[tuple[int, int, int, int]] = []

    for row, color in zip(rows, h_colors, strict=True):
        left_room = min(2, min_col)
        right_room = min(2, width - 1 - max_col)
        left_ext = rng.randint(0, left_room)
        right_ext = rng.randint(0, right_room)
        horizontals.append((row, min_col - left_ext, max_col + right_ext, color))

    for col, color in zip(cols, v_colors, strict=True):
        top_room = min(2, min_row - 1)
        bottom_room = min(2, height - 1 - max_row)
        top_ext = rng.randint(0, top_room)
        bottom_ext = rng.randint(0, bottom_room)
        verticals.append((col, min_row - top_ext, max_row + bottom_ext, color))

    distinct_crosses = [(row, col, h_color, v_color) for row, h_color in zip(rows, h_colors, strict=True) for col, v_color in zip(cols, v_colors, strict=True) if h_color != v_color]
    if not distinct_crosses:
        return None

    visible_at_cross: dict[tuple[int, int], int] = {}
    first_row, first_col, first_h, first_v = distinct_crosses[0]
    visible_at_cross[(first_row, first_col)] = first_h
    if len(distinct_crosses) > 1:
        second_row, second_col, second_h, second_v = distinct_crosses[1]
        visible_at_cross[(second_row, second_col)] = second_v

    for row, h_color in zip(rows, h_colors, strict=True):
        for col, v_color in zip(cols, v_colors, strict=True):
            key = (row, col)
            if key in visible_at_cross:
                continue
            visible_at_cross[key] = rng.choice((h_color, v_color))

    legend = _build_legend(rng, width, legend_groups)
    grid = _draw_input((height, width), legend, horizontals, verticals, visible_at_cross)
    expected = _expected_output((height, width), rows, cols, h_colors, v_colors, mix_lookup)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
