from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MARKER_COLORS = (3, 4, 5, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
PAIR_COUNT_RANGE = (3, 5)
MIN_SEGMENT_LEN = 3
MAX_SEGMENT_LEN = 11
MAX_TARGET_CROSSINGS = 3
GEN_ATTEMPTS = 400
SPAN_PICK_ATTEMPTS = 80


HorizontalSegment = tuple[int, int, int, int]  # color, row, c0, c1
VerticalSegment = tuple[int, int, int, int]  # color, col, r0, r1


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _extract_segments(grid: np.ndarray) -> tuple[int, list[HorizontalSegment], list[VerticalSegment]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    horizontal: list[HorizontalSegment] = []
    vertical: list[VerticalSegment] = []

    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each marker color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        assert (r0 == r1) ^ (c0 == c1), "same-color markers must share one row or one column"

        if r0 == r1:
            c0, c1 = sorted((int(c0), int(c1)))
            row = int(r0)
            assert c1 - c0 + 1 >= MIN_SEGMENT_LEN and (c1 - c0) % 2 == 0
            assert np.all(g[row, c0 + 1 : c1] == bg), "horizontal path must be unobstructed"
            horizontal.append((int(color), row, c0, c1))
        else:
            r0, r1 = sorted((int(r0), int(r1)))
            col = int(c0)
            assert r1 - r0 + 1 >= MIN_SEGMENT_LEN and (r1 - r0) % 2 == 0
            assert np.all(g[r0 + 1 : r1, col] == bg), "vertical path must be unobstructed"
            vertical.append((int(color), col, r0, r1))

    assert horizontal and vertical, "the puzzle needs both horizontal and vertical segments"
    assert len(horizontal) != len(vertical), "global direction vote must not tie"
    return bg, horizontal, vertical


def _render_output(
    shape: tuple[int, int],
    bg: int,
    horizontal: list[HorizontalSegment],
    vertical: list[VerticalSegment],
) -> np.ndarray:
    recolor = RED if len(horizontal) > len(vertical) else BLUE
    out = np.full(shape, bg, dtype=int)

    for color, row, c0, c1 in horizontal:
        mid = (c0 + c1) // 2
        out[row, c0:mid] = recolor
        out[row, mid : c1 + 1] = color

    for color, col, r0, r1 in vertical:
        mid = (r0 + r1) // 2
        out[r0:mid, col] = recolor
        out[mid : r1 + 1, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color marker pairs, split each segment by its own direction, and recolor the leading half using the global direction majority.

    Input structure:
    - The background is black and each non-background color appears in exactly two marker pixels.
    - Each same-color pair shares exactly one row or exactly one column, with an unobstructed odd-length gap between the endpoints.
    - Different pairs may cross after completion, and vertical segments conceptually sit above horizontal ones at crossings.

    Rule:
    - Connect every same-color pair with its full axis-aligned segment, drawing all horizontal segments first and all vertical segments afterward.
    - Count horizontal pairs H and vertical pairs V; if H > V the recolor color is red, otherwise the recolor color is blue.
    - For each horizontal segment, recolor only the cells strictly left of its midpoint; for each vertical segment, recolor only the cells strictly above its midpoint.
    - Because vertical segments are drawn last, any crossing cell follows the vertical segment's own split rule and final color.

    Key insight:
    - The same recovered segments must be interpreted twice: locally, each segment's direction picks its split axis, while globally, the majority direction picks the recolor used on every segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, horizontal, vertical = _extract_segments(g)
    return _render_output(g.shape, bg, horizontal, vertical)


def _spaced_positions(size: int, count: int) -> list[int]:
    choices = list(range(1, size - 1))
    buckets = [bucket.tolist() for bucket in np.array_split(np.array(choices, dtype=int), count)]
    positions = [int(random.choice(bucket)) for bucket in buckets if bucket]
    random.shuffle(positions)
    return positions


def _choose_orientation_counts(total_pairs: int) -> tuple[int, int]:
    if total_pairs == 3:
        return random.choice(((2, 1), (1, 2)))
    if total_pairs == 4:
        return random.choice(((3, 1), (1, 3)))
    if random.random() < 0.8:
        return random.choice(((3, 2), (2, 3)))
    return random.choice(((4, 1), (1, 4)))


def _span_options(limit: int, include: list[int], exclude: list[int]) -> list[tuple[int, int]]:
    max_len = min(limit, MAX_SEGMENT_LEN)
    options: list[tuple[int, int]] = []
    for length in range(MIN_SEGMENT_LEN, max_len + 1, 2):
        for start in range(0, limit - length + 1):
            end = start + length - 1
            if include and not all(start < pos < end for pos in include):
                continue
            if any(start <= pos <= end for pos in exclude):
                continue
            options.append((start, end))
    return options


def _find_plans(
    height: int,
    width: int,
    h_rows: list[int],
    v_cols: list[int],
) -> list[tuple[set[tuple[int, int]], list[list[tuple[int, int]]], list[list[tuple[int, int]]]]]:
    h_count = len(h_rows)
    v_count = len(v_cols)
    all_pairs = [(hi, vi) for hi in range(h_count) for vi in range(v_count)]
    max_crossings = min(MAX_TARGET_CROSSINGS, len(all_pairs))

    crossing_counts = [n for n in range(2, max_crossings + 1)]
    random.shuffle(crossing_counts)
    if 1 <= max_crossings:
        crossing_counts.append(1)

    plans = []
    for target_count in crossing_counts:
        for combo in itertools.combinations(all_pairs, target_count):
            target = set(combo)

            horizontal_options: list[list[tuple[int, int]]] = []
            ok = True
            for hi in range(h_count):
                include = [v_cols[vi] for vi in range(v_count) if (hi, vi) in target]
                exclude = [v_cols[vi] for vi in range(v_count) if (hi, vi) not in target]
                options = _span_options(width, include, exclude)
                if not options:
                    ok = False
                    break
                horizontal_options.append(options)
            if not ok:
                continue

            vertical_options: list[list[tuple[int, int]]] = []
            for vi in range(v_count):
                include = [h_rows[hi] for hi in range(h_count) if (hi, vi) in target]
                exclude = [h_rows[hi] for hi in range(h_count) if (hi, vi) not in target]
                options = _span_options(height, include, exclude)
                if not options:
                    ok = False
                    break
                vertical_options.append(options)
            if ok:
                plans.append((target, horizontal_options, vertical_options))
    random.shuffle(plans)
    return plans


def _build_segments(
    colors: list[int],
    h_rows: list[int],
    v_cols: list[int],
    horizontal_options: list[list[tuple[int, int]]],
    vertical_options: list[list[tuple[int, int]]],
) -> tuple[list[HorizontalSegment], list[VerticalSegment]]:
    horizontal: list[HorizontalSegment] = []
    vertical: list[VerticalSegment] = []

    for hi, row in enumerate(h_rows):
        c0, c1 = random.choice(horizontal_options[hi])
        horizontal.append((colors[hi], row, c0, c1))
    for vi, col in enumerate(v_cols):
        r0, r1 = random.choice(vertical_options[vi])
        vertical.append((colors[len(h_rows) + vi], col, r0, r1))
    return horizontal, vertical


def _crossings(
    horizontal: list[HorizontalSegment],
    vertical: list[VerticalSegment],
) -> list[tuple[int, int, int]]:
    cells = []
    for h_index, (_color, row, c0, c1) in enumerate(horizontal):
        for v_index, (_vcolor, col, r0, r1) in enumerate(vertical):
            if c0 <= col <= c1 and r0 <= row <= r1:
                cells.append((h_index, v_index, row))
    return cells


def _render_input(
    shape: tuple[int, int],
    colors: list[int],
    horizontal: list[HorizontalSegment],
    vertical: list[VerticalSegment],
) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    for color, row, c0, c1 in horizontal:
        inp[row, c0] = color
        inp[row, c1] = color
    for color, col, r0, r1 in vertical:
        inp[r0, col] = color
        inp[r1, col] = color
    return inp


def _endpoints_are_unique(horizontal: list[HorizontalSegment], vertical: list[VerticalSegment]) -> bool:
    endpoints = []
    for color, row, c0, c1 in horizontal:
        endpoints.extend([(row, c0, color), (row, c1, color)])
    for color, col, r0, r1 in vertical:
        endpoints.extend([(r0, col, color), (r1, col, color)])
    return len({(r, c) for r, c, _ in endpoints}) == len(endpoints)


def _visible_half_segments(horizontal: list[HorizontalSegment], vertical: list[VerticalSegment]) -> bool:
    crossing_cells = {(row, vertical[vi][1]) for _hi, vi, row in _crossings(horizontal, vertical)}

    has_visible_horizontal = False
    for _color, row, c0, c1 in horizontal:
        mid = (c0 + c1) // 2
        if any((row, c) not in crossing_cells for c in range(c0, mid)):
            has_visible_horizontal = True
            break

    has_visible_vertical = False
    for _color, col, r0, r1 in vertical:
        mid = (r0 + r1) // 2
        if any((r, col) not in crossing_cells for r in range(r0, mid)):
            has_visible_vertical = True
            break

    return has_visible_horizontal and has_visible_vertical


def _crossings_cover_both_vertical_halves(
    horizontal: list[HorizontalSegment],
    vertical: list[VerticalSegment],
) -> bool:
    crossing_info = _crossings(horizontal, vertical)
    if len(crossing_info) < 2:
        return False

    above_mid = False
    at_or_below_mid = False
    for _hi, vi, row in crossing_info:
        _color, _col, r0, r1 = vertical[vi]
        mid = (r0 + r1) // 2
        if row < mid:
            above_mid = True
        else:
            at_or_below_mid = True
    return above_mid and at_or_below_mid


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    total_pairs = random.randint(*PAIR_COUNT_RANGE)
    h_count, v_count = _choose_orientation_counts(total_pairs)

    h_rows = _spaced_positions(height, h_count)
    v_cols = _spaced_positions(width, v_count)
    plans = _find_plans(height, width, h_rows, v_cols)
    if not plans:
        return None

    colors = random.sample(list(MARKER_COLORS), total_pairs)
    for target, horizontal_options, vertical_options in plans:
        _ = target
        for _ in range(SPAN_PICK_ATTEMPTS):
            horizontal, vertical = _build_segments(colors, h_rows, v_cols, horizontal_options, vertical_options)
            if not _endpoints_are_unique(horizontal, vertical):
                continue
            if not _visible_half_segments(horizontal, vertical):
                continue
            if not _crossings_cover_both_vertical_halves(horizontal, vertical):
                continue

            inp = _render_input((height, width), colors, horizontal, vertical)
            expected = _render_output((height, width), BG, horizontal, vertical)
            if not np.array_equal(main(inp), expected):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
