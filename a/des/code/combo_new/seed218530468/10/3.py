from __future__ import annotations

import random

import numpy as np


COLOR_IDS = tuple(range(10))
GRID_H_RANGE = (10, 24)
GRID_W_MAX = 24
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_MIN_HEIGHT = 2
RECT_WIDTH_RANGE = (2, 7)
TOTAL_RECT_MAX = 4
LEFT_GAP_CHOICES = (0, 1, 2)
PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    inner_mask = grid[:, 1:] != bg
    components = _components4(inner_mask)
    assert components, "expected at least one interior object"

    rectangles: list[tuple[int, int, int, int]] = []
    for coords in components:
        rows = coords[:, 0]
        cols = coords[:, 1] + 1
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert coords.shape[0] == area, "each interior object must be a solid rectangle"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] != bg)
        rectangles.append((r0, r1, c0, c1))

    return rectangles


def _apply_projection(
    out: np.ndarray,
    ruler: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    for r0, r1, c0, c1 in rectangles:
        for r in range(r0, r1 + 1):
            out[r, c0 : c1 + 1] = int(ruler[r])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Project the left-edge ruler colors row-by-row through each interior rectangle.

    Input structure:
    - The background is the most common color.
    - The leftmost column is a full-height vertical ruler made of contiguous color segments.
    - To the right are one or more solid rectangular objects on the background.

    Rule:
    - Keep the ruler unchanged.
    - Find each solid rectangle to the right of the ruler.
    - Repaint every row of that rectangle with the ruler color from the same row, producing horizontal stripes inside the rectangle.

    Key insight:
    - The rectangles act as windows onto the left-edge ruler: row alignment, not the rectangles' original fill color, determines the output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    bg = _bg_color(g)
    ruler = g[:, 0].copy()
    rectangles = _extract_rectangles(g, bg)

    out = g.copy()
    return _apply_projection(out, ruler, rectangles)


def _weak_composition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _composition_with_minimum(total: int, parts: int, minimum: int) -> list[int] | None:
    remainder = total - parts * minimum
    if remainder < 0:
        return None
    extras = _weak_composition(remainder, parts)
    return [minimum + extra for extra in extras]


def _positive_composition(total: int, parts: int) -> list[int]:
    result = _composition_with_minimum(total, parts, 1)
    assert result is not None
    return result


def _sample_segments(
    height: int,
    colors: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    k = len(colors)
    lengths = _composition_with_minimum(height, k, SEGMENT_MIN_HEIGHT)
    assert lengths is not None

    ruler = np.empty(height, dtype=int)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    for length, color in zip(lengths, colors):
        end = start + length - 1
        ruler[start : end + 1] = color
        ranges.append((start, end, color))
        start = end + 1
    assert start == height
    return ruler, ranges


def _interval_within_segment(r0: int, r1: int) -> tuple[int, int]:
    top = random.randint(r0, r1 - 1)
    bottom = random.randint(top + 1, r1)
    return top, bottom


def _interval_across_segments(
    segment_ranges: list[tuple[int, int, int]],
    first_idx: int,
    last_idx: int,
) -> tuple[int, int]:
    if first_idx == last_idx:
        r0, r1, _ = segment_ranges[first_idx]
        return _interval_within_segment(r0, r1)

    top0, top1, _ = segment_ranges[first_idx]
    bot0, bot1, _ = segment_ranges[last_idx]
    top = random.randint(top0, top1)
    bottom = random.randint(bot0, bot1)
    assert top < bottom
    return top, bottom


def _coverage_intervals(segment_ranges: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    seg_count = len(segment_ranges)
    part_count = random.randint(1, min(TOTAL_RECT_MAX, seg_count - 1))
    sizes = _positive_composition(seg_count, part_count)

    intervals: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        first_idx = cursor
        last_idx = cursor + size - 1
        intervals.append(_interval_across_segments(segment_ranges, first_idx, last_idx))
        cursor = last_idx + 1
    assert cursor == seg_count
    return intervals


def _extra_intervals(
    segment_ranges: list[tuple[int, int, int]],
    count: int,
) -> list[tuple[int, int]]:
    seg_count = len(segment_ranges)
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        if random.random() < 0.5:
            idx = random.randrange(seg_count)
            r0, r1, _ = segment_ranges[idx]
            intervals.append(_interval_within_segment(r0, r1))
        else:
            first_idx = random.randrange(seg_count - 1)
            last_idx = random.randint(first_idx + 1, seg_count - 1)
            intervals.append(_interval_across_segments(segment_ranges, first_idx, last_idx))
    return intervals


def _place_rectangles(
    shape: tuple[int, int],
    row_intervals: list[tuple[int, int]],
    left_gap: int,
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    rectangles: list[tuple[int, int, int, int]] = []
    min_col = 1 + left_gap
    max_rect_width = min(RECT_WIDTH_RANGE[1], w - min_col)
    if max_rect_width < RECT_WIDTH_RANGE[0]:
        return None

    for r0, r1 in sorted(row_intervals, key=lambda t: (-(t[1] - t[0]), t[0])):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rect_w = random.randint(RECT_WIDTH_RANGE[0], max_rect_width)
            c0 = random.randint(min_col, w - rect_w)
            c1 = c0 + rect_w - 1

            rr0 = max(0, r0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc0 = max(1, c0 - 1)
            cc1 = min(w - 1, c1 + 1)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            rectangles.append((r0, r1, c0, c1))
            blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return rectangles


def _all_segment_colors_used(
    rectangles: list[tuple[int, int, int, int]],
    ruler: np.ndarray,
    segment_colors: set[int],
) -> bool:
    used = set()
    for r0, r1, _c0, _c1 in rectangles:
        used.update(int(ruler[r]) for r in range(r0, r1 + 1))
    return used == segment_colors


def _has_boundary_spanning_rectangle(
    rectangles: list[tuple[int, int, int, int]],
    ruler: np.ndarray,
) -> bool:
    for r0, r1, _c0, _c1 in rectangles:
        if len(set(map(int, ruler[r0 : r1 + 1]))) >= 2:
            return True
    return False


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > max(int(counts[c]) for c in range(10) if c != bg)


def _gen_attempt() -> np.ndarray | None:
    seg_count = random.randint(*SEGMENT_COUNT_RANGE)
    min_height = max(GRID_H_RANGE[0], seg_count * SEGMENT_MIN_HEIGHT)
    if min_height > GRID_H_RANGE[1]:
        return None
    height = random.randint(min_height, GRID_H_RANGE[1])

    needed_colors = seg_count + 2
    chosen = random.sample(COLOR_IDS, needed_colors)
    bg = chosen[0]
    object_color = chosen[1]
    ruler_colors = chosen[2:]
    ruler, segment_ranges = _sample_segments(height, ruler_colors)

    cover_intervals = _coverage_intervals(segment_ranges)
    extra_count = random.randint(0, TOTAL_RECT_MAX - len(cover_intervals))
    row_intervals = cover_intervals + _extra_intervals(segment_ranges, extra_count)

    left_gap = random.choice(LEFT_GAP_CHOICES)
    min_width = max(10, left_gap + 3 * len(row_intervals) + 4)
    if min_width > GRID_W_MAX:
        return None
    width = random.randint(min_width, GRID_W_MAX)

    rectangles = _place_rectangles((height, width), row_intervals, left_gap)
    if rectangles is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[:, 0] = ruler
    for r0, r1, c0, c1 in rectangles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = object_color

    segment_colors = {color for _r0, _r1, color in segment_ranges}
    if not _all_segment_colors_used(rectangles, ruler, segment_colors):
        return None
    if not _has_boundary_spanning_rectangle(rectangles, ruler):
        return None
    if not _background_is_dominant(grid, bg):
        return None

    expected = _apply_projection(grid.copy(), ruler, rectangles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-projection input")
