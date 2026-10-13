from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

AXIS_COUNT_CHOICES = (2, 3)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 6
GAP_MIN = 2
GAP_MAX = 4
MARGIN_MIN = 1
MARGIN_MAX = 4
AXIS_MAX = 28

RECT_COUNT_CHOICES = (3, 4)
PAIR_COUNT_CHOICES = (3, 4)
MIN_SEGMENT_LEN = 6
MAX_GENERATION_ATTEMPTS = 400


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("marker pairs must share a row or a column")


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0 = int(np.min(coords[:, 0]))
    c0 = int(np.min(coords[:, 1]))
    r1 = int(np.max(coords[:, 0]))
    c1 = int(np.max(coords[:, 1]))
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _classify_colors(
    grid: np.ndarray,
) -> tuple[list[tuple[int, tuple[int, int], tuple[int, int]]], set[int], np.ndarray]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    rectangle_colors: set[int] = set()
    rect_mask = np.zeros_like(grid, dtype=bool)

    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BACKGROUND):
        coords = np.argwhere(grid == color)
        if coords.shape[0] == 2:
            a = tuple(map(int, coords[0]))
            b = tuple(map(int, coords[1]))
            assert a[0] == b[0] or a[1] == b[1], "marker pair must align horizontally or vertically"
            pairs.append((color, a, b))
            continue

        assert coords.shape[0] >= 3, "foreground objects must be rectangles or marker pairs"
        assert _is_solid_rectangle(coords), "non-marker colors must form solid rectangles"
        rectangle_colors.add(color)
        rect_mask[grid == color] = True

    assert pairs, "expected at least one marker pair"
    assert rectangle_colors, "expected at least one rectangle"
    return pairs, rectangle_colors, rect_mask


def _draw_pairs(
    grid: np.ndarray,
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]],
    rect_mask: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for color, a, b in pairs:
        for r, c in _segment_cells(a, b):
            base = int(grid[r, c])
            if rect_mask[r, c]:
                desired = base
            else:
                assert base in (BACKGROUND, color), "line path may only traverse background outside rectangles"
                desired = color
            assert int(out[r, c]) in (base, desired), "different marker lines must not conflict"
            out[r, c] = desired
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned marker pair with a straight line whose color changes inside crossed rectangles.

    Input structure:
    - The background is black.
    - Several disjoint solid rectangles of distinct colors sit on the background.
    - Outside those rectangles, each marker color appears exactly twice as singleton pixels.
    - The two pixels of every marker color share one row or one column.

    Rule:
    - Keep the original rectangles and marker pixels in place.
    - For each marker pair, draw the horizontal or vertical segment joining the two markers.
    - Use the marker color on background cells of that segment.
    - Whenever the segment passes through a colored rectangle, keep that part in the rectangle's own color.

    Key insight:
    - The pairing rule comes from matching equal-colored singleton markers, while the visible line color is determined locally by the region the line is crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BACKGROUND), "expected black background cells"

    pairs, _rectangle_colors, rect_mask = _classify_colors(g)
    return _draw_pairs(g, pairs, rect_mask)


def _sample_axis_bands(count: int) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(200):
        sizes = [random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX) for _ in range(count)]
        gaps = [random.randint(GAP_MIN, GAP_MAX) for _ in range(count - 1)]
        margins = [random.randint(MARGIN_MIN, MARGIN_MAX), random.randint(MARGIN_MIN, MARGIN_MAX)]
        base_total = sum(sizes) + sum(gaps) + margins[0] + margins[1]
        if base_total > AXIS_MAX:
            continue

        slots = [margins[0], *gaps, margins[1]]
        extra = random.randint(0, AXIS_MAX - base_total)
        for _ in range(extra):
            slots[random.randrange(len(slots))] += 1

        bands: list[tuple[int, int]] = []
        pos = slots[0]
        for i, size in enumerate(sizes):
            bands.append((pos, pos + size - 1))
            pos += size
            if i < count - 1:
                pos += slots[i + 1]
        total = pos + slots[-1]
        if 14 <= total <= AXIS_MAX:
            return bands, total
    return None


def _choose_rectangle_positions(row_count: int, col_count: int, rect_count: int) -> list[tuple[int, int]]:
    row_a, row_b = random.sample(range(row_count), 2)
    col_a, col_b = random.sample(range(col_count), 2)
    base = {(row_a, col_a), (row_a, col_b), (row_b, col_a)}
    remaining = [(r, c) for r in range(row_count) for c in range(col_count) if (r, c) not in base]
    extras = random.sample(remaining, rect_count - len(base))
    return list(base) + extras


def _build_scene() -> tuple[np.ndarray, np.ndarray, list[tuple[int, tuple[int, int], tuple[int, int]]]] | None:
    row_count = random.choice(AXIS_COUNT_CHOICES)
    col_count = random.choice(AXIS_COUNT_CHOICES)

    row_result = _sample_axis_bands(row_count)
    col_result = _sample_axis_bands(col_count)
    if row_result is None or col_result is None:
        return None
    row_bands, height = row_result
    col_bands, width = col_result

    rect_count = random.choice(RECT_COUNT_CHOICES)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    if rect_count > row_count * col_count or rect_count + pair_count > len(PALETTE):
        return None

    colors = list(PALETTE)
    random.shuffle(colors)
    rect_colors = colors[:rect_count]
    pair_colors = colors[rect_count : rect_count + pair_count]

    grid = np.zeros((height, width), dtype=int)
    rect_id_grid = np.full((height, width), -1, dtype=int)

    positions = _choose_rectangle_positions(row_count, col_count, rect_count)
    for rect_id, ((row_idx, col_idx), color) in enumerate(zip(positions, rect_colors)):
        r0, r1 = row_bands[row_idx]
        c0, c1 = col_bands[col_idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        rect_id_grid[r0 : r1 + 1, c0 : c1 + 1] = rect_id

    pairs = _plan_pairs(grid, rect_id_grid, pair_colors)
    if pairs is None:
        return None

    rect_mask = rect_id_grid >= 0
    inp = grid.copy()
    for color, a, b in pairs:
        inp[a] = color
        inp[b] = color
    expected = _draw_pairs(inp, pairs, rect_mask)
    return inp, expected, pairs


def _candidate_buckets(pair_count: int) -> list[int]:
    buckets = [0, 1, 2]
    if pair_count == 4:
        buckets.append(random.choice((0, 1, 2)))
    random.shuffle(buckets)
    return buckets


def _candidate_orientations(pair_count: int) -> list[str]:
    if pair_count == 3:
        orientations = ["h", "h", "v"] if random.random() < 0.5 else ["h", "v", "v"]
    else:
        orientations = ["h", "h", "v", "v"]
    random.shuffle(orientations)
    return orientations


def _enumerate_candidates(
    rect_id_grid: np.ndarray,
    reserved: np.ndarray,
    orientation: str,
    bucket: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    height, width = rect_id_grid.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if orientation == "h":
        for r in range(height):
            for c0 in range(width):
                if rect_id_grid[r, c0] >= 0 or reserved[r, c0]:
                    continue
                for c1 in range(c0 + MIN_SEGMENT_LEN - 1, width):
                    if rect_id_grid[r, c1] >= 0 or reserved[r, c1]:
                        continue
                    segment_rects = rect_id_grid[r, c0 : c1 + 1]
                    segment_reserved = reserved[r, c0 : c1 + 1]
                    if np.any(segment_reserved):
                        continue
                    crossed = {int(x) for x in segment_rects if int(x) >= 0}
                    if min(2, len(crossed)) != bucket:
                        continue
                    if rect_id_grid[r, c0 + 1] >= 0 or rect_id_grid[r, c1 - 1] >= 0:
                        continue
                    background_count = int(segment_rects.size - np.count_nonzero(segment_rects >= 0))
                    if background_count < 3:
                        continue
                    candidates.append(((r, c0), (r, c1)))
    else:
        for c in range(width):
            for r0 in range(height):
                if rect_id_grid[r0, c] >= 0 or reserved[r0, c]:
                    continue
                for r1 in range(r0 + MIN_SEGMENT_LEN - 1, height):
                    if rect_id_grid[r1, c] >= 0 or reserved[r1, c]:
                        continue
                    segment_rects = rect_id_grid[r0 : r1 + 1, c]
                    segment_reserved = reserved[r0 : r1 + 1, c]
                    if np.any(segment_reserved):
                        continue
                    crossed = {int(x) for x in segment_rects if int(x) >= 0}
                    if min(2, len(crossed)) != bucket:
                        continue
                    if rect_id_grid[r0 + 1, c] >= 0 or rect_id_grid[r1 - 1, c] >= 0:
                        continue
                    background_count = int(segment_rects.size - np.count_nonzero(segment_rects >= 0))
                    if background_count < 3:
                        continue
                    candidates.append(((r0, c), (r1, c)))

    random.shuffle(candidates)
    return candidates


def _plan_pairs(
    grid: np.ndarray,
    rect_id_grid: np.ndarray,
    pair_colors: list[int],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]] | None:
    reserved = np.zeros_like(grid, dtype=bool)
    bucket_order = _candidate_buckets(len(pair_colors))
    orientation_order = _candidate_orientations(len(pair_colors))

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, bucket, orientation in zip(pair_colors, bucket_order, orientation_order):
        candidates = _enumerate_candidates(rect_id_grid, reserved, orientation, bucket)
        if not candidates:
            return None
        a, b = random.choice(candidates)
        for r, c in _segment_cells(a, b):
            reserved[r, c] = True
        pairs.append((color, a, b))
    return pairs


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected, _pairs = built
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
