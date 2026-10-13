from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
LINE = 5
MARKER_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 400
GEN_MIN_SPAN = 4
GEN_SIZE_RANGE = (14, 19)


def _boundary_marker_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    vertical_cols: set[int] = set()
    horizontal_rows: set[int] = set()
    for r, c in np.argwhere(grid == LINE):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1):
            vertical_cols.add(cc)
        if cc in (0, w - 1):
            horizontal_rows.add(rr)
    return sorted(vertical_cols), sorted(horizontal_rows)


def _region_spans(size: int, line_positions: list[int], axis: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for pos in list(line_positions) + [size]:
        start = prev + 1
        stop = pos - 1
        if start > stop:
            raise AssertionError(f"empty {axis}-region between lines")
        spans.append((start, stop))
        prev = pos
    return spans


def _dominant_color(region: np.ndarray) -> int:
    coords = np.argwhere((region != BG) & (region != LINE))
    if len(coords) == 0:
        return BG

    counts: Counter[int] = Counter(int(region[r, c]) for r, c in coords)
    best_count = max(counts.values())
    tied = {color for color, count in counts.items() if count == best_count}
    if len(tied) == 1:
        return next(iter(tied))

    for r, c in coords:
        color = int(region[r, c])
        if color in tied:
            return color
    raise AssertionError("unreachable")


def _best_output_shape(n_cells: int) -> tuple[int, int]:
    best: tuple[int, int, int, int] | None = None
    for rows in range(1, n_cells + 1):
        cols = (n_cells + rows - 1) // rows
        area = rows * cols
        candidate = (area, abs(rows - cols), cols, rows)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _area, _diff, cols, rows = best
    return rows, cols


def _snake_fill(values: list[int]) -> np.ndarray:
    rows, cols = _best_output_shape(len(values))
    out = np.full((rows, cols), BG, dtype=int)
    for index, color in enumerate(values):
        r = index // cols
        offset = index % cols
        c = offset if r % 2 == 0 else cols - 1 - offset
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use grey boundary markers to partition the canvas, then summarize the regions as snake-ordered color runs.

    Input structure:
    - The background is black, with grey marker pixels placed only on the outer border.
    - A top/bottom grey marker defines a full vertical partition line through its column, and a left/right grey marker defines a full horizontal partition line through its row.
    - Colored singleton markers in red, blue, green, or yellow are scattered inside the rectangular regions formed by those lines and never lie on the partition lines.

    Rule:
    - Recover the full grey spanning lines implied by the border markers and use them as region boundaries.
    - Enumerate the rectangular regions in reading order by their top-left corners.
    - For each region, count its non-black non-grey markers and find the region's most frequent marker color.
    - Emit one snake-ordered output sequence segment per region: repeat the dominant color once for every marker in that region, or emit a single black cell if the region is empty.

    Key insight:
    - The grey lines are not themselves the final target; they first carve the plane into regions, and each region then contributes a variable-length color run whose position in the snake summary depends on the region's spatial order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    vertical_cols, horizontal_rows = _boundary_marker_lines(g)
    assert vertical_cols and horizontal_rows

    row_spans = _region_spans(g.shape[0], horizontal_rows, "row")
    col_spans = _region_spans(g.shape[1], vertical_cols, "col")

    sequence: list[int] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            count = int(np.count_nonzero((region != BG) & (region != LINE)))
            if count == 0:
                sequence.append(BG)
                continue
            sequence.extend([_dominant_color(region)] * count)

    return _snake_fill(sequence)


def _sample_region_widths(total: int, n_lines: int, min_span: int) -> list[int] | None:
    n_regions = n_lines + 1
    remaining = int(total) - int(n_lines)
    if remaining < n_regions * min_span:
        return None

    widths = [min_span] * n_regions
    extra = remaining - n_regions * min_span
    for _ in range(extra):
        widths[random.randrange(n_regions)] += 1
    random.shuffle(widths)
    return widths


def _lines_from_widths(widths: list[int]) -> list[int]:
    lines: list[int] = []
    cursor = widths[0]
    for width in widths[1:]:
        lines.append(cursor)
        cursor += 1 + width
    return lines


def _sample_counts(n_regions: int) -> list[int]:
    include_zero = n_regions == 9 or random.random() < 0.55
    if include_zero:
        pool = [0] + random.sample(list(range(1, 10)), n_regions - 1)
    else:
        pool = random.sample(list(range(1, 10)), n_regions)
    random.shuffle(pool)
    return pool


def _region_color_multiset(count: int) -> list[int]:
    if count == 0:
        return []

    dominant = random.choice(MARKER_COLORS)
    if count == 1:
        return [dominant]

    dominant_count = random.randint(count // 2 + 1, count)
    colors = [dominant] * dominant_count
    others = [color for color in MARKER_COLORS if color != dominant]
    colors.extend(random.choice(others) for _ in range(count - dominant_count))
    random.shuffle(colors)

    counts = Counter(colors)
    assert counts[dominant] > max((v for k, v in counts.items() if k != dominant), default=0)
    return colors


def _place_region_markers(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    colors: list[int],
) -> None:
    cells = [(r, c) for r in range(row_span[0], row_span[1] + 1) for c in range(col_span[0], col_span[1] + 1)]
    chosen = random.sample(cells, len(colors))
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = int(color)


def _build_input() -> np.ndarray | None:
    vertical_count = random.choice((1, 2))
    horizontal_count = random.choice((1, 2))
    h = random.randrange(*GEN_SIZE_RANGE)
    w = random.randrange(*GEN_SIZE_RANGE)

    row_widths = _sample_region_widths(h, horizontal_count, GEN_MIN_SPAN)
    col_widths = _sample_region_widths(w, vertical_count, GEN_MIN_SPAN)
    if row_widths is None or col_widths is None:
        return None

    horizontal_rows = _lines_from_widths(row_widths)
    vertical_cols = _lines_from_widths(col_widths)
    row_spans = _region_spans(h, horizontal_rows, "row")
    col_spans = _region_spans(w, vertical_cols, "col")
    region_count = len(row_spans) * len(col_spans)

    counts = _sample_counts(region_count)
    region_areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1 in row_spans for c0, c1 in col_spans]
    if any(count > area for count, area in zip(counts, region_areas)):
        return None

    grid = np.zeros((h, w), dtype=int)

    vertical_sides = random.sample(("top", "bottom"), vertical_count)
    horizontal_sides = random.sample(("left", "right"), horizontal_count)
    for side, col in zip(vertical_sides, vertical_cols):
        grid[0 if side == "top" else h - 1, col] = LINE
    for side, row in zip(horizontal_sides, horizontal_rows):
        grid[row, 0 if side == "left" else w - 1] = LINE

    region_index = 0
    for row_span in row_spans:
        for col_span in col_spans:
            count = counts[region_index]
            colors = _region_color_multiset(count)
            _place_region_markers(grid, row_span, col_span, colors)
            region_index += 1

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    observed_counts = [
        int(np.count_nonzero((grid[r0 : r1 + 1, c0 : c1 + 1] != BG) & (grid[r0 : r1 + 1, c0 : c1 + 1] != LINE)))
        for r0, r1 in row_spans
        for c0, c1 in col_spans
    ]
    if len(set(observed_counts)) != len(observed_counts) or sorted(observed_counts) != sorted(counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partition-snake ARC input")
