import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GRID = 30
MAX_GENERATION_ATTEMPTS = 400


def _segment_cells(segment: dict) -> list[tuple[int, int]]:
    if segment["orientation"] == "h":
        return [(segment["row"], c) for c in range(segment["start"], segment["start"] + segment["length"])]
    return [(r, segment["col"]) for r in range(segment["start"], segment["start"] + segment["length"])]


def _extract_segments(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    for color in sorted(int(color) for color in np.unique(grid) if int(color) != BG):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue

        rows = sorted(int(r) for r in np.unique(coords[:, 0]))
        cols = sorted(int(c) for c in np.unique(coords[:, 1]))

        if len(rows) == 1:
            row = rows[0]
            start = cols[0]
            assert len(cols) >= 2, "horizontal segments must have length at least 2"
            assert cols == list(range(start, start + len(cols))), "horizontal segment must be contiguous"
            segments.append(
                {
                    "color": color,
                    "orientation": "h",
                    "row": row,
                    "start": start,
                    "length": len(cols),
                }
            )
        elif len(cols) == 1:
            col = cols[0]
            start = rows[0]
            assert len(rows) >= 2, "vertical segments must have length at least 2"
            assert rows == list(range(start, start + len(rows))), "vertical segment must be contiguous"
            segments.append(
                {
                    "color": color,
                    "orientation": "v",
                    "col": col,
                    "start": start,
                    "length": len(rows),
                }
            )
        else:
            raise AssertionError("each foreground color must form a single straight line segment")
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack horizontal segments to the left and vertical segments to the top.

    Input structure:
    - The background is black.
    - Each foreground color forms one isolated straight line segment of length at least 2.
    - Every segment is horizontal or vertical, and the input segments do not touch the grid border.

    Rule:
    - Detect every colored line segment and classify it by orientation.
    - For each row, keep every horizontal segment on that row and sort them by original left endpoint, then repack them flush against column 0 with no gaps.
    - For each column, keep every vertical segment in that column and sort them by original top endpoint, then repack them flush against row 0 with no gaps.
    - Preserve every segment's color, length, and fixed row or column.

    Key insight:
    - The same objects move in different axes depending on orientation: horizontal segments slide left within their rows, while vertical segments slide up within their columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full(g.shape, BG, dtype=int)
    segments = _extract_segments(g)

    by_row: dict[int, list[dict]] = defaultdict(list)
    by_col: dict[int, list[dict]] = defaultdict(list)
    for segment in segments:
        if segment["orientation"] == "h":
            by_row[segment["row"]].append(segment)
        else:
            by_col[segment["col"]].append(segment)

    for row, row_segments in by_row.items():
        cursor = 0
        for segment in sorted(row_segments, key=lambda item: item["start"]):
            out[row, cursor : cursor + segment["length"]] = segment["color"]
            cursor += segment["length"]

    for col, col_segments in by_col.items():
        cursor = 0
        for segment in sorted(col_segments, key=lambda item: item["start"]):
            current = out[cursor : cursor + segment["length"], col]
            assert not np.any(current != BG), "packed horizontal and vertical segments must not overlap"
            out[cursor : cursor + segment["length"], col] = segment["color"]
            cursor += segment["length"]

    return out


def _split_nonnegative(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _sample_spaced_positions(low: int, high: int, count: int, gap: int = 2) -> list[int]:
    assert count >= 1
    span = high - low + 1
    required = count + (count - 1) * (gap - 1)
    assert span >= required
    compressed_high = high - (gap - 1) * (count - 1)
    bases = sorted(random.sample(range(low, compressed_high + 1), count))
    return [base + i * (gap - 1) for i, base in enumerate(bases)]


def _place_group(lengths: list[int], band_size: int) -> list[int]:
    min_used = sum(lengths) + len(lengths) - 1
    assert min_used <= band_size
    slack = band_size - min_used
    extras = _split_nonnegative(slack, len(lengths) + 1)

    starts: list[int] = []
    cursor = 1 + extras[0]
    for idx, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if idx + 1 < len(lengths):
            cursor += 1 + extras[idx + 1]
    return starts


def _sample_group_sizes(total: int, force_multi: bool) -> list[int]:
    if not force_multi:
        return [1] * total

    group_count = random.randint(1, min(3, total - 1))
    sizes = [1] * group_count
    for _ in range(total - group_count):
        sizes[random.randrange(group_count)] += 1
    random.shuffle(sizes)
    return sizes


def _segments_are_separated(segments: list[dict]) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    for idx, segment in enumerate(segments):
        for cell in _segment_cells(segment):
            if cell in occupied:
                return False
            r, c = cell
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    other = occupied.get((r + dr, c + dc))
                    if other is not None and other != idx:
                        return False
            occupied[cell] = idx
    return True


def _build_grid(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for segment in segments:
        for r, c in _segment_cells(segment):
            grid[r, c] = segment["color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("none", "row", "col", "both"))
    row_multi = mode in {"row", "both"}
    col_multi = mode in {"col", "both"}

    h_total = random.randint(2, 3)
    v_total = random.randint(2, 3)

    h_group_sizes = _sample_group_sizes(h_total, row_multi)
    v_group_sizes = _sample_group_sizes(v_total, col_multi)

    for _ in range(80):
        h_groups = [[random.randint(2, 5) for _ in range(size)] for size in h_group_sizes]
        v_groups = [[random.randint(2, 5) for _ in range(size)] for size in v_group_sizes]

        left_band_min = max(sum(group) + len(group) - 1 for group in h_groups)
        top_band_min = max(sum(group) + len(group) - 1 for group in v_groups)

        width_min = left_band_min + 2 * len(v_groups) + 2
        height_min = top_band_min + 2 * len(h_groups) + 2
        if width_min > MAX_GRID or height_min > MAX_GRID:
            continue

        left_band_max = min(left_band_min + 4, MAX_GRID - (2 * len(v_groups) + 2))
        top_band_max = min(top_band_min + 4, MAX_GRID - (2 * len(h_groups) + 2))
        left_band = random.randint(left_band_min, left_band_max)
        top_band = random.randint(top_band_min, top_band_max)

        width_low = left_band + 2 * len(v_groups) + 2
        height_low = top_band + 2 * len(h_groups) + 2
        width_high = min(MAX_GRID, width_low + 4)
        height_high = min(MAX_GRID, height_low + 4)
        width = random.randint(width_low, width_high)
        height = random.randint(height_low, height_high)

        row_low, row_high = top_band + 2, height - 2
        col_low, col_high = left_band + 2, width - 2
        if row_high < row_low or col_high < col_low:
            continue

        try:
            h_rows = _sample_spaced_positions(row_low, row_high, len(h_groups), gap=2)
            v_cols = _sample_spaced_positions(col_low, col_high, len(v_groups), gap=2)
        except (AssertionError, ValueError):
            continue

        random.shuffle(h_rows)
        random.shuffle(v_cols)

        colors = random.sample(PALETTE, h_total + v_total)
        segments: list[dict] = []
        color_idx = 0

        for row, lengths in zip(h_rows, h_groups):
            starts = _place_group(lengths, left_band)
            for start, length in zip(starts, lengths):
                segments.append(
                    {
                        "color": colors[color_idx],
                        "orientation": "h",
                        "row": row,
                        "start": start,
                        "length": length,
                    }
                )
                color_idx += 1

        for col, lengths in zip(v_cols, v_groups):
            starts = _place_group(lengths, top_band)
            for start, length in zip(starts, lengths):
                segments.append(
                    {
                        "color": colors[color_idx],
                        "orientation": "v",
                        "col": col,
                        "start": start,
                        "length": length,
                    }
                )
                color_idx += 1

        if not _segments_are_separated(segments):
            continue

        grid = _build_grid((height, width), segments)
        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        if max(grid.shape) > MAX_GRID or max(out.shape) > MAX_GRID:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
