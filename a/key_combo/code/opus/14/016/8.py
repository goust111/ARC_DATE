from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


PALETTE = tuple(range(1, 10))
BACKGROUND = 0
MAX_ATTEMPTS = 400

HEIGHT_RANGE = (10, 17)
WIDTH_RANGE = (11, 18)
LEGEND_LEN_RANGE = (3, 6)
BAND_COUNT_RANGE = (3, 5)
BAND_WIDTH_MIN = 2
MARKER_COUNT_RANGE = (2, 5)
ORDINARY_NOISE_RANGE = (2, 8)
SAME_ROW_PAIR_PROB = 0.35
CLIPPED_MARKER_PROB = 0.45


def _legend_info(last_row: np.ndarray) -> tuple[int, np.ndarray]:
    nz = np.flatnonzero(last_row != BACKGROUND)
    assert nz.size >= 3, "expected a bottom legend string"
    start = int(nz[0])
    end = int(nz[-1])
    assert end - start + 1 == nz.size, "legend must be contiguous"
    seq = np.asarray(last_row[start : end + 1], dtype=int)
    assert len(set(map(int, seq))) == len(seq), "legend colors must be unique"
    return start, seq


def _dominant_color(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, values.tolist()))
    ranked = counts.most_common()
    assert ranked, "expected nonempty column"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0]), int(ranked[0][1])


def _column_modes(grid: np.ndarray, legend_colors: set[int]) -> list[int]:
    top = grid[:-1]
    min_support = math.ceil(top.shape[0] * 0.7)
    modes: list[int] = []
    for c in range(grid.shape[1]):
        color, support = _dominant_color(top[:, c])
        assert color != BACKGROUND, "stripe dominant colors must be non-black"
        assert color not in legend_colors, "stripe colors must be disjoint from legend colors"
        assert support >= min_support, "each column must keep its stripe dominant in at least 70% of cells"
        modes.append(color)
    return modes


def _analyze(grid: np.ndarray) -> tuple[int, np.ndarray, dict[int, int], list[int], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    start, legend = _legend_info(grid[-1])
    legend_index = {int(color): idx for idx, color in enumerate(map(int, legend))}
    modes = _column_modes(grid, set(legend_index))

    markers: list[tuple[int, int, int]] = []
    ordinary_noise: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0] - 1):
        for c, mode in enumerate(modes):
            value = int(grid[r, c])
            if value == mode:
                continue
            if value in legend_index:
                markers.append((r, c, value))
            else:
                ordinary_noise.append((r, c, value))
    return start, legend, legend_index, modes, markers, ordinary_noise


def _solve_from_analysis(
    grid: np.ndarray,
    legend: np.ndarray,
    legend_index: dict[int, int],
    modes: list[int],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for c, mode in enumerate(modes):
        out[:-1, c] = mode

    width = out.shape[1]
    for r, c, color in sorted(markers, key=lambda item: (item[0], item[1])):
        start_c = c - legend_index[int(color)]
        for offset, value in enumerate(legend):
            cc = start_c + offset
            if 0 <= cc < width:
                out[r, cc] = int(value)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify legend-marker noise hidden in vertical stripes, denoise the stripes, then expand the full bottom legend at each marker row.

    Input structure:
    - All rows except the last form vertical non-black stripes: each column has one non-black dominant color.
    - Some non-bottom cells are noise pixels whose colors differ from their column dominant color.
    - The last row contains one contiguous horizontal legend string of distinct non-black colors, with black elsewhere.
    - A subset of the noise pixels use legend colors; those pixels are semantic markers rather than ordinary noise.

    Rule:
    - Read the bottom legend string and treat its colors as the marker vocabulary.
    - For every non-bottom cell, compare it with the dominant color of its column; differing cells are noise.
    - Record only those noise cells whose colors appear in the legend, then denoise every non-bottom column back to its dominant stripe color.
    - For each recorded marker, copy the full legend onto that marker row, shifted so the marker color keeps the same legend position, clipping at the side borders.

    Key insight:
    - Marker pixels are indistinguishable from ordinary stripe noise until their colors are cross-checked with the bottom legend, so the marker extraction must happen before denoising destroys them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 4 <= g.shape[0] <= 30 and 4 <= g.shape[1] <= 30, "grid must fit ARC limits"

    _start, legend, legend_index, modes, markers, _ordinary_noise = _analyze(g)
    return _solve_from_analysis(g, legend, legend_index, modes, markers)


def _partition(total: int, parts: int, min_width: int) -> tuple[int, ...] | None:
    options: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: tuple[int, ...]) -> None:
        if slots == 0:
            if remaining == 0:
                options.append(prefix)
            return
        max_width = remaining - min_width * (slots - 1)
        for width in range(min_width, max_width + 1):
            rec(remaining - width, slots - 1, prefix + (width,))

    rec(total, parts, ())
    if not options:
        return None
    return random.choice(options)


def _make_stripes(height: int, width: int, legend_len: int) -> tuple[np.ndarray, list[int], list[int], int]:
    legend_colors = random.sample(PALETTE, legend_len)
    remaining = [color for color in PALETTE if color not in legend_colors]
    max_band_count = min(BAND_COUNT_RANGE[1], len(remaining), width // BAND_WIDTH_MIN)
    band_count = random.randint(BAND_COUNT_RANGE[0], max_band_count)
    band_widths = _partition(width, band_count, BAND_WIDTH_MIN)
    assert band_widths is not None
    band_colors = random.sample(remaining, band_count)

    grid = np.zeros((height, width), dtype=int)
    column_modes: list[int] = []
    c0 = 0
    for band_width, color in zip(band_widths, band_colors):
        grid[:-1, c0 : c0 + band_width] = int(color)
        column_modes.extend([int(color)] * band_width)
        c0 += band_width

    legend_start = random.randint(0, width - legend_len)
    grid[-1, legend_start : legend_start + legend_len] = np.asarray(legend_colors, dtype=int)
    return grid, column_modes, legend_colors, legend_start


def _visible_interval(start_c: int, legend_len: int, width: int) -> tuple[int, int]:
    return max(0, start_c), min(width - 1, start_c + legend_len - 1)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int], gap: int = 0) -> bool:
    return not (a[1] + gap < b[0] or b[1] + gap < a[0])


def _candidate_marker_columns(
    width: int,
    legend_len: int,
    anchor: int,
    clip_kind: str,
    occupied: list[tuple[int, int]],
) -> list[int]:
    cols: list[int] = []
    for c in range(width):
        start_c = c - anchor
        end_c = start_c + legend_len - 1
        if clip_kind == "none" and not (0 <= start_c and end_c < width):
            continue
        if clip_kind == "left" and not (start_c < 0 <= end_c):
            continue
        if clip_kind == "right" and not (start_c <= width - 1 < end_c):
            continue
        interval = _visible_interval(start_c, legend_len, width)
        if any(_intervals_overlap(interval, old, gap=1) for old in occupied):
            continue
        cols.append(c)
    random.shuffle(cols)
    return cols


def _marker_row_plan(row_pool: list[int], marker_count: int) -> list[int]:
    rows = random.sample(row_pool, marker_count)
    if marker_count >= 2 and random.random() < SAME_ROW_PAIR_PROB:
        a, b = random.sample(range(marker_count), 2)
        rows[b] = rows[a]
    random.shuffle(rows)
    return rows


def _place_markers(
    grid: np.ndarray,
    legend_colors: list[int],
    column_modes: list[int],
) -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int]]], bool] | None:
    height, width = grid.shape
    usable_rows = list(range(height - 1))
    marker_count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(usable_rows)))
    row_plan = _marker_row_plan(usable_rows, marker_count)

    anchor_choices = list(range(len(legend_colors)))
    if marker_count <= len(anchor_choices):
        anchors = random.sample(anchor_choices, marker_count)
    else:
        anchors = [random.choice(anchor_choices) for _ in range(marker_count)]
    if len(set(anchors)) == 1 and len(anchor_choices) > 1:
        anchors[-1] = random.choice([a for a in anchor_choices if a != anchors[0]])

    clipped_used = False
    force_clip = random.random() < CLIPPED_MARKER_PROB and len(legend_colors) >= 3
    clip_slots = []
    if force_clip:
        clip_slots.append(random.randrange(marker_count))

    col_noise_counts = [0] * width
    capacity = max(1, math.floor((height - 1) * 0.3))
    occupied_by_row: dict[int, list[tuple[int, int]]] = {}
    markers: list[tuple[int, int, int]] = []

    for i, (row, anchor) in enumerate(zip(row_plan, anchors)):
        clip_kind = "none"
        if i in clip_slots:
            options = []
            if anchor > 0:
                options.append("left")
            if anchor < len(legend_colors) - 1:
                options.append("right")
            if options:
                clip_kind = random.choice(options)

        candidates = _candidate_marker_columns(
            width,
            len(legend_colors),
            anchor,
            clip_kind,
            occupied_by_row.get(row, []),
        )
        placed = False
        for c in candidates:
            if col_noise_counts[c] >= capacity:
                continue
            if legend_colors[anchor] == column_modes[c]:
                continue
            color = int(legend_colors[anchor])
            grid[row, c] = color
            start_c = c - anchor
            occupied_by_row.setdefault(row, []).append(_visible_interval(start_c, len(legend_colors), width))
            col_noise_counts[c] += 1
            markers.append((row, c, color))
            clipped_used = clipped_used or clip_kind != "none"
            placed = True
            break
        if not placed:
            return None

    return markers, occupied_by_row, clipped_used


def _ordinary_noise_colors(column_mode: int, legend_colors: set[int]) -> list[int]:
    preferred = [color for color in PALETTE if color not in legend_colors and color != column_mode]
    if BACKGROUND != column_mode:
        preferred.append(BACKGROUND)
    return preferred


def _place_ordinary_noise(
    grid: np.ndarray,
    column_modes: list[int],
    markers: list[tuple[int, int, int]],
) -> bool:
    height, width = grid.shape
    legend_set = set(map(int, grid[-1, grid[-1] != BACKGROUND]))
    capacity = max(1, math.floor((height - 1) * 0.3))
    col_noise_counts = [0] * width
    marker_cells = {(r, c) for r, c, _ in markers}
    for _r, c, _color in markers:
        col_noise_counts[c] += 1

    candidate_cells = [(r, c) for r in range(height - 1) for c in range(width) if (r, c) not in marker_cells]
    random.shuffle(candidate_cells)
    target = random.randint(ORDINARY_NOISE_RANGE[0], min(ORDINARY_NOISE_RANGE[1], len(candidate_cells)))
    placed = 0
    for r, c in candidate_cells:
        if col_noise_counts[c] >= capacity:
            continue
        choices = _ordinary_noise_colors(column_modes[c], legend_set)
        if not choices:
            continue
        color = int(random.choice(choices))
        if color == column_modes[c] or color in legend_set:
            continue
        grid[r, c] = color
        col_noise_counts[c] += 1
        placed += 1
        if placed >= target:
            break
    return placed >= 1


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        legend_start, legend, legend_index, modes, markers, ordinary_noise = _analyze(inp)
    except AssertionError:
        return False

    if len(legend) < 3 or len(markers) < 2 or len(ordinary_noise) < 1:
        return False
    if len({legend_index[color] for _r, _c, color in markers}) < 2 and len(legend) > 1:
        return False
    if np.array_equal(inp[:-1], out[:-1]):
        return False

    expected = _solve_from_analysis(inp, legend, legend_index, modes, markers)
    if not np.array_equal(expected, out):
        return False

    clean = inp.copy()
    for c, mode in enumerate(modes):
        clean[:-1, c] = mode
    if np.array_equal(clean, out):
        return False

    rows: dict[int, list[tuple[int, int, int, tuple[int, int]]]] = {}
    clipped = 0
    for r, c, color in markers:
        anchor = legend_index[color]
        start_c = c - anchor
        interval = _visible_interval(start_c, len(legend), inp.shape[1])
        if interval != (start_c, start_c + len(legend) - 1):
            clipped += 1
        rows.setdefault(r, []).append((c, color, anchor, interval))

    for row, entries in rows.items():
        intervals = [entry[3] for entry in entries]
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                if _intervals_overlap(intervals[i], intervals[j]):
                    return False

        reconstructed = clean[row].copy()
        for c, _color, anchor, _interval in sorted(entries, key=lambda item: item[0]):
            start_c = c - anchor
            for offset, value in enumerate(legend):
                cc = start_c + offset
                if 0 <= cc < inp.shape[1]:
                    reconstructed[cc] = int(value)
        if not np.array_equal(out[row], reconstructed):
            return False

    if any(int(inp[-1, c]) != BACKGROUND for c in range(legend_start)):
        return False
    if any(int(inp[-1, c]) != BACKGROUND for c in range(legend_start + len(legend), inp.shape[1])):
        return False
    return clipped <= len(markers)


def _generate_one() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    if width < LEGEND_LEN_RANGE[0] or height < 4:
        return None

    legend_len = random.randint(LEGEND_LEN_RANGE[0], min(LEGEND_LEN_RANGE[1], width))
    grid, column_modes, legend_colors, _legend_start = _make_stripes(height, width, legend_len)

    marker_result = _place_markers(grid, legend_colors, column_modes)
    if marker_result is None:
        return None
    markers, _occupied_by_row, _clipped_used = marker_result

    if not _place_ordinary_noise(grid, column_modes, markers):
        return None

    out = main(grid)
    if not _validate_semantics(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy-legend ARC input")
