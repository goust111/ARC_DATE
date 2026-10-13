import random

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7)

HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (12, 18)
BAND_COUNT_RANGE = (3, 5)
BAND_WIDTH_RANGE = (2, 4)
MIN_COLUMN_PIXELS = 2
MIN_FILL_RATIO = 0.40
GENERATION_ATTEMPTS = 400


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    col = 0
    while col < len(mask):
        if not mask[col]:
            col += 1
            continue
        start = col
        while col < len(mask) and mask[col]:
            col += 1
        spans.append((start, col))
    return spans


def _extract_bands(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    spans = _contiguous_spans(np.any(g != BACKGROUND, axis=0))
    assert 3 <= len(spans) <= 5, "expected three to five vertical bands"

    bands: list[dict[str, int]] = []
    seen_colors: set[int] = set()
    for c0, c1 in spans:
        block = g[:, c0:c1]
        colors = [int(color) for color in np.unique(block) if int(color) != BACKGROUND]
        assert len(colors) == 1, "each band span must contain exactly one foreground color"
        color = colors[0]
        assert color not in seen_colors, "band colors must be unique"
        seen_colors.add(color)

        count = int(np.count_nonzero(block == color))
        area = int(block.size)
        assert 0 < count < area, "each band must contain both foreground cells and gaps"
        assert np.all(np.any(block == color, axis=0)), "each band column must contain foreground"

        bands.append(
            {
                "color": color,
                "c0": int(c0),
                "c1": int(c1),
                "width": int(c1 - c0),
                "count": count,
            }
        )

    counts = [band["count"] for band in bands]
    assert len(set(counts)) == len(counts), "foreground counts must be unique"
    return bands


def _successor_map(order: list[int]) -> dict[int, int]:
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def _predecessor_map(order: list[int]) -> dict[int, int]:
    return {order[i]: order[(i - 1) % len(order)] for i in range(len(order))}


def _fill_by_order(grid: np.ndarray, bands: list[dict[str, int]], order: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ordered_bands = [bands[index] for index in order]
    for band, next_band in zip(ordered_bands, ordered_bands[1:] + ordered_bands[:1]):
        view = out[:, band["c0"] : band["c1"]]
        view[view == BACKGROUND] = next_band["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the pixels in each vertical band, rank the bands by frequency, and fill each band's gaps with its next ranked color.

    Input structure:
    - The background is black.
    - Three to five separated vertical bands occupy disjoint contiguous column intervals.
    - Each band uses one unique foreground color and contains black gap cells inside its own columns.
    - The bands' foreground pixel counts are all different.

    Rule:
    - Detect each band's full column span and count only its own colored cells inside that span.
    - Sort the bands from most colored cells to fewest to form a cyclic order.
    - For each band, recolor every black gap cell inside its span to the color of the next band in that frequency cycle.
    - Keep all original foreground pixels and all background outside the band spans unchanged.

    Key insight:
    - The borrowed color is defined by the hidden frequency ranking, not by spatial left-right adjacency.
    """
    g = np.asarray(grid, dtype=int)
    bands = _extract_bands(g)
    ranked = sorted(range(len(bands)), key=lambda index: bands[index]["count"], reverse=True)
    return _fill_by_order(g, bands, ranked)


def _random_composition(total: int, parts: int) -> list[int]:
    pieces = [0] * parts
    for _ in range(total):
        pieces[random.randrange(parts)] += 1
    return pieces


def _has_width_count_inversion(widths: list[int], counts: list[int]) -> bool:
    for i, width_i in enumerate(widths):
        for j, width_j in enumerate(widths):
            if width_i < width_j and counts[i] > counts[j]:
                return True
    return False


def _sample_layout() -> tuple[int, int, list[tuple[int, int]]]:
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        band_count = random.randint(*BAND_COUNT_RANGE)
        widths = [random.randint(*BAND_WIDTH_RANGE) for _ in range(band_count)]
        if len(set(widths)) < 2:
            continue

        min_total_width = sum(widths) + (band_count - 1)
        if min_total_width > WIDTH_RANGE[1]:
            continue

        width = random.randint(max(WIDTH_RANGE[0], min_total_width), WIDTH_RANGE[1])
        extras = _random_composition(width - min_total_width, band_count + 1)

        spans: list[tuple[int, int]] = []
        col = extras[0]
        for index, band_width in enumerate(widths):
            spans.append((col, col + band_width))
            col += band_width
            if index < band_count - 1:
                col += 1 + extras[index + 1]

        if col + extras[-1] != width:
            continue
        return height, width, spans

    raise RuntimeError("failed to sample a valid band layout")


def _sample_counts(height: int, spans: list[tuple[int, int]]) -> list[int]:
    widths = [c1 - c0 for c0, c1 in spans]
    areas = [height * band_width for band_width in widths]
    spatial_order = list(range(len(spans)))
    spatial_successors = _successor_map(spatial_order)
    reversed_successors = _successor_map(list(reversed(spatial_order)))

    for _ in range(GENERATION_ATTEMPTS):
        counts = []
        for area, band_width in zip(areas, widths):
            low = max(MIN_COLUMN_PIXELS * band_width, int(np.ceil(area * MIN_FILL_RATIO)))
            high = area - 1
            if low > high:
                break
            counts.append(random.randint(low, high))

        if len(counts) != len(spans) or len(set(counts)) != len(counts):
            continue
        if not _has_width_count_inversion(widths, counts):
            continue

        ranked = sorted(spatial_order, key=lambda index: counts[index], reverse=True)
        ranked_successors = _successor_map(ranked)
        if ranked_successors == spatial_successors:
            continue
        if len(spans) >= 4 and ranked_successors == reversed_successors:
            continue

        ranked_predecessors = _predecessor_map(ranked)
        spatial_predecessors = _predecessor_map(spatial_order)
        if not any(ranked_predecessors[i] != spatial_predecessors[i] for i in spatial_order):
            continue

        return counts

    raise RuntimeError("failed to sample count ordering that distinguishes the rule")


def _band_gap_scores(height: int, width: int) -> list[tuple[float, float, int, int]]:
    mode = random.choice(("noise", "row_clusters", "column_clusters", "diagonal", "center"))
    row_bias = [random.random() for _ in range(height)]
    col_bias = [random.random() for _ in range(width)]
    diag_shift = random.uniform(-0.6, 0.6)
    center_r = random.uniform(0, max(1, height - 1))
    center_c = random.uniform(0, max(1, width - 1))

    scores: list[tuple[float, float, int, int]] = []
    for r in range(height):
        for c in range(width):
            noise = 0.15 * random.random()
            if mode == "noise":
                score = random.random()
            elif mode == "row_clusters":
                score = row_bias[r] + 0.2 * col_bias[c] + noise
            elif mode == "column_clusters":
                score = col_bias[c] + 0.2 * row_bias[r] + noise
            elif mode == "diagonal":
                rr = 0.0 if height == 1 else r / (height - 1)
                cc = 0.0 if width == 1 else c / (width - 1)
                score = -abs(rr - cc - diag_shift) + noise
            else:
                score = -((r - center_r) ** 2 + (c - center_c) ** 2) + noise
            scores.append((score, random.random(), r, c))
    scores.sort(reverse=True)
    return scores


def _make_band_mask(height: int, width: int, target_count: int) -> np.ndarray:
    area = height * width
    assert MIN_COLUMN_PIXELS * width <= target_count < area

    mask = np.ones((height, width), dtype=bool)
    keep_per_column = [height] * width
    removals_needed = area - target_count
    removed = 0

    for _score, _tie_break, r, c in _band_gap_scores(height, width):
        if removed == removals_needed:
            break
        if keep_per_column[c] <= MIN_COLUMN_PIXELS:
            continue
        mask[r, c] = False
        keep_per_column[c] -= 1
        removed += 1

    assert removed == removals_needed, "failed to realize requested foreground count"
    assert np.all(mask.sum(axis=0) >= MIN_COLUMN_PIXELS), "band columns became too sparse"
    return mask


def _build_grid(height: int, width: int, spans: list[tuple[int, int]], counts: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(PALETTE, len(spans))
    for (c0, c1), count, color in zip(spans, counts, colors):
        band_mask = _make_band_mask(height, c1 - c0, count)
        grid[:, c0:c1][band_mask] = color
    return grid


def _is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        bands = _extract_bands(grid)
    except AssertionError:
        return False

    colors = [band["color"] for band in bands]
    widths = [band["width"] for band in bands]
    counts = [band["count"] for band in bands]
    if any(color not in PALETTE for color in colors):
        return False
    if len(set(colors)) != len(colors):
        return False
    if not _has_width_count_inversion(widths, counts):
        return False

    spatial_order = list(range(len(bands)))
    ranked = sorted(spatial_order, key=lambda index: bands[index]["count"], reverse=True)
    ranked_predecessors = _predecessor_map(ranked)
    spatial_predecessors = _predecessor_map(spatial_order)
    if not any(ranked_predecessors[i] != spatial_predecessors[i] for i in spatial_order):
        return False

    expected = _fill_by_order(grid, bands, ranked)
    if np.array_equal(expected, grid):
        return False

    spatial_fill = _fill_by_order(grid, bands, spatial_order)
    if np.array_equal(expected, spatial_fill):
        return False

    if len(bands) >= 4:
        reverse_spatial_fill = _fill_by_order(grid, bands, list(reversed(spatial_order)))
        if np.array_equal(expected, reverse_spatial_fill):
            return False

    ascending = sorted(spatial_order, key=lambda index: bands[index]["count"])
    ascending_fill = _fill_by_order(grid, bands, ascending)
    if np.array_equal(expected, ascending_fill):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        height, width, spans = _sample_layout()
        try:
            counts = _sample_counts(height, spans)
        except RuntimeError:
            continue
        grid = _build_grid(height, width, spans, counts)
        if _is_semantically_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid frequency-ranked band puzzle")
