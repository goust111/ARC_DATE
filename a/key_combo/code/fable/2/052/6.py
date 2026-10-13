from __future__ import annotations

import random

import numpy as np


ALL_COLORS = tuple(range(10))
GRID_LIMIT = 30
MIN_BANDS = 2
MAX_BANDS = 4
MIN_REGION_SIZE = 4
MAX_REGION_SIZE = 8
MIN_BASE_SIZE = 2
MAX_BASE_SIZE = 4
GEN_ATTEMPTS = 400
RANDOM_REGION_ATTEMPTS = 120
BASE_PATTERN_ATTEMPTS = 80


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    return runs


def _spans_between_runs(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if cursor < start:
            spans.append((cursor, start))
        cursor = end
    if cursor < length:
        spans.append((cursor, length))
    return spans


def _partition_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(grid):
        divider = int(color)
        row_runs = _true_runs(np.all(grid == divider, axis=1))
        col_runs = _true_runs(np.all(grid == divider, axis=0))
        if not row_runs or not col_runs:
            continue
        row_spans = _spans_between_runs(grid.shape[0], row_runs)
        col_spans = _spans_between_runs(grid.shape[1], col_runs)
        if not row_spans or not col_spans:
            continue
        if all(np.any(grid[r0:r1, c0:c1] != divider) for r0, r1 in row_spans for c0, c1 in col_spans):
            candidates.append((divider, row_spans, col_spans))

    assert len(candidates) == 1, "expected exactly one divider-grid color"
    divider, row_spans, col_spans = candidates[0]
    return divider, row_spans, col_spans


def _compress_patch(patch: np.ndarray) -> np.ndarray:
    assert patch.ndim == 2 and patch.size > 0

    keep_rows = [0]
    for r in range(1, patch.shape[0]):
        if not np.array_equal(patch[r], patch[r - 1]):
            keep_rows.append(r)
    row_compressed = patch[keep_rows]

    keep_cols = [0]
    for c in range(1, row_compressed.shape[1]):
        if not np.array_equal(row_compressed[:, c], row_compressed[:, c - 1]):
            keep_cols.append(c)
    return row_compressed[:, keep_cols]


def _unique_mode(patch: np.ndarray) -> int:
    colors, counts = np.unique(patch, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) >= 1
    if len(order) >= 2:
        assert int(counts[order[0]]) > int(counts[order[1]]), "expected a unique majority color"
    return int(colors[order[0]])


def _adjacent_duplicate_rows(patch: np.ndarray) -> bool:
    return any(np.array_equal(patch[r], patch[r - 1]) for r in range(1, patch.shape[0]))


def _adjacent_duplicate_cols(patch: np.ndarray) -> bool:
    return any(np.array_equal(patch[:, c], patch[:, c - 1]) for c in range(1, patch.shape[1]))


def _stretch_patch(base: np.ndarray, row_factors: list[int], col_factors: list[int]) -> np.ndarray:
    out = np.repeat(base, row_factors, axis=0)
    out = np.repeat(out, col_factors, axis=1)
    return out


def _row_run_lengths(patch: np.ndarray) -> list[int]:
    lengths: list[int] = []
    start = 0
    for r in range(1, patch.shape[0] + 1):
        if r == patch.shape[0] or not np.array_equal(patch[r], patch[r - 1]):
            lengths.append(r - start)
            start = r
    return lengths


def _col_run_lengths(patch: np.ndarray) -> list[int]:
    lengths: list[int] = []
    start = 0
    for c in range(1, patch.shape[1] + 1):
        if c == patch.shape[1] or not np.array_equal(patch[:, c], patch[:, c - 1]):
            lengths.append(c - start)
            start = c
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reduce each divider-grid region back to its unstretched pattern, then vote its majority color into a 1-cell summary grid.

    Input structure:
    - One color forms full-row and full-column divider strips that partition the canvas into rectangular regions.
    - Inside each region sits a multicolor small pattern whose original rows and columns were repeated by independent run lengths.
    - The compressed pattern inside every region has a unique most frequent color.

    Rule:
    - Detect the divider-grid partition and process every region independently.
    - Within one region, merge each run of identical adjacent rows to one row, then merge each run of identical adjacent columns to one column.
    - Count colors in that recovered unstretched pattern and take its unique majority color.
    - Output an m x n grid whose cell (i, j) is the majority color of region (i, j).

    Key insight:
    - The apparent area inside a stretched region is misleading: votes must be counted after undoing row and column run-length stretching, not on the visible pixel totals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _divider, row_spans, col_spans = _partition_regions(g)
    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            region = g[r0:r1, c0:c1]
            assert region.size > 0, "each divider cell must contain a non-empty region"
            compressed = _compress_patch(region)
            out[i, j] = _unique_mode(compressed)
    return out


def _sample_band_sizes(count: int, total_limit: int) -> list[int]:
    usable = total_limit - (count + 1)
    min_total = count * MIN_REGION_SIZE
    max_total = min(usable, count * MAX_REGION_SIZE)
    assert min_total <= max_total
    total = random.randint(min_total, max_total)
    parts = [MIN_REGION_SIZE] * count
    remaining = total - min_total
    while remaining > 0:
        idxs = [idx for idx, value in enumerate(parts) if value < MAX_REGION_SIZE]
        idx = random.choice(idxs)
        add = random.randint(1, min(remaining, MAX_REGION_SIZE - parts[idx]))
        parts[idx] += add
        remaining -= add
    return parts


def _sample_factors(total: int, count: int) -> list[int]:
    assert 1 <= count <= total
    parts = [1] * count
    for _ in range(total - count):
        parts[random.randrange(count)] += 1
    return parts


def _sample_base_pattern(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    color_cap = min(4, len(palette), height * width)
    for _ in range(BASE_PATTERN_ATTEMPTS):
        n_colors = random.randint(2, color_cap)
        chosen = random.sample(palette, n_colors)
        cells = chosen[:] + random.choices(chosen, k=height * width - n_colors)
        random.shuffle(cells)
        base = np.array(cells, dtype=int).reshape(height, width)
        if _adjacent_duplicate_rows(base) or _adjacent_duplicate_cols(base):
            continue
        try:
            _unique_mode(base)
        except AssertionError:
            continue
        return base
    return None


def _relation_ok(relation: str, compressed_mode: int, raw_mode: int) -> bool:
    if relation == "same":
        return raw_mode == compressed_mode
    if relation == "different":
        return raw_mode != compressed_mode
    raise ValueError(f"unknown relation {relation}")


def _region_info(region: np.ndarray) -> dict:
    compressed = _compress_patch(region)
    return {
        "region_shape": tuple(int(x) for x in region.shape),
        "compressed_shape": tuple(int(x) for x in compressed.shape),
        "compressed_mode": _unique_mode(compressed),
        "raw_mode": _unique_mode(region),
        "row_factors": _row_run_lengths(region),
        "col_factors": _col_run_lengths(region),
        "n_colors": int(len(np.unique(compressed))),
    }


def _random_region(height: int, width: int, palette: list[int], relation: str, prefer_large: bool) -> np.ndarray | None:
    h_choices = list(range(MIN_BASE_SIZE, min(MAX_BASE_SIZE, height) + 1))
    w_choices = list(range(MIN_BASE_SIZE, min(MAX_BASE_SIZE, width) + 1))
    dim_choices = [(h, w) for h in h_choices for w in w_choices]
    if prefer_large:
        dim_choices = [(h, w) for h, w in dim_choices if h > 2 or w > 2]
    if not dim_choices:
        return None

    for _ in range(RANDOM_REGION_ATTEMPTS):
        base_h, base_w = random.choice(dim_choices)
        row_factors = _sample_factors(height, base_h)
        col_factors = _sample_factors(width, base_w)
        if max(row_factors) == 1 and max(col_factors) == 1:
            continue
        if len(set(row_factors)) == 1 and len(set(col_factors)) == 1:
            continue

        base = _sample_base_pattern(base_h, base_w, palette)
        if base is None:
            continue

        region = _stretch_patch(base, row_factors, col_factors)
        if not np.array_equal(_compress_patch(region), base):
            continue

        compressed_mode = _unique_mode(base)
        try:
            raw_mode = _unique_mode(region)
        except AssertionError:
            continue
        if _relation_ok(relation, compressed_mode, raw_mode):
            return region
    return None


def _template_region(height: int, width: int, palette: list[int], relation: str) -> np.ndarray:
    mode_color, other_a, other_b = random.sample(palette, 3)
    base = np.array([[mode_color, other_a], [mode_color, other_b]], dtype=int)

    if relation == "same":
        row_factors = [random.randint(1, height - 1), 0]
        row_factors[1] = height - row_factors[0]
        col_factors = [width - 1, 1]
    else:
        row_factors = [1, height - 1]
        col_factors = [1, width - 1]

    if random.random() < 0.5:
        base = base[::-1, :]
        row_factors = row_factors[::-1]
    if random.random() < 0.5:
        base = base[:, ::-1]
        col_factors = col_factors[::-1]

    region = _stretch_patch(base, row_factors, col_factors)
    assert np.array_equal(_compress_patch(region), base)
    assert _relation_ok(relation, _unique_mode(base), _unique_mode(region))
    return region


def _generate_region(height: int, width: int, palette: list[int], relation: str, prefer_large: bool) -> np.ndarray:
    region = _random_region(height, width, palette, relation, prefer_large)
    if region is None:
        region = _template_region(height, width, palette, relation)
    return region


def _assemble_grid(divider: int, row_heights: list[int], col_widths: list[int], regions: list[list[np.ndarray]]) -> np.ndarray:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    grid = np.full((height, width), divider, dtype=int)

    r = 1
    for row_height, row_regions in zip(row_heights, regions):
        c = 1
        for col_width, region in zip(col_widths, row_regions):
            assert region.shape == (row_height, col_width)
            grid[r : r + row_height, c : c + col_width] = region
            c += col_width + 1
        r += row_height + 1
    return grid


def _validate_case(grid: np.ndarray) -> None:
    divider, row_spans, col_spans = _partition_regions(grid)
    assert max(grid.shape) <= GRID_LIMIT
    assert np.any(grid != divider)

    out = main(grid)
    assert out.shape == (len(row_spans), len(col_spans))

    same_count = 0
    different_count = 0
    output_colors: list[int] = []
    large_pattern_seen = False

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            region = grid[r0:r1, c0:c1]
            info = _region_info(region)

            assert region.size > 0
            assert out[i, j] == info["compressed_mode"]
            assert np.array_equal(_compress_patch(_compress_patch(region)), _compress_patch(region))
            assert info["compressed_shape"][0] < region.shape[0] or info["compressed_shape"][1] < region.shape[1]
            assert len(set(info["row_factors"])) > 1 or len(set(info["col_factors"])) > 1
            assert max(info["row_factors"]) > 1 or max(info["col_factors"]) > 1
            assert not _adjacent_duplicate_rows(_compress_patch(region))
            assert not _adjacent_duplicate_cols(_compress_patch(region))

            if info["compressed_mode"] == info["raw_mode"]:
                same_count += 1
            else:
                different_count += 1

            if info["compressed_shape"] != (2, 2) or info["n_colors"] >= 4:
                large_pattern_seen = True
            output_colors.append(info["compressed_mode"])

    assert same_count >= 1, "need at least one bait region where raw and compressed votes coincide"
    assert different_count >= 1, "need at least one trap region where raw and compressed votes differ"
    assert len(set(output_colors)) >= 2, "summary grid must contain at least two colors"
    assert large_pattern_seen, "generator should not collapse to one tiny template everywhere"


def _build_case() -> np.ndarray:
    n_rows = random.randint(MIN_BANDS, MAX_BANDS)
    n_cols = random.randint(MIN_BANDS, MAX_BANDS)
    row_heights = _sample_band_sizes(n_rows, GRID_LIMIT)
    col_widths = _sample_band_sizes(n_cols, GRID_LIMIT)

    height = sum(row_heights) + n_rows + 1
    width = sum(col_widths) + n_cols + 1
    assert height <= GRID_LIMIT and width <= GRID_LIMIT

    divider = random.choice(ALL_COLORS)
    available = [color for color in ALL_COLORS if color != divider]
    palette_size = random.randint(4, min(7, len(available)))
    palette = random.sample(available, palette_size)
    region_count = n_rows * n_cols

    relations = ["different"] * ((region_count + 1) // 2) + ["same"] * (region_count // 2)
    random.shuffle(relations)
    prefer_large_index = random.randrange(region_count)

    regions: list[list[np.ndarray]] = []
    rel_idx = 0
    for i in range(n_rows):
        row_regions: list[np.ndarray] = []
        for j in range(n_cols):
            region = _generate_region(
                row_heights[i],
                col_widths[j],
                palette,
                relations[rel_idx],
                prefer_large=(rel_idx == prefer_large_index),
            )
            row_regions.append(region)
            rel_idx += 1
        regions.append(row_regions)

    return _assemble_grid(divider, row_heights, col_widths, regions)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        try:
            _validate_case(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid partition-compress-vote ARC input")
