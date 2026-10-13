import random

import numpy as np


BG = 0
AZURE = 8
PROPAGATOR_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GEN_RETRIES = 300
ROW_BAND_COUNT_RANGE = (3, 5)
COL_BAND_COUNT_RANGE = (3, 5)
BAND_SIZE_MIN = 2
BAND_SIZE_MAX = 7
EXTRA_PROPAGATOR_PROB = 0.35
OUTER_BORDER_PROB = 0.5


def _separator_indices(mask: np.ndarray) -> list[int]:
    return [int(i) for i, is_sep in enumerate(mask) if bool(is_sep)]


def _non_separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if start <= sep - 1:
            spans.append((int(start), int(sep - 1)))
        start = int(sep + 1)
    if start <= length - 1:
        spans.append((int(start), int(length - 1)))
    return spans


def _region_data(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    row_seps = _separator_indices(np.all(g == AZURE, axis=1))
    col_seps = _separator_indices(np.all(g == AZURE, axis=0))
    row_spans = _non_separator_spans(g.shape[0], row_seps)
    col_spans = _non_separator_spans(g.shape[1], col_seps)

    regions: list[dict] = []
    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == AZURE), "azure separators cannot pass through region interiors"

            cells = np.argwhere((patch != BG) & (patch != AZURE))
            assert len(cells) <= 1, "each region must contain at most one propagator"

            propagator = None
            if len(cells) == 1:
                pr, pc = map(int, cells[0])
                propagator = (int(r0 + pr), int(c0 + pc), int(patch[pr, pc]))

            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if width > height:
                shape = "horizontal"
            elif height > width:
                shape = "vertical"
            else:
                shape = "square"

            regions.append(
                {
                    "row_idx": int(row_idx),
                    "col_idx": int(col_idx),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "height": int(height),
                    "width": int(width),
                    "shape": shape,
                    "propagator": propagator,
                }
            )

    return row_spans, col_spans, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by full-row/full-column azure separators, then fill each
    occupied rectangle according to its aspect ratio and its propagator position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _row_spans, _col_spans, regions = _region_data(g)

    for region in regions:
        propagator = region["propagator"]
        if propagator is None:
            continue

        r0 = int(region["r0"])
        r1 = int(region["r1"])
        c0 = int(region["c0"])
        c1 = int(region["c1"])
        pr, pc, color = propagator

        if region["shape"] == "square":
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        elif region["shape"] == "horizontal":
            for r in range(r0, r1 + 1):
                if (r - pr) % 2 == 0:
                    out[r, c0 : c1 + 1] = color
                else:
                    out[r, c0 : c1 + 1] = BG
        else:
            for c in range(c0, c1 + 1):
                if (c - pc) % 2 == 0:
                    out[r0 : r1 + 1, c] = color
                else:
                    out[r0 : r1 + 1, c] = BG

    return out


def _shuffle_bands(
    rng: random.Random, bands: list[tuple[str, int]]
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    shuffled = list(bands)
    rng.shuffle(shuffled)
    positions = {role: idx for idx, (role, _size) in enumerate(shuffled)}
    return shuffled, positions


def _build_bands(
    rng: random.Random,
) -> tuple[list[tuple[str, int]], dict[str, int], list[tuple[str, int]], dict[str, int]]:
    square = rng.randint(2, 5)
    short = rng.randint(2, min(4, BAND_SIZE_MAX - 1))
    narrow = rng.randint(2, min(4, BAND_SIZE_MAX - 1))

    tall_min = max(square + 1, narrow + 1, 3)
    tall_max = min(BAND_SIZE_MAX, tall_min + 2)
    wide_min = max(square + 1, short + 1, 3)
    wide_max = min(BAND_SIZE_MAX, wide_min + 2)
    tall = rng.randint(tall_min, tall_max)
    wide = rng.randint(wide_min, wide_max)

    row_extra_count = rng.randint(ROW_BAND_COUNT_RANGE[0], ROW_BAND_COUNT_RANGE[1]) - 3
    col_extra_count = rng.randint(COL_BAND_COUNT_RANGE[0], COL_BAND_COUNT_RANGE[1]) - 3

    row_bands = [("square", square), ("tall", tall), ("short", short)]
    row_bands.extend((f"extra_row_{i}", rng.randint(BAND_SIZE_MIN, BAND_SIZE_MAX)) for i in range(row_extra_count))

    col_bands = [("square", square), ("wide", wide), ("narrow", narrow)]
    col_bands.extend((f"extra_col_{i}", rng.randint(BAND_SIZE_MIN, BAND_SIZE_MAX)) for i in range(col_extra_count))

    row_bands, row_positions = _shuffle_bands(rng, row_bands)
    col_bands, col_positions = _shuffle_bands(rng, col_bands)
    return row_bands, row_positions, col_bands, col_positions


def _assemble_grid(
    row_bands: list[tuple[str, int]],
    col_bands: list[tuple[str, int]],
    outer_border: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_sizes = [size for _role, size in row_bands]
    col_sizes = [size for _role, size in col_bands]
    row_sep_count = max(0, len(row_sizes) - 1) + (2 if outer_border else 0)
    col_sep_count = max(0, len(col_sizes) - 1) + (2 if outer_border else 0)
    height = int(sum(row_sizes) + row_sep_count)
    width = int(sum(col_sizes) + col_sep_count)

    grid = np.full((height, width), BG, dtype=int)
    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []

    cursor = 1 if outer_border else 0
    if outer_border:
        grid[0, :] = AZURE
    for idx, size in enumerate(row_sizes):
        r0 = int(cursor)
        r1 = int(cursor + size - 1)
        row_spans.append((r0, r1))
        cursor = int(r1 + 1)
        if idx != len(row_sizes) - 1:
            grid[cursor, :] = AZURE
            cursor += 1
    if outer_border:
        grid[cursor, :] = AZURE

    cursor = 1 if outer_border else 0
    if outer_border:
        grid[:, 0] = AZURE
    for idx, size in enumerate(col_sizes):
        c0 = int(cursor)
        c1 = int(cursor + size - 1)
        col_spans.append((c0, c1))
        cursor = int(c1 + 1)
        if idx != len(col_sizes) - 1:
            grid[:, cursor] = AZURE
            cursor += 1
    if outer_border:
        grid[:, cursor] = AZURE

    return grid, row_spans, col_spans


def _place_propagator(
    grid: np.ndarray, span_r: tuple[int, int], span_c: tuple[int, int], color: int, rng: random.Random
) -> None:
    r = rng.randint(span_r[0], span_r[1])
    c = rng.randint(span_c[0], span_c[1])
    grid[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_bands, row_positions, col_bands, col_positions = _build_bands(rng)
    outer_border = rng.random() < OUTER_BORDER_PROB

    grid, row_spans, col_spans = _assemble_grid(row_bands, col_bands, outer_border)
    available_colors = list(PROPAGATOR_COLORS)
    rng.shuffle(available_colors)

    required_regions = [
        (row_positions["square"], col_positions["square"]),
        (row_positions["short"], col_positions["wide"]),
        (row_positions["tall"], col_positions["narrow"]),
    ]
    required_colors = available_colors[: len(required_regions)]
    occupied = set(required_regions)

    for (row_idx, col_idx), color in zip(required_regions, required_colors):
        _place_propagator(grid, row_spans[row_idx], col_spans[col_idx], color, rng)

    all_regions = [(ri, ci) for ri in range(len(row_spans)) for ci in range(len(col_spans))]
    rng.shuffle(all_regions)
    for row_idx, col_idx in all_regions:
        if (row_idx, col_idx) in occupied:
            continue
        if rng.random() >= EXTRA_PROPAGATOR_PROB:
            continue
        color = rng.choice(PROPAGATOR_COLORS)
        _place_propagator(grid, row_spans[row_idx], col_spans[col_idx], color, rng)
        occupied.add((row_idx, col_idx))

    try:
        _row_spans, _col_spans, regions = _region_data(grid)
    except AssertionError:
        return None

    seen_shapes = {region["shape"] for region in regions if region["propagator"] is not None}
    if seen_shapes != {"horizontal", "vertical", "square"}:
        return None

    empty_count = sum(1 for region in regions if region["propagator"] is None)
    if empty_count == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
