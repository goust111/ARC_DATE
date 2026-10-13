from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands), "separator lines must split non-empty regions"
    return bands


def _separator_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert rows and cols, "expected full grey separator rows and columns"
    return rows, cols


def _ranked_counts(region: np.ndarray) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    flat_index = 0
    for r in range(region.shape[0]):
        for c in range(region.shape[1]):
            color = int(region[r, c])
            if color in (BLACK, GREY):
                flat_index += 1
                continue
            counts[color] += 1
            first_seen.setdefault(color, flat_index)
            flat_index += 1
    return sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))


def _transform_region(region: np.ndarray) -> np.ndarray:
    out = np.full(region.shape, BLACK, dtype=int)
    for col, (color, count) in enumerate(_ranked_counts(region)):
        out[:count, col] = int(color)
    return out


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep_rows, sep_cols = _separator_rows_cols(g)
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _transform_region(region)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each separator-delimited region with its own descending frequency histogram.

    Input structure:
    - Black background with one or more full grey separator rows and columns.
    - The separator lines partition the grid into rectangular regions.
    - Inside each region, 2-3 non-black non-grey colors appear as scattered pixels.

    Rule:
    - Use the full grey rows and columns as region boundaries.
    - Within each region, count each non-black non-grey color and sort the colors by descending frequency.
    - Clear the region to black, then rebuild it as a top-aligned local histogram: the most frequent color fills the leftmost column from the top for its count, the next color fills the next column, and so on.
    - Leave all grey separator lines unchanged.

    Key insight:
    - The separators define independent counting scopes, so each region becomes its own local histogram rather than contributing to one global count.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _layout_bands(lengths: list[int]) -> list[tuple[int, int]]:
    start = 0
    bands = []
    for i, width in enumerate(lengths):
        bands.append((start, start + width))
        start += width
        if i != len(lengths) - 1:
            start += 1
    return bands


def _make_color_spec(
    rng: random.Random,
    k: int,
    *,
    shared_color: int | None = None,
    shared_count: int | None = None,
    shared_must_not_be_top: bool = False,
) -> list[tuple[int, int]]:
    if shared_color is None:
        colors = rng.sample(PALETTE, k)
        counts = rng.sample([1, 2, 3, 4], k)
        return list(zip(colors, counts))

    others = [color for color in PALETTE if color != shared_color]
    colors = [shared_color, *rng.sample(others, k - 1)]

    assert shared_count is not None
    remaining = [count for count in [1, 2, 3, 4] if count != shared_count]
    if shared_must_not_be_top:
        higher = [count for count in remaining if count > shared_count]
        assert higher, "shared color needs a higher competitor"
        picked = [rng.choice(higher)]
        leftovers = [count for count in remaining if count != picked[0]]
        if k == 3:
            picked.append(rng.choice(leftovers))
        counts = [shared_count, *picked]
    else:
        if k == 2:
            counts = [shared_count, rng.choice(remaining)]
        else:
            counts = [shared_count, *rng.sample(remaining, 2)]
    return list(zip(colors, counts))


def _scatter_region(
    height: int,
    width: int,
    color_counts: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    total = sum(count for _, count in color_counts)
    cells = [(r, c) for r in range(height) for c in range(width)]
    expected = np.full((height, width), BLACK, dtype=int)
    for idx, (color, count) in enumerate(sorted(color_counts, key=lambda item: -item[1])):
        expected[:count, idx] = int(color)

    for _ in range(200):
        region = np.full((height, width), BLACK, dtype=int)
        chosen = rng.sample(cells, total)
        rng.shuffle(chosen)
        offset = 0
        for color, count in color_counts:
            for r, c in chosen[offset : offset + count]:
                region[r, c] = int(color)
            offset += count
        if np.array_equal(region, expected):
            continue
        return region
    return None


def _validate_regions(inp: np.ndarray, out: np.ndarray) -> bool:
    sep_rows, sep_cols = _separator_rows_cols(inp)
    row_bands = _bands(inp.shape[0], sep_rows)
    col_bands = _bands(inp.shape[1], sep_cols)

    if len(row_bands) != 2 or len(col_bands) not in (2, 3):
        return False
    if not np.array_equal(inp[sep_rows, :], np.full((len(sep_rows), inp.shape[1]), GREY, dtype=int)):
        return False
    if not np.array_equal(inp[:, sep_cols], np.full((inp.shape[0], len(sep_cols)), GREY, dtype=int)):
        return False

    changed_regions = 0
    all_region_counts: list[dict[int, int]] = []
    for r0, r1 in row_bands:
        if r1 - r0 < 4:
            return False
        for c0, c1 in col_bands:
            if c1 - c0 < 4:
                return False
            region_in = inp[r0:r1, c0:c1]
            region_out = out[r0:r1, c0:c1]
            if np.any(region_in == GREY):
                return False
            ranked = _ranked_counts(region_in)
            if len(ranked) not in (2, 3):
                return False
            counts = [count for _, count in ranked]
            if len(set(counts)) != len(counts):
                return False
            if max(counts) > region_in.shape[0] or len(ranked) > region_in.shape[1]:
                return False
            if not np.array_equal(region_out, _transform_region(region_in)):
                return False
            if not np.array_equal(region_in, region_out):
                changed_regions += 1
            all_region_counts.append(dict(ranked))

    shared_variation = False
    seen: dict[int, set[int]] = {}
    for region_counts in all_region_counts:
        for color, count in region_counts.items():
            seen.setdefault(int(color), set()).add(int(count))
    for counts in seen.values():
        if len(counts) >= 2:
            shared_variation = True
            break
    return changed_regions >= 2 and shared_variation


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    n_col_regions = rng.choice([2, 3])
    row_heights = [rng.randint(4, 8), rng.randint(4, 8)]
    col_widths = [rng.randint(4, 8) for _ in range(n_col_regions)]

    height = sum(row_heights) + 1
    width = sum(col_widths) + (n_col_regions - 1)
    if height > 30 or width > 30:
        return None

    row_bands = _layout_bands(row_heights)
    col_bands = _layout_bands(col_widths)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_bands[0][1], :] = GREY
    for band in col_bands[:-1]:
        grid[:, band[1]] = GREY

    shared_color = rng.choice(PALETTE)
    region_total = len(row_bands) * len(col_bands)
    ks = [rng.choice([2, 3]) for _ in range(region_total)]
    ks[0] = rng.choice([2, 3])
    ks[1] = 5 - ks[0]

    specs: list[list[tuple[int, int]]] = []
    specs.append(_make_color_spec(rng, ks[0], shared_color=shared_color, shared_count=4))
    shared_low = rng.choice([1, 2])
    specs.append(
        _make_color_spec(
            rng,
            ks[1],
            shared_color=shared_color,
            shared_count=shared_low,
            shared_must_not_be_top=True,
        )
    )
    for k in ks[2:]:
        specs.append(_make_color_spec(rng, k))

    for index, ((r0, r1), (c0, c1)) in enumerate(
        [(rb, cb) for rb in row_bands for cb in col_bands]
    ):
        region = _scatter_region(r1 - r0, c1 - c0, specs[index], rng)
        if region is None:
            return None
        grid[r0:r1, c0:c1] = region

    out = _apply_rule(grid)
    if np.array_equal(out, grid):
        return None
    if not _validate_regions(grid, out):
        return None

    first_region = out[row_bands[0][0] : row_bands[0][1], col_bands[0][0] : col_bands[0][1]]
    second_region = out[row_bands[0][0] : row_bands[0][1], col_bands[1][0] : col_bands[1][1]]
    if first_region[0, 0] != shared_color:
        return None
    if second_region[0, 0] == shared_color:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
