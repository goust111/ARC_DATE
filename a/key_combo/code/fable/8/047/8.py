from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, int(mask.size) - 1))
    return runs


def _regions_between(length: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    regions: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in bands:
        if start <= prev_end + 1:
            return None
        regions.append((prev_end + 1, start - 1))
        prev_end = end
    if prev_end >= length - 1:
        return None
    regions.append((prev_end + 1, length - 1))
    return regions


def _extract_compact(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    candidates: list[np.ndarray] = []
    for sep_color in map(int, np.unique(g)):
        row_bands = _runs_from_mask(np.all(g == sep_color, axis=1))
        col_bands = _runs_from_mask(np.all(g == sep_color, axis=0))
        k_rows = len(row_bands) + 1
        k_cols = len(col_bands) + 1
        if k_rows != k_cols or k_rows not in (3, 4):
            continue

        row_regions = _regions_between(h, row_bands)
        col_regions = _regions_between(w, col_bands)
        if row_regions is None or col_regions is None:
            continue
        if len(row_regions) != k_rows or len(col_regions) != k_cols:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_regions}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_regions}
        row_band_sizes = {r1 - r0 + 1 for r0, r1 in row_bands} or {0}
        col_band_sizes = {c1 - c0 + 1 for c0, c1 in col_bands} or {0}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if len(row_band_sizes) != 1 or len(col_band_sizes) != 1:
            continue

        compact = np.empty((k_rows, k_cols), dtype=int)
        valid = True
        for i, (r0, r1) in enumerate(row_regions):
            for j, (c0, c1) in enumerate(col_regions):
                block = g[r0 : r1 + 1, c0 : c1 + 1]
                colors = np.unique(block)
                if colors.size != 1:
                    valid = False
                    break
                color = int(colors[0])
                if color in (BLACK, sep_color):
                    valid = False
                    break
                compact[i, j] = color
            if not valid:
                break

        if valid:
            candidates.append(compact)

    assert len(candidates) == 1, "expected one valid separator partition"
    return candidates[0]


def _unique_least_color(tile: np.ndarray) -> int:
    colors, counts = np.unique(tile, return_counts=True)
    least = int(counts.min())
    rare = colors[counts == least]
    assert rare.size == 1, "expected one uniquely least frequent color"
    return int(rare[0])


def _stamp_from_compact(tile: np.ndarray) -> np.ndarray:
    k = int(tile.shape[0])
    rare_color = _unique_least_color(tile)
    out = np.full((k * k, k * k), BLACK, dtype=int)
    for r, c in np.argwhere(tile == rare_color):
        r0 = int(r) * k
        c0 = int(c) * k
        out[r0 : r0 + k, c0 : c0 + k] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-partitioned board, then self-stamp the compressed image onto rare-color slots.

    Input structure:
    - The grid is partitioned by one separator color into a 3x3 or 4x4 array of equal-size rectangular regions.
    - Every separator row and separator column is a solid full-span bar of that separator color.
    - Each region is uniformly filled with one non-black color distinct from the separator.

    Rule:
    - Collapse the partitioned board to its k x k region-color image by taking one color per region.
    - Find the unique least frequent color in that compressed image.
    - Create a black canvas made of k x k blocks, each block itself being k x k cells.
    - In every block whose corresponding compressed cell has the least frequent color, copy the entire compressed image.
      Leave all other blocks black.

    Key insight:
    - The same compressed image plays two roles at once: it is both the sprite being stamped and the mask whose rare
      color selects the stamp locations, so neither the partition compression nor the rare-color lookup can be skipped.
    """
    compact = _extract_compact(grid)
    return _stamp_from_compact(compact)


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cut_points = sorted(random.sample(range(1, int(total)), parts - 1))
    values = []
    prev = 0
    for cut in cut_points + [int(total)]:
        values.append(cut - prev)
        prev = cut
    return values


def _nonnegative_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    dividers = sorted(random.sample(range(int(total) + parts - 1), parts - 1))
    out = []
    prev = -1
    for divider in dividers + [int(total) + parts - 1]:
        out.append(divider - prev - 1)
        prev = divider
    return out


def _feasible_count_plans(k: int, branch: str) -> list[tuple[int, int]]:
    total = k * k
    plans: list[tuple[int, int]] = []
    for n_colors in range(3, min(6, total) + 1):
        for rare_count in range(1, total):
            if branch == "single" and rare_count != 1:
                continue
            if branch == "multi" and rare_count <= 1:
                continue
            if total - rare_count >= (n_colors - 1) * (rare_count + 1):
                plans.append((n_colors, rare_count))
    return plans


def _sample_compact(k: int) -> np.ndarray | None:
    branch = random.choice(("single", "multi"))
    plans = _feasible_count_plans(k, branch)
    if not plans:
        plans = _feasible_count_plans(k, "single")
    n_colors, rare_count = random.choice(plans)

    palette = random.sample(NON_BLACK_COLORS, n_colors)
    rare_color = palette[0]
    other_colors = palette[1:]
    extra = k * k - rare_count - len(other_colors) * (rare_count + 1)
    counts = [rare_count] + [rare_count + 1 + x for x in _nonnegative_composition(extra, len(other_colors))]

    cells: list[int] = []
    for color, count in zip([rare_color] + other_colors, counts):
        cells.extend([int(color)] * int(count))
    random.shuffle(cells)
    compact = np.array(cells, dtype=int).reshape(k, k)

    if not _compact_is_useful(compact):
        return None
    return compact


def _compact_is_useful(compact: np.ndarray) -> bool:
    colors = np.unique(compact)
    if colors.size < 3:
        return False

    try:
        rare_color = _unique_least_color(compact)
    except AssertionError:
        return False

    row_varied = any(np.unique(compact[r, :]).size >= 2 for r in range(compact.shape[0]))
    col_varied = any(np.unique(compact[:, c]).size >= 2 for c in range(compact.shape[1]))
    if not row_varied or not col_varied:
        return False

    rare_positions = np.argwhere(compact == rare_color)
    if rare_positions.shape[0] == 0:
        return False

    all_on_border = all(
        r in (0, compact.shape[0] - 1) or c in (0, compact.shape[1] - 1)
        for r, c in rare_positions.tolist()
    )
    non_rare_colors = np.unique(compact[compact != rare_color])
    return non_rare_colors.size >= 2 and not (rare_positions.shape[0] == 1 and all_on_border)


def _expand_compact(
    compact: np.ndarray,
    sep_color: int,
    cell_h: int,
    cell_w: int,
    bar_h: int,
    bar_w: int,
) -> np.ndarray:
    k = int(compact.shape[0])
    h = k * cell_h + (k - 1) * bar_h
    w = k * cell_w + (k - 1) * bar_w
    grid = np.full((h, w), int(sep_color), dtype=int)

    for i in range(k):
        for j in range(k):
            r0 = i * (cell_h + bar_h)
            c0 = j * (cell_w + bar_w)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = int(compact[i, j])
    return grid


def _validate_generated_instance(
    grid: np.ndarray,
    compact: np.ndarray,
    sep_color: int,
    cell_h: int,
    cell_w: int,
    bar_h: int,
    bar_w: int,
) -> bool:
    k = int(compact.shape[0])
    expected_h = k * cell_h + (k - 1) * bar_h
    expected_w = k * cell_w + (k - 1) * bar_w
    if grid.shape != (expected_h, expected_w):
        return False
    if max(main(grid).shape) > 30:
        return False

    for split in range(k - 1):
        row0 = (split + 1) * cell_h + split * bar_h
        col0 = (split + 1) * cell_w + split * bar_w
        if not np.all(grid[row0 : row0 + bar_h, :] == sep_color):
            return False
        if not np.all(grid[:, col0 : col0 + bar_w] == sep_color):
            return False

    for i in range(k):
        for j in range(k):
            r0 = i * (cell_h + bar_h)
            c0 = j * (cell_w + bar_w)
            block = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            if np.unique(block).tolist() != [int(compact[i, j])]:
                return False

    expected = _stamp_from_compact(compact)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    k = random.choice((3, 4))
    compact = _sample_compact(k)
    if compact is None:
        return None

    palette = set(map(int, np.unique(compact)))
    sep_choices = [color for color in NON_BLACK_COLORS if color not in palette]
    if not sep_choices:
        return None
    sep_color = random.choice(sep_choices)

    cell_h = random.randint(2, 5)
    cell_w = random.randint(2, 5)
    bar_h = random.randint(1, 2)
    bar_w = random.randint(1, 2)
    grid = _expand_compact(compact, sep_color, cell_h, cell_w, bar_h, bar_w)

    if max(grid.shape) > 30 or max(_stamp_from_compact(compact).shape) > 30:
        return None
    if not _validate_generated_instance(grid, compact, sep_color, cell_h, cell_w, bar_h, bar_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
