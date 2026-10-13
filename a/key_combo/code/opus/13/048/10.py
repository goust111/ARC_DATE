from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

SEED_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_RETRIES = 500


def _layout_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        spans.append((pos, pos + size))
        pos += size
        if idx < len(sizes) - 1:
            dividers.append(pos)
            pos += 1
    return spans, dividers, pos


def _divider_indices(grid: np.ndarray, axis: int) -> list[int]:
    line_mask = np.isin(grid, (GREY, PINK))
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if bool(np.all(line_mask[r, :]))]
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(line_mask[:, c]))]


def _spans_from_dividers(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for d in dividers:
        assert d - start >= 2, "each region band must be at least 2 cells thick"
        spans.append((start, d))
        start = d + 1
    assert length - start >= 2, "each region band must be at least 2 cells thick"
    spans.append((start, length))
    return spans


def _parse_regions(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    row_dividers = _divider_indices(grid, axis=0)
    col_dividers = _divider_indices(grid, axis=1)
    assert 2 <= len(row_dividers) <= 4
    assert 2 <= len(col_dividers) <= 4

    row_spans = _spans_from_dividers(grid.shape[0], row_dividers)
    col_spans = _spans_from_dividers(grid.shape[1], col_dividers)
    assert 3 <= len(row_spans) <= 5
    assert 3 <= len(col_spans) <= 5

    region_grid = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            vals = np.unique(grid[r0:r1, c0:c1])
            assert vals.size == 1, "each region interior must be uniform"
            color = int(vals[0])
            assert color in (BLACK, *SEED_COLORS), "regions may only be empty or seed-filled"
            region_grid[i, j] = color

    for r in row_dividers:
        for c in range(grid.shape[1]):
            expected = PINK if c in col_dividers else GREY
            assert int(grid[r, c]) in (GREY, PINK)
            if c not in col_dividers:
                assert int(grid[r, c]) == GREY
    for c in col_dividers:
        for r in range(grid.shape[0]):
            if r not in row_dividers:
                assert int(grid[r, c]) == GREY

    return row_spans, col_spans, region_grid


def _resolve_regions(region_grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    seeds = [
        (i, j, int(region_grid[i, j]))
        for i in range(region_grid.shape[0])
        for j in range(region_grid.shape[1])
        if int(region_grid[i, j]) != BLACK
    ]
    assert len(seeds) >= 2, "expected at least two seed regions"

    out = region_grid.copy()
    for i in range(region_grid.shape[0]):
        for j in range(region_grid.shape[1]):
            if int(region_grid[i, j]) != BLACK:
                continue
            distances = [abs(i - si) + abs(j - sj) for si, sj, _color in seeds]
            nearest = [seed for seed, dist in zip(seeds, distances) if dist == min(distances)]
            nearest_colors = {color for _si, _sj, color in nearest}
            out[i, j] = next(iter(nearest_colors)) if len(nearest_colors) == 1 else GREY
    return out, seeds


def _validate_markers(grid: np.ndarray, final_regions: np.ndarray) -> None:
    row_dividers = _divider_indices(grid, axis=0)
    col_dividers = _divider_indices(grid, axis=1)
    for i, r in enumerate(row_dividers):
        for j, c in enumerate(col_dividers):
            around = {
                int(final_regions[i, j]),
                int(final_regions[i + 1, j]),
                int(final_regions[i, j + 1]),
                int(final_regions[i + 1, j + 1]),
            }
            expected = PINK if len(around) > 1 else GREY
            assert int(grid[r, c]) == expected, "corner markers must match final region disagreements"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a divider grid into one pixel per region after assigning every empty region to its nearest seed region.

    Input structure:
    - Grey one-cell divider lines partition the canvas into a 3x3 to 5x5 grid of rectangular regions with at least 2x2 interior cells.
    - Some regions are fully filled with one seed color from {blue, red, green, yellow}; the remaining regions are black.
    - Divider intersections may carry magenta corner markers instead of grey.

    Rule:
    - Parse the divider lines to recover the region lattice and read each seed region's color.
    - For every empty region, compare its region-grid Manhattan distance to all seed regions: copy the unique nearest seed color, or use grey when the nearest tied seeds have different colors.
    - Keep original seed regions unchanged, validate that each magenta intersection marks a corner where the surrounding final region colors are not all the same, and output the final region lattice compressed to one pixel per region.

    Key insight:
    - The diffusion happens on the abstract region grid, not in pixel space, so irregular region sizes are irrelevant once the divider structure is identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    _row_spans, _col_spans, region_grid = _parse_regions(g)
    final_regions, _seeds = _resolve_regions(region_grid)
    _validate_markers(g, final_regions)
    return final_regions


def _choose_band_sizes(count: int, rng: np.random.Generator) -> list[int]:
    while True:
        sizes = [int(x) for x in rng.integers(2, 7, size=count)]
        if sum(sizes) + (count - 1) > 30:
            continue
        if max(sizes) - min(sizes) < 2:
            continue
        return sizes


def _sample_seed_positions(
    rows: int, cols: int, count: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    cells = [(i, j) for i in range(rows) for j in range(cols)]
    rng.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in seeds):
            seeds.append(cell)
        if len(seeds) == count:
            return seeds
    return None


def _assign_seed_colors(
    seed_count: int, mode: str, rng: np.random.Generator
) -> list[int]:
    num_colors = 3 if rng.random() < 0.45 else 2
    chosen = [int(c) for c in rng.choice(SEED_COLORS, size=num_colors, replace=False)]
    min_seed_count = max(num_colors + (1 if mode in {"same_tie", "mixed"} else 0), 3)
    assert seed_count >= min_seed_count

    colors = chosen[:]
    while len(colors) < seed_count:
        colors.append(int(rng.choice(chosen)))
    rng.shuffle(colors)
    return colors


def _center_rule_output(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    region_grid: np.ndarray,
) -> np.ndarray:
    centers_r = np.array([(r0 + r1 - 1) / 2.0 for r0, r1 in row_spans], dtype=float)
    centers_c = np.array([(c0 + c1 - 1) / 2.0 for c0, c1 in col_spans], dtype=float)
    seed_info = [
        (i, j, float(centers_r[i]), float(centers_c[j]), int(region_grid[i, j]))
        for i in range(region_grid.shape[0])
        for j in range(region_grid.shape[1])
        if int(region_grid[i, j]) != BLACK
    ]
    out = region_grid.copy()
    for i in range(region_grid.shape[0]):
        for j in range(region_grid.shape[1]):
            if int(region_grid[i, j]) != BLACK:
                continue
            dists = [
                abs(centers_r[i] - sr) + abs(centers_c[j] - sc)
                for _si, _sj, sr, sc, _color in seed_info
            ]
            best = min(dists)
            nearest = [
                seed for seed, dist in zip(seed_info, dists) if abs(dist - best) < 1e-9
            ]
            colors = {color for _si, _sj, _sr, _sc, color in nearest}
            out[i, j] = next(iter(colors)) if len(colors) == 1 else GREY
    return out


def _marker_grid(final_regions: np.ndarray) -> np.ndarray:
    marks = np.full((final_regions.shape[0] - 1, final_regions.shape[1] - 1), GREY, dtype=int)
    for i in range(marks.shape[0]):
        for j in range(marks.shape[1]):
            around = {
                int(final_regions[i, j]),
                int(final_regions[i + 1, j]),
                int(final_regions[i, j + 1]),
                int(final_regions[i + 1, j + 1]),
            }
            if len(around) > 1:
                marks[i, j] = PINK
    return marks


def _empty_region_ties(
    region_grid: np.ndarray, seeds: list[tuple[int, int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    diff_ties: list[tuple[int, int]] = []
    same_ties: list[tuple[int, int]] = []
    for i in range(region_grid.shape[0]):
        for j in range(region_grid.shape[1]):
            if int(region_grid[i, j]) != BLACK:
                continue
            distances = [abs(i - si) + abs(j - sj) for si, sj, _color in seeds]
            best = min(distances)
            nearest = [seed for seed, dist in zip(seeds, distances) if dist == best]
            if len(nearest) > 2:
                return [], []
            if len(nearest) == 2:
                colors = {color for _si, _sj, color in nearest}
                if len(colors) == 1:
                    same_ties.append((i, j))
                else:
                    diff_ties.append((i, j))
    return diff_ties, same_ties


def _render_input(
    row_spans: list[tuple[int, int]],
    row_dividers: list[int],
    col_spans: list[tuple[int, int]],
    col_dividers: list[int],
    region_grid: np.ndarray,
    final_regions: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((row_spans[-1][1], col_spans[-1][1]), dtype=int)
    for r in row_dividers:
        grid[r, :] = GREY
    for c in col_dividers:
        grid[:, c] = GREY
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            grid[r0:r1, c0:c1] = int(region_grid[i, j])
    marks = _marker_grid(final_regions)
    for i, r in enumerate(row_dividers):
        for j, c in enumerate(col_dividers):
            grid[r, c] = int(marks[i, j])
    return grid


def _make_candidate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    mode = str(rng.choice(["diff_tie", "same_tie", "mixed", "free"], p=[0.34, 0.24, 0.24, 0.18]))
    rows = int(rng.integers(3, 6))
    cols = int(rng.integers(3, 6))

    row_sizes = _choose_band_sizes(rows, rng)
    col_sizes = _choose_band_sizes(cols, rng)
    row_spans, row_dividers, height = _layout_from_sizes(row_sizes)
    col_spans, col_dividers, width = _layout_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    min_seed_count = 3
    if mode in {"same_tie", "mixed"}:
        min_seed_count = 4
    max_seed_count = min(5, rows * cols - 1)
    if min_seed_count > max_seed_count:
        return None

    seed_count = int(rng.integers(min_seed_count, max_seed_count + 1))
    seed_positions = _sample_seed_positions(rows, cols, seed_count, rng)
    if seed_positions is None:
        return None

    seed_colors = _assign_seed_colors(seed_count, mode, rng)
    region_grid = np.zeros((rows, cols), dtype=int)
    for (i, j), color in zip(seed_positions, seed_colors):
        region_grid[i, j] = int(color)

    final_regions, seeds = _resolve_regions(region_grid)
    diff_ties, same_ties = _empty_region_ties(region_grid, seeds)
    if final_regions.shape != (rows, cols):
        return None
    if not np.any(region_grid == BLACK):
        return None
    if np.array_equal(final_regions, region_grid):
        return None
    if len(set(map(int, np.unique(final_regions)))) < 3:
        return None
    if mode == "diff_tie" and not diff_ties:
        return None
    if mode == "same_tie" and not same_ties:
        return None
    if mode == "mixed" and (not diff_ties or not same_ties):
        return None
    if not diff_ties and not same_ties and mode != "free":
        return None

    markers = _marker_grid(final_regions)
    pink_count = int(np.count_nonzero(markers == PINK))
    if pink_count == 0 or pink_count == markers.size:
        return None

    center_out = _center_rule_output(row_spans, col_spans, region_grid)
    if np.array_equal(center_out, final_regions):
        return None

    grid = _render_input(
        row_spans=row_spans,
        row_dividers=row_dividers,
        col_spans=col_spans,
        col_dividers=col_dividers,
        region_grid=region_grid,
        final_regions=final_regions,
    )
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    return grid, final_regions


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        made = _make_candidate(rng)
        if made is None:
            continue
        grid, expected = made
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
