from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
VOTE_COLORS = (BLUE, RED, GREEN, YELLOW)
GENERATION_ATTEMPTS = 400


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            keep.append(i)
    return keep


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(grid)
    col_starts = _run_start_indices(grid.T)
    return grid[np.ix_(row_starts, col_starts)].copy()


def _split_spans(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in dividers:
        if start < idx:
            spans.append((start, idx - 1))
        start = idx + 1
    if start < size:
        spans.append((start, size - 1))
    return spans


def _region_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _split_spans(grid.shape[0], divider_rows)
    col_spans = _split_spans(grid.shape[1], divider_cols)
    assert row_spans and col_spans, "expected at least one region"
    for r0, r1 in row_spans:
        assert r1 - r0 + 1 >= 2, "regions must be at least 2 cells tall"
    for c0, c1 in col_spans:
        assert c1 - c0 + 1 >= 2, "regions must be at least 2 cells wide"
    return row_spans, col_spans


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    row_spans, col_spans = _region_spans(grid)
    out = np.full_like(grid, BLACK)
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if divider_rows:
        out[divider_rows, :] = GREY
    if divider_cols:
        out[:, divider_cols] = GREY

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            corners = [
                int(grid[r0, c0]),
                int(grid[r0, c1]),
                int(grid[r1, c0]),
                int(grid[r1, c1]),
            ]
            if len(set(corners)) == 1 and corners[0] not in (BLACK, GREY):
                out[r0 : r1 + 1, c0 : c1 + 1] = corners[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a row/column-stretched divider grid, then fill each region by unanimous corner-marker vote.

    Input structure:
    - The input is a 2-D grid whose canonical form is a black region grid split by full grey divider rows and columns.
    - Some adjacent rows and adjacent columns are exact duplicates because the canonical grid was expanded by copying whole rows and columns.
    - Inside each rectangular black region, only the four region-corner cells may carry non-black marker colors from {blue, red, green, yellow}.

    Rule:
    - Collapse each maximal run of identical neighboring rows to its first row, and do the same for columns, to recover the canonical grid.
    - In the canonical grid, treat full grey rows and columns as dividers that partition the grid into rectangular regions.
    - For each region, inspect its four corner cells; if all four exist and share one non-black, non-grey color, fill the whole region with that color, otherwise leave the region black. Keep the grey dividers unchanged.

    Key insight:
    - Row/column duplication changes the visible corner locations, so the corner vote is only meaningful after the duplicate runs are removed and the true divider topology is restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    canonical = _compress_grid(g)
    return _solve_canonical(canonical)


def _adjacent_runs_are_unique(grid: np.ndarray) -> bool:
    for r in range(1, grid.shape[0]):
        if np.array_equal(grid[r], grid[r - 1]):
            return False
    for c in range(1, grid.shape[1]):
        if np.array_equal(grid[:, c], grid[:, c - 1]):
            return False
    return True


def _breaker_positions(region_rows: int, region_cols: int) -> list[tuple[int, int]]:
    positions = {(r, r % region_cols) for r in range(region_rows)}
    for c in range(region_cols):
        positions.add((c % region_rows, c))
    return sorted(positions)


def _sample_consistent_colors(rng: np.random.Generator, count: int) -> list[int]:
    if count == 0:
        return []
    colors = rng.choice(VOTE_COLORS, size=count, replace=True).astype(int).tolist()
    if count >= 2 and len(set(colors)) == 1:
        colors[1] = int([c for c in VOTE_COLORS if c != colors[0]][int(rng.integers(3))])
    rng.shuffle(colors)
    return colors


def _missing_pattern(rng: np.random.Generator) -> np.ndarray:
    color = int(rng.choice(VOTE_COLORS))
    if bool(rng.integers(2)):
        return np.array([[color, BLACK], [BLACK, color]], dtype=int)
    return np.array([[BLACK, color], [color, BLACK]], dtype=int)


def _mixed_pattern(rng: np.random.Generator) -> np.ndarray:
    a, b = rng.choice(VOTE_COLORS, size=2, replace=False).astype(int).tolist()
    if bool(rng.integers(2)):
        return np.array([[a, b], [b, a]], dtype=int)
    return np.array([[a, a], [b, b]], dtype=int)


def _inconsistent_pattern(rng: np.random.Generator, family: str) -> np.ndarray:
    if family == "missing":
        return _missing_pattern(rng)
    if family == "mixed":
        return _mixed_pattern(rng)
    raise AssertionError("unknown inconsistent family")


def _sample_run_lengths(rng: np.random.Generator, base_len: int) -> np.ndarray:
    while True:
        runs = rng.integers(1, 4, size=base_len, endpoint=False) + 0
        if runs.max() == 1:
            continue
        if len(set(map(int, runs))) == 1:
            continue
        if runs.sum() > 30:
            continue
        return runs.astype(int)


def _expand_grid(grid: np.ndarray, row_runs: np.ndarray, col_runs: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, row_runs, axis=0), col_runs, axis=1)


def _build_candidate(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    region_rows = int(rng.integers(2, 5))
    region_cols = int(rng.integers(2, 5))
    canonical_h = region_rows * 2 + (region_rows - 1)
    canonical_w = region_cols * 2 + (region_cols - 1)
    if canonical_h > 11 or canonical_w > 11:
        return None

    canonical = np.full((canonical_h, canonical_w), BLACK, dtype=int)
    expected = np.full_like(canonical, BLACK)

    row_starts = [r * 3 for r in range(region_rows)]
    col_starts = [c * 3 for c in range(region_cols)]
    divider_rows = [r * 3 + 2 for r in range(region_rows - 1)]
    divider_cols = [c * 3 + 2 for c in range(region_cols - 1)]
    if divider_rows:
        canonical[divider_rows, :] = GREY
        expected[divider_rows, :] = GREY
    if divider_cols:
        canonical[:, divider_cols] = GREY
        expected[:, divider_cols] = GREY

    all_regions = [(r, c) for r in range(region_rows) for c in range(region_cols)]
    breakers = _breaker_positions(region_rows, region_cols)
    breaker_set = set(breakers)
    remaining = [rc for rc in all_regions if rc not in breaker_set]
    max_extra_inconsistent = len(all_regions) - len(breakers) - 2
    if max_extra_inconsistent < 0:
        return None
    extra_count = int(rng.integers(0, max_extra_inconsistent + 1))
    extra_inconsistent = set()
    if extra_count:
        picks = rng.choice(len(remaining), size=extra_count, replace=False)
        extra_inconsistent = {remaining[int(i)] for i in np.atleast_1d(picks)}
    inconsistent = set(breakers) | extra_inconsistent
    consistent = [rc for rc in all_regions if rc not in inconsistent]
    if len(consistent) < 2 or len(inconsistent) < 1:
        return None

    consistent_colors = _sample_consistent_colors(rng, len(consistent))
    consistent_map = {rc: color for rc, color in zip(consistent, consistent_colors)}

    inconsistent_list = sorted(inconsistent)
    family_order = []
    if len(inconsistent_list) == 1:
        family_order = ["missing" if bool(rng.integers(2)) else "mixed"]
    else:
        family_order = ["missing", "mixed"]
        for _ in range(len(inconsistent_list) - 2):
            family_order.append("missing" if bool(rng.integers(2)) else "mixed")
        rng.shuffle(family_order)

    for rr, cc in consistent:
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        color = int(consistent_map[(rr, cc)])
        canonical[r0 : r0 + 2, c0 : c0 + 2] = color
        expected[r0 : r0 + 2, c0 : c0 + 2] = color

    for (rr, cc), family in zip(inconsistent_list, family_order):
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        canonical[r0 : r0 + 2, c0 : c0 + 2] = _inconsistent_pattern(rng, family)
        expected[r0 : r0 + 2, c0 : c0 + 2] = BLACK

    if not _adjacent_runs_are_unique(canonical):
        return None
    if not np.array_equal(_solve_canonical(canonical), expected):
        return None

    row_runs = _sample_run_lengths(rng, canonical.shape[0])
    col_runs = _sample_run_lengths(rng, canonical.shape[1])
    expanded = _expand_grid(canonical, row_runs, col_runs)
    if max(expanded.shape) > 30:
        return None
    if row_runs.max() == 1 or col_runs.max() == 1:
        return None
    if expanded.shape == canonical.shape:
        return None

    compressed = _compress_grid(expanded)
    if not np.array_equal(compressed, canonical):
        return None
    if np.array_equal(compressed, expected):
        return None
    if not np.array_equal(main(expanded), expected):
        return None
    return expanded, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_candidate(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC input")
