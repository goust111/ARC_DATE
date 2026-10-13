from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GEN_MAX_ATTEMPTS = 400


def _nondivider_intervals(full_black_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal non-divider intervals as half-open [start, end) slices."""
    intervals: list[tuple[int, int]] = []
    start = 0
    i = 0
    size = int(full_black_mask.shape[0])
    while i < size:
        if full_black_mask[i]:
            if start < i:
                intervals.append((start, i))
            while i < size and full_black_mask[i]:
                i += 1
            start = i
        else:
            i += 1
    if start < size:
        intervals.append((start, size))
    return intervals


def _compress_seed(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_dividers = np.all(g == BLACK, axis=1)
    col_dividers = np.all(g == BLACK, axis=0)
    row_intervals = _nondivider_intervals(row_dividers)
    col_intervals = _nondivider_intervals(col_dividers)

    assert row_intervals and col_intervals
    assert len(row_intervals) == len(col_intervals), "expected a square region grid"

    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "regions must be equal-sized"

    n = len(row_intervals)
    seed = np.zeros((n, n), dtype=int)
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            region = g[r0:r1, c0:c1]
            colors = sorted(set(map(int, region.flatten())) - {BLACK})
            assert len(colors) <= 1, "each region must use at most one non-black color"
            seed[i, j] = colors[0] if colors else BLACK
    return seed


def _expand_inverted(seed: np.ndarray) -> np.ndarray:
    s = np.asarray(seed, dtype=int)
    assert s.ndim == 2 and s.shape[0] == s.shape[1]
    n = int(s.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for br in range(n):
        for bc in range(n):
            if s[br, bc] == BLACK:
                out[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = s
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a partitioned color grid, then invert the seed mask during self-Kronecker expansion.

    Input structure:
    - Full-black rows and columns partition the grid into an N x N array of equal-sized rectangular regions.
    - Each region is either fully black or contains scattered pixels of exactly one non-black color.
    - The partition layout, not the scatter pattern, defines the compressed seed coordinates.

    Rule:
    - Detect the divider rows and columns, then reduce each region to one seed cell using its unique non-black color, or black if the region is empty.
    - Treat the seed's non-black cells as a foreground mask over an N x N block layout.
    - Build an (N*N) x (N*N) output where foreground-mask blocks become all black, while background-mask blocks are filled with a full copy of the seed grid.

    Key insight:
    - The compressed seed is reused twice: its non-black support chooses which Kronecker blocks are erased, and its full color pattern is pasted only into the seed's own background positions.
    """
    seed = _compress_seed(grid)
    return _expand_inverted(seed)


def _sample_seed(rng: random.Random, n: int) -> np.ndarray | None:
    mask = np.array(
        [[rng.random() < rng.uniform(0.45, 0.72) for _ in range(n)] for _ in range(n)],
        dtype=bool,
    )
    nonzero_count = int(mask.sum())
    if nonzero_count == 0 or nonzero_count == n * n:
        return None
    if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
        return None

    palette_size = rng.randint(1, min(4, nonzero_count))
    palette = rng.sample(COLORS, palette_size)
    seed = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            if mask[r, c]:
                seed[r, c] = rng.choice(palette)

    # Avoid overly trivial instances where every colored cell is identical and there is only one hole.
    if np.count_nonzero(seed == BLACK) == 1 and len(set(map(int, seed[seed != BLACK]))) == 1:
        return None
    return seed


def _region_pattern(rng: random.Random, size: int, color: int) -> np.ndarray:
    """
    Make a scattered monochrome region that touches every local row and every local column.

    This prevents accidental extra full-black divider rows/cols outside the intended separators.
    """
    patch = np.zeros((size, size), dtype=int)
    perm = list(range(size))
    rng.shuffle(perm)
    for r, c in enumerate(perm):
        patch[r, c] = color

    remaining = [(r, c) for r in range(size) for c in range(size) if patch[r, c] == BLACK]
    rng.shuffle(remaining)
    extra = rng.randint(0, len(remaining))
    for r, c in remaining[:extra]:
        patch[r, c] = color
    return patch


def _render_input(seed: np.ndarray, cell_size: int, rng: random.Random) -> np.ndarray:
    n = int(seed.shape[0])
    grid_size = n * cell_size + (n - 1)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(n):
        for c in range(n):
            color = int(seed[r, c])
            if color == BLACK:
                continue
            r0 = r * (cell_size + 1)
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = _region_pattern(
                random.Random(rng.randint(0, 10**9)),
                cell_size,
                color,
            )
    return grid


def _expected_output_from_seed(seed: np.ndarray) -> np.ndarray:
    return _expand_inverted(seed)


def _intended_dividers(n: int, cell_size: int) -> list[int]:
    return [(cell_size + 1) * k - 1 for k in range(1, n)]


def _semantic_checks(inp: np.ndarray, seed: np.ndarray, cell_size: int) -> bool:
    n = int(seed.shape[0])
    expected_dividers = _intended_dividers(n, cell_size)
    actual_row_dividers = np.flatnonzero(np.all(inp == BLACK, axis=1)).tolist()
    actual_col_dividers = np.flatnonzero(np.all(inp == BLACK, axis=0)).tolist()
    if actual_row_dividers != expected_dividers or actual_col_dividers != expected_dividers:
        return False

    if not np.array_equal(_compress_seed(inp), seed):
        return False

    expected = _expected_output_from_seed(seed)
    if not np.array_equal(main(inp), expected):
        return False

    if np.any(seed.sum(axis=1) == 0) or np.any(seed.sum(axis=0) == 0):
        return False
    if not np.any(seed == BLACK) or not np.any(seed != BLACK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.choice((3, 4))
    cell_size = rng.choice((3, 4))
    seed = _sample_seed(rng, n)
    if seed is None:
        return None

    grid_size = n * cell_size + (n - 1)
    if grid_size > 30 or n * n > 30:
        return None

    inp = _render_input(seed, cell_size, rng)
    if not _semantic_checks(inp, seed, cell_size):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
