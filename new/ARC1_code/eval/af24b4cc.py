from collections import Counter

import numpy as np

from arc_utils import get_bg
import random


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs = []
    start = None
    prev = None
    for idx in indices:
        if start is None:
            start = prev = int(idx)
        elif idx == prev + 1:
            prev = int(idx)
        else:
            runs.append((start, prev))
            start = prev = int(idx)
    if start is not None:
        runs.append((start, prev))
    return runs


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a nonempty block"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each foreground block in a background-separated lattice to its unique majority color, then add a background border.

    Input structure
    - The dominant background color forms complete separator rows and columns,
      including the outer border.
    - The separator lines divide the foreground into a rectangular lattice of
      same-sized filled blocks.
    - Every block has one unique majority color and contains no background cells.

    Rule
    - Find the contiguous non-separator row bands and column bands.
    - For each band intersection, write the unique majority color of that input
      block to the corresponding interior cell of the output.
    - Leave a one-cell background border around the summarized lattice.

    Key insight
    - The output is a coarse color map of the block lattice; the internal shape
      noise inside each block is discarded by taking the block majority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    full_bg_rows = np.all(g == bg, axis=1)
    full_bg_cols = np.all(g == bg, axis=0)
    assert full_bg_rows[0] and full_bg_rows[-1], "expected background top and bottom borders"
    assert full_bg_cols[0] and full_bg_cols[-1], "expected background left and right borders"

    row_runs = _runs([r for r, is_bg in enumerate(full_bg_rows) if not is_bg])
    col_runs = _runs([c for c, is_bg in enumerate(full_bg_cols) if not is_bg])
    assert row_runs and col_runs, "expected at least one foreground block"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1, "expected equal-height row bands"
    assert len(col_widths) == 1, "expected equal-width column bands"

    out = np.full((len(row_runs) + 2, len(col_runs) + 2), bg, dtype=int)
    for out_r, (r0, r1) in enumerate(row_runs, start=1):
        for out_c, (c0, c1) in enumerate(col_runs, start=1):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == bg), "expected foreground blocks with no background holes"
            out[out_r, out_c] = _unique_mode(block)

    return out

# --------------- generator (o5x) ---------------
# Lattice geometry.
BG, SEP = 0, 1
ROWS, COLS = 2, 3
BLOCK_H, BLOCK_W = 3, 2

# Color sampling.
PALETTE = tuple(range(1, 10))
TARGET_DISTINCT = (7, 8)
MAJORITY_DISTINCT_RANGE = (4, 6)

# Block noise and retry limits.
FORCED_NOISE_COUNTS, OPTIONAL_NOISE_COUNTS = (1, 2), (0, 1, 2)
MAX_ATTEMPTS = 300


def _make_output_and_pools():
    """Return the oracle output, required minority colors, and allowed noise colors."""
    out = np.full((ROWS + 2, COLS + 2), BG, dtype=int)
    target_distinct = random.choice(TARGET_DISTINCT)
    majority_pool = random.sample(PALETTE, random.randint(*MAJORITY_DISTINCT_RANGE))
    positions = list(range(ROWS * COLS))
    random.shuffle(positions)
    for color, pos in zip(majority_pool, positions):
        out[1 + pos // COLS, 1 + pos % COLS] = color
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if out[r, c] == BG:
                out[r, c] = random.choice(majority_pool)
    random.shuffle(majority_pool)
    interior_colors = set(map(int, out[1:-1, 1:-1].ravel()))
    missing = [color for color in PALETTE if color not in interior_colors]
    minority_pool = random.sample(missing, max(0, target_distinct - len(interior_colors)))
    return out, minority_pool, interior_colors | set(minority_pool)


def _render_block(majority, minority, noise_count):
    block = np.full((BLOCK_H, BLOCK_W), majority, dtype=int)
    if noise_count:
        cells = np.array(random.sample(range(BLOCK_H * BLOCK_W), noise_count), dtype=int)
        block[cells // BLOCK_W, cells % BLOCK_W] = minority
    return block


def _render_input(out, minority_pool, target_colors):
    """Interleave noisy blocks with fixed-width background separators."""
    height = ROWS * BLOCK_H + (ROWS + 1) * SEP
    width = COLS * BLOCK_W + (COLS + 1) * SEP
    grid = np.full((height, width), BG, dtype=int)
    pending_minorities = minority_pool[:]
    block_positions = [(br, bc) for br in range(ROWS) for bc in range(COLS)]
    random.shuffle(block_positions)
    for br, bc in block_positions:
        majority = int(out[br + 1, bc + 1])
        if pending_minorities:
            minority = pending_minorities.pop()
            noise_count = random.choice(FORCED_NOISE_COUNTS)
        else:
            noise_count = random.choice(OPTIONAL_NOISE_COUNTS)
            choices = [color for color in target_colors if color != majority]
            minority = random.choice(choices)
        r0 = SEP + br * (BLOCK_H + SEP)
        c0 = SEP + bc * (BLOCK_W + SEP)
        grid[r0 : r0 + BLOCK_H, c0 : c0 + BLOCK_W] = _render_block(majority, minority, noise_count)
    return grid


def _gen_attempt() -> np.ndarray | None:
    out, minority_pool, target_colors = _make_output_and_pools()
    grid = _render_input(out, minority_pool, target_colors)
    if np.array_equal(main(grid), out):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid af24b4cc input")
