from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the scene at the red divider and output the union of the two orange masks in blue.

    Input structure:
    - The grid contains two same-sized black/orange masks.
    - A single full-height red column separates the left and right masks.

    Rule:
    - Locate the red separator column and split the input into the two aligned masks.
    - For each corresponding cell, output blue if either mask has an orange pixel there.
    - Output black where both masks have black.

    Key insight:
    - The red column is only a delimiter; the visible operation is a pixelwise OR of the two
      aligned orange masks, with active pixels recolored blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_cols = np.flatnonzero(np.all(g == Color.RED, axis=0))
    assert len(separator_cols) == 1, "expected one full-height red separator column"
    c = int(separator_cols[0])

    left = g[:, :c]
    right = g[:, c + 1 :]
    assert left.shape == right.shape, "expected equal-sized masks on both sides"
    assert left.size > 0, "expected nonempty masks"
    assert set(np.unique(left)) <= {Color.BLACK, Color.ORANGE}
    assert set(np.unique(right)) <= {Color.BLACK, Color.ORANGE}

    out = np.full(left.shape, Color.BLACK, dtype=int)
    out[(left == Color.ORANGE) | (right == Color.ORANGE)] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
MIN_HEIGHT, MAX_HEIGHT, MIN_WIDTH, MAX_WIDTH = 3, 7, 4, 8
# Density bounds.
UNION_DENSITY_RANGE, MASK_DENSITY_RANGE = (0.65, 0.80), (0.30, 0.60)
# Structural count constraints.
MIN_UNION_CELLS, MIN_MASK_CELLS, MIN_OVERLAP_CELLS = 3, 1, 1
MIN_LEFT_ONLY_CELLS, MIN_RIGHT_ONLY_CELLS = 1, 1
# Palette and retry budget.
OUTPUT_COLOR, SEPARATOR_COLOR, MASK_COLOR = Color.BLUE, Color.RED, Color.ORANGE
MAX_GENERATION_ATTEMPTS = 300


def _choose_mask_counts(n: int, union_count: int) -> tuple[int, int] | None:
    """Sample left/right counts that keep both unique sides and overlap."""
    lo, hi = MASK_DENSITY_RANGE
    mask_lo = max(MIN_MASK_CELLS, int(np.ceil(lo * n)))
    mask_hi = int(np.floor(hi * n))
    feasible_counts = [
        (left_count, right_count)
        for left_count in range(mask_lo, min(mask_hi, union_count - MIN_RIGHT_ONLY_CELLS) + 1)
        for right_count in range(mask_lo, min(mask_hi, union_count - MIN_LEFT_ONLY_CELLS) + 1)
        if left_count + right_count >= union_count + MIN_OVERLAP_CELLS
    ]
    if not feasible_counts: return None
    return feasible_counts[int(np.random.randint(len(feasible_counts)))]


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(MIN_HEIGHT, MAX_HEIGHT + 1))
    w = int(np.random.randint(MIN_WIDTH, MAX_WIDTH + 1))
    n = h * w

    lo, hi = UNION_DENSITY_RANGE
    union_lo = max(MIN_UNION_CELLS, int(np.ceil(lo * n)))
    union_hi = int(np.floor(hi * n))
    if union_lo > union_hi: return None

    k = int(np.random.randint(union_lo, union_hi + 1))
    counts = _choose_mask_counts(n, k)
    if counts is None: return None

    left_count, right_count = counts
    left_only_count = k - right_count
    right_only_count = k - left_count

    out = np.zeros((h, w), dtype=int)
    out.reshape(-1)[np.random.choice(n, size=k, replace=False)] = OUTPUT_COLOR
    active = np.flatnonzero(out.reshape(-1))
    np.random.shuffle(active)

    left, right = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    left_only = active[:left_only_count]
    right_only = active[left_only_count : left_only_count + right_only_count]
    both = active[left_only_count + right_only_count :]

    left[left_only] = MASK_COLOR
    right[right_only] = MASK_COLOR
    left[both] = MASK_COLOR
    right[both] = MASK_COLOR

    left, right = left.reshape(h, w), right.reshape(h, w)
    sep = np.full((h, 1), SEPARATOR_COLOR, dtype=int)
    inp = np.concatenate([left, sep, right], axis=1)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 195ba7dc input")
