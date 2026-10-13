from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
STRIPE_COLORS = (1, 2, 3, 4, 5)
MAX_GEN_ATTEMPTS = 300


def _column_kind(col: np.ndarray) -> str:
    vals = set(map(int, np.unique(col)))
    if len(vals) == 1 and BLACK not in vals and ORANGE not in vals:
        return "stripe"
    if vals <= {BLACK, ORANGE}:
        return "gap"
    raise ValueError(f"unsupported mixed column values: {sorted(vals)}")


def _extract_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    kinds = [_column_kind(grid[:, c]) for c in range(grid.shape[1])]
    runs: list[tuple[str, int, int]] = []
    start = 0
    for c in range(1, len(kinds) + 1):
        if c == len(kinds) or kinds[c] != kinds[start]:
            runs.append((kinds[start], start, c - 1))
            start = c

    if not runs or runs[0][0] != "stripe" or runs[-1][0] != "stripe":
        raise ValueError("columns must start and end with stripes")

    stripes = [(c0, c1) for kind, c0, c1 in runs if kind == "stripe"]
    gaps = [(c0, c1) for kind, c0, c1 in runs if kind == "gap"]

    if len(stripes) < 2 or len(gaps) != len(stripes) - 1:
        raise ValueError("expected alternating stripe/gap runs")
    if any(runs[i][0] == runs[i + 1][0] for i in range(len(runs) - 1)):
        raise ValueError("runs must alternate")

    stripe_widths = {c1 - c0 + 1 for c0, c1 in stripes}
    if len(stripe_widths) != 1:
        raise ValueError("all stripes must have equal width")

    return stripes, gaps


def _gap_is_open(gap_block: np.ndarray) -> bool:
    """
    A gap is open iff its black channel contains an 8-connected path
    from the top edge of the grid to the bottom edge.
    """
    h, w = gap_block.shape
    open_mask = gap_block == BLACK
    seen = np.zeros((h, w), dtype=bool)
    stack = [(0, c) for c in range(w) if open_mask[0, c]]

    while stack:
        r, c = stack.pop()
        if seen[r, c]:
            continue
        seen[r, c] = True
        if r == h - 1:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and open_mask[rr, cc] and not seen[rr, cc]:
                    stack.append((rr, cc))
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder equal-width vertical color stripes by counting which inter-stripe channels remain open from top to bottom.

    Input structure:
    - The grid has a black background with 4-5 vertical solid color stripes, separated by black 2-column gaps.
    - Each stripe is filled with one non-black main color and all stripes have equal width.
    - Some gaps have orange cells filling their top 1-2 rows, while other gaps remain fully black.

    Rule:
    - Treat each gap as the channel induced by the two neighboring stripe-edge columns and test whether its black cells contain a continuous top-to-bottom path.
    - Count the number M of gaps whose channel is fully open from the top edge to the bottom edge.
    - If M = 0, keep the stripe order unchanged.
    - If M >= 1, move the leftmost M stripes to the far right, preserving the relative order within both moved and unmoved groups.
    - Keep every stripe's internal contents unchanged, and leave each gap's black/orange blockage pattern fixed in place.

    Key insight:
    - The orange blockage pattern is not output decoration; it implicitly encodes a count of open inter-stripe channels, and that count alone determines the stripe rotation.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    stripes, gaps = _extract_runs(g)
    stripe_blocks = [g[:, c0 : c1 + 1].copy() for c0, c1 in stripes]
    open_count = sum(_gap_is_open(g[:, c0 : c1 + 1]) for c0, c1 in gaps)

    if open_count == 0:
        return g.copy()

    shift = open_count % len(stripe_blocks)
    reordered = stripe_blocks[shift:] + stripe_blocks[:shift]

    out = g.copy()
    for (c0, c1), block in zip(stripes, reordered):
        out[:, c0 : c1 + 1] = block
    return out


def _build_expected(
    height: int,
    stripe_width: int,
    stripe_colors: list[int],
    open_gap_indices: set[int],
    blocked_heights: dict[int, int],
) -> np.ndarray:
    n = len(stripe_colors)
    width = n * stripe_width + (n - 1) * 2
    grid = np.zeros((height, width), dtype=int)

    c = 0
    for i, color in enumerate(stripe_colors):
        grid[:, c : c + stripe_width] = color
        c += stripe_width
        if i == n - 1:
            break
        if i not in open_gap_indices:
            block_h = blocked_heights[i]
            grid[:block_h, c : c + 2] = ORANGE
        c += 2
    return grid


def _sample_open_gaps(n_gaps: int) -> set[int]:
    open_count = random.randint(1, n_gaps - 1)
    return set(random.sample(range(n_gaps), open_count))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        n_stripes = random.choice((4, 5))
        n_gaps = n_stripes - 1
        height = random.randint(5, 8)
        stripe_width = random.randint(2, 4)
        width = n_stripes * stripe_width + n_gaps * 2
        if width > 30:
            continue

        stripe_colors = random.sample(STRIPE_COLORS, n_stripes)
        open_gaps = _sample_open_gaps(n_gaps)
        blocked_heights = {
            gap_idx: random.randint(1, min(2, height - 1))
            for gap_idx in range(n_gaps)
            if gap_idx not in open_gaps
        }

        grid = _build_expected(height, stripe_width, stripe_colors, open_gaps, blocked_heights)

        if not any(grid[:, c].min() == grid[:, c].max() != BLACK and grid[:, c].min() != ORANGE for c in range(grid.shape[1])):
            continue
        if not any(np.all(grid[:, c0 : c1 + 1] == BLACK) for c0, c1 in _extract_runs(grid)[1]):
            continue
        if not any(np.any(grid[:, c0 : c1 + 1] == ORANGE) for c0, c1 in _extract_runs(grid)[1]):
            continue

        out = main(grid)
        if out.shape == grid.shape and not np.array_equal(out, grid):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
