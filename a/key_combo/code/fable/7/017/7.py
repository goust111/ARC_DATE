"""ARC task 7: reconstruct a linked list encoded by vertical color stripes."""

from __future__ import annotations

import numpy as np


def _strict_mode(values: np.ndarray) -> int:
    """Return the unique most frequent integer in a one-dimensional array."""
    colors, counts = np.unique(values, return_counts=True)
    best = int(counts.max())
    winners = colors[counts == best]
    assert len(winners) == 1, "stripe main color must be visually unambiguous"
    return int(winners[0])


def _vertical_stripes(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Find (left, right-exclusive, main-color) stripes from column modes."""
    h, w = grid.shape
    assert h >= 2 and w >= 2
    column_colors = [_strict_mode(grid[:, c]) for c in range(w)]

    runs: list[tuple[int, int, int]] = []
    start = 0
    for c in range(1, w + 1):
        if c == w or column_colors[c] != column_colors[start]:
            runs.append((start, c, column_colors[start]))
            start = c

    assert len(runs) >= 2, "expected several vertical stripes"
    widths = {right - left for left, right, _ in runs}
    assert len(widths) == 1, "vertical stripes must have equal widths"
    mains = [color for _, _, color in runs]
    assert len(set(mains)) == len(mains), "each stripe needs a unique main color"
    return runs


def _chain_order(
    grid: np.ndarray, stripes: list[tuple[int, int, int]]
) -> list[int]:
    """Return source-stripe indices in the order encoded by foreign colors."""
    main_to_index = {main: i for i, (_, _, main) in enumerate(stripes)}
    parent: dict[int, int] = {}
    heads: list[int] = []

    for i, (left, right, main) in enumerate(stripes):
        visible = {int(v) for v in np.unique(grid[:, left:right])}
        foreign = visible - {main}
        assert len(foreign) <= 1, "a stripe may contain only one marker color"
        if not foreign:
            heads.append(i)
            continue
        marker = foreign.pop()
        assert marker in main_to_index, "marker must name another stripe's main color"
        parent_i = main_to_index[marker]
        assert parent_i != i
        parent[i] = parent_i

    assert len(heads) == 1, "the chain must have exactly one unmarked head"
    assert len(parent) == len(stripes) - 1

    # In a single unbranched list each stripe has at most one direct successor.
    child: dict[int, int] = {}
    for node, parent_i in parent.items():
        assert parent_i not in child, "marker relation must not branch"
        child[parent_i] = node

    order = [heads[0]]
    seen = {heads[0]}
    while order[-1] in child:
        nxt = child[order[-1]]
        assert nxt not in seen, "marker relation must not contain a cycle"
        order.append(nxt)
        seen.add(nxt)
    assert len(order) == len(stripes), "markers must form one complete chain"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder intact vertical stripes by following their distributed left-neighbor markers.

    Input structure:
    - The grid is partitioned into several equal-width vertical stripes, each with
      a unique visually dominant main color.
    - Exactly one stripe is solid.  Every other stripe contains one foreign
      marker color, which is the main color of another stripe.
    - These marker relations form one unbranched, acyclic chain.

    Rule:
    - A stripe whose marker has color X must be immediately right of the stripe
      whose main color is X.  Put the unmarked stripe first, follow successors,
      and move every full stripe to its resulting slot without changing any of
      its pixels.  Output dimensions equal input dimensions.

    Key insight:
    - Each foreign pixel is a distributed pointer to that stripe's left
      neighbor; the complete order must be reconstructed by following the chain,
      while marker pixels travel with their containing stripe.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional grid"
    stripes = _vertical_stripes(g)
    order = _chain_order(g, stripes)
    pieces = [g[:, stripes[i][0] : stripes[i][1]] for i in order]
    return np.concatenate(pieces, axis=1).copy()


def _nontrivial_physical_order(rng: np.random.Generator, n: int) -> list[int]:
    """Choose an order that is neither the answer nor a simple rotation/reversal."""
    target = tuple(range(n))
    forbidden = {tuple(target[k:] + target[:k]) for k in range(n)}
    rev = tuple(reversed(target))
    forbidden |= {tuple(rev[k:] + rev[:k]) for k in range(n)}
    for _ in range(100):
        candidate = tuple(int(x) for x in rng.permutation(n))
        if candidate not in forbidden:
            return list(candidate)
    raise RuntimeError("could not sample a nontrivial stripe permutation")


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked linked-stripe instance."""
    rng = np.random.default_rng()
    n = int(rng.integers(4, 8))
    max_width = min(5, 30 // n)
    stripe_width = int(rng.integers(2, max_width + 1))
    height = int(rng.integers(7, 19))

    # Numeric color order carries no meaning: this random list is the hidden chain.
    chain_colors = [int(x) for x in rng.choice(10, size=n, replace=False)]
    chain_blocks: list[np.ndarray] = []
    forced_multi_marker = int(rng.integers(1, n))

    for chain_i, main_color in enumerate(chain_colors):
        block = np.full((height, stripe_width), main_color, dtype=int)
        if chain_i > 0:
            max_markers = min(3, stripe_width)
            marker_count = (
                max_markers
                if chain_i == forced_multi_marker
                else int(rng.integers(1, max_markers + 1))
            )
            # Distinct columns and interior, independently shuffled rows keep every
            # column's strict mode equal to the stripe's main color.
            columns = rng.choice(stripe_width, size=marker_count, replace=False)
            rows = rng.choice(np.arange(1, height - 1), size=marker_count, replace=False)
            rows = rows[rng.permutation(marker_count)]
            marker_color = chain_colors[chain_i - 1]
            for r, c in zip(rows, columns):
                block[int(r), int(c)] = marker_color
        chain_blocks.append(block)

    physical_order = _nontrivial_physical_order(rng, n)
    grid = np.concatenate([chain_blocks[i] for i in physical_order], axis=1)
    expected = np.concatenate(chain_blocks, axis=1)

    # Independent semantic checks: dimensions, true movement, intact blocks,
    # unique complete chain, and agreement with the solver.
    assert max(grid.shape) <= 30
    assert not np.array_equal(grid, expected)
    stripes = _vertical_stripes(grid)
    recovered = _chain_order(grid, stripes)
    recovered_colors = [stripes[i][2] for i in recovered]
    assert recovered_colors == chain_colors
    solved = main(grid)
    assert np.array_equal(solved, expected)
    width = stripe_width
    for i, source_i in enumerate(recovered):
        left, right, _ = stripes[source_i]
        assert np.array_equal(solved[:, i * width : (i + 1) * width], grid[:, left:right])
    return grid
