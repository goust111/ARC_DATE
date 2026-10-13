from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
SIGNAL_WIDTH = 3
STRIPE_COLORS = (1, 2, 3, 5, 6, 7, 8)
ALL_OFF_PATTERN = (MAROON, MAROON, MAROON)
MAX_GENERATION_ATTEMPTS = 300


def _signal_pattern(cells: np.ndarray) -> tuple[int, int, int]:
    values = tuple(int(v) for v in cells)
    assert len(values) == SIGNAL_WIDTH
    assert set(values) <= {MAROON, YELLOW}, "signal cells must be Maroon or Yellow"
    return values


def _stripe_values(cells: np.ndarray) -> list[int]:
    values = [int(v) for v in cells]
    assert len(values) == len(set(values)), "stripe colors must be pairwise distinct"
    assert not ({BLACK, MAROON, YELLOW} & set(values)), "stripe colors must exclude Black/Maroon/Yellow"
    return values


def _separator_row(g: np.ndarray) -> int:
    rows = np.where(np.all(g == BLACK, axis=1))[0]
    assert len(rows) == 1, "expected exactly one solid Black separator row"
    sep = int(rows[0])
    assert 0 < sep < g.shape[0] - 1, "separator must split legend and operation panels"
    return sep


def _apply_permutation(values: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    return values[np.array(perm, dtype=int)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each lower stripe row by the permutation indexed by its 3-light signal.

    Input structure:
    - Each row has a left stripe zone of N colored cells and a right signal zone of 3 cells.
    - Signal cells are Maroon or Yellow; stripe cells are distinct colors that are not Black, Maroon, or Yellow.
    - One solid Black row splits the grid into a top legend panel and a lower operation panel.
    - Every legend row uses the same stripe-color set, and the legend row whose signal is all Maroon shows the unpermuted reference order.

    Rule:
    - Read the all-Maroon legend row as the reference stripe order.
    - For every legend row, compare its stripe order with that reference to derive the row's column permutation.
    - For every operation row, look up the matching 3-light pattern in the legend and reorder that row's stripe cells by the same permutation.
    - Leave the legend rows, separator row, and all signal cells unchanged.

    Key insight:
    - The signal is only a table key; the actual action is the column permutation revealed by how each legend row rearranges the shared reference stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width > SIGNAL_WIDTH, "grid must contain a stripe zone and a 3-cell signal zone"

    stripe_w = width - SIGNAL_WIDTH
    sep = _separator_row(g)
    legend = g[:sep]
    assert legend.shape[0] >= 2, "legend must contain a reference row and at least one mapped row"

    legend_patterns = [_signal_pattern(row[stripe_w:]) for row in legend]
    baseline_rows = [i for i, pattern in enumerate(legend_patterns) if pattern == ALL_OFF_PATTERN]
    assert len(baseline_rows) == 1, "legend must contain exactly one all-off reference row"

    reference = legend[baseline_rows[0], :stripe_w]
    reference_values = _stripe_values(reference)
    index_by_color = {color: i for i, color in enumerate(reference_values)}

    lookup: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for row, pattern in zip(legend, legend_patterns):
        stripe_values = _stripe_values(row[:stripe_w])
        assert set(stripe_values) == set(reference_values), "every legend row must reuse the reference palette"
        perm = tuple(index_by_color[color] for color in stripe_values)
        assert pattern not in lookup, "legend signal patterns must be unique"
        lookup[pattern] = perm

    out = g.copy()
    for r in range(sep + 1, height):
        signal = _signal_pattern(g[r, stripe_w:])
        assert signal in lookup, "every operation signal must appear in the legend"
        stripe_values = np.array(_stripe_values(g[r, :stripe_w]), dtype=int)
        out[r, :stripe_w] = _apply_permutation(stripe_values, lookup[signal])

    return out


def _all_signal_patterns() -> list[tuple[int, int, int]]:
    patterns = []
    for bits in range(1 << SIGNAL_WIDTH):
        pattern = []
        for shift in range(SIGNAL_WIDTH - 1, -1, -1):
            pattern.append(YELLOW if bits & (1 << shift) else MAROON)
        patterns.append(tuple(pattern))
    return patterns


def _expected_output(
    grid: np.ndarray,
    lookup: dict[tuple[int, int, int], tuple[int, ...]],
) -> np.ndarray:
    out = grid.copy()
    stripe_w = grid.shape[1] - SIGNAL_WIDTH
    sep = _separator_row(grid)
    for r in range(sep + 1, grid.shape[0]):
        pattern = _signal_pattern(grid[r, stripe_w:])
        out[r, :stripe_w] = _apply_permutation(grid[r, :stripe_w], lookup[pattern])
    return out


def _gen_attempt() -> np.ndarray | None:
    stripe_w = random.randint(3, len(STRIPE_COLORS))
    legend_count = random.randint(3, 6)
    extra_operation_rows = random.randint(1, 4)
    operation_count = legend_count + extra_operation_rows

    reference = np.array(random.sample(STRIPE_COLORS, stripe_w), dtype=int)
    identity = tuple(range(stripe_w))
    all_perms = list(permutations(range(stripe_w)))
    nontrivial = [perm for perm in all_perms if perm != identity]

    other_patterns = [pattern for pattern in _all_signal_patterns() if pattern != ALL_OFF_PATTERN]
    chosen_patterns = [ALL_OFF_PATTERN, *random.sample(other_patterns, legend_count - 1)]
    chosen_perms = [identity, *random.sample(nontrivial, legend_count - 1)]
    lookup = dict(zip(chosen_patterns, chosen_perms))

    legend_items = list(lookup.items())
    random.shuffle(legend_items)
    legend_rows = []
    for pattern, perm in legend_items:
        left = _apply_permutation(reference, perm)
        right = np.array(pattern, dtype=int)
        legend_rows.append(np.concatenate([left, right]))

    operation_patterns = chosen_patterns[:]
    for _ in range(extra_operation_rows):
        operation_patterns.append(random.choice(chosen_patterns))
    random.shuffle(operation_patterns)

    operation_rows = []
    for pattern in operation_patterns:
        left = np.array(random.sample(STRIPE_COLORS, stripe_w), dtype=int)
        right = np.array(pattern, dtype=int)
        operation_rows.append(np.concatenate([left, right]))

    separator = np.full((1, stripe_w + SIGNAL_WIDTH), BLACK, dtype=int)
    grid = np.vstack([np.array(legend_rows, dtype=int), separator, np.array(operation_rows, dtype=int)])

    if max(grid.shape) > 30:
        return None

    expected = _expected_output(grid, lookup)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-lookup permutation puzzle")
