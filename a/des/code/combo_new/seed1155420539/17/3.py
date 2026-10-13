from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

COLOR_COUNT_RANGE = (4, 8)
PARTITION_COUNT_RANGE = (2, 5)
PARTITION_WIDTH_RANGE = (2, 6)
GENERATION_MODES = ("all_full", "all_missing", "mixed")
MAX_GENERATION_ATTEMPTS = 400


def _partition_runs(body: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    width = int(body.shape[1])
    while start < width:
        end = start + 1
        while end < width and np.array_equal(body[:, end], body[:, start]):
            end += 1
        runs.append((start, end))
        start = end

    assert len(runs) >= 2, "expected multiple hidden partitions"
    run_widths = {end - start for start, end in runs}
    assert len(run_widths) == 1, "all partitions must have the same inferred width"
    return runs


def _sorted_partition_column(order: list[int], column: np.ndarray) -> np.ndarray:
    present = [int(value) for value in column if int(value) != BG]
    assert len(present) == len(set(present)), "partition strips must use distinct colors"
    assert set(present) <= set(order), "partition colors must come from the marker palette"

    sorted_present = [color for color in order if color in set(present)]
    out = np.full_like(column, BG)
    out[: len(sorted_present)] = np.asarray(sorted_present, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack each hidden partition's horizontal strips into the left-column color order.

    Input structure:
    - Black background.
    - The leftmost column is a vertical marker that lists one non-black color per row.
    - The remaining area is tiled by equal-width rectangular partitions whose boundaries are not drawn.
    - Inside each partition, every occupied row is a one-pixel-tall monochrome bar spanning the full partition width, while missing colors appear as black rows.

    Rule:
    - Read the leftmost column from top to bottom to obtain the target strip order.
    - Infer partition boundaries by grouping consecutive identical full-height columns; each equal-width run is one partition.
    - For each partition, keep only the colors that are present, reorder them by the left-column order, and pack them upward without gaps.
    - Leave the leftmost marker column unchanged.

    Key insight:
    - The sorting key is explicit in the marker column, but the sortable objects live inside implicit same-width column blocks that must be recovered from repeated column vectors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 3, "grid is too small for a marker and multiple partitions"

    order = [int(value) for value in g[:, 0]]
    assert all(color != BG for color in order), "marker column must be fully colored"
    assert len(order) == len(set(order)), "marker colors must define a unique visible order"

    body = g[:, 1:]
    runs = _partition_runs(body)
    out = g.copy()

    for start, end in runs:
        block = body[:, start:end]
        reference = block[:, 0]
        assert np.all(block == reference[:, None]), "each strip row must span the full partition width"

        sorted_column = _sorted_partition_column(order, reference)
        out[:, start + 1 : end + 1] = sorted_column[:, None]

    return out


def _choose_dimensions() -> tuple[int, int, int]:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    partition_count = random.randint(*PARTITION_COUNT_RANGE)
    max_width = min(PARTITION_WIDTH_RANGE[1], (30 - 1) // partition_count)
    min_width = PARTITION_WIDTH_RANGE[0]
    assert min_width <= max_width
    partition_width = random.randint(min_width, max_width)
    return color_count, partition_count, partition_width


def _sample_partition_vector(order: list[int], is_full: bool) -> np.ndarray:
    height = len(order)
    if is_full:
        chosen_colors = order[:]
    else:
        size = random.randint(1, height - 1)
        chosen_colors = random.sample(order, size)

    random.shuffle(chosen_colors)
    rows = random.sample(range(height), len(chosen_colors))
    vector = np.full(height, BG, dtype=int)
    for row, color in zip(rows, chosen_colors):
        vector[int(row)] = int(color)
    return vector


def _sample_partition_vectors(order: list[int], partition_count: int, mode: str) -> list[np.ndarray] | None:
    if mode == "all_full":
        full_flags = [True] * partition_count
    elif mode == "all_missing":
        full_flags = [False] * partition_count
    else:
        full_count = random.randint(1, partition_count - 1)
        full_indices = set(random.sample(range(partition_count), full_count))
        full_flags = [idx in full_indices for idx in range(partition_count)]

    vectors: list[np.ndarray] = []
    for is_full in full_flags:
        for _ in range(50):
            candidate = _sample_partition_vector(order, is_full)
            if vectors and np.array_equal(candidate, vectors[-1]):
                continue
            vectors.append(candidate)
            break
        else:
            return None

    used_colors = {int(value) for vector in vectors for value in vector if int(value) != BG}
    if used_colors != set(order):
        return None
    return vectors


def _render_input(order: list[int], vectors: list[np.ndarray], partition_width: int) -> np.ndarray:
    height = len(order)
    width = 1 + len(vectors) * partition_width
    grid = np.full((height, width), BG, dtype=int)
    grid[:, 0] = np.asarray(order, dtype=int)

    for idx, vector in enumerate(vectors):
        c0 = 1 + idx * partition_width
        grid[:, c0 : c0 + partition_width] = vector[:, None]
    return grid


def _gen_attempt() -> np.ndarray | None:
    color_count, partition_count, partition_width = _choose_dimensions()
    order = random.sample(PALETTE, color_count)
    mode = random.choice(GENERATION_MODES)

    vectors = _sample_partition_vectors(order, partition_count, mode)
    if vectors is None:
        return None

    grid = _render_input(order, vectors, partition_width)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-partition sorting input")
