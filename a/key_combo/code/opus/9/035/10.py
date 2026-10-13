from __future__ import annotations

from collections import Counter

import numpy as np


GRID_SHAPE = (10, 8)
PALETTE = (1, 2, 3, 4, 7)
MAX_GENERATION_ATTEMPTS = 400
DIFF_PATTERNS = {
    3: ((4, 3, 1), (5, 3, 2), (6, 4, 2), (7, 4, 3)),
    4: ((4, 3, 2, 1), (5, 4, 2, 1), (6, 3, 2, 1), (5, 4, 3, 2)),
}


def _pack_counts(counts: Counter[int], shape: tuple[int, int]) -> np.ndarray:
    """Pack color runs column-major, bottom-to-top inside each column."""
    height, width = map(int, shape)
    out = np.zeros((height, width), dtype=int)
    ranked = sorted(counts, key=lambda color: -counts[color])
    scan = ((r, c) for c in range(width) for r in range(height - 1, -1, -1))
    for color in ranked:
        for _ in range(int(counts[color])):
            r, c = next(scan)
            out[r, c] = int(color)
    return out


def _difference_counts(grid: np.ndarray) -> Counter[int]:
    """Return original-grid color counts at cells that disagree with their vertical flip."""
    flipped = np.flipud(grid)
    diff_mask = grid != flipped
    diff_colors = grid[diff_mask]
    assert diff_colors.size > 0, "expected at least one asymmetric pixel"
    assert bool(np.all(diff_colors != 0)), "difference pixels must be non-black for a visible output"
    counts = Counter(map(int, diff_colors.tolist()))
    assert len(set(counts.values())) == len(counts), "difference-color frequencies must have a unique rank"
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect vertically asymmetric pixels via row reversal, then pack their colors by descending frequency.

    Input structure:
    - A 2D grid with black background and several non-black pixels.
    - Most colored pixels are vertically mirrored by row reversal, but a small set of mirrored positions disagree.

    Rule:
    - Reverse the row order to form the comparison grid.
    - Mark every coordinate where the original grid and the reversed grid have different colors.
    - Collect the original-grid colors at those differing coordinates and count each color.
    - Build an all-black output of the same shape, then place the colors in descending frequency order while scanning columns left-to-right and cells bottom-to-top within each column.

    Key insight:
    - The vertical mirror is only a detector: once asymmetric pixels are found, the output keeps only their color frequencies and discards all original positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    counts = _difference_counts(g)
    return _pack_counts(counts, g.shape)


def _pair_color_occurrences(
    counts: dict[int, int], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    """
    Pair colored occurrences into mismatched mirror pairs.

    Each pair contributes two asymmetric cells whose colors must differ.
    """
    remaining = {int(color): int(count) for color, count in counts.items()}
    pairs: list[tuple[int, int]] = []
    while sum(remaining.values()) > 0:
        active = [color for color, count in remaining.items() if count > 0]
        total = sum(remaining[color] for color in active)
        if max(remaining[color] for color in active) > total - max(remaining[color] for color in active):
            return None
        active.sort(key=lambda color: (-remaining[color], color))
        a = active[0]
        best_other = max(remaining[color] for color in active[1:])
        candidates = [color for color in active[1:] if remaining[color] == best_other]
        b = int(rng.choice(np.array(candidates, dtype=int)))
        pairs.append((int(a), int(b)))
        remaining[a] -= 1
        remaining[b] -= 1
    rng.shuffle(pairs)
    return pairs


def _top_positions() -> list[tuple[int, int]]:
    height, width = GRID_SHAPE
    return [(r, c) for r in range(height // 2) for c in range(width)]


def _sample_difference_plan(
    rng: np.random.Generator,
) -> tuple[list[int], Counter[int], list[tuple[int, int]]]:
    n_colors = int(rng.choice(np.array([3, 4], dtype=int)))
    colors = rng.choice(np.array(PALETTE, dtype=int), size=n_colors, replace=False).tolist()
    pattern = DIFF_PATTERNS[n_colors][int(rng.integers(len(DIFF_PATTERNS[n_colors])))]
    counts = Counter({int(color): int(count) for color, count in zip(colors, pattern)})
    pairs = _pair_color_occurrences(dict(counts), rng)
    assert pairs is not None
    return colors, counts, pairs


def _sample_symmetric_fillers(
    rng: np.random.Generator, diff_colors: list[int]
) -> list[int]:
    extra_colors = [color for color in PALETTE if color not in diff_colors]
    primary_extra = int(rng.choice(np.array(extra_colors, dtype=int)))
    filler_colors: list[int] = [primary_extra] * int(rng.integers(4, 7))

    diff_echo = rng.choice(np.array(diff_colors, dtype=int), size=2, replace=False).tolist()
    for color in diff_echo:
        filler_colors.extend([int(color)] * int(rng.integers(1, 4)))

    if len(extra_colors) > 1 and rng.random() < 0.6:
        secondary_choices = [color for color in extra_colors if color != primary_extra]
        secondary_extra = int(rng.choice(np.array(secondary_choices, dtype=int)))
        filler_colors.extend([secondary_extra] * int(rng.integers(1, 3)))

    extra_count = int(rng.integers(1, 5))
    noise_pool = np.array(PALETTE, dtype=int)
    filler_colors.extend(rng.choice(noise_pool, size=extra_count, replace=True).astype(int).tolist())
    rng.shuffle(filler_colors)
    return filler_colors


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    height, width = GRID_SHAPE
    diff_colors, target_counts, diff_pairs = _sample_difference_plan(rng)
    filler_colors = _sample_symmetric_fillers(rng, diff_colors)

    top_cells = _top_positions()
    needed = len(diff_pairs) + len(filler_colors)
    if needed > len(top_cells):
        return None

    chosen_ids = rng.choice(len(top_cells), size=needed, replace=False)
    diff_cells = [top_cells[int(i)] for i in chosen_ids[: len(diff_pairs)]]
    filler_cells = [top_cells[int(i)] for i in chosen_ids[len(diff_pairs) :]]

    grid = np.zeros(GRID_SHAPE, dtype=int)
    for (r, c), color in zip(filler_cells, filler_colors):
        mr = height - 1 - int(r)
        grid[r, c] = int(color)
        grid[mr, c] = int(color)

    for (r, c), (a, b) in zip(diff_cells, diff_pairs):
        mr = height - 1 - int(r)
        if rng.random() < 0.5:
            grid[r, c], grid[mr, c] = int(a), int(b)
        else:
            grid[r, c], grid[mr, c] = int(b), int(a)

    if _validate_generated_grid(grid, target_counts):
        return grid
    return None


def _validate_generated_grid(grid: np.ndarray, target_counts: Counter[int]) -> bool:
    """Reject degenerate samples that admit obvious shortcuts."""
    if grid.shape != GRID_SHAPE:
        return False

    flipped = np.flipud(grid)
    diff_mask = grid != flipped
    if int(diff_mask.sum()) not in {8, 10, 12, 14}:
        return False

    try:
        diff_counts = _difference_counts(grid)
    except AssertionError:
        return False
    if diff_counts != target_counts:
        return False
    if not 3 <= len(diff_counts) <= 4:
        return False

    symmetric_nonzero = (grid == flipped) & (grid != 0)
    if int(symmetric_nonzero.sum()) < 14:
        return False

    symmetric_colors = Counter(map(int, grid[symmetric_nonzero].tolist()))
    echoed = sum(1 for color in diff_counts if symmetric_colors.get(int(color), 0) > 0)
    if echoed < 2:
        return False

    nonzero_colors = set(map(int, grid[grid != 0].tolist()))
    if len(nonzero_colors - set(diff_counts)) < 1:
        return False

    global_nonzero_counts = Counter(map(int, grid[grid != 0].tolist()))
    if global_nonzero_counts == diff_counts:
        return False

    out = main(grid)
    if not np.array_equal(out, _pack_counts(diff_counts, grid.shape)):
        return False
    if np.array_equal(out, np.flipud(grid)):
        return False

    top_half_counts = Counter(map(int, grid[: grid.shape[0] // 2][grid[: grid.shape[0] // 2] != 0].tolist()))
    if top_half_counts and np.array_equal(out, _pack_counts(top_half_counts, grid.shape)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid asymmetric-mirror ARC input")
