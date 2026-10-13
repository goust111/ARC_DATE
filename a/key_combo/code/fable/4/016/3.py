from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

import numpy as np


PROFILE_MAX_REPEAT = 4
QUADRANT_EXTENT_MAX = 11
MATRIX_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), tuple(map(int, arr.ravel()))


def _expand(compact: np.ndarray, row_counts: tuple[int, ...], col_counts: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(compact, row_counts, axis=0), col_counts, axis=1)


def _run_starts_and_counts(lines: np.ndarray) -> tuple[list[int], tuple[int, ...]]:
    starts = [0]
    counts: list[int] = []
    count = 1
    for index in range(1, lines.shape[0]):
        if np.array_equal(lines[index], lines[index - 1]):
            count += 1
        else:
            counts.append(count)
            starts.append(index)
            count = 1
    counts.append(count)
    return starts, tuple(counts)


def _compress_quadrant(quadrant: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    patch = np.asarray(quadrant, dtype=int)
    if patch.ndim != 2 or patch.shape[0] == 0 or patch.shape[1] == 0:
        raise ValueError("quadrant must be non-empty")

    row_starts, row_counts = _run_starts_and_counts(patch)
    row_reduced = patch[row_starts, :]
    col_starts, col_counts = _run_starts_and_counts(row_reduced.T)
    compact = row_reduced[:, col_starts]

    rebuilt = _expand(compact, row_counts, col_counts)
    if not np.array_equal(rebuilt, patch):
        raise ValueError("quadrant is not a separable row/column stretch")
    return compact, row_counts, col_counts


def _solve_candidate(grid: np.ndarray, split_r: int, split_c: int) -> np.ndarray:
    sep_color = int(grid[split_r, split_c])
    if not np.all(grid[split_r, :] == sep_color):
        raise ValueError("separator row is not solid")
    if not np.all(grid[:, split_c] == sep_color):
        raise ValueError("separator column is not solid")
    if split_r == 0 or split_r == grid.shape[0] - 1 or split_c == 0 or split_c == grid.shape[1] - 1:
        raise ValueError("separator must leave four non-empty quadrants")

    quadrants = [
        grid[:split_r, :split_c],
        grid[:split_r, split_c + 1 :],
        grid[split_r + 1 :, :split_c],
        grid[split_r + 1 :, split_c + 1 :],
    ]

    infos = []
    for quadrant in quadrants:
        compact, row_counts, col_counts = _compress_quadrant(quadrant)
        infos.append(
            {
                "quadrant": quadrant,
                "compact": compact,
                "compact_sig": _signature(compact),
                "row_counts": row_counts,
                "col_counts": col_counts,
                "profile_sig": (row_counts, col_counts),
            }
        )

    compact_shapes = {info["compact"].shape for info in infos}
    if len(compact_shapes) != 1:
        raise ValueError("all compressed quadrants must share one shape")

    if len({info["profile_sig"] for info in infos}) != 4:
        raise ValueError("stretch profiles must be pairwise distinct")

    compact_counter = Counter(info["compact_sig"] for info in infos)
    if sorted(compact_counter.values()) != [1, 3]:
        raise ValueError("expected one odd compact matrix against a 3-way majority")

    majority_sig = next(sig for sig, count in compact_counter.items() if count == 3)
    odd_sig = next(sig for sig, count in compact_counter.items() if count == 1)
    majority_compact = next(info["compact"] for info in infos if info["compact_sig"] == majority_sig)
    odd_info = next(info for info in infos if info["compact_sig"] == odd_sig)

    out = _expand(majority_compact, odd_info["row_counts"], odd_info["col_counts"])
    if out.shape != odd_info["quadrant"].shape:
        raise ValueError("expanded output must match the odd quadrant size")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the odd quadrant's compressed content with the majority quadrant pattern while keeping the odd quadrant's stretch profile.

    Input structure:
    - One solid-color row and one solid-color column of the same color form a separator cross, splitting the grid into four non-empty quadrants.
    - Each quadrant is a stretched version of a smaller color matrix, produced by repeating adjacent rows and columns.
    - Three quadrants compress to the same small matrix, while the remaining quadrant compresses to a different matrix of the same shape.

    Rule:
    - Find the separator cross and extract the four quadrants.
    - For each quadrant, collapse adjacent duplicate rows and adjacent duplicate columns, while recording the row-run and column-run lengths.
    - Identify the odd quadrant by its compressed matrix.
    - Expand the majority compressed matrix using the odd quadrant's recorded row and column run lengths.

    Key insight:
    - The odd quadrant contributes only its stretch rhythm, while the majority quadrants contribute the content that must be re-expanded.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError("expected a 2D grid with room for a separator cross")

    candidate_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0])]
    candidate_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == g[0, c])]

    solutions: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for split_r in candidate_rows:
        for split_c in candidate_cols:
            if int(g[split_r, split_c]) != int(g[split_r, 0]):
                continue
            try:
                out = _solve_candidate(g, split_r, split_c)
            except ValueError:
                continue
            solutions[_signature(out)] = out

    if len(solutions) != 1:
        raise ValueError("failed to identify a unique valid separator cross")
    return next(iter(solutions.values()))


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int, max_part: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        if 1 <= total <= max_part:
            return ((total,),)
        return ()

    out = []
    lo = max(1, total - max_part * (parts - 1))
    hi = min(max_part, total - (parts - 1))
    for first in range(lo, hi + 1):
        for rest in _compositions(total - first, parts - 1, max_part):
            out.append((first, *rest))
    return tuple(out)


def _sample_profile_pair(parts: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    totals = [
        total
        for total in range(parts + 1, min(QUADRANT_EXTENT_MAX, parts * PROFILE_MAX_REPEAT) + 1)
        if len(_compositions(total, parts, PROFILE_MAX_REPEAT)) >= 2
    ]
    if not totals:
        raise RuntimeError("no valid stretch totals")
    total = random.choice(totals)
    choices = list(_compositions(total, parts, PROFILE_MAX_REPEAT))
    a, b = random.sample(choices, 2)
    return a, b


def _has_adjacent_duplicate_lines(grid: np.ndarray) -> bool:
    for index in range(1, grid.shape[0]):
        if np.array_equal(grid[index], grid[index - 1]):
            return True
    for index in range(1, grid.shape[1]):
        if np.array_equal(grid[:, index], grid[:, index - 1]):
            return True
    return False


def _sample_compact_matrix(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    for _ in range(MATRIX_ATTEMPTS):
        grid = np.array(
            [[random.choice(palette) for _ in range(cols)] for _ in range(rows)],
            dtype=int,
        )
        if len(set(map(int, grid.ravel()))) < 2:
            continue
        if _has_adjacent_duplicate_lines(grid):
            continue
        return grid
    raise RuntimeError("failed to sample a compact matrix")


def _mutate_matrix(base: np.ndarray, palette: list[int]) -> np.ndarray:
    rows, cols = map(int, base.shape)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(MATRIX_ATTEMPTS):
        mutated = base.copy()
        change_count = random.randint(1, max(1, (rows * cols) // 2))
        for r, c in random.sample(cells, change_count):
            options = [color for color in palette if color != int(mutated[r, c])]
            mutated[r, c] = random.choice(options)
        if np.array_equal(mutated, base):
            continue
        if _has_adjacent_duplicate_lines(mutated):
            continue
        return mutated
    for _ in range(MATRIX_ATTEMPTS):
        mutated = _sample_compact_matrix(rows, cols, palette)
        if not np.array_equal(mutated, base):
            return mutated
    raise RuntimeError("failed to build a distinct odd compact matrix")


def _build_input() -> np.ndarray:
    separator = random.randrange(10)
    palette_size = random.randint(3, 5)
    palette = random.sample([color for color in range(10) if color != separator], palette_size)

    base_rows = random.randint(2, 4)
    base_cols = random.randint(2, 4)
    majority_compact = _sample_compact_matrix(base_rows, base_cols, palette)
    odd_compact = _mutate_matrix(majority_compact, palette)

    top_rows = _sample_profile_pair(base_rows)
    bottom_rows = _sample_profile_pair(base_rows)
    left_cols = _sample_profile_pair(base_cols)
    right_cols = _sample_profile_pair(base_cols)

    profiles = [
        (top_rows[0], left_cols[0]),
        (top_rows[1], right_cols[0]),
        (bottom_rows[0], left_cols[1]),
        (bottom_rows[1], right_cols[1]),
    ]
    if len(set(profiles)) != 4:
        raise RuntimeError("quadrant profiles must be distinct")

    odd_index = random.randrange(4)
    quadrants = []
    for index, (row_counts, col_counts) in enumerate(profiles):
        compact = odd_compact if index == odd_index else majority_compact
        quadrants.append(_expand(compact, row_counts, col_counts))

    top_height = quadrants[0].shape[0]
    bottom_height = quadrants[2].shape[0]
    left_width = quadrants[0].shape[1]
    right_width = quadrants[1].shape[1]
    height = top_height + 1 + bottom_height
    width = left_width + 1 + right_width
    if height > 30 or width > 30:
        raise RuntimeError("grid exceeds ARC size limit")

    grid = np.full((height, width), separator, dtype=int)
    grid[:top_height, :left_width] = quadrants[0]
    grid[:top_height, left_width + 1 :] = quadrants[1]
    grid[top_height + 1 :, :left_width] = quadrants[2]
    grid[top_height + 1 :, left_width + 1 :] = quadrants[3]

    expected = _expand(majority_compact, profiles[odd_index][0], profiles[odd_index][1])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("oracle mismatch")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _build_input()
        except (RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
