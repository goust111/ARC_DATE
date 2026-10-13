from __future__ import annotations

import random
from itertools import combinations

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
INNER_SIZE_RANGE = (2, 4)
GRID_PADDING_RANGE = (2, 4)
GUIDE_PROBABILITY = 0.5
PALETTE = list(range(10))


def _infer_separator_grid(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, list[int], list[int]]] = []

    for color in map(int, np.unique(grid)):
        row_divs = [r for r in range(height) if bool(np.all(grid[r, :] == color))]
        col_divs = [c for c in range(width) if bool(np.all(grid[:, c] == color))]
        if len(row_divs) < 2 or len(col_divs) < 2:
            continue
        if row_divs[0] != 0 or row_divs[-1] != height - 1:
            continue
        if col_divs[0] != 0 or col_divs[-1] != width - 1:
            continue

        row_steps = {b - a for a, b in zip(row_divs, row_divs[1:])}
        col_steps = {b - a for a, b in zip(col_divs, col_divs[1:])}
        if len(row_steps) != 1 or len(col_steps) != 1:
            continue

        row_step = next(iter(row_steps))
        col_step = next(iter(col_steps))
        if row_step != col_step or row_step <= 1:
            continue
        if (height - 1) % row_step != 0 or (width - 1) % col_step != 0:
            continue
        expected_rows = (height - 1) // row_step + 1
        expected_cols = (width - 1) // col_step + 1
        if len(row_divs) != expected_rows or len(col_divs) != expected_cols:
            continue

        candidates.append((row_step, color, row_divs, col_divs))

    assert candidates, "expected one regular separator lattice"
    candidates.sort(key=lambda item: (item[0], item[1]))
    step, color, row_divs, col_divs = candidates[0]
    return int(color), int(step), row_divs, col_divs


def _extract_cells(
    grid: np.ndarray, row_divs: list[int], col_divs: list[int]
) -> tuple[int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    n_rows = len(row_divs) - 1
    n_cols = len(col_divs) - 1
    assert n_rows == n_cols and n_rows >= 2, "expected an NxN cell grid"

    inner_h = row_divs[1] - row_divs[0] - 1
    inner_w = col_divs[1] - col_divs[0] - 1
    assert inner_h == inner_w and inner_h >= 2, "expected square cell interiors"

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cells: list[list[tuple[int, ...]]] = []
    for i in range(n_rows):
        row: list[tuple[int, ...]] = []
        for j in range(n_cols):
            patch = grid[row_divs[i] + 1 : row_divs[i + 1], col_divs[j] + 1 : col_divs[j + 1]].copy()
            key = tuple(map(int, patch.ravel()))
            if key not in patterns:
                patterns[key] = patch
            row.append(key)
        cells.append(row)

    return inner_h, patterns, cells


def _placement_candidates(existing: np.ndarray, active_mask: np.ndarray) -> list[tuple[int, int]]:
    n = existing.shape[0]
    inner_h, inner_w = active_mask.shape
    candidates: list[tuple[int, int]] = []

    for top in range(n - inner_h + 1):
        for left in range(n - inner_w + 1):
            block = np.zeros_like(existing, dtype=bool)
            block[top : top + inner_h, left : left + inner_w] = True
            if bool(np.any(existing & ~block)):
                continue

            sub = existing[top : top + inner_h, left : left + inner_w]
            if bool(np.any(sub & ~active_mask)):
                continue
            candidates.append((top, left))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal minority-cell pattern on a regular separator grid.

    The cell-grid contains one repeated majority interior pattern and a rarer
    minority interior pattern. The minority pattern acts as a cell-scale
    blueprint: positions where it differs from the majority pattern mark where
    minority cells should appear in a larger placement of that pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator, _step, row_divs, col_divs = _infer_separator_grid(g)
    inner, patterns, cells = _extract_cells(g, row_divs, col_divs)
    n = len(cells)

    counts: dict[tuple[int, ...], int] = {}
    for row in cells:
        for key in row:
            counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 2, "expected exactly one majority and one minority pattern"
    ordered = sorted(counts.items(), key=lambda item: item[1])
    minority_key, minority_count = ordered[0]
    majority_key, majority_count = ordered[1]
    assert minority_count < majority_count, "minority pattern must be less frequent"

    majority = patterns[majority_key]
    minority = patterns[minority_key]
    active_mask = minority != majority
    assert bool(active_mask.any()), "minority pattern must visibly differ from the majority"
    assert active_mask.shape == (inner, inner)
    assert inner <= n, "cell blueprint must fit within the cell grid"

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cells[i][j] == minority_key
    assert bool(existing.any()), "expected at least one minority cell"

    candidates = _placement_candidates(existing, active_mask)
    assert len(candidates) == 1, "minority placement must be uniquely determined"
    top, left = candidates[0]

    out = g.copy()
    for i in range(inner):
        for j in range(inner):
            r0 = row_divs[top + i] + 1
            r1 = row_divs[top + i + 1]
            c0 = col_divs[left + j] + 1
            c1 = col_divs[left + j + 1]
            out[r0:r1, c0:c1] = minority if active_mask[i, j] else majority
    return out


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _random_active_mask(inner: int) -> np.ndarray | None:
    min_active = 2
    max_active = min(inner + 1, inner * inner - 1)
    active_count = random.randint(min_active, max_active)
    flat = random.sample(range(inner * inner), active_count)

    mask = np.zeros((inner, inner), dtype=bool)
    for idx in flat:
        mask[idx // inner, idx % inner] = True

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    return mask


def _random_majority_pattern(inner: int, base: int, guide: int | None) -> np.ndarray:
    pattern = np.full((inner, inner), base, dtype=int)
    if guide is None:
        return pattern

    guide_count = random.randint(1, max(1, inner - 1))
    for idx in random.sample(range(inner * inner), guide_count):
        pattern[idx // inner, idx % inner] = guide
    return pattern


def _build_grid(
    n: int,
    inner: int,
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    minority_cells: list[tuple[int, int]],
) -> np.ndarray:
    step = inner + 1
    size = n * step + 1
    grid = np.full((size, size), separator, dtype=int)

    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            grid[r0 : r0 + inner, c0 : c0 + inner] = majority

    for i, j in minority_cells:
        r0 = i * step + 1
        c0 = j * step + 1
        grid[r0 : r0 + inner, c0 : c0 + inner] = minority

    return grid


def _subset_candidates(
    n: int, active_mask: np.ndarray, top: int, left: int, subset: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    existing = np.zeros((n, n), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True
    return _placement_candidates(existing, active_mask)


def _choose_given_subset(
    n: int, active_mask: np.ndarray, top: int, left: int
) -> list[tuple[int, int]] | None:
    active = [tuple(map(int, rc)) for rc in np.argwhere(active_mask)]
    size_options = list(range(1, len(active)))
    random.shuffle(size_options)

    for size in size_options:
        valid: list[tuple[tuple[int, int], ...]] = []
        for subset in combinations(active, size):
            if _subset_candidates(n, active_mask, top, left, subset) == [(top, left)]:
                valid.append(subset)
        if valid:
            return list(random.choice(valid))
    return None


def _gen_attempt() -> np.ndarray | None:
    inner = random.randint(*INNER_SIZE_RANGE)
    n = inner + random.randint(*GRID_PADDING_RANGE)

    separator, base, minority_color, guide = random.sample(PALETTE, 4)
    guide_color = guide if random.random() < GUIDE_PROBABILITY else None

    majority = _random_majority_pattern(inner, base, guide_color)
    active_mask = _random_active_mask(inner)
    if active_mask is None:
        return None

    minority = majority.copy()
    minority[active_mask] = minority_color
    if np.array_equal(minority, majority):
        return None

    top = random.randint(0, n - inner)
    left = random.randint(0, n - inner)
    all_active_cells = [(top + r, left + c) for r, c in np.argwhere(active_mask)]
    given_local = _choose_given_subset(n, active_mask, top, left)
    if given_local is None:
        return None
    given_cells = [(top + r, left + c) for r, c in given_local]

    inp = _build_grid(n, inner, separator, majority, minority, given_cells)
    out = _build_grid(n, inner, separator, majority, minority, all_active_cells)

    if _dominant_color(inp) != base:
        return None
    if len(given_cells) >= len(all_active_cells):
        return None
    if len(all_active_cells) >= (n * n + 1) // 2:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractal-grid input")
