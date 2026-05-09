# Variant of 38007db0
# Dimension: D10 — 多步处理与排序 (multi_step)
# Change: 单步处理 — 直接识别并提取每行唯一项... → 两阶段处理：先识别基础精灵再找差异

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the D10 variant
# ----------------------------------
# spec:
#   task: Infer the shared base sprite first, then extract each row's unique sprite.
#   input:
#     - Same lattice layout as the original task: 1-pixel gridlines, square cells, one odd cell per row.
#     - Every non-odd cell in every row is the same base sprite, so there is one global shared pattern.
#   output:
#     - Single-column lattice.
#     - Top section: one cell containing the inferred global base sprite.
#     - Bottom section: one cell per original row, in order, containing that row's odd sprite.
#     - Therefore output has (input_rows_of_cells + 1) cell rows instead of just input_rows_of_cells.
# behavior_trace:
#   - Original row pattern: [B, B, U] -> output contributed only [U].
#   - Variant row pattern: [B, B, U] still requires finding U, but output now starts with [B] once,
#     then appends [U_row0], [U_row1], ...
#   - The extra top cell makes the "infer the shared pattern" step visible and necessary.
# complexity_calibration:
#   - Keeping 3-4 columns preserves an unambiguous majority in every row.
#   - Keeping 3-4 rows gives repeated evidence that the same majority sprite is shared globally.
#   - Unique variants still differ by a small number of flipped pixels, so the base/variant relation
#     is visible rather than looking like unrelated random cells.
# implicit_assumptions_audit:
#   - Uniform full rows/cols still identify lattice lines: preserved from the original construction.
#   - Cells are still equal-sized square slices between lattice lines: preserved.
#   - Old code assumed only the unique cells matter. That is no longer enough because the output must
#     also include the shared base cell, so we explicitly compute each row majority cell first.
#   - New solver assumes every row has exactly one singleton signature and one repeated majority
#     signature, and that all row-majority signatures are identical across rows. The generator enforces
#     this directly.
# placement_legality:
#   - Cells are inserted only into lattice slots produced by `_build_lattice`, so there is no overlap or
#     out-of-bounds placement.
#   - Output simply adds one extra cell row on top of the single output column, preserving clean spacing.
# visual_observability:
#   - A human can see the top output cell is the pattern repeated many times in the input, while each
#     lower output cell is the per-row exception. The rule is visible from 2-3 examples without relying
#     on hidden numeric color ordering.
# visual_cleanliness:
#   - No extra markers are added; the lattice already separates the "base" row from the list of odd rows.
# color_reasoning:
#   - Colors are unchanged semantically. Only three colors are used, so the new behavior is carried by
#     structure and placement, not by arbitrary color remapping.


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = inner.shape[0] + 2
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full(
        (rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1),
        int(bg),
        dtype=int,
    )
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _parse_lattice(grid: np.ndarray) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == lattice])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == lattice])

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    return lattice, bg, row_lines, col_lines, r_slices, c_slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Two-stage variant:
    1. Infer the shared base sprite by taking the repeated cell in each row and confirming these agree.
    2. Extract each row's unique cell and place them below the base sprite in a single output column.
    """
    g = np.array(grid, dtype=int)
    lattice, bg, _row_lines, _col_lines, r_slices, c_slices = _parse_lattice(g)

    row_base_cells: list[np.ndarray] = []
    row_unique_cells: list[np.ndarray] = []

    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        majority_sig, majority_count = counts.most_common(1)[0]
        assert int(majority_count) >= 2

        unique_indices = [idx for idx, sig in enumerate(sigs) if int(counts[sig]) == 1]
        assert len(unique_indices) == 1

        base_idx = next(idx for idx, sig in enumerate(sigs) if sig == majority_sig)
        row_base_cells.append(cells[int(base_idx)])
        row_unique_cells.append(cells[int(unique_indices[0])])

    base_sigs = [cell.tobytes() for cell in row_base_cells]
    base_sig, base_count = Counter(base_sigs).most_common(1)[0]
    assert int(base_count) == len(row_base_cells)
    base_cell = next(cell for cell in row_base_cells if cell.tobytes() == base_sig)

    out_cells = [[base_cell.copy()]] + [[cell.copy()] for cell in row_unique_cells]
    return _build_lattice(out_cells, lattice, bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    base_cell = _make_cell(base_inner, bg)

    unique_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _ in range(n_rows):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [flip_set for flip_set in candidates if flip_set not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)

        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == bg else bg
        unique_cells.append(_make_cell(variant_inner, bg))

    expected = _build_lattice([[base_cell.copy()]] + [[cell.copy()] for cell in unique_cells], lattice, bg)

    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
