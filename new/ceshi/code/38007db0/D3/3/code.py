# Variant of 38007db0
# Dimension: D3 — 目标位置：单列格子输出 (dst)
# Change: 输出为单列格子，保留原始行高，每行放置该行的唯一单元格... → 原位标记（保留输入网格）

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the D3 variant
# - Source task: none. Only the original 38007db0 solver/generator is relevant.
# - Concrete behavior:
#   Input is still a uniform square-cell lattice. Each lattice row contains one odd cell and
#   several exact copies of a base cell. The output no longer compresses rows into a single
#   destination column. Instead, it keeps the full original lattice footprint and blanks every
#   non-unique cell to the cell background color, leaving the unique cell untouched in its
#   original column.
# - Behavior trace on one row:
#   Original: [base, odd, base, base] -> extract [odd] into a new 1-column lattice.
#   Variant:  [base, odd, base, base] -> [blank, odd, blank, blank] in the same row/columns.
#   The odd-one-out detection stage is unchanged; only the write-out stage changes.
# - Complexity calibration:
#   The lattice geometry, color count, sprite generation, and per-row single oddball structure
#   stay unchanged. That preserves repeated evidence for the "one unique pattern per row" rule.
#   The variant stays visually obvious because outputs keep the full width and show most cells
#   erased, which is a strong contrast from the dense input.
# - Hidden assumptions audit against the original main():
#   1. Uniform full rows/cols still identify lattice lines. This remains valid because inputs are
#      generated with 1-pixel lattice lines and square cells.
#   2. Lattice color as the most common color among uniform rows/cols still holds because only
#      actual grid lines are fully uniform across a full row/column.
#   3. Background as the most common non-lattice color still holds because each cell has a full
#      background border and the sprite fill density is capped below 100%.
#   4. Per-row uniqueness is still defined by exact cell-array equality; generator keeps exactly
#      one variant per row and all other cells are exact base copies.
#   5. The old solver assumed the destination width should shrink to one cell. That assumption is
#      invalid in this variant and is replaced with "output shape equals input shape".
#   6. Starting the output from a blank background grid is now safer than copying the input,
#      because copying would require destructive erasure of majority cells while preserving
#      lattice lines; rebuilding avoids accidental leftovers.
# - Placement legality:
#   Cells are still placed only inside the lattice gaps computed from row/column lines, so there
#   is no overlap or out-of-bounds risk. The output uses the same lattice geometry as the input.
# - Visual observability:
#   Across 3-4 columns, clearing all non-odd cells makes the rule legible from a few examples:
#   the unique pattern is not moved, only isolated in-place.
# - Visual cleanliness:
#   No new marks are introduced. Blanking uses the existing cell background color, so the output
#   remains a clean lattice with some cells empty and some retained.
# - New solver assumptions introduced by this variant:
#   main() now assumes there is a single shared lattice geometry to reuse in the output and that
#   "blank cell" means filling the whole cell area with bg. generate_input() guarantees both by
#   using uniform square cells and by constructing expected outputs from explicit blank cells.


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
    assert len(row_lines) >= 2 and len(col_lines) >= 2

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
    Keep only the unique cell from each lattice row at its original position.

    The lattice geometry is preserved. Every non-unique cell is replaced by a blank background
    cell, so the output has the same shape as the input.
    """
    g = np.array(grid, dtype=int)
    lattice, bg, row_lines, col_lines, r_slices, c_slices = _parse_lattice(g)

    out = np.full(g.shape, int(bg), dtype=int)
    for r in row_lines:
        out[int(r), :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)

    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(cnt[sig]) == 1), None)
        assert uniq_idx is not None
        ca, cb = c_slices[int(uniq_idx)]
        out[int(ra) : int(rb), int(ca) : int(cb)] = cells[int(uniq_idx)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = inner.shape[0] + 2
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _blank_cell(cell_size: int, bg: int) -> np.ndarray:
    return np.full((cell_size, cell_size), int(bg), dtype=int)


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


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
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        unique_cells.append(_make_cell(variant_inner, bg))

    base_cell = _make_cell(base_inner, bg)
    blank_cell = _blank_cell(cell_size, bg)

    odd_cols = [int(rng.integers(0, n_cols)) for _ in range(n_rows)]

    input_cells: list[list[np.ndarray]] = []
    output_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        inp_row = [base_cell.copy() for _ in range(n_cols)]
        out_row = [blank_cell.copy() for _ in range(n_cols)]
        inp_row[odd_cols[row]] = unique_cells[row].copy()
        out_row[odd_cols[row]] = unique_cells[row].copy()
        input_cells.append(inp_row)
        output_cells.append(out_row)

    inp = _build_lattice(input_cells, lattice, bg)
    out = _build_lattice(output_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
