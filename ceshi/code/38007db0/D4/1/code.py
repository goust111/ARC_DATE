# Variant of 38007db0
# Dimension: D4 — 变换：直接复制无变换 (transform)
# Change: 唯一单元格原样复制到输出，无颜色或形状变换... → 取反精灵（精灵色与背景色互换）

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for this D4 variant
# -----------------------------------
# spec_variant:
#   core_rule:
#     input parsing is unchanged from 38007db0:
#       - a square lattice is divided by 1px gridlines of color L
#       - each cell has a 1px bg padding ring and an inner sprite using color sp
#       - in each lattice row, exactly one cell is the odd-one-out
#     output extraction changes only at the final transform step:
#       - original task: copy the unique cell into the output column as-is
#       - variant: after extracting that unique cell, swap bg and sp everywhere inside
#         the cell area, while keeping lattice lines L unchanged
#   concrete_behavior_trace:
#     - if a chosen cell is
#         bg bg bg bg bg
#         bg sp sp bg bg
#         bg bg sp bg bg
#         bg bg bg bg bg
#         bg bg bg bg bg
#       then original output pastes that exact 5x5 cell
#       but this variant outputs
#         sp sp sp sp sp
#         sp bg bg sp sp
#         sp sp bg sp sp
#         sp sp sp sp sp
#         sp sp sp sp sp
#     - the "find unique cell per row" stage is unchanged; only the pasted cell is inverted
#   complexity_calibration:
#     - keep exactly 3 colors in the input: lattice L, cell background bg, sprite sp
#     - keep 3-4 rows and 3-4 cols so each row still has a clear majority pattern
#     - keep 5x5 or 6x6 cells with a 1px bg ring; when inverted, that ring becomes sp,
#       making the D4 change visually obvious instead of collapsing to near-original output
#     - keep unique variants as 1-3 pixel flips from one shared base sprite so the
#       odd-one-out logic remains the same task rather than a new recognition problem
#   implicit_assumptions_review:
#     - uniform full rows/cols identify lattice lines: still valid because input structure is unchanged
#     - most common non-lattice color is bg: still valid because every cell contains a bg ring
#     - there is exactly one other non-lattice color sp: generator enforces exactly 3 colors
#     - each row has exactly one unique full-cell byte pattern: generator still creates one odd cell
#     - chosen cells contain only bg/sp, so np.where(cell == bg, sp, bg) is safe: generator guarantees it
#   placement_legality:
#     - cells are still pasted into disjoint lattice slots built from a regular grid
#     - output remains a single-column lattice with the same row slices as the input
#     - inversion happens after extraction and does not alter dimensions, so no overlap or bounds risk
#   visual_observability:
#     - across examples, the extracted output cell always becomes a negative of the seen odd cell
#     - because the whole cell area swaps bg/sp, especially the padding ring, the transformation is
#       obvious from 2-3 examples and does not rely on hidden numeric color identities
#   source_task_note:
#     - the referenced source task a25697e4 solver file is not present in this workspace,
#       so only its color-swap idea (visible in the provided JSON examples) is borrowed


def _invert_cell(cell: np.ndarray, bg: int, sp: int) -> np.ndarray:
    """Swap bg and sp inside one extracted cell; lattice color never appears here."""
    return np.where(cell == int(bg), int(sp), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice,
    then invert bg/sp inside each extracted cell before writing it to the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # Gridlines appear as fully-uniform rows/cols.
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    # Lattice color = most common value among uniform rows/cols.
    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == int(L)])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == int(L)])

    # Cell slices are the gaps between consecutive grid lines.
    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    # Background inside cells = most common non-lattice color; sprite is the only other color.
    non_lattice = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert non_lattice
    bg = int(Counter(non_lattice).most_common(1)[0][0])
    sprite_colors = sorted({int(x) for x in non_lattice if int(x) != int(bg)})
    assert len(sprite_colors) == 1
    sp = int(sprite_colors[0])

    chosen_cells: list[np.ndarray] = []
    for (ra, rb) in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for (ca, cb) in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(_invert_cell(cells[int(uniq_idx)], bg, sp))

    # Output: same height, single lattice column => width = cell_w + 2 (outer gridlines).
    outW = int(cell_w + 2)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


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
    inverted_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _ in range(n_rows):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [fs for fs in candidates if fs not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)
        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        cell = _make_cell(variant_inner, bg)
        unique_cells.append(cell)
        inverted_cells.append(_invert_cell(cell, bg, sp))

    direct_out = _build_lattice([[cell] for cell in unique_cells], lattice, bg)
    out = _build_lattice([[cell] for cell in inverted_cells], lattice, bg)
    if np.array_equal(out, direct_out):
        return None

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
