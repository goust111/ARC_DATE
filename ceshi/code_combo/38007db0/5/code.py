# Combo variant of 38007db0 (index 5)
# combo_spec: D3/3+D4/2
# Name: 原位差异标记

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for combo D3/3 + D4/2
# --------------------------------------
# a) Concrete behavior and trace:
#    Original task:
#      each lattice row looks like [base, base, odd, base] and the output extracts [odd]
#      into a new single-column lattice.
#    D3/3 changes destination:
#      keep the full lattice footprint and blank all non-odd cells.
#    D4/2 changes the copied content:
#      do not copy the full odd cell; compare it to the repeated majority cell in the same
#      row and keep only the differing pixels.
#    Combined rule on one row therefore becomes:
#      input  row: [base, odd, base, base]
#      output row: [blank, diff(odd-base), blank, blank]
#    The diff is shown in the odd cell's original position, so the solver must first detect
#    the unique cell and the majority template, then compute a per-pixel subtraction mask.
#
# b) Complexity calibration:
#    A raw diff would hide sp->bg removals because those become background-colored and vanish.
#    To keep the fused rule visible, the generator only creates unique cells by adding 2-4
#    sprite pixels onto a shared base sprite. That gives repeated support for the rule:
#    input rows show one slightly denser cell, while outputs keep only those extra pixels.
#    Rows still have 3-4 columns so the majority pattern is visually obvious.
#
# c) Hidden-assumption audit against the original main():
#    - Gridline detection via fully uniform rows/cols still works because geometry is unchanged.
#    - Inferring the lattice color from those uniform lines still works because only true
#      lattice lines are globally uniform.
#    - Inferring bg as the most common non-lattice color still works because every cell keeps
#      a bg border and the base sprite density is capped, leaving bg dominant.
#    - The original solver assumed the destination shrinks to one column; that is invalid here.
#      The new solver must rebuild the full input-sized lattice and place output only in the
#      original odd-cell coordinates.
#    - The original solver assumed the odd cell itself is the desired payload; that is invalid
#      here. We now also need the row-majority cell as a reference before writing the output.
#    - Because each row has exactly one unique signature and all other cells are exact copies,
#      Counter(...).most_common(1) still recovers the majority reference safely.
#
# d) Placement legality:
#    The generator still uses a regular 1-pixel lattice and square cells of one shared size.
#    Added difference pixels are chosen only from bg locations inside the inner sprite area, so
#    no change crosses the cell border, no cells overlap, and all placements remain in-bounds.
#
# e) Visual observability:
#    Across 2-4 examples, a human can see that the odd cell stays in place but becomes sparse in
#    the output instead of being copied whole. Because the repeated base cells remain visible in
#    the input, the "subtract the row template from the odd cell" rule is recoverable by eye.
#
# f) Visual cleanliness:
#    Output contains only lattice lines, blank cells, and one compact sparse marker per row.
#    There is no overlay conflict because each row writes into exactly one cell and all other
#    cells are cleared to the existing background color.
#
# g) Change inferability / color mapping:
#    No new colors are introduced. Differing pixels keep the existing sprite color, and every
#    other cell pixel becomes bg. The only change is structural, not a hidden value-based color
#    remapping.
#
# New solver assumptions introduced by the combo and how generate_input() guarantees them:
#    1. Each row has exactly one unique cell and at least two majority copies. Guaranteed by
#       choosing 3-4 columns and replacing exactly one base cell per row.
#    2. The majority cell is a strict row template for diffing. Guaranteed because all non-odd
#       cells in the row are exact copies of one shared base cell.
#    3. The visible diff should contain at least two sprite pixels. Guaranteed by generating
#       every odd cell as a compact 2-4 pixel addition on bg positions only.
#    4. The fused output should not collapse to D3-only behavior. Guaranteed because the base
#       cell always contains some sprite pixels, so the full odd cell is strictly denser than
#       the diff-only output.


def _parse_lattice(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    row_lines = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)
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


def _diff_cell(base_cell: np.ndarray, unique_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(unique_cell.shape, int(bg), dtype=int)
    mask = unique_cell != base_cell
    diff[mask] = unique_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the original lattice shape, blank all majority cells, and in each odd cell keep only
    the pixels that differ from that row's repeated base cell.
    """
    g = np.array(grid, dtype=int)
    lattice, bg, row_lines, col_lines, r_slices, c_slices = _parse_lattice(g)

    out = np.full(g.shape, int(bg), dtype=int)
    for r in row_lines:
        out[int(r), :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)

    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)

        uniq_idx = next((j for j, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
        base_sig, base_count = counts.most_common(1)[0]
        base_idx = next((j for j, sig in enumerate(sigs) if sig == base_sig), None)
        assert uniq_idx is not None
        assert base_idx is not None and int(base_count) >= 2

        diff = _diff_cell(cells[int(base_idx)], cells[int(uniq_idx)], bg)
        ca, cb = c_slices[int(uniq_idx)]
        out[int(ra):int(rb), int(ca):int(cb)] = diff

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


def _compact_addition_sets(base_inner: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    inner_size = int(base_inner.shape[0])
    bg_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(bg)]

    candidate_sets: list[tuple[int, ...]] = []
    max_count = min(4, len(bg_positions))
    for count in range(2, max_count + 1):
        for combo in itertools.combinations(bg_positions, count):
            coords = [divmod(idx, inner_size) for idx in combo]
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if bbox_area <= count + 2:
                candidate_sets.append(tuple(int(x) for x in combo))
    return candidate_sets


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    # Keep the base visible but leave enough empty space for sparse in-place diffs.
    min_fill = math.ceil(0.30 * inner_area)
    max_fill = math.floor(0.60 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    base_inner.flat[rng.choice(inner_area, size=fill, replace=False)] = int(sp)

    candidate_sets = _compact_addition_sets(base_inner, bg)
    if len(candidate_sets) < n_rows:
        return None

    chosen_ids = rng.choice(len(candidate_sets), size=n_rows, replace=False)
    base_cell = _make_cell(base_inner, bg)
    blank_cell = _blank_cell(cell_size, bg)

    odd_cols = [int(rng.integers(0, n_cols)) for _ in range(n_rows)]
    input_cells: list[list[np.ndarray]] = []
    output_cells: list[list[np.ndarray]] = []
    for row, choice_id in enumerate(chosen_ids):
        add_set = candidate_sets[int(choice_id)]

        variant_inner = base_inner.copy()
        diff_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        for flat_idx in add_set:
            variant_inner.flat[int(flat_idx)] = int(sp)
            diff_inner.flat[int(flat_idx)] = int(sp)

        unique_cell = _make_cell(variant_inner, bg)
        diff_cell = _make_cell(diff_inner, bg)

        inp_row = [base_cell.copy() for _ in range(n_cols)]
        out_row = [blank_cell.copy() for _ in range(n_cols)]
        inp_row[odd_cols[row]] = unique_cell
        out_row[odd_cols[row]] = diff_cell
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
