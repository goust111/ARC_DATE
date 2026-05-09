# Combo variant of 38007db0 (index 6)
# combo_spec: D1/0+D4/2
# Name: 列对齐差异提取
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    The original task scans each lattice row, finds the one odd cell in that row,
#    and copies the whole odd cell into a single-column output. This combo changes
#    two linked stages at once:
#    - D1/0 switches the search direction from rows to columns.
#    - D4/2 changes the copied content from "full odd cell" to "odd-minus-majority diff".
#    Example on a 3x4 cell grid:
#    - Original: row 0 has [base, oddA, base, base] -> output row 0 = oddA.
#    - Combo: column 0 might have [base, base, oddB] -> output column 0 is not oddB
#      itself, but only the pixels where oddB differs from the base cell in that column.
#    Repeating this for every column yields a single-row lattice of sparse diff cells.
#
# b) Complexity calibration:
#    To keep the diff rule human-visible, the generator only uses bg->sp additions.
#    If sp->bg removals were allowed, those removals would vanish in a raw diff because
#    the output keeps bg elsewhere. Each odd cell therefore adds a compact 2-4 pixel set.
#    To keep the column rule visible, at least one input row contains zero odd cells,
#    so the sample cannot be reinterpreted as the original "one odd per row" task.
#
# c) Implicit assumption audit against the original solver:
#    - Uniform full-color rows/cols still identify lattice lines, so that parsing stays valid.
#    - The original loop over `r_slices` assumed every row has exactly one odd cell; that
#      assumption is false here and must move to a loop over `c_slices`.
#    - The original `chosen_cells.append(unique_cell)` assumed the output wants the full odd
#      sprite; that is false here. We must also recover the majority cell in the same column
#      and write only the differing pixels from the odd cell.
#    - The original output frame was a single column with original height. After D1/0 the
#      output instead keeps original width and collapses to a single row.
#    - `Counter.most_common(1)` is only safe if every column has a strict majority cell; the
#      generator guarantees exactly one odd cell per column and `n_rows >= 3`.
#
# d) Placement legality:
#    All cells are placed by `_build_lattice()` into non-overlapping slots separated by 1-pixel
#    lattice lines. Inner edits only touch the `(cell_size-2)x(cell_size-2)` sprite area, so
#    no edit crosses a border or overwrites lattice pixels.
#
# e) Visual observability:
#    A human sees each column as a stack of repeated base cells with one discrepant cell. The
#    output is a single-row strip where each cell keeps only the discrepant pixels. Because at
#    least one row has no odd cell at all, the evidence points away from row scanning and toward
#    column scanning plus subtraction of the shared template.
#
# f) Visual cleanliness:
#    The output uses the same tidy lattice geometry and only three colors. Diff cells are sparse
#    but compact, so they read as highlights rather than noise. No object overlap or extra marker
#    layers are introduced.
#
# g) Change inferability:
#    No hidden numeric ordering or color remapping is used. The sprite color stays the sprite
#    color; only the selection axis and copied shape change. Across 2-4 examples, the repeated
#    base cells provide the structural cue needed to infer "compare odd cell against its column's
#    majority cell and keep only the added pixels".
#
# h) New solver assumptions and generator guarantees:
#    - Assumption: every column has exactly one unique signature and one majority signature.
#      Guarantee: generator fills each column with one variant and the rest exact base copies.
#    - Assumption: the diff is visible in output.
#      Guarantee: generator uses additions only, with 2-4 added pixels.
#    - Assumption: bg remains the most common non-lattice color.
#      Guarantee: base fill is capped and cells keep a bg border plus enough inner bg space.

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    return H, W, lattice, bg, row_lines, col_lines, r_slices, c_slices


def _diff_cell(base_cell: np.ndarray, unique_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(unique_cell.shape, int(bg), dtype=int)
    mask = unique_cell != base_cell
    diff[mask] = unique_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scan lattice columns, find the odd cell in each column, and output only the
    pixels where that odd cell differs from the column-majority cell.
    """
    g = np.array(grid, dtype=int)
    H, W, lattice, bg, _, col_lines, r_slices, c_slices = _lattice_slices(g)
    cell_h = int(r_slices[0][1] - r_slices[0][0])

    chosen_diffs: list[np.ndarray] = []
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)

        uniq_idx = next((i for i, sig in enumerate(sigs) if int(cnt[sig]) == 1), None)
        base_sig, base_count = cnt.most_common(1)[0]
        base_idx = next((i for i, sig in enumerate(sigs) if sig == base_sig), None)

        assert uniq_idx is not None
        assert base_idx is not None and int(base_count) >= 2
        assert sum(1 for v in cnt.values() if int(v) == 1) == 1

        chosen_diffs.append(_diff_cell(cells[int(base_idx)], cells[int(uniq_idx)], bg))

    out_h = int(cell_h + 2)
    out = np.full((out_h, W), int(bg), dtype=int)
    out[0, :] = int(lattice)
    out[-1, :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)

    for j, (ca, cb) in enumerate(c_slices):
        out[1 : 1 + int(cell_h), int(ca) : int(cb)] = chosen_diffs[int(j)]

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


def _compact_addition_sets(base_inner: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    inner_size = int(base_inner.shape[0])
    bg_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(bg)]

    candidate_sets: list[tuple[int, ...]] = []
    for count in range(2, min(4, len(bg_positions)) + 1):
        for combo in itertools.combinations(bg_positions, count):
            coords = [divmod(int(idx), inner_size) for idx in combo]
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if bbox_area <= count + 2:
                candidate_sets.append(tuple(int(x) for x in combo))
    return candidate_sets


def _choose_odd_rows(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray | None:
    for _ in range(120):
        odd_rows = rng.integers(0, n_rows, size=n_cols)
        counts = np.bincount(odd_rows, minlength=n_rows)
        if int(np.min(counts)) == 0:
            return odd_rows
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    min_fill = math.ceil(0.3 * inner_area)
    max_fill = math.floor(0.6 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    if int(np.sum(base_inner == int(bg))) < 4:
        return None

    candidate_sets = _compact_addition_sets(base_inner, bg)
    if len(candidate_sets) < n_cols:
        return None

    chosen_ids = rng.choice(len(candidate_sets), size=n_cols, replace=False)
    diff_cells: list[np.ndarray] = []
    unique_cells: list[np.ndarray] = []
    for idx in chosen_ids:
        add_set = candidate_sets[int(idx)]
        diff_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        unique_inner = base_inner.copy()
        for flat_idx in add_set:
            diff_inner.flat[int(flat_idx)] = int(sp)
            unique_inner.flat[int(flat_idx)] = int(sp)
        diff_cells.append(_make_cell(diff_inner, bg))
        unique_cells.append(_make_cell(unique_inner, bg))

    odd_rows = _choose_odd_rows(rng, n_rows, n_cols)
    if odd_rows is None:
        return None

    out = _build_lattice([diff_cells], lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells = [[base_cell.copy() for _ in range(n_cols)] for _ in range(n_rows)]
    for col in range(n_cols):
        in_cells[int(odd_rows[col])][col] = unique_cells[col].copy()
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
