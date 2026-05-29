# Variant of 38007db0
# Dimension: D4 — 变换：直接复制无变换 (transform)
# Change: 唯一单元格原样复制到输出，无颜色或形状变换... → 只保留差异像素

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    The lattice layout, row count, column count, cell size, lattice color, and
#    cell background remain unchanged. Each input row still contains one odd cell
#    among repeated copies of a base cell. The original task copied that whole
#    odd cell into the single-column output. This variant instead compares the
#    odd cell against the repeated base cell from the same row and keeps only the
#    pixels where they differ. Example:
#      base inner   = [[sp,bg,bg],[bg,sp,bg],[bg,bg,sp]]
#      odd inner    = [[sp,sp,bg],[bg,sp,bg],[bg,sp,sp]]
#      old output   = odd inner as-is
#      new output   = [[bg,sp,bg],[bg,bg,bg],[bg,sp,bg]]
#    So the pipeline changes exactly at "row oddball extraction": after finding
#    the odd cell, we also recover the majority cell and compute a per-pixel diff.
#
# b) Complexity calibration:
#    If the odd cell were allowed to remove sprite pixels (sp->bg), then a raw
#    "copy differing pixels" display would hide those removals because bg is
#    invisible. To keep the D4 rule visually observable without adding a new
#    color (which would violate direct-copy semantics), the generator only uses
#    bg->sp additions. Each row adds 2-4 pixels, never just 1, and prefers
#    compact subsets so humans can perceive the output as a sparse highlight
#    rather than noise.
#
# c) Implicit-assumption audit against the original main():
#    - Uniform-row/col gridline detection still holds because the lattice layout
#      is unchanged.
#    - Inferring the lattice color as the most common color on uniform lines
#      still holds.
#    - Inferring bg as the most common non-lattice color still holds because bg
#      remains dominant in the input.
#    - The original chosen_cells.append(cells[uniq_idx]) depended on the old
#      invariant "output wants the full odd cell". That invariant is false now.
#      We must also identify the majority signature in the same row, then replace
#      every non-differing pixel with bg before writing to the output.
#    - Writing the output frame with lattice lines is still safe because those
#      cells stay cell-sized and aligned with the original slices.
#
# d) Placement legality:
#    Cells are still tiled into a regular non-overlapping 1-pixel lattice. Inner
#    additions only modify coordinates inside the (cell_size-2)x(cell_size-2)
#    sprite area, so nothing crosses cell borders or overwrites lattice pixels.
#
# e) Visual observability:
#    Across 2-4 examples, the human sees that each output row preserves the odd
#    cell's location pattern but strips away the shared base shape, leaving only
#    a few highlighted pixels. Because every input row also contains repeated
#    base cells, the "subtract common pattern from oddball" rule is directly
#    inferable from the examples.
#
# f) Visual cleanliness:
#    Output cells contain only lattice, bg, and a small compact set of sprite
#    pixels. There is no overlap or multi-layer marking; the result is sparse but
#    still aligned to the same tidy single-column lattice as the source task.
#
# g) Color inference:
#    No new color mapping is introduced. The differing pixels keep their original
#    sprite color, so the only conceptual change is structural: copy only the
#    delta, not the whole odd cell.


def _lattice_slices(g: np.ndarray) -> tuple[int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
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

    return H, W, row_lines, r_slices, c_slices


def _diff_cell(base_cell: np.ndarray, unique_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(unique_cell.shape, int(bg), dtype=int)
    diff[unique_cell != base_cell] = unique_cell[unique_cell != base_cell]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the per-row odd cell, but keep only the pixels that differ from that
    row's repeated base cell.

    New key insight: the repeated cell in each row is the reference template.
    The output shows the odd cell minus that template, not the whole odd cell.
    """
    g = np.array(grid, dtype=int)
    H, _, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(g[row_lines[0], 0])]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])
    lattice = int(g[row_lines[0], 0])

    chosen_diffs: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)

        base_sig, base_count = cnt.most_common(1)[0]
        assert sum(1 for v in cnt.values() if int(v) == 1) == 1
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        base_idx = next((j for j, s in enumerate(sigs) if s == base_sig), None)
        assert uniq_idx is not None and base_idx is not None and int(base_count) >= 2

        base_cell = cells[int(base_idx)]
        unique_cell = cells[int(uniq_idx)]
        chosen_diffs.append(_diff_cell(base_cell, unique_cell, bg))

    outW = int(cell_w + 2)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_diffs[int(i)]

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

    # Leave enough bg space so the difference-only output stays visible.
    min_fill = math.ceil(0.3 * inner_area)
    max_fill = math.floor(0.65 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    empty_count = int(np.sum(base_inner == int(bg)))
    if empty_count < 3:
        return None

    candidate_sets = _compact_addition_sets(base_inner, bg)
    if len(candidate_sets) < n_rows:
        return None

    chosen_ids = rng.choice(len(candidate_sets), size=n_rows, replace=False)
    diff_cells: list[np.ndarray] = []
    unique_cells: list[np.ndarray] = []
    for idx in chosen_ids:
        add_set = candidate_sets[int(idx)]
        diff_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        variant_inner = base_inner.copy()
        for flat_idx in add_set:
            diff_inner.flat[int(flat_idx)] = int(sp)
            variant_inner.flat[int(flat_idx)] = int(sp)
        diff_cells.append(_make_cell(diff_inner, bg))
        unique_cells.append(_make_cell(variant_inner, bg))

    out = _build_lattice([[cell] for cell in diff_cells], lattice, bg)

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
