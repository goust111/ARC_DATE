from __future__ import annotations

# Variant of 38007db0
# Dimension: D3 — 目标位置：单列格子输出 (dst)
# Change: 输出为单列格子，保留原始行高，每行放置该行的唯一单元格... → 按唯一项列位置排序输出
#
# Design analysis
# a) Concrete behavior:
#    Input is still a lattice of square cells. In each lattice row, all cells share one base sprite
#    except a single odd cell. The original task outputs those odd cells in the same top-to-bottom
#    order as the source rows. This variant changes only the destination ordering: after detecting
#    each row's odd cell, rank rows by the odd cell's source column from left to right, then place
#    the extracted cells into the single output column in that ranked order.
#    Example with three source rows:
#    - row 0 odd cell at column 2 -> unique A
#    - row 1 odd cell at column 0 -> unique B
#    - row 2 odd cell at column 1 -> unique C
#    Original output order: A, B, C.
#    Variant output order:  B, C, A.
#    The lattice framing and row-band geometry stay unchanged; only which extracted cell occupies
#    the first/second/third output slot changes.
#
# b) Complexity calibration:
#    The new reasoning step is a single global ordering over already-detected odd cells, mirroring
#    the reference task's "rank objects by a visible key, then redistribute into fixed slots".
#    To keep this visible and unambiguous, the generator makes odd-column positions distinct and
#    forces a non-identity permutation, so the output does not accidentally match the original
#    row order. Other dimensions remain unchanged: same 3 colors, same lattice format, same
#    one-odd-cell-per-row rule.
#
# c) Implicit assumption audit for the original main():
#    - Gridline detection via uniform rows/cols still holds unchanged.
#    - Cell slicing between gridlines still holds unchanged.
#    - Per-row odd-cell detection via exact byte signature counts still holds unchanged.
#    - The original code's hidden assumption was in the final placement:
#         chosen_cells.append(cells[uniq_idx])
#         ...
#         out[row_band_i] = chosen_cells[i]
#      This assumes output band i corresponds to input row i. That is no longer valid. The fix is
#      to store both the odd cell and its source column index, then sort those records before
#      populating the output row bands.
#    - A secondary assumption is tie handling. The variant rule sorts by odd-column position; if
#      two rows shared the same odd column, the task would be visually weaker. The solver therefore
#      uses a deterministic tie-break on original row index, while the generator avoids ties.
#
# d) Placement legality:
#    The generator still builds a regular non-overlapping lattice, so cells never overlap or leave
#    bounds. Unique variants are created only by flipping inner pixels inside a cell's sprite area.
#    Distinct odd columns are sampled from valid lattice columns, and the output is built with the
#    same row count and cell size as the input, so the solver's slicing logic remains valid.
#
# e) Visual observability:
#    The sorting key is a visible spatial property: the x-position of the odd cell. With 3-4 rows,
#    humans can compare which oddball is leftmost, next, rightmost, then see those cells reordered
#    top-to-bottom in the output. This avoids any hidden dependence on numeric color ids.
#
# f) Visual cleanliness:
#    No extra markers, colors, or overlays are added. The scene remains a clean lattice of repeated
#    motifs with one oddball per row, and the output remains a clean single-column lattice.
#
# g) Color inference:
#    No color remapping is introduced. Colors keep the original roles: lattice, cell background,
#    and sprite color.

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice,
    then sort the extracted cells by the source odd-cell column from left to right.
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
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(L))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(L))

    # Cell slices are the gaps between consecutive grid lines.
    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    # Background inside cells = most common non-lattice color.
    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    odd_records: list[tuple[int, int, np.ndarray]] = []
    for row_idx, (ra, rb) in enumerate(r_slices):
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for (ca, cb) in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        odd_records.append((int(uniq_idx), int(row_idx), cells[int(uniq_idx)]))

    # Sort by source column. Stable row-order tie-break keeps behavior deterministic if ties appear.
    odd_records.sort(key=lambda rec: (int(rec[0]), int(rec[1])))
    chosen_cells = [cell for _odd_col, _row_idx, cell in odd_records]

    # Output: same height, single lattice column => width = cell_w + 2 (outer gridlines).
    out_w = int(cell_w + 2)
    out = np.full((H, out_w), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for out_row_idx, (ra, rb) in enumerate(r_slices):
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(out_row_idx)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = int(inner.shape[0] + 2)
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _make_unique_cells(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    bg: int,
    sp: int,
    n_rows: int,
) -> list[np.ndarray] | None:
    inner_area = int(base_inner.size)
    flip_sets = [
        fs
        for k in range(1, min(3, inner_area) + 1)
        for fs in itertools.combinations(range(inner_area), k)
    ]
    if len(flip_sets) < int(n_rows):
        return None

    pick_ids = rng.choice(len(flip_sets), size=int(n_rows), replace=False)
    unique_cells: list[np.ndarray] = []
    for pick_id in pick_ids:
        variant_inner = base_inner.copy()
        for idx in flip_sets[int(pick_id)]:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        unique_cells.append(_make_cell(variant_inner, bg))
    return unique_cells


def _sample_distinct_odd_cols(rng: np.random.Generator, n_rows: int, n_cols: int) -> list[int]:
    slots = sorted(int(x) for x in rng.choice(n_cols, size=int(n_rows), replace=False))
    perm = rng.permutation(int(n_rows))
    if np.array_equal(perm, np.arange(int(n_rows))):
        a, b = rng.choice(int(n_rows), size=2, replace=False)
        perm[int(a)], perm[int(b)] = perm[int(b)], perm[int(a)]
    return [int(slots[int(i)]) for i in perm]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(n_rows, 5))

    inner_size = int(cell_size - 2)
    inner_area = int(inner_size * inner_size)
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    unique_cells = _make_unique_cells(rng, base_inner, bg, sp, n_rows)
    if unique_cells is None:
        return None

    odd_cols = _sample_distinct_odd_cols(rng, n_rows, n_cols)
    sort_order = sorted(range(n_rows), key=lambda r: int(odd_cols[r]))
    if sort_order == list(range(n_rows)):
        return None

    out_cells = [[unique_cells[int(r)].copy()] for r in sort_order]
    out = _build_lattice(out_cells, lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[int(odd_cols[row])] = unique_cells[int(row)].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
