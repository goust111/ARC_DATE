# Combo variant of 38007db0 (index 11)
# combo_spec: D3/2+D4/2+D8/0
# Name: 位置排序差异提取

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

# Design analysis
# ---------------
# a) Concrete behavior:
#    Original 38007db0 extracts the full odd cell from each lattice row and keeps
#    source row order. This combo changes two reasoning steps at once:
#    1. Per row, recover both the odd cell and the repeated majority cell, then
#       keep only the odd-vs-majority differing pixels.
#    2. Across rows, sort those difference cells by the odd cell's source column,
#       left to right, before placing them into the single output column.
#    Small example with 3 rows and odd columns [2, 0, 1]:
#    - row 0 odd = base + pixels A
#    - row 1 odd = base + pixels B
#    - row 2 odd = base + pixels C
#    Original output order would be [odd0, odd1, odd2].
#    This combo outputs [diff1, diff2, diff0], because the visible ordering key
#    is column position 0 < 1 < 2 and each cell shows only the added pixels.
#
# b) Complexity calibration:
#    Raw "difference only" can become visually ambiguous if unique cells remove
#    pixels, because removals collapse back to bg. To keep the transform legible,
#    the generator uses only bg->sp additions, with 2-4 pixels per row and compact
#    local shapes. Sorting remains observable because odd columns are all distinct
#    and the generator rejects the identity permutation, so D3 is always active.
#
# c) Implicit assumption audit:
#    - Uniform full rows/cols still identify lattice lines; layout is unchanged.
#    - Consecutive lattice lines still delimit equal square cells.
#    - Each row still contains exactly one signature with count 1 and one majority
#      signature with count >= 2. The solver now relies on both, not only uniq_idx.
#    - The original placement assumption "output row i comes from input row i" is
#      false after D3/2. We therefore carry odd_col together with the diff cell and
#      sort records before writing output bands.
#    - The new diff step assumes that visible differences are recoverable from the
#      odd cell alone. The generator guarantees this by using additive variants only.
#    - D8/0 removes column ties, so the sort key is strict. The solver still keeps
#      a row-index tie break for robustness, but generated inputs never need it.
#
# d) Placement legality:
#    Cells stay on a regular 1-pixel lattice. Unique variants are formed only by
#    toggling interior bg pixels to sp inside the cell's padded inner sprite area.
#    Distinct odd columns are sampled without replacement from valid columns, so
#    no overlap, no out-of-bounds placement, and no duplicate-column violations.
#
# e) Visual observability:
#    A human can see two independent cues in 2-4 examples:
#    - within each row, the odd cell differs from repeated neighbors by a small
#      added patch, and the output keeps only that patch;
#    - across rows, the output order matches the left-to-right odd-cell positions,
#      not the top-to-bottom source rows.
#    No hidden numeric-color ordering is used; the key is purely spatial.
#
# f) Visual cleanliness:
#    The scene remains a clean lattice with only three colors. Output cells are
#    sparse but not noisy because difference patches are compact. There are no
#    extra overlays or markers competing with the lattice structure.
#
# g) Color inference:
#    No color remapping is introduced. The output uses the original lattice color,
#    the original cell background, and the original sprite color. The only changes
#    are structural: subtract common content, then reorder by visible column.


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape

    row_lines = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(lattice))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(lattice))

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    return int(h), int(w), int(lattice), row_lines, r_slices, c_slices


def _diff_cell(base_cell: np.ndarray, unique_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(unique_cell.shape, int(bg), dtype=int)
    mask = unique_cell != base_cell
    diff[mask] = unique_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract per-row odd cells as difference-only sprites, then sort those sprites
    by the odd cell's source column before placing them into a single output column.
    """
    g = np.array(grid, dtype=int)
    h, _, lattice, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    odd_records: list[tuple[int, int, np.ndarray]] = []
    for row_idx, (ra, rb) in enumerate(r_slices):
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)

        uniq_count = sum(1 for v in cnt.values() if int(v) == 1)
        assert uniq_count == 1
        base_sig, base_count = cnt.most_common(1)[0]
        assert int(base_count) >= 2

        odd_col = next(j for j, s in enumerate(sigs) if int(cnt[s]) == 1)
        base_idx = next(j for j, s in enumerate(sigs) if s == base_sig)
        diff = _diff_cell(cells[int(base_idx)], cells[int(odd_col)], bg)
        odd_records.append((int(odd_col), int(row_idx), diff))

    odd_records.sort(key=lambda rec: (int(rec[0]), int(rec[1])))

    out_w = int(cell_w + 2)
    out = np.full((h, out_w), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for out_row_idx, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = odd_records[int(out_row_idx)][2]

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


def _compact_addition_sets(base_inner: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    inner_size = int(base_inner.shape[0])
    bg_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(bg)]

    candidate_sets: list[tuple[int, ...]] = []
    max_count = min(4, len(bg_positions))
    for count in range(2, max_count + 1):
        for combo in itertools.combinations(bg_positions, count):
            coords = [divmod(int(idx), inner_size) for idx in combo]
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if bbox_area <= count + 2:
                candidate_sets.append(tuple(int(x) for x in combo))
    return candidate_sets


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

    min_fill = math.ceil(0.3 * inner_area)
    max_fill = math.floor(0.65 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    candidate_sets = _compact_addition_sets(base_inner, bg)
    if len(candidate_sets) < int(n_rows):
        return None

    chosen_ids = rng.choice(len(candidate_sets), size=int(n_rows), replace=False)
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

    odd_cols = _sample_distinct_odd_cols(rng, n_rows, n_cols)
    sort_order = sorted(range(n_rows), key=lambda row: int(odd_cols[row]))
    if sort_order == list(range(n_rows)):
        return None

    out_cells = [[diff_cells[int(row)].copy()] for row in sort_order]
    out = _build_lattice(out_cells, lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[int(odd_cols[row])] = unique_cells[int(row)].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
