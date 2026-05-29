# Combo variant of 38007db0 (index 22)
# combo_spec: D1/0+D2/2+D4/2
# Name: 列对齐旋转异常差异提取

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste

# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    The original task scans each lattice row, finds the one exact oddball cell,
#    and copies that full cell into a single-column output. This combo changes
#    all three stages of that pipeline:
#    1. Scan by columns instead of rows.
#    2. In each column, compare cells by 180-degree rotation class instead of by
#       exact bitmap, so {0,180} counts as one family and {90,270} as the other.
#    3. After selecting the anomalous rotated cell, compare it against the
#       column's repeated base orientation and keep only the pixels that are
#       present in the anomaly but not in the base.
#    A concrete 4x3-cell column can look like [rot0, rot0, rot180, rot90].
#    - Old exact-byte logic sees two singleton bitmaps (rot180 and rot90), so it
#      cannot cleanly identify the intended source.
#    - New logic groups rot0+rot180 into one majority half-turn family of size 3
#      and rot90 into the singleton family of size 1, so rot90 is selected.
#    - The output cell is not rot90 copied as-is; it is rot90 minus rot0, keeping
#      only those sprite pixels where rot90 differs from rot0.
#    Repeating this for each column yields a single-row lattice output.
#
# b) Complexity calibration:
#    To make the rotation rule essential instead of cosmetic, each generated
#    column is constrained to contain exactly:
#    - two copies of rot0(base),
#    - one copy of rot180(base) as a same-family distractor,
#    - one copy of rot90(base) or rot270(base) as the true anomaly.
#    This keeps the hypothesis space small (only two rotation families) while
#    ensuring exact-frequency counting is insufficient. The row count is fixed to
#    4 so the generator can simultaneously provide a repeated exact base cell for
#    D4-style differencing and a same-family rotation distractor for D2.
#
# c) Implicit-assumption review of the original solver:
#    - Uniform lattice-row / lattice-column detection still works unchanged.
#    - Inferring the lattice color from uniform lines still works unchanged.
#    - Inferring bg as the most common non-lattice color still works because the
#      padding and empty output differences remain bg.
#    - The original loop `for (ra, rb) in r_slices` assumed one odd cell per row;
#      that is false now, so selection must iterate over columns.
#    - The original `Counter(cell.tobytes())` assumed all normal cells in the
#      comparison group are exact copies. That is false now because rot180 is a
#      deliberate distractor in the same semantic family as rot0.
#    - The original output geometry assumed width shrinks to one cell. That is
#      false now because D1/0 keeps all input columns and collapses height.
#    - The original output wrote the chosen cell directly. That is false now
#      because D4/2 requires recovering the repeated base orientation and writing
#      only anomaly-minus-base pixels.
#
# d) Placement legality:
#    `_build_lattice()` still places disjoint square cells separated by 1-pixel
#    lattice lines, so there is no overlap or boundary risk. Rotations happen
#    inside the inner sprite only, preserving the bg padding and cell footprint.
#
# e) Visual observability:
#    Across examples, a human sees column-wise repetition with one rotated oddball
#    per column, plus outputs that are sparse highlights rather than full copied
#    cells. At least one input row is forced to contain zero anomalies so a row
#    interpretation is visually contradicted. The rot180 distractor ensures the
#    anomaly is not merely "the only different bitmap"; it is the only member of
#    the other rotation family.
#
# f) Visual cleanliness:
#    The scene still uses only three colors and the same tidy lattice geometry as
#    the source task. Output cells contain only bg, lattice, and a few sprite
#    pixels, so there is no marker clutter or overlap ambiguity.
#
# g) Inferability of the transformation:
#    No new color mapping is introduced. The only color transformation is the
#    structural subtraction "keep odd-cell sprite pixels that the base cell lacks".
#    To keep that subtraction visible, the generator samples asymmetric base
#    sprites with enough empty space that rot90/rot270 create at least two
#    visible sprite additions relative to rot0.


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
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

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    return H, W, lattice, bg, row_lines, col_lines, r_slices, c_slices


def _half_turn_signature(arr: np.ndarray) -> bytes:
    a = arr.tobytes()
    b = np.rot90(arr, 2).tobytes()
    return a if a <= b else b


def _diff_cell(base_cell: np.ndarray, odd_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(odd_cell.shape, int(bg), dtype=int)
    mask = odd_cell != base_cell
    diff[mask] = odd_cell[mask]
    return diff


def _column_solution_cells(
    grid: np.ndarray,
) -> tuple[list[np.ndarray], int, int, int, list[int], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    H, W, lattice, bg, _row_lines, col_lines, r_slices, c_slices = _lattice_slices(g)
    assert len(r_slices) >= 4

    chosen_diffs: list[np.ndarray] = []
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        class_sigs = [_half_turn_signature(cell) for cell in cells]
        class_counts = Counter(class_sigs)

        uniq_classes = [sig for sig, n in class_counts.items() if int(n) == 1]
        assert len(class_counts) == 2 and len(uniq_classes) == 1
        uniq_idx = next(i for i, sig in enumerate(class_sigs) if sig == uniq_classes[0])

        majority_cells = [cell for i, cell in enumerate(cells) if int(i) != int(uniq_idx)]
        exact_counts = Counter(cell.tobytes() for cell in majority_cells)
        base_sig, base_count = exact_counts.most_common(1)[0]
        assert int(base_count) >= 2
        base_cell = next(cell for cell in majority_cells if cell.tobytes() == base_sig)
        odd_cell = cells[int(uniq_idx)]
        chosen_diffs.append(_diff_cell(base_cell, odd_cell, bg))

    return chosen_diffs, W, lattice, bg, col_lines, c_slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scan each lattice column, find the singleton 180-degree rotation family, and
    output only the anomaly-vs-base difference pixels in a single-row lattice.
    """
    chosen_diffs, W, lattice, bg, col_lines, c_slices = _column_solution_cells(grid)
    cell_h = int(chosen_diffs[0].shape[0])

    out_h = int(cell_h + 2)
    out_w = int(W)
    out = np.full((out_h, out_w), int(bg), dtype=int)
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


def _visible_additions(base_inner: np.ndarray, odd_inner: np.ndarray, sp: int, bg: int) -> int:
    return int(np.sum((odd_inner == int(sp)) & (base_inner == int(bg))))


def _is_good_base(inner: np.ndarray, bg: int, sp: int) -> bool:
    rots = [np.rot90(inner, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False

    area = int(inner.size)
    if int(np.count_nonzero(inner == int(bg))) < 3:
        return False
    if int(np.count_nonzero(inner == int(sp))) < 3:
        return False
    if int(np.count_nonzero(rots[0] != rots[1])) < max(4, area // 3):
        return False
    if int(np.count_nonzero(rots[0] != rots[2])) < max(3, area // 4):
        return False
    if _half_turn_signature(rots[0]) == _half_turn_signature(rots[1]):
        return False

    viable_odds = [k for k in (1, 3) if _visible_additions(rots[0], rots[k], sp, bg) >= 2]
    return len(viable_odds) >= 1


def _sample_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.3 * inner_area)
    max_fill = math.floor(0.65 * inner_area)

    for _ in range(250):
        fill = int(rng.integers(min_fill, max_fill + 1))
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        inner.flat[on_idx] = int(sp)
        if _is_good_base(inner, bg, sp):
            return inner
    return None


def _choose_odd_rows(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray | None:
    for _ in range(120):
        odd_rows = rng.integers(0, n_rows, size=n_cols)
        counts = np.bincount(odd_rows, minlength=n_rows)
        if int(np.min(counts)) == 0:
            return odd_rows
    return None


def _choose_rot180_rows(rng: np.random.Generator, n_rows: int, odd_rows: np.ndarray) -> np.ndarray:
    rows = np.empty_like(odd_rows)
    for idx, odd_row in enumerate(odd_rows):
        choices = [r for r in range(n_rows) if int(r) != int(odd_row)]
        rows[int(idx)] = int(rng.choice(choices))
    return rows


def _raw_singleton_count(cells: list[np.ndarray]) -> int:
    sigs = [cell.tobytes() for cell in cells]
    cnt = Counter(sigs)
    return sum(1 for sig in sigs if int(cnt[sig]) == 1)


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    _, _, lattice, bg, _row_lines, _col_lines, r_slices, c_slices = _lattice_slices(g)

    raw_singletons_per_col: list[int] = []
    visible_diff_pixels: list[int] = []
    odd_row_hist = [0 for _ in r_slices]
    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        raw_singletons_per_col.append(_raw_singleton_count(cells))

        class_sigs = [_half_turn_signature(cell) for cell in cells]
        class_counts = Counter(class_sigs)
        uniq_class = next(sig for sig, n in class_counts.items() if int(n) == 1)
        odd_idx = next(i for i, sig in enumerate(class_sigs) if sig == uniq_class)
        odd_row_hist[int(odd_idx)] += 1

        majority_cells = [cell for i, cell in enumerate(cells) if int(i) != int(odd_idx)]
        base_sig = Counter(cell.tobytes() for cell in majority_cells).most_common(1)[0][0]
        base_cell = next(cell for cell in majority_cells if cell.tobytes() == base_sig)
        diff = _diff_cell(base_cell, cells[int(odd_idx)], bg)
        visible_diff_pixels.append(int(np.sum(diff != int(bg))))

    return {
        "lattice": lattice,
        "raw_singletons_per_col": raw_singletons_per_col,
        "visible_diff_pixels": visible_diff_pixels,
        "odd_row_hist": odd_row_hist,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = 4
    n_cols = int(rng.integers(3, 5))

    base_inner = _sample_base_inner(rng, cell_size - 2, bg, sp)
    if base_inner is None:
        return None

    rotated_inners = [np.rot90(base_inner, k) for k in range(4)]
    viable_odd_ks = [k for k in (1, 3) if _visible_additions(base_inner, rotated_inners[k], sp, bg) >= 2]
    if not viable_odd_ks:
        return None

    odd_rows = _choose_odd_rows(rng, n_rows, n_cols)
    if odd_rows is None:
        return None
    rot180_rows = _choose_rot180_rows(rng, n_rows, odd_rows)

    base_cell = _make_cell(rotated_inners[0], bg)
    rot180_cell = _make_cell(rotated_inners[2], bg)
    in_cells = [[base_cell.copy() for _ in range(n_cols)] for _ in range(n_rows)]
    out_cells: list[np.ndarray] = []

    for col in range(n_cols):
        odd_k = int(rng.choice(viable_odd_ks))
        odd_cell = _make_cell(rotated_inners[odd_k], bg)
        diff_cell = _diff_cell(base_cell, odd_cell, bg)
        if int(np.sum(diff_cell != int(bg))) < 2:
            return None

        odd_row = int(odd_rows[col])
        rot180_row = int(rot180_rows[col])
        if int(odd_row) == int(rot180_row):
            return None

        in_cells[int(rot180_row)][col] = rot180_cell.copy()
        in_cells[int(odd_row)][col] = odd_cell.copy()
        out_cells.append(diff_cell)

        col_cells = [in_cells[r][col] for r in range(n_rows)]
        class_counts = Counter(_half_turn_signature(cell) for cell in col_cells)
        if sorted(map(int, class_counts.values())) != [1, n_rows - 1]:
            return None
        if _raw_singleton_count(col_cells) != 2:
            return None
        exact_counts = Counter(cell.tobytes() for cell in col_cells)
        if int(exact_counts.most_common(1)[0][1]) != 2:
            return None

    out = _build_lattice([out_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 400 attempts")
