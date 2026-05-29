# Combo variant of 38007db0 (index 17)
# combo_spec: D4/2+D7/0
# Name: 极小差异提取

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The original task copies each row's odd cell into a single-column lattice.
#    D4/2 changes the copied object from "the whole odd cell" to "only the
#    pixels where the odd cell differs from that row's repeated base cell".
#    D7/0 then tightens the oddness magnitude so the unique cell differs by
#    exactly one inner pixel. In the fused task, those two changes reinforce each
#    other: subtracting the base from the odd cell leaves a 1-pixel dot.
#    Example with a 3x3 inner sprite:
#      base inner   = [[sp,bg,sp],
#                      [bg,sp,bg],
#                      [sp,bg,bg]]
#      unique inner = [[sp,bg,sp],
#                      [bg,sp,sp],
#                      [sp,bg,bg]]
#      original output cell = unique cell with the full shape
#      fused output cell    = only the extra pixel at (1,2)
#    So the solver must still detect the per-row oddball, but it now also needs
#    the repeated majority cell as a template and must emit just their delta.
#
# b) Complexity calibration:
#    A 1-pixel diff is intentionally subtle, so the generator must avoid making
#    the rule invisible. If the unique cell removed a pixel (sp->bg), then the
#    diff-only output would become blank because the differing value would be bg.
#    To keep the fusion visually inferable while honoring "exactly 1 pixel
#    differs", the generator only uses bg->sp additions. This keeps the control
#    variable minimal without collapsing the task into empty outputs.
#
# c) Implicit-assumption audit:
#    - Uniform lattice-row/column detection still works because the geometry is
#      unchanged.
#    - Inferring the lattice color from uniform rows/cols still works.
#    - The original solver assumed "odd cell itself is the output payload". That
#      is false now; main() must identify both the unique cell and the repeated
#      base cell in the same row, then compute a diff mask.
#    - New assumption: each row has exactly one signature occurring once and one
#      majority signature occurring at least twice. The generator guarantees
#      n_cols >= 3 and places only one odd cell per row.
#    - New assumption: visible diff pixels come from unique_cell != base_cell and
#      equal the sprite color. The generator guarantees exactly one bg->sp flip.
#
# d) Placement legality:
#    The lattice is still regular, all cells stay square, and the changed pixel
#    is chosen from the inner sprite area only. No edit touches lattice lines or
#    crosses cell boundaries. Distinct rows use distinct addition sites so their
#    unique variants remain different.
#
# e) Visual observability:
#    Across examples, the input still shows a repeated base sprite and one almost
#    identical odd cell in each row. The output's single-column lattice contains
#    exactly one dot per row in the position where the odd cell gained an extra
#    sprite pixel. Because the repeated base sprite is visible beside the odd
#    cell, a human can infer "compare against the row template, keep only the
#    extra pixel" from a few samples.
#
# f) Visual cleanliness:
#    The output becomes sparser than the source task, but it stays tidy because
#    every row still occupies one framed cell and each output cell contains at
#    most one non-background pixel. There is no overlap, recoloring, or layered
#    annotation.
#
# g) Change inferability:
#    No new colors are introduced. The only non-background output mark uses the
#    same sprite color already seen inside the input cells. The structure, not a
#    hidden color remapping, carries the rule.


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
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

    return H, W, lattice, row_lines, r_slices, c_slices


def _diff_cell(base_cell: np.ndarray, unique_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(unique_cell.shape, int(bg), dtype=int)
    mask = unique_cell != base_cell
    diff[mask] = unique_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the odd cell from each lattice row, then keep only its single
    differing pixel relative to the repeated base cell in that row.
    """
    g = np.array(grid, dtype=int)
    H, _, lattice, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_diffs: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)

        assert sum(1 for v in cnt.values() if int(v) == 1) == 1
        base_sig, base_count = cnt.most_common(1)[0]
        uniq_idx = next(i for i, s in enumerate(sigs) if int(cnt[s]) == 1)
        base_idx = next(i for i, s in enumerate(sigs) if s == base_sig)
        assert int(base_count) >= 2

        base_cell = cells[int(base_idx)]
        unique_cell = cells[int(uniq_idx)]
        diff_cell = _diff_cell(base_cell, unique_cell, bg)
        assert int(np.count_nonzero(diff_cell != int(bg))) == 1
        chosen_diffs.append(diff_cell)

    out = np.full((H, int(cell_w + 2)), int(bg), dtype=int)
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    min_fill = max(3, math.ceil(0.35 * inner_area))
    max_fill = min(math.floor(0.6 * inner_area), inner_area - n_rows)
    if max_fill < min_fill:
        return None
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    add_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(bg)]
    if len(add_positions) < n_rows:
        return None

    chosen_positions = rng.choice(add_positions, size=n_rows, replace=False)

    diff_cells: list[np.ndarray] = []
    unique_cells: list[np.ndarray] = []
    for flat_idx in chosen_positions:
        diff_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        diff_inner.flat[int(flat_idx)] = int(sp)

        variant_inner = base_inner.copy()
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
