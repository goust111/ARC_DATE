# Variant of 38007db0
# Dimension: D1 — 匹配条件：行对齐选取唯一项 (match_cond)
# Change: s.pos.row == d.pos.row — 每行中出现一次的奇异单元格被提取到输出对应行... → 频率最少项选取（而非唯一项）
#
# Design analysis
# spec:
#   task: Per-row least-frequent-pattern extraction from a lattice grid into a single-column lattice.
#   preserved_structure:
#     - Input/output are still square-cell lattices with 1px lattice lines.
#     - Each cell still has 1px bg padding and a small inner sprite using a single sprite color.
#     - Output is still a single-column lattice, row-aligned with the input lattice.
#   concrete_variant:
#     - Original per-row pattern counts were [1 oddball, C-1 majority].
#     - This variant changes each row to exactly two pattern types with counts [2 minority, 3 majority].
#     - The minority pattern is row-specific and appears twice in the input row.
#     - The output keeps one representative copy of that minority pattern in the same row.
#   behavior_trace:
#     - Example row before: [A, A, B, A] and solver used "find count == 1", so B is extracted.
#     - Example row now:    [A, B, A, B, A] and solver must compare counts, observe B:2 < A:3, then extract B.
#     - Difference: uniqueness is no longer sufficient; the selected pattern can occupy multiple cells.
#   complexity_calibration:
#     - A strict non-unique minority needs at least 5 columns; with 4 columns the only non-unique split is 2 vs 2,
#       which is tied and ambiguous.
#     - ARC grids are bounded, so to fit 5 columns inside the usual <=30 canvas, cell size is reduced to 4x4.
#       That still leaves a visible 2x2 inner sprite plus the original 1px bg frame.
#     - Each row contains only two signatures, avoiding underconstrained "which rare pattern?" ambiguity.
#   implicit_assumption_audit:
#     - Gridline detection via fully uniform rows/cols still holds unchanged.
#     - Background inference as the most common non-lattice color still holds because every 4x4 cell contains a
#       thick bg border (12 pixels) and the 2x2 sprite uses at most 3 foreground pixels.
#     - The original `count == 1` assumption is invalid in this variant and is replaced with
#       `count == min(counts)` plus an assertion that exactly one signature attains that minimum.
#     - Copying the first matched minority cell into the output is safe because the two minority cells in a row are exact copies.
#   placement_legality:
#     - All cells are placed on the regular lattice; no overlap is possible because placement uses fixed cell slots.
#     - The generator chooses exactly two distinct minority columns per row, so every row satisfies the 3-vs-2 rule.
#   visual_observability:
#     - Humans can see three repeated cells and two repeated "other" cells in the same row, so the shift from
#       "unique oddball" to "least frequent pattern" is visible across examples.
#   cleanliness:
#     - Only three colors are used, and each row has only two cell patterns, so the image stays easy to parse.
#   color_inference:
#     - No rule depends on numeric color ids; colors are only used as visible lattice/bg/sprite roles.
#   new_solver_assumptions_checked_against_generator:
#     - Solver assumes a strict least-frequency signature exists in every row.
#     - Generator guarantees that by constructing each row as exactly 3 copies of one pattern and 2 copies of another.

from __future__ import annotations

from collections import Counter
import itertools

import numpy as np

from arc_utils import paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the least-frequent cell pattern from each lattice row into a single-column lattice.

    The lattice parsing and output framing are unchanged from the original task.
    The only semantic change is row selection: instead of looking for a signature with
    frequency 1, choose the signature with the minimum frequency in that row.
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

    # Background inside cells = most common non-lattice color.
    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for (ra, rb) in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for (ca, cb) in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        assert len(cnt) >= 2
        min_freq = min(int(v) for v in cnt.values())
        rare_sigs = [sig for sig, freq in cnt.items() if int(freq) == int(min_freq)]
        assert len(rare_sigs) == 1
        rare_sig = rare_sigs[0]
        rare_idx = next(j for j, sig in enumerate(sigs) if sig == rare_sig)
        chosen_cells.append(cells[int(rare_idx)])

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
    size = inner.shape[0] + 2
    cell = np.full((size, size), int(bg), dtype=int)
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
    cell_size = 4
    n_rows = int(rng.integers(3, 5))
    n_cols = 5

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    # Ensure both bg and sp appear in the base sprite; 2x2 is the smallest size that still leaves
    # enough distinct variants for 3-4 rows while fitting 5 columns inside ARC's canvas budget.
    fill = int(rng.integers(1, inner_area))
    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    minority_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    all_flip_sets = [
        flip_set
        for flip_count in range(1, inner_area + 1)
        for flip_set in itertools.combinations(range(inner_area), flip_count)
    ]
    rng.shuffle(all_flip_sets)
    for _ in range(n_rows):
        flip_set = next((fs for fs in all_flip_sets if fs not in used_flip_sets), None)
        if flip_set is None:
            return None
        used_flip_sets.add(flip_set)
        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        minority_cells.append(_make_cell(variant_inner, bg))

    out = _build_lattice([[cell] for cell in minority_cells], lattice, bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        minority_cols = sorted(map(int, rng.choice(n_cols, size=2, replace=False)))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        for col in minority_cols:
            row_cells[int(col)] = minority_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
