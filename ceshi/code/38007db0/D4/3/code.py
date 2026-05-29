# Variant of 38007db0
# Dimension: D4 — 变换：直接复制无变换 (transform)
# Change: 唯一单元格原样复制到输出，无颜色或形状变换... → 旋转后输出

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior
#    The input stays a lattice of square cells. In each lattice row, exactly one
#    cell is the odd one out. The original task copied that cell unchanged into a
#    single output column. This variant changes only the final transform stage:
#    after the odd cell is found, the whole square cell is rotated 90 degrees
#    counterclockwise before being placed in the output lattice.
#    Example trace on one row:
#    - Original: find odd cell [border + asymmetric inner sprite] -> paste as-is.
#    - Variant:  find the same odd cell -> apply np.rot90(cell, 1) -> paste.
#    The extraction logic is unchanged; only the output representation changes.
#
# b) Complexity calibration
#    The transform is a single fixed rotation, not a per-row choice among many
#    orientations. To keep that rule visually observable, the generator rejects
#    odd cells whose 90-degree rotation looks identical to the original, and also
#    rejects cases where the rotated odd cell would collapse back to the majority
#    base cell. This adds one spatial-reasoning layer without changing the core
#    odd-one-out search.
#
# c) Hidden assumptions review
#    - Gridline detection via fully uniform rows/cols still holds.
#    - Background detection via the most common non-lattice color still holds.
#    - Odd-cell detection by exact cell bytes still holds because the input
#      structure is unchanged.
#    - The old write-back step implicitly assumed identity transform. That is the
#      only behavior that must change.
#    - Rotation now relies on cells being square. The original spec already says
#      5x5 or 6x6 cells, but the solver now makes that assumption explicit.
#
# d) Placement legality
#    Rotating a square cell preserves its dimensions, so the original output
#    lattice geometry remains valid: same row slices, same output height, same
#    single output column width = cell_size + 2.
#
# e) Visual observability
#    ARC examples must let a human see the new rule. The generator therefore
#    guarantees that every odd cell changes under the fixed 90-degree rotation.
#    Several rows per sample repeat the same "find odd cell first" behavior while
#    showing the same rotation transform afterward.
#
# f) Visual cleanliness
#    No extra markers or overlays are introduced. The output remains the same
#    tidy single-column lattice, so the only new signal is the rotated sprite.
#
# g) Color-change inferability
#    No color mapping is introduced. Colors are preserved exactly; only spatial
#    orientation changes.
#
# New solver assumptions introduced by this variant
# -------------------------------------------------
# - Cells are square; otherwise a fixed 90-degree rotation would not fit back
#   into the original output layout.
# - The generator ensures every sample still has exactly one odd cell per row
#   and keeps the lattice geometry uniform, so the row/column slicing logic
#   remains valid.


def _rot90_ccw(cell: np.ndarray) -> np.ndarray:
    return np.rot90(cell, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice,
    then rotate each extracted cell 90 degrees counterclockwise.
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

    cell_h = int(r_slices[0][1] - r_slices[0][0])
    cell_w = int(c_slices[0][1] - c_slices[0][0])
    assert all(int(rb - ra) == int(cell_h) for ra, rb in r_slices)
    assert all(int(cb - ca) == int(cell_w) for ca, cb in c_slices)
    assert int(cell_h) == int(cell_w)

    # Background inside cells = most common non-lattice color.
    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(_rot90_ccw(cells[int(uniq_idx)]))

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


def _is_visibly_rotated(cell: np.ndarray, base_cell: np.ndarray) -> bool:
    rotated = _rot90_ccw(cell)
    return not np.array_equal(rotated, cell) and not np.array_equal(rotated, base_cell)


def _collect_visible_variants(base_inner: np.ndarray, bg: int, sp: int) -> list[np.ndarray]:
    inner_area = int(base_inner.size)
    base_cell = _make_cell(base_inner, bg)
    variants: list[np.ndarray] = []
    for flip_count in range(1, 4):
        for flip_set in itertools.combinations(range(inner_area), flip_count):
            variant_inner = base_inner.copy()
            for idx in flip_set:
                variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
            variant_cell = _make_cell(variant_inner, bg)
            if _is_visibly_rotated(variant_cell, base_cell):
                variants.append(variant_cell)
    return variants


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

    base_inner = None
    visible_variants: list[np.ndarray] = []
    for _ in range(40):
        fill = int(rng.integers(min_fill, max_fill + 1))
        candidate = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        candidate.flat[on_idx] = int(sp)
        variants = _collect_visible_variants(candidate, bg, sp)
        if len(variants) >= int(n_rows):
            base_inner = candidate
            visible_variants = variants
            break
    if base_inner is None:
        return None

    pick_idx = rng.choice(len(visible_variants), size=n_rows, replace=False)
    unique_cells = [visible_variants[int(i)].copy() for i in pick_idx]
    rotated_cells = [_rot90_ccw(cell) for cell in unique_cells]
    out = _build_lattice([[cell] for cell in rotated_cells], lattice, bg)

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
