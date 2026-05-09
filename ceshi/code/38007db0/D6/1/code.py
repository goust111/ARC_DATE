# Variant of 38007db0
# Dimension: D6 — 格子结构参数：单元格尺寸 (meta)
# Change: 所有单元格为统一的5x5或6x6正方形... → 矩形单元格（非正方形）

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from arc_utils import paste

# Design analysis for the D6 variant
# ----------------------------------
# variant_spec:
#   task: Per-row odd-one-out extraction from a lattice grid into a single-column lattice.
#   colors: exactly 3 distinct -- lattice L, background bg, sprite sp
#   lattice:
#     gridline_width: 1
#     cell_rows: 3-4
#     cell_cols: 3-4
#     cell_size: one uniform non-square rectangle per instance
#     allowed_sizes:
#       - 5x7
#       - 5x8
#       - 6x8
#       - 7x5
#       - 8x5
#       - 8x6
#   cell_interior:
#     border: 1px bg padding on all four sides of each cell
#     inner_sprite: drawn in sp on bg inside a (cell_h-2)x(cell_w-2) rectangle
#     fill_density: 40-90% of inner area
#   row_structure:
#     base_sprite: one rectangular base sprite shared by all majority cells
#     unique_cell: exactly 1 per row, differs from the base by 1-3 inner flips
#     unique_cells_differ: each row uses a distinct variant
#   output:
#     single-column lattice preserving the original row heights
#     width: cell_w + 2
#
# a) Concretization and behavior trace
#    The visible change is that every cell is a rectangle rather than a square.
#    Example: with cell_h=5 and cell_w=7, each cell interior is 3x5. The input is
#    a 3-4 by 3-4 lattice of 5x7 rectangles separated by 1px lattice lines. Each row
#    still has one odd cell, but that odd cell now lives in a rectangular frame.
#    Original flow: generator used one scalar cell_size, _make_cell made an s x s cell,
#    _build_lattice stepped by the same amount in both axes, and main only tracked cell_w.
#    Variant flow: generator carries cell_h/cell_w separately, _make_cell returns
#    (inner_h+2) x (inner_w+2), _build_lattice places cells on independent vertical and
#    horizontal strides, and main explicitly validates one uniform cell height and one
#    uniform cell width instead of relying on a square layout.
#
# b) Complexity calibration
#    The rule remains the same so the task is not made easier; the visual change is clear
#    because width and height differ by at least 2. The size choices are a small discrete
#    set, so examples repeat enough structure for humans to infer "same task, rectangular
#    cells now" rather than facing arbitrary free-form geometry.
#
# c) Hidden assumptions audit on the original main()
#    - row_lines / col_lines detection only assumes 1px full-length lattice lines: still valid.
#    - r_slices / c_slices already recover rectangular gaps: still valid.
#    - cell_w = c_slices[0][1] - c_slices[0][0] implicitly ignored cell height. That was
#      harmless for squares, but for rectangles we should also compute cell_h and assert
#      uniformity on both axes.
#    - cell.tobytes() comparison is safe for rectangles as long as every cell slice has the
#      same shape; the new generator guarantees that and main now asserts it directly.
#    - _make_cell used inner.shape[0] for both axes and _build_lattice used one cell_size
#      for both axes. Those are the main square-only generator assumptions and must change.
#
# d) Placement legality
#    The generator uses one cell shape across the whole lattice, 1px gridlines, and bg-filled
#    padding inside each cell. Lattice dimensions are computed from cell_h/cell_w, so cells
#    never overlap and never leave the canvas. The output uses the same row slices and one
#    column of the chosen rectangular cells, so the solver reads it consistently.
#
# e) Visual observability
#    Humans can see the dimension change immediately because the cell frame itself is
#    rectangular. The odd-one-out rule is unchanged and still supported by repeated majority
#    copies within each row. No invisible numeric ordering or hidden metadata is used.
#
# f) Visual neatness
#    The scene remains a clean lattice with only three colors and one sprite layer inside
#    each cell. No overlapping objects or secondary markers are added, so the grid stays easy
#    to parse.
#
# g) Color inference
#    Colors are unchanged from the original task: one lattice color, one background color,
#    one sprite color. The variant depends only on geometry, not on any new color mapping.
#
# New solver assumptions introduced here and matched by the generator
# - All lattice rows have the same cell height.
# - All lattice columns have the same cell width.
# - Every row contains exactly one cell pattern whose full rectangular bitmap appears once.
# - The lattice keeps 1px outer border lines on all four sides.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique rectangular cell from each lattice row into a single-column lattice.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(L))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(L))
    assert len(row_lines) >= 2 and len(col_lines) >= 2

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_heights = {int(rb - ra) for ra, rb in r_slices}
    cell_widths = {int(cb - ca) for ca, cb in c_slices}
    assert len(cell_heights) == 1 and len(cell_widths) == 1
    cell_h = next(iter(cell_heights))
    cell_w = next(iter(cell_widths))

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        assert all(cell.shape == (cell_h, cell_w) for cell in cells)
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)])

    out = np.full((H, int(cell_w) + 2), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        assert int(rb - ra) == int(cell_h)
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    inner_h, inner_w = map(int, inner.shape)
    cell = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_h, cell_w = map(int, cells[0][0].shape)
    assert all(cell.shape == (cell_h, cell_w) for row in cells for cell in row)

    grid_h = rows * (cell_h + 1) + 1
    grid_w = cols * (cell_w + 1) + 1
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    for r in range(rows + 1):
        grid[r * (cell_h + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_w + 1)] = int(lattice)

    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_h + 1), 1 + c * (cell_w + 1))
    return grid


def _sample_variant(
    base_inner: np.ndarray,
    bg: int,
    sp: int,
    rng: np.random.Generator,
    used: set[bytes],
) -> np.ndarray | None:
    inner_area = int(base_inner.size)
    for _ in range(80):
        flip_count = int(rng.integers(1, 4))
        flip_idx = rng.choice(inner_area, size=flip_count, replace=False)
        variant = base_inner.copy()
        for idx in flip_idx:
            idx = int(idx)
            variant.flat[idx] = int(sp) if int(variant.flat[idx]) == int(bg) else int(bg)
        sig = variant.tobytes()
        if sig != base_inner.tobytes() and sig not in used:
            used.add(sig)
            return variant
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_h, cell_w = map(int, rng.choice(np.array([(5, 7), (5, 8), (6, 8), (7, 5), (8, 5), (8, 6)])))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_h = int(cell_h - 2)
    inner_w = int(cell_w - 2)
    inner_area = int(inner_h * inner_w)
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = max(min_fill, math.floor(0.9 * inner_area))
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_h, inner_w), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    unique_cells: list[np.ndarray] = []
    used_variants = {base_inner.tobytes()}
    for _ in range(n_rows):
        variant_inner = _sample_variant(base_inner, bg, sp, rng, used_variants)
        if variant_inner is None:
            return None
        unique_cells.append(_make_cell(variant_inner, bg))

    out = _build_lattice([[cell] for cell in unique_cells], lattice, bg)

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
