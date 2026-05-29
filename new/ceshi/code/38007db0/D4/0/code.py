# Variant of 38007db0
# Dimension: D4 — 变换：直接复制无变换 (transform)
# Change: 唯一单元格原样复制到输出，无颜色或形状变换... → 重新着色为统一颜色

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for this D4 variant
# ===================================
# a) Concrete behavior:
#    The input keeps the original lattice layout: each row contains several same-size cells,
#    and exactly one cell per row has an inner sprite that differs from the row's repeated base.
#    The output still extracts those unique cells into a single-column lattice, but now the
#    extracted sprite pixels are recolored to the lattice color L. Only the foreground sprite
#    changes color; the bg padding inside each cell stays bg, so the recolored shape remains
#    visually separated from the gridlines.
#    Small trace:
#    - Original row cells: [base(sp), unique(sp), base(sp)] -> output unique(sp)
#    - Variant row cells:  [base(sp), unique(sp), base(sp)] -> output unique(L)
#    The difference happens after odd-one-out selection, during rendering of the output cell.
#
# b) Complexity calibration:
#    The recolor target is fixed to the visible lattice color rather than chosen from many
#    possibilities. That keeps the added rule simple but still observable: the output has the
#    same extracted shapes, yet their foreground color no longer matches the input sprite color.
#    The 1px bg padding around the sprite is preserved so recoloring does not visually merge
#    the sprite with the surrounding lattice frame.
#
# c) Implicit assumptions review:
#    - Uniform-row/col detection assumes gridlines are solid rows/cols of a single color.
#      This still holds because the input construction is unchanged.
#    - Lattice color inference assumes the dominant uniform row/col color is the true lattice.
#      Still valid.
#    - bg inference assumes bg is the most common non-lattice color in the input. Still valid
#      because only the output is recolored; input frequencies remain in the original regime.
#    - Odd-one-out detection uses exact cell bytes. Still valid because input cells keep their
#      original sprite color sp.
#    - New assumption: cell interiors contain no lattice-colored foreground before recoloring.
#      Generator enforces distinct colors for lattice/bg/sprite, so recoloring all non-bg pixels
#      inside a chosen cell is safe and cannot accidentally preserve old colors.
#
# d) Placement legality:
#    Cells remain square, uniformly spaced by 1px lattice lines, and each unique sprite variant
#    is created by flipping inner pixels only. No object overlaps or boundary issues are added.
#
# e) Visual observability:
#    Across 2-4 examples, a human sees the same extraction behavior as the original task, but
#    every extracted shape switches from sprite color sp to lattice color L. Because the shape
#    remains unchanged and the color target is structurally anchored to the gridlines, the new
#    rule is inferable from few examples.
#
# f) Visual cleanliness:
#    Recoloring to L could be messy if the sprite touched the cell border, but the original cell
#    template has a guaranteed bg border of width 1, so the recolored sprite stays isolated.
#
# g) Color-rule inferability:
#    The mapping is single-target and repeated across all rows within a sample: extracted
#    foreground -> lattice color. There are only three input colors and two output colors, which
#    provides strong evidence for the recolor rule rather than an arbitrary color remapping.


def _recolor_extracted_cell(cell: np.ndarray, bg: int, target: int) -> np.ndarray:
    """Recolor the extracted cell's foreground sprite to a uniform target color."""
    out = np.array(cell, copy=True)
    out[out != int(bg)] = int(target)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice,
    then recolor the extracted sprite to the lattice color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == int(L)])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == int(L)])

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

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
        chosen_cells.append(_recolor_extracted_cell(cells[int(uniq_idx)], bg, L))

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
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    unique_cells: list[np.ndarray] = []
    recolored_output_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _row in range(n_rows):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [fs for fs in candidates if fs not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)
        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg
        unique_cell = _make_cell(variant_inner, bg)
        unique_cells.append(unique_cell)
        recolored_output_cells.append(_recolor_extracted_cell(unique_cell, bg, lattice))

    out = _build_lattice([[cell] for cell in recolored_output_cells], lattice, bg)

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
