# Combo variant of 38007db0 (index 4)
# combo_spec: D3/3+D4/0
# Name: 原位重新着色标记

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


MARKER_COLOR = 3


# Design analysis for combo D3/3 + D4/0
# a) Concrete behavior and trace:
#    The source task finds one odd cell per lattice row and extracts those cells into a new
#    single-column lattice. This combo changes two stages at once:
#    - D3/3 changes destination geometry: do not extract to a new column; keep the full input
#      lattice footprint and blank every non-odd cell.
#    - D4/0 changes rendering of the kept odd cells: recolor their sprite pixels to one fixed
#      marker color that does not appear in the input.
#    Row trace:
#      Input row:  [base(sp), odd(sp), base(sp), base(sp)]
#      Original :  [odd(sp)] in a 1-column output lattice
#      Combo    :  [blank(bg), odd(marker), blank(bg), blank(bg)] in the same row positions
#    The odd cell is still detected by exact equality in the input. Only after selection do we
#    recolor it, and only the selected cell survives in the output.
# b) Complexity calibration:
#    The lattice geometry, number of rows/cols, and odd-one-out structure stay unchanged, so the
#    puzzle still has repeated evidence for the selection rule. The output is visibly different
#    from the original because width is preserved and from D3-only because the kept odd cells no
#    longer retain the input sprite color. Using one fixed marker color across samples keeps the
#    added rule low-entropy and learnable.
# c) Hidden assumptions audit:
#    - Uniform full rows/cols still identify lattice lines; generator keeps 1px solid gridlines.
#    - The dominant uniform-row/col color is still the lattice color.
#    - The most common non-lattice input color is still bg because each cell has a 1px bg border
#      and the marker color never appears in the input.
#    - Odd-cell detection still depends on exact cell-array equality in the input; generator keeps
#      one variant and C-1 exact copies of the base cell per row.
#    - The old assumption "output width shrinks to one cell" is invalid; new output shape equals
#      input shape.
#    - New assumption: the marker color is absent from the input and distinct from lattice/bg/sp,
#      otherwise the recolor dimension could visually collapse. generate_input() enforces that.
#    - Recoloring only non-bg pixels inside the preserved cell is safe because the lattice lines
#      are outside the cell slices and each cell has guaranteed bg padding around the sprite.
# d) Placement legality:
#    Cells remain square and are pasted strictly into lattice gaps. Unique variants are made by
#    flipping inner pixels only, so there is no overlap and no boundary violation.
# e) Visual observability:
#    From a few examples, a human can see two coupled regularities: all repeated cells disappear,
#    and the survivor in each row is marked by the same new color at its original location. The
#    new color is not an arbitrary per-cell mapping; it repeats everywhere and never appears in
#    the input, which makes the transform inferable.
# f) Visual cleanliness:
#    Output cells are either fully blank or contain one recolored sprite with bg padding, so the
#    lattice stays uncluttered. No overlapping annotations are introduced.
# g) Non-degeneration check target:
#    This combo should differ from D3-only on every valid sample by at least one pixel because the
#    preserved odd cell contains sprite pixels and those pixels are recolored to MARKER_COLOR.


def _parse_lattice(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
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

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])
    return lattice, bg, row_lines, col_lines, r_slices, c_slices


def _find_unique_index(cells: list[np.ndarray]) -> int:
    sigs = [cell.tobytes() for cell in cells]
    counts = Counter(sigs)
    uniq_idx = next((idx for idx, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
    assert uniq_idx is not None
    return int(uniq_idx)


def _recolor_cell(cell: np.ndarray, bg: int, marker: int) -> np.ndarray:
    out = np.array(cell, copy=True)
    out[out != int(bg)] = int(marker)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each row's odd cell in place, blank all other cells, and recolor the kept cell to a
    fixed marker color that is absent from the input.
    """
    g = np.array(grid, dtype=int)
    lattice, bg, row_lines, col_lines, r_slices, c_slices = _parse_lattice(g)

    out = np.full(g.shape, int(bg), dtype=int)
    for r in row_lines:
        out[int(r), :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)

    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        uniq_idx = _find_unique_index(cells)
        ca, cb = c_slices[int(uniq_idx)]
        out[int(ra) : int(rb), int(ca) : int(cb)] = _recolor_cell(cells[int(uniq_idx)], bg, MARKER_COLOR)

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = inner.shape[0] + 2
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _blank_cell(cell_size: int, bg: int) -> np.ndarray:
    return np.full((cell_size, cell_size), int(bg), dtype=int)


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

    palette = [color for color in range(10) if color != MARKER_COLOR]
    lattice, bg, sp = map(int, rng.choice(palette, size=3, replace=False))
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
    recolored_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _ in range(n_rows):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [flip_set for flip_set in candidates if flip_set not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)

        variant_inner = base_inner.copy()
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(variant_inner.flat[int(idx)]) == int(bg) else bg

        unique_cell = _make_cell(variant_inner, bg)
        unique_cells.append(unique_cell)
        recolored_cells.append(_recolor_cell(unique_cell, bg, MARKER_COLOR))

    base_cell = _make_cell(base_inner, bg)
    blank_cell = _blank_cell(cell_size, bg)
    odd_cols = [int(rng.integers(0, n_cols)) for _ in range(n_rows)]

    input_cells: list[list[np.ndarray]] = []
    output_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        inp_row = [base_cell.copy() for _ in range(n_cols)]
        out_row = [blank_cell.copy() for _ in range(n_cols)]
        inp_row[odd_cols[row]] = unique_cells[row].copy()
        out_row[odd_cols[row]] = recolored_cells[row].copy()
        input_cells.append(inp_row)
        output_cells.append(out_row)

    inp = _build_lattice(input_cells, lattice, bg)
    out = _build_lattice(output_cells, lattice, bg)

    if MARKER_COLOR in set(map(int, inp.flatten())):
        return None
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
