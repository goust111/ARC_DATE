# Combo variant of 38007db0 (index 16)
# combo_spec: D1/0+D3/3+D4/0
# Name: 列对齐原位重新着色
#
# Design analysis
# a) Concrete behavior and trace:
#    The original task scans each lattice row, finds the single odd cell in that row, and copies
#    those cells into a new one-column lattice. This combo changes three linked stages together:
#    scan by column instead of row, keep the original lattice footprint instead of extracting, and
#    recolor each kept odd cell's sprite to the lattice color.
#    Example on a 3x4 cell lattice:
#    - Original: each row has exactly one odd cell, output is a 3x1 strip.
#    - Combo input: each column has exactly one odd cell, but rows may have 0, 1, or 2 odd cells.
#      If column 0's odd cell is at row 2 and column 1's odd cell is at row 0, then output keeps
#      those two cells in their original coordinates, blanks every other cell, and recolors the
#      kept sprites from sprite color `sp` to lattice color `L`.
#    - Difference: detection axis changes, destination stays in-place, and rendering recolors the
#      surviving sprite instead of copying it verbatim.
#
# b) Complexity calibration:
#    The object vocabulary stays tight: exactly one shared base sprite, 3 colors total, 3-4 rows,
#    and 3-4 columns. The generator forces at least one row with zero odd cells and at least one
#    row with two or more odd cells. That makes the column rule visible and prevents accidental
#    degeneration into the original per-row problem.
#
# c) Hidden-assumption audit of the original solver:
#    - Uniform full rows/cols are still valid lattice detectors because the lattice geometry is
#      unchanged.
#    - Lattice color as the dominant uniform-line color still holds.
#    - Background as the most common non-lattice color still holds because every cell preserves a
#      1-pixel bg border and the sprite fill remains below 100%.
#    - The original loop `for each row slice` assumed every row contains exactly one unique cell.
#      That is false here; the unique-counting loop must move to column slices.
#    - The original output builder assumed width shrinks to one cell. That is false here; output
#      shape must equal input shape, with most cell interiors blanked.
#    - Recoloring introduces one new assumption: no input cell interior already uses the lattice
#      color. The generator guarantees this by choosing distinct `lattice`, `bg`, and `sp`.
#
# d) Placement legality:
#    Cells are still placed only into lattice gaps produced by `_build_lattice`, so there is no
#    overlap or out-of-bounds risk. Each column receives exactly one unique variant; all other
#    cells in that column are exact copies of the base cell.
#
# e) Visual observability:
#    A human can infer the rule from a few examples because every output preserves the full grid,
#    leaves some entire rows empty, leaves some rows with multiple marked cells, and recolors all
#    surviving odd sprites to the same visible lattice color. Those cues jointly point to
#    "scan vertically, keep in place, recolor."
#
# f) Visual cleanliness:
#    The kept cell is separated from the lattice lines by the original 1-pixel bg padding, so
#    recoloring to `L` does not visually fuse the sprite with the frame. Blanked cells are plain
#    bg, so the result stays sparse and readable.
#
# g) Color-rule inferability:
#    The recolor target is fixed to the already salient lattice color rather than an arbitrary new
#    mapping. Because all kept cells switch from `sp` to `L` while blank cells remain bg, the
#    color transformation has a single repeated explanation.
#
# New solver assumptions checked against the generator:
# - Every column has exactly one cell signature with frequency 1.
# - At least one row has zero odd cells, so row-wise extraction is invalid on generated samples.
# - At least one row has two or more odd cells, strengthening the evidence for column scanning.
# - All unique variants are distinct byte patterns, so frequency-1 matching is unambiguous.

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


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

    r_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
    c_slices = [(col_lines[j] + 1, col_lines[j + 1]) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if b > a]
    c_slices = [(a, b) for a, b in c_slices if b > a]
    assert r_slices and c_slices

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])
    return lattice, bg, row_lines, col_lines, r_slices, c_slices


def _recolor_kept_cell(cell: np.ndarray, bg: int, target: int) -> np.ndarray:
    out = np.array(cell, copy=True)
    out[out != int(bg)] = int(target)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scan each lattice column, keep only the odd cell in-place, and recolor its sprite to the
    lattice color. All non-odd cells are blanked to the cell background.
    """
    g = np.array(grid, dtype=int)
    lattice, bg, row_lines, col_lines, r_slices, c_slices = _parse_lattice(g)

    out = np.full(g.shape, int(bg), dtype=int)
    for r in row_lines:
        out[int(r), :] = int(lattice)
    for c in col_lines:
        out[:, int(c)] = int(lattice)

    for ca, cb in c_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ra, rb in r_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((i for i, sig in enumerate(sigs) if int(cnt[sig]) == 1), None)
        assert uniq_idx is not None
        ra, rb = r_slices[int(uniq_idx)]
        out[int(ra) : int(rb), int(ca) : int(cb)] = _recolor_kept_cell(cells[int(uniq_idx)], bg, lattice)

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


def _make_unique_variants(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    bg: int,
    sp: int,
    count: int,
) -> list[np.ndarray] | None:
    inner_area = int(base_inner.size)
    unique_cells: list[np.ndarray] = []
    used_flip_sets: set[tuple[int, ...]] = set()
    for _ in range(count):
        flip_count = int(rng.integers(1, 4))
        candidates = list(itertools.combinations(range(inner_area), flip_count))
        available = [flip_set for flip_set in candidates if flip_set not in used_flip_sets]
        if not available:
            return None
        flip_set = available[int(rng.integers(len(available)))]
        used_flip_sets.add(flip_set)

        variant_inner = np.array(base_inner, copy=True)
        for idx in flip_set:
            variant_inner.flat[int(idx)] = sp if int(base_inner.flat[int(idx)]) == int(bg) else int(bg)

        unique_cells.append(_make_cell(variant_inner, bg))
    return unique_cells


def _choose_odd_rows(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray | None:
    for _ in range(120):
        odd_rows = rng.integers(0, n_rows, size=n_cols)
        counts = np.bincount(odd_rows, minlength=n_rows)
        if int(np.min(counts)) == 0 and int(np.max(counts)) >= 2:
            return odd_rows
    return None


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

    unique_cells = _make_unique_variants(rng, base_inner, bg, sp, n_cols)
    if unique_cells is None:
        return None

    odd_rows = _choose_odd_rows(rng, n_rows, n_cols)
    if odd_rows is None:
        return None

    base_cell = _make_cell(base_inner, bg)
    blank_cell = _blank_cell(cell_size, bg)

    input_cells = [[base_cell.copy() for _ in range(n_cols)] for _ in range(n_rows)]
    output_cells = [[blank_cell.copy() for _ in range(n_cols)] for _ in range(n_rows)]
    for col in range(n_cols):
        row = int(odd_rows[col])
        input_cells[row][col] = unique_cells[col].copy()
        output_cells[row][col] = _recolor_kept_cell(unique_cells[col], bg, lattice)

    inp = _build_lattice(input_cells, lattice, bg)
    out = _build_lattice(output_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
