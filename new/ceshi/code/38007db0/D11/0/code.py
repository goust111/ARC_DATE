# Variant of 38007db0
# Dimension: D11 — 格子线宽度 (meta)
# Change: 格子线宽度为1像素的点阵格线 → 格子线宽度为2像素的连续格线带

from __future__ import annotations

from collections import Counter
import itertools
import math
import numpy as np

# Design analysis for the D11 variant
# -----------------------------------
# variant_spec:
#   task: Per-row odd-one-out extraction from a lattice grid into a single-column lattice.
#   preserved_rule:
#     - Every lattice row still contains exactly one unique cell.
#     - The output still keeps one cell per row, in the same top-to-bottom order.
#   changed_dimension:
#     - The lattice/gridline width changes from 1 px to 2 px.
#     - This affects both the input and the output visuals and the parsing logic.
#   concrete_pixel_structure:
#     - Original: line rows/cols were isolated single indices like [0, 6, 12, 18].
#     - Variant: each line becomes a 2-row / 2-col band like [0,1], [7,8], [14,15].
#     - Cells remain square 5x5 or 6x6 blocks between those bands.
#   behavior_trace:
#     - Original solver assumed consecutive uniform line indices define cells by +1/-1 offsets.
#     - In the variant, uniform indices come in runs. Using raw indices directly would treat each
#       band row as a separate line, which breaks output reconstruction width and relies on
#       accidental filtering of empty slices.
#     - New logic first groups consecutive uniform rows/cols into bands, then uses the gaps between
#       bands as cell slices. Example with 2px lines:
#         row line indices: [0,1,7,8,14,15]
#         grouped bands: [(0,1), (7,8), (14,15)]
#         cell slices: rows [2:7], [9:14]
#   complexity_calibration:
#     - The change is visually obvious in every example because all separators become thick bands.
#     - The underlying object vocabulary stays small: exactly 3 colors and one repeated base sprite.
#     - We avoid adding new shape logic, so the task remains the same ARC concept with a single
#       meta change rather than becoming a new puzzle.
#   implicit_assumption_review:
#     - "uniform rows/cols == individual lines": no longer valid; replaced with contiguous bands.
#     - "output width = cell_w + 2": only valid for 1px borders; now it must be cell_w + 2*line_w.
#     - "set out[:,0] and out[:,-1] to L": only paints 1px side borders; now we paint full bands.
#     - "for r in row_lines: out[r,:]=L": raw line indices still works, but grouping is safer and
#       lets us validate consistent band thickness.
#   placement_legality:
#     - Generator builds grids with exact dimensions rows*cell + (rows+1)*line_w, so nothing
#       overlaps or goes out of bounds.
#     - Cells are pasted only into the interior gaps between horizontal/vertical lattice bands.
#     - The solver assumes every row/col band has the same width and every cell slice has the same
#       size; the generator guarantees both.
#   visual_observability:
#     - Humans can still infer the same row-wise odd-one-out rule from 2-4 examples.
#     - The D11 change is directly visible because every boundary is thickened, not hidden in any
#       color encoding or invisible metadata.
#   cleanliness:
#     - Thick bands do not overlap sprites; they simply consume separator space.
#     - The sprite language remains sparse enough to keep the lattice legible.


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev))
        start = prev = idx
    bands.append((start, prev))
    return bands


def _line_bands(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])

    row_bands = _group_consecutive([r for r in row_lines if int(g[r, 0]) == lattice])
    col_bands = _group_consecutive([c for c in col_lines if int(g[0, c]) == lattice])
    assert row_bands and col_bands

    row_widths = {int(b - a + 1) for a, b in row_bands}
    col_widths = {int(b - a + 1) for a, b in col_bands}
    assert len(row_widths) == 1 and len(col_widths) == 1
    assert next(iter(row_widths)) == next(iter(col_widths))
    return lattice, row_bands, col_bands


def _cell_slices(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    slices = [(int(bands[i][1]) + 1, int(bands[i + 1][0])) for i in range(len(bands) - 1)]
    slices = [(a, b) for a, b in slices if int(b) > int(a)]
    assert slices
    span_sizes = {int(b - a) for a, b in slices}
    assert len(span_sizes) == 1
    return slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice.

    Variant handling:
    - Lattice separators are contiguous 2-pixel-wide bands rather than single rows/cols.
    - The solver groups uniform lattice rows/cols into bands, then slices cells between bands.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    lattice, row_bands, col_bands = _line_bands(g)
    line_w = int(row_bands[0][1] - row_bands[0][0] + 1)

    r_slices = _cell_slices(row_bands)
    c_slices = _cell_slices(col_bands)
    cell_h = int(r_slices[0][1] - r_slices[0][0])
    cell_w = int(c_slices[0][1] - c_slices[0][0])
    assert cell_h == cell_w

    bg_candidates = [int(x) for x in g.flatten() if int(x) != lattice]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)])

    out_w = int(cell_w + 2 * line_w)
    out = np.full((H, out_w), int(bg), dtype=int)

    for a, b in row_bands:
        out[int(a) : int(b) + 1, :] = int(lattice)
    out[:, :line_w] = int(lattice)
    out[:, out_w - line_w :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), line_w : line_w + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int, line_w: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    step = cell_size + line_w
    grid = np.full((rows * cell_size + (rows + 1) * line_w, cols * cell_size + (cols + 1) * line_w), int(bg), dtype=int)

    for r in range(rows + 1):
        start = r * step
        grid[start : start + line_w, :] = int(lattice)
    for c in range(cols + 1):
        start = c * step
        grid[:, start : start + line_w] = int(lattice)

    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, line_w + r * step, line_w + c * step)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    line_w = 2
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
    used_flip_sets: set[tuple[int, ...]] = set()
    for _ in range(n_rows):
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
        unique_cells.append(_make_cell(variant_inner, bg))

    out = _build_lattice([[cell] for cell in unique_cells], lattice, bg, line_w)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg, line_w)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
