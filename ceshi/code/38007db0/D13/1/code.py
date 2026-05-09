# Variant of 38007db0
# Dimension: D13 — 干扰机制 (meta)
# Change: 无干扰项 — 每行恰好一个唯一项，其余完全相同... → 跨行共享变体作为干扰

from __future__ import annotations

from collections import Counter
import math
import numpy as np

from arc_utils import paste

# Design analysis
# - Concrete behavior:
#   The lattice format stays the same: 1-pixel gridlines, square cells, a 1-pixel bg border
#   inside each cell, and a single sp/bg sprite in the inner square. The key change is that the
#   "odd" cell in a row is no longer a row-specific mutation of one global base. Instead, each row
#   has its own majority sprite type, and its odd cell is an exact copy of a different row's
#   majority sprite. Example with three row types A/B/C:
#   row1 = A A B A, row2 = B C B B, row3 = C C A C. Output is [B, C, A] in a single column.
#   Original logic could rely on "odd pattern is globally rare". The variant breaks that because B
#   is common elsewhere when it is row2's majority.
# - Flow impact:
#   The affected stage is row interpretation, not lattice parsing or output formatting. The solver
#   must compare cells only inside each row: identify the row-majority template, then take the one
#   cell that differs from it. A concrete trace:
#   original row = [A, A, A, A*] where A* is a 1-3 pixel mutation of A; variant row = [A, A, B, A]
#   where B may be another row's majority. In both cases output keeps the non-majority cell, but in
#   the variant "global rarity" is false, so the row majority must be found locally.
# - Complexity calibration:
#   I keep only 2-3 sprite types per grid. That is enough to create real cross-row reuse while
#   keeping the examples legible. Too many types would make the grid noisy; too few with no reuse
#   support would hide the D13 change. Using repeated row-majority types also gives the solver and
#   the human repeated evidence that a cell can be common globally yet still be the local odd one.
# - Old implicit assumptions that fail:
#   The original generator assumes one global base sprite shared by all rows and row-specific odd
#   variants created by flipping a few pixels. That assumption is exactly what D13 changes. The
#   original solver's row-local Counter still happens to work on this variant, but it does not make
#   the required reasoning explicit. The new solver therefore computes the row-majority cell first
#   and selects the one non-matching cell, which is the intended invariant.
# - Placement legality:
#   Cells are still pasted into the same regular lattice. Each row has exactly one odd column, all
#   cells are full-cell replacements, and every odd sprite is chosen from a different existing base
#   type, so there is no overlap, no out-of-bounds placement, and no ambiguous 2-vs-2 row split.
# - Visual observability:
#   Humans can see the change because the odd cell often visibly matches another row's repeated
#   majority motif. Across 2-4 examples this supports the rule "pick the row exception" while also
#   showing that the exception is not defined by global uniqueness.
# - Visual cleanliness and color inference:
#   The task still uses exactly three colors (lattice, bg, sprite). No color ordering or numeric
#   color semantics are used. Patterns are separated by gridlines and differ only by visible shape,
#   keeping the display tidy.
# - New solver assumptions and generator guarantees:
#   (1) every row has one majority signature repeated n_cols-1 times; generator enforces this.
#   (2) the odd cell differs from the row-majority cell; generator picks a different sprite type.
#   (3) every odd sprite equals some other row's majority sprite in the same grid; generator uses a
#   small library of row-majority sprites and maps each base type to a different existing type.


def _parse_lattice(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(L))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(L))

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    return g, H, W, L, bg, row_lines, col_lines, r_slices, c_slices


def _row_majority_and_odd(cells: list[np.ndarray]) -> tuple[bytes, int]:
    sigs = [cell.tobytes() for cell in cells]
    cnt = Counter(sigs)
    majority_sig, majority_count = cnt.most_common(1)[0]
    assert int(majority_count) == len(cells) - 1
    odd_indices = [j for j, sig in enumerate(sigs) if sig != majority_sig]
    assert len(odd_indices) == 1
    return majority_sig, int(odd_indices[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the row exception from each lattice row.

    Variant rule:
    - Each row has one majority cell type and exactly one cell of a different type.
    - The odd cell may be globally common because it can match another row's majority sprite.
    - Therefore the decision must be made within each row: find the repeated template first, then
      select the one cell that differs from that template.
    """
    g, H, _, L, bg, row_lines, _, r_slices, c_slices = _parse_lattice(grid)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        _, odd_idx = _row_majority_and_odd(cells)
        chosen_cells.append(cells[int(odd_idx)])

    out = np.full((H, int(cell_w + 2)), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)
    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(i)]
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


def _random_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.35 * inner_area)
    max_fill = math.floor(0.75 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))
    inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    inner.flat[on_idx] = int(sp)
    return inner


def _sample_base_inners(rng: np.random.Generator, n_types: int, inner_size: int, bg: int, sp: int) -> list[np.ndarray] | None:
    inners: list[np.ndarray] = []
    min_dist = max(3, inner_size)
    for _ in range(400):
        if len(inners) == n_types:
            return inners
        inner = _random_inner(rng, inner_size, bg, sp)
        if np.array_equal(inner, np.flipud(inner)) and np.array_equal(inner, np.fliplr(inner)):
            continue
        if any(np.array_equal(inner, other) for other in inners):
            continue
        if any(int(np.count_nonzero(inner != other)) < min_dist for other in inners):
            continue
        inners.append(inner)
    return None


def _count_cross_row_reuse_rows(grid: np.ndarray) -> int:
    g, _, _, _, _, _, _, r_slices, c_slices = _parse_lattice(grid)
    majority_sigs: list[bytes] = []
    odd_sigs: list[bytes] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        majority_sig, odd_idx = _row_majority_and_odd(cells)
        majority_sigs.append(majority_sig)
        odd_sigs.append(cells[int(odd_idx)].tobytes())
    return sum(
        1
        for i, odd_sig in enumerate(odd_sigs)
        if any(odd_sig == majority_sigs[j] for j in range(len(majority_sigs)) if j != i)
    )


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    n_types = int(rng.choice([2, min(3, n_rows)]))

    inner_size = cell_size - 2
    base_inners = _sample_base_inners(rng, n_types, inner_size, bg, sp)
    if base_inners is None:
        return None
    base_cells = [_make_cell(inner, bg) for inner in base_inners]

    base_ids = list(range(n_types))
    while len(base_ids) < n_rows:
        base_ids.append(int(rng.integers(0, n_types)))
    rng.shuffle(base_ids)

    type_cycle = list(range(n_types))
    rng.shuffle(type_cycle)
    odd_of = {type_cycle[i]: type_cycle[(i + 1) % n_types] for i in range(n_types)}
    odd_ids = [int(odd_of[base_id]) for base_id in base_ids]

    out_cells = [[base_cells[odd_id].copy()] for odd_id in odd_ids]
    out = _build_lattice(out_cells, lattice, bg)

    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        base_cell = base_cells[base_ids[row]]
        odd_cell = base_cells[odd_ids[row]]
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = odd_cell.copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    if _count_cross_row_reuse_rows(inp) != n_rows:
        return None
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
