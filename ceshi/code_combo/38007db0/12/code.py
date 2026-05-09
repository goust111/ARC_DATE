# Combo variant of 38007db0 (index 12)
# combo_spec: D1/2+D4/2
# Name: 多数派差异提取

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Original task:
#      - Each lattice row contains one odd cell and several identical copies.
#      - Output copies the odd cell for each row into a single output column.
#    Combo behavior:
#      - We invert the selection target: the row's repeated cell is now the
#        conceptual object of interest.
#      - But we do not display that full majority cell. We compare majority vs
#        odd cell and keep only the majority-side pixels that differ.
#    Concrete example with a 3x3 inner sprite (`#`=sp, `.`=bg):
#      majority inner:
#        ###      odd inner:
#        #.#        #.#
#        .##        ..#
#      Original output would be the whole odd cell:
#        ### / #.# / ..#
#      This combo output is only the majority-only difference:
#        ... / ... / .#.
#      So the reasoning chain is "find the repeated template, subtract the odd
#      cell from it, then show only what the repeated template has extra."
#
# b) Complexity calibration:
#    A raw symmetric diff would be visually weak because majority-side bg pixels
#    are invisible. To make the fusion legible, the generator only creates odd
#    cells by removing 2-4 sprite pixels from the majority template. Then the
#    output becomes a sparse highlight of those removed pixels. This keeps the
#    inversion visible without introducing extra colors or too many cases.
#
# c) Implicit-assumption audit against the original main():
#    - Uniform row/column gridline detection still holds: the lattice geometry
#      and gridline color are unchanged.
#    - Lattice color as "most common uniform-line color" still holds.
#    - Background as "most common non-lattice color" still holds because bg is
#      still dominant in both input and output-side constructions.
#    - The original code assumed "the count-1 signature is the thing to copy".
#      That invariant is broken. We now need both signatures in the row:
#      the majority signature to select the template, and the minority one only
#      as a comparison target.
#    - The original code could ignore the majority cell once the oddball was
#      found. That is no longer valid because the output depends on majority
#      pixel values at differing coordinates.
#    - New fused assumption: each row should contain exactly two signatures,
#      with counts (n_cols-1, 1). The generator enforces this directly.
#
# d) Placement legality:
#    The generator still uses a regular 1-pixel lattice, square cells of size
#    5 or 6, and edits only the inner (cell_size-2)x(cell_size-2) sprite area.
#    Removal sets are chosen from existing sprite pixels, so nothing can cross a
#    border, overlap another cell, or leave the grid bounds.
#
# e) Visual observability:
#    Across 2-4 examples, a human sees that the input row mostly repeats one
#    template while a single cell is missing a few pixels. The output does not
#    copy that damaged cell; instead it marks the missing pixels alone. This is
#    distinguishable from both the original task (copy the odd cell) and the D4
#    single variant (copy odd-minus-base additions) because here the highlighted
#    pixels belong to the repeated template, not to an augmented oddball.
#
# f) Visual cleanliness:
#    Output cells stay sparse and use only the original three colors. There is
#    no overlay ambiguity: lattice lines frame each cell, bg stays blank, and
#    the few sprite pixels form a clean per-row marker.
#
# g) Change inferability:
#    No color remapping is used. The only change is structural: the solver must
#    infer that repeated cells define the reference, and output shows the pixels
#    that disappear in the singleton relative to that reference.
#
# Additional audit of new main() assumptions:
#    - Each row loop assumes at least 3 columns so a majority count >= 2 exists.
#      The generator keeps n_cols in [3, 4].
#    - `cnt.most_common(1)` assumes a unique strict majority signature; the
#      generator creates exactly one singleton and all other cells identical.
#    - `_majority_diff_cell` assumes cell shapes match; lattice slicing guarantees
#      equal cell rectangles, and the generator uses a single cell size.
#    - For visual legibility, removed pixels must be sprite pixels in the
#      majority template; `_compact_removal_sets` enforces that during synthesis.


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
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

    return H, W, lattice, row_lines, r_slices, c_slices


def _majority_diff_cell(majority_cell: np.ndarray, odd_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(majority_cell.shape, int(bg), dtype=int)
    mask = majority_cell != odd_cell
    diff[mask] = majority_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each lattice row, identify the repeated majority cell and output only
    the majority-side pixels that differ from the row's singleton odd cell.
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

        assert len(cnt) == 2
        majority_sig, majority_count = cnt.most_common(1)[0]
        assert int(majority_count) >= 2
        odd_sig = next(sig for sig, count in cnt.items() if int(count) == 1)

        majority_idx = next(i for i, sig in enumerate(sigs) if sig == majority_sig)
        odd_idx = next(i for i, sig in enumerate(sigs) if sig == odd_sig)
        chosen_diffs.append(_majority_diff_cell(cells[int(majority_idx)], cells[int(odd_idx)], bg))

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


def _compact_removal_sets(base_inner: np.ndarray, sp: int) -> list[tuple[int, ...]]:
    inner_size = int(base_inner.shape[0])
    sp_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(sp)]

    candidate_sets: list[tuple[int, ...]] = []
    max_remove = min(4, len(sp_positions))
    for count in range(2, max_remove + 1):
        for combo in itertools.combinations(sp_positions, count):
            coords = [divmod(idx, inner_size) for idx in combo]
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if bbox_area <= count + 2:
                candidate_sets.append(tuple(int(x) for x in combo))
    return candidate_sets


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    min_fill = math.ceil(0.45 * inner_area)
    max_fill = math.floor(0.8 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    sprite_count = int(np.sum(base_inner == int(sp)))
    if sprite_count < 4:
        return None

    candidate_sets = _compact_removal_sets(base_inner, sp)
    if len(candidate_sets) < n_rows:
        return None

    chosen_ids = rng.choice(len(candidate_sets), size=n_rows, replace=False)
    diff_cells: list[np.ndarray] = []
    odd_cells: list[np.ndarray] = []
    for idx in chosen_ids:
        remove_set = candidate_sets[int(idx)]
        diff_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        odd_inner = base_inner.copy()
        for flat_idx in remove_set:
            diff_inner.flat[int(flat_idx)] = int(sp)
            odd_inner.flat[int(flat_idx)] = int(bg)
        diff_cells.append(_make_cell(diff_inner, bg))
        odd_cells.append(_make_cell(odd_inner, bg))

    out = _build_lattice([[cell] for cell in diff_cells], lattice, bg)

    majority_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [majority_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = odd_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
