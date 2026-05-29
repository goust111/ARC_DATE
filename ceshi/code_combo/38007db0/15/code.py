# Combo variant of 38007db0 (index 15)
# combo_spec: D3/3+D4/2+D13/0
# Name: 原位差异标记含干扰

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    Original row logic:
#      [base, base, unique, base] -> extract the whole unique cell into a 1-column output.
#    Combo row logic:
#      [base, near, base, far, base] where both near/far differ from base,
#      and near changes fewer pixels than far.
#      The solver must:
#      1. recover the majority base cell from the row,
#      2. compare both singleton distractors to that base,
#      3. choose the closer distractor (near),
#      4. keep only the pixels where near differs from base,
#      5. write that sparse difference cell back at near's original column,
#         blanking every other cell in the row.
#    So D13 changes selection, D4 changes what is rendered, and D3 changes where it is rendered.
#
# b) Complexity calibration
#    The row should visibly contain two odd cells, not one, otherwise the D13 branch disappears.
#    To keep D4 visible, both near and far are generated as bg->sp additions only, so their
#    difference-only outputs contain visible sprite pixels instead of invisible deletions.
#    Near uses only 2-3 added pixels and far uses strictly more, which keeps the comparison
#    human-legible while avoiding too many free distance values.
#
# c) Hidden-assumption audit
#    The original main() assumed:
#    - exactly one singleton per row,
#    - any singleton is the target,
#    - output width shrinks to one cell,
#    - copying the whole target cell is correct.
#    All four assumptions are false now. The new solver instead assumes:
#    - each row has one unique majority signature (the base),
#    - each row has exactly two non-base cells,
#    - their Hamming distances to the base are different,
#    - blanking a cell means filling the whole cell rectangle with bg.
#    The generator explicitly constructs rows to satisfy those assumptions.
#
# d) Placement legality
#    The lattice is still regular: 1-pixel grid lines, equal square cells, uniform slicing.
#    The generator only mutates inner sprite pixels, never lattice lines or coordinates outside
#    the cell, so cells cannot overlap or escape their slices.
#
# e) Visual observability
#    From 2-4 examples a human can see:
#    - each row contains two odd cells,
#    - the output keeps only one row position instead of compressing columns,
#    - the kept cell is reduced to just the new pixels relative to the common base.
#    No invisible metadata such as color ids or coordinate ordering is used for selection.
#
# f) Visual cleanliness
#    Output cells are either blank or contain a tiny sparse highlight in the chosen location.
#    There is no multi-layer overlap because only one cell per row survives in the output.
#
# g) Change inferability
#    Color roles stay fixed: lattice, bg, sprite. There is no remapping. The only change is
#    structural subtraction against the row base, which is directly supported by repeated base
#    copies in every row.
#
# h) Generator validation
#    The generator is output-first in the sense that it explicitly decides the near-difference
#    cell that should appear in each output row, then places base/near/far cells into the input
#    and validates with main(inp) == out.
#    A separate metric pass after implementation checks row distance pairs and confirms that every
#    generated row really has two singleton distractors with a strict near/far gap.
#
# i) main()/generator contract
#    main() requires recoverable lattice lines, a dominant base signature per row, exactly two
#    distractors, and a unique closest distractor. generate_input() guarantees all of these by
#    using 4-5 columns, planting exactly two special cells per row, and separating their
#    difference counts.


def _parse_lattice(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    row_lines = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
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


def _cell_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _diff_cell(base_cell: np.ndarray, variant_cell: np.ndarray, bg: int) -> np.ndarray:
    diff = np.full(variant_cell.shape, int(bg), dtype=int)
    mask = variant_cell != base_cell
    diff[mask] = variant_cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the row-wise closer distractor in place, but only render its pixels that differ from the
    row's repeated base cell.
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
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs).most_common()

        assert len(counts) == 3
        base_sig, base_count = counts[0]
        assert int(base_count) >= 2
        assert counts[1][1] == 1 and counts[2][1] == 1

        base_cell = next(cell for cell, sig in zip(cells, sigs) if sig == base_sig)
        candidates: list[tuple[int, int, np.ndarray]] = []
        for idx, (cell, sig) in enumerate(zip(cells, sigs)):
            if sig != base_sig:
                candidates.append((_cell_distance(cell, base_cell), idx, cell))

        assert len(candidates) == 2
        candidates.sort(key=lambda item: item[0])
        assert candidates[0][0] < candidates[1][0]

        _, chosen_idx, chosen_cell = candidates[0]
        diff = _diff_cell(base_cell, chosen_cell, bg)
        ca, cb = c_slices[int(chosen_idx)]
        out[int(ra) : int(rb), int(ca) : int(cb)] = diff

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


def _compact_addition_sets(base_inner: np.ndarray, bg: int, counts: tuple[int, ...]) -> list[tuple[int, ...]]:
    inner_size = int(base_inner.shape[0])
    bg_positions = [idx for idx, val in enumerate(base_inner.flat) if int(val) == int(bg)]

    candidate_sets: list[tuple[int, ...]] = []
    for count in counts:
        if count > len(bg_positions):
            continue
        for combo in itertools.combinations(bg_positions, int(count)):
            coords = [divmod(idx, inner_size) for idx in combo]
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if bbox_area <= count + 2:
                candidate_sets.append(tuple(int(x) for x in combo))
    return candidate_sets


def _add_pixels(base_inner: np.ndarray, add_set: tuple[int, ...], sp: int) -> np.ndarray:
    variant = base_inner.copy()
    for flat_idx in add_set:
        variant.flat[int(flat_idx)] = int(sp)
    return variant


def _pick_row_variants(
    rng: np.random.Generator,
    near_sets: list[tuple[int, ...]],
    far_sets: list[tuple[int, ...]],
    n_rows: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    rng.shuffle(near_sets)
    rng.shuffle(far_sets)

    chosen_rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    used_near: set[tuple[int, ...]] = set()
    used_far: set[tuple[int, ...]] = set()

    for near in near_sets:
        if near in used_near:
            continue
        far = next(
            (
                cand
                for cand in far_sets
                if cand not in used_far and cand != near and len(cand) > len(near)
            ),
            None,
        )
        if far is None:
            continue
        used_near.add(near)
        used_far.add(far)
        chosen_rows.append((near, far))
        if len(chosen_rows) == n_rows:
            return chosen_rows

    raise ValueError("not enough distinct near/far variant pairs")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(4, 6))

    inner_size = cell_size - 2
    inner_area = inner_size * inner_size

    # Keep enough empty space so additive differences are both available and visible.
    min_fill = math.ceil(0.25 * inner_area)
    max_fill = math.floor(0.55 * inner_area)
    fill = int(rng.integers(min_fill, max_fill + 1))

    base_inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    on_idx = rng.choice(inner_area, size=fill, replace=False)
    base_inner.flat[on_idx] = int(sp)

    empty_count = int(np.sum(base_inner == int(bg)))
    if empty_count < 5:
        return None

    near_sets = _compact_addition_sets(base_inner, bg, (2, 3))
    far_sets = _compact_addition_sets(base_inner, bg, (4, 5, 6))
    if len(near_sets) < n_rows or len(far_sets) < n_rows:
        return None

    try:
        row_specs = _pick_row_variants(rng, near_sets, far_sets, n_rows)
    except ValueError:
        return None

    base_cell = _make_cell(base_inner, bg)
    blank_cell = _blank_cell(cell_size, bg)

    input_cells: list[list[np.ndarray]] = []
    output_cells: list[list[np.ndarray]] = []

    for near_set, far_set in row_specs:
        near_inner = _add_pixels(base_inner, near_set, sp)
        far_inner = _add_pixels(base_inner, far_set, sp)

        near_cell = _make_cell(near_inner, bg)
        far_cell = _make_cell(far_inner, bg)
        diff_cell = _make_cell(_diff_cell(base_inner, near_inner, bg), bg)

        near_col, far_col = map(int, rng.choice(n_cols, size=2, replace=False))
        inp_row = [base_cell.copy() for _ in range(n_cols)]
        out_row = [blank_cell.copy() for _ in range(n_cols)]
        inp_row[near_col] = near_cell
        inp_row[far_col] = far_cell
        out_row[near_col] = diff_cell
        input_cells.append(inp_row)
        output_cells.append(out_row)

    inp = _build_lattice(input_cells, lattice, bg)
    out = _build_lattice(output_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
