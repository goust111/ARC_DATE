# Combo variant of 38007db0 (index 18)
# combo_spec: D11/2+D12/0
# Name: 无格线大网格检测

from __future__ import annotations

from collections import Counter
import itertools
import math
from pathlib import Path
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and trace:
#    Original behavior: a 3-4 x 3-4 lattice uses a dedicated gridline color L.
#    Each row has C cells, C-1 are identical, and 1 is the odd sprite. The solver
#    detects uniform L rows/cols, slices cells, and outputs a 1-column lattice.
#    Fused behavior here:
#    - explicit lattice color disappears completely;
#    - the logical grid becomes larger: 5-6 rows x 5-6 cols;
#    - cells may be directly adjacent (gap 0) or separated by a 1px bg gap, with
#      row-gap and col-gap chosen independently per example.
#    Concrete toy trace with cell size 5 (inner size 3), col_gap = 0:
#      input row layout:
#        [bg border + sprite + bg border][bg border + sprite + bg border]...
#      There is no visible colored lattice. The only stable cue is that each cell
#      contributes one consecutive 3-row by 3-col non-bg band, and these bands
#      recur on a regular pitch. Expanding each band by 1px on all sides recovers
#      the full 5x5 cell. After slicing each logical row, the per-row odd-one-out
#      rule is unchanged. Output keeps the same no-grid rendering style but only
#      one column of recovered cells.
#    Difference from the original is therefore synergistic rather than additive:
#    the bigger 5-6 x 5-6 search area makes the missing lattice materially harder,
#    because the solver must infer more cell boundaries from repeated geometry.
#
# b) Complexity calibration:
#    If we only removed the lattice without extra guards, some sprites could contain
#    all-bg inner rows/cols and create fake separators. To keep the no-grid parsing
#    observable but not ambiguous, generator enforces:
#    - base sprite has at least one sprite pixel in every inner row and every inner
#      column;
#    - every unique variant also preserves that property;
#    - row/col gaps are only from the small set {0, 1};
#    - odd-cell columns use at least two distinct positions across the big grid.
#    This keeps the visual change obvious while preventing uncontrolled geometry.
#
# c) Implicit-assumption audit against the original main():
#    - `row_lines = uniform rows` and `col_lines = uniform cols` break immediately,
#      because there is no dedicated lattice color anymore.
#    - "most common uniform-line color is L" also fails; many uniform rows/cols are
#      just background borders/gaps.
#    - `r_slices/c_slices = gaps between gridlines` no longer works at all.
#    - `outW = cell_w + 2` assumed a 1px outer lattice on both sides. In this combo
#      the outer horizontal margin is either 0 or 1, so output width must be inferred
#      from the actual spacing geometry instead of hardcoded.
#    New solver assumptions introduced here:
#    - bg is the most common color;
#    - every logical cell row/col yields exactly one consecutive non-bg band;
#    - cell padding remains exactly 1px on all four sides;
#    - cell size is square and uniform, with inner size 3 or 4;
#    - row gaps and col gaps are globally uniform and each lie in {0, 1}.
#    Generator explicitly guarantees each of these assumptions.
#
# d) Placement legality:
#    Cells are pasted on a bg canvas with pitch = cell_size + gap along each axis,
#    where each axis gap is 0 or 1. Dimensions are derived from row/col counts and
#    pitch, so cells never overlap or go out of bounds.
#
# e) Visual observability:
#    A human still sees repeated padded sprite tiles aligned in a large regular array,
#    but the colored lattice has vanished. Because there are 5-6 rows and 5-6 cols,
#    the regularity itself becomes the cue for inferring the hidden grid structure.
#    The output visibly extracts one tile per input row, so the rule remains learnable
#    from a few examples without relying on color ids.
#
# f) Visual cleanliness:
#    Only two colors appear (bg and sprite). There are no extra markers or overlays.
#    Even when gap = 0, the 1px bg border inside every cell keeps neighboring sprites
#    visually separated, so the scene stays readable instead of collapsing into noise.
#
# g) Change inferability:
#    There is no recoloring rule. Colors are preserved exactly; only grouping and
#    extraction change. The structural signal is carried by repeated spacing and
#    per-row frequency, which is visible in 2-3 examples.
#
# Post-edit audit of new assumptions:
#    - `_infer_spaced_cells` assumes every non-bg row run/col run corresponds to one
#      logical lattice band. Generator guarantees this via full row/col coverage.
#    - The per-row Counter assumes exactly one unique cell and a majority template.
#      Generator constructs each row from C-1 base cells plus 1 unique cell.
#    - Output reconstruction assumes the input outer horizontal margins are reused.
#      Generator uses symmetric margins equal to the sampled col_gap.


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    return runs


def _infer_spaced_cells(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int, int]:
    H, W = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    content_rows = [r for r in range(H) if np.any(g[r, :] != bg)]
    content_cols = [c for c in range(W) if np.any(g[:, c] != bg)]
    assert content_rows and content_cols

    row_runs = _group_runs(content_rows)
    col_runs = _group_runs(content_cols)
    assert 5 <= len(row_runs) <= 6
    assert 5 <= len(col_runs) <= 6

    inner_heights = {int(b - a + 1) for a, b in row_runs}
    inner_widths = {int(b - a + 1) for a, b in col_runs}
    assert len(inner_heights) == 1 and len(inner_widths) == 1
    inner_size = int(next(iter(inner_heights)))
    assert inner_size == int(next(iter(inner_widths)))
    assert inner_size in (3, 4)
    cell_size = int(inner_size + 2)

    r_slices: list[tuple[int, int]] = []
    for a, b in row_runs:
        assert 1 <= a <= b <= H - 2
        r_slices.append((int(a - 1), int(b + 2)))

    c_slices: list[tuple[int, int]] = []
    for a, b in col_runs:
        assert 1 <= a <= b <= W - 2
        c_slices.append((int(a - 1), int(b + 2)))

    assert len({rb - ra for ra, rb in r_slices}) == 1
    assert len({cb - ca for ca, cb in c_slices}) == 1
    assert int(r_slices[0][1] - r_slices[0][0]) == cell_size
    assert int(c_slices[0][1] - c_slices[0][0]) == cell_size

    top_margin = int(r_slices[0][0])
    left_margin = int(c_slices[0][0])
    row_gap = int(top_margin)
    col_gap = int(left_margin)
    assert row_gap in (0, 1)
    assert col_gap in (0, 1)
    assert int(H - r_slices[-1][1]) == row_gap
    assert int(W - c_slices[-1][1]) == col_gap

    row_pitches = [int(r_slices[i + 1][0] - r_slices[i][0]) for i in range(len(r_slices) - 1)]
    col_pitches = [int(c_slices[i + 1][0] - c_slices[i][0]) for i in range(len(c_slices) - 1)]
    assert all(p == cell_size + row_gap for p in row_pitches)
    assert all(p == cell_size + col_gap for p in col_pitches)

    return bg, r_slices, c_slices, row_gap, col_gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each logical row of a large no-grid lattice.

    Variant-specific parsing:
    - no explicit lattice color exists;
    - logical rows/cols are inferred from repeated non-bg content bands;
    - the grid is larger (5-6 x 5-6);
    - cells can be directly adjacent or separated by a 1px bg gap.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg, r_slices, c_slices, _row_gap, _col_gap = _infer_spaced_cells(g)
    cell_size = int(r_slices[0][1] - r_slices[0][0])
    left_margin = int(c_slices[0][0])
    right_margin = int(W - c_slices[-1][1])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)])

    out = np.full((H, int(left_margin + cell_size + right_margin)), int(bg), dtype=int)
    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), int(left_margin) : int(left_margin + cell_size)] = chosen_cells[int(i)]
    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = int(inner.shape[0] + 2)
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_spaced(
    cells: list[list[np.ndarray]],
    bg: int,
    row_gap: int,
    col_gap: int,
) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    H = int(rows * cell_size + (rows + 1) * row_gap)
    W = int(cols * cell_size + (cols + 1) * col_gap)
    grid = np.full((H, W), int(bg), dtype=int)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            top = int(row_gap + r * (cell_size + row_gap))
            left = int(col_gap + c * (cell_size + col_gap))
            paste(grid, cell, top, left)
    return grid


def _has_full_inner_coverage(inner: np.ndarray, bg: int) -> bool:
    for r in range(inner.shape[0]):
        if not np.any(inner[r, :] != bg):
            return False
    for c in range(inner.shape[1]):
        if not np.any(inner[:, c] != bg):
            return False
    return True


def _make_base_inner(
    rng: np.random.Generator,
    inner_size: int,
    bg: int,
    sp: int,
) -> np.ndarray | None:
    inner_area = int(inner_size * inner_size)
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    for _ in range(200):
        fill = int(rng.integers(min_fill, max_fill + 1))
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        inner.flat[on_idx] = int(sp)
        if _has_full_inner_coverage(inner, bg):
            return inner
    return None


def _make_unique_variants(
    base_inner: np.ndarray,
    n_rows: int,
    bg: int,
    sp: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    inner_area = int(base_inner.size)
    seen: set[bytes] = set()
    candidates: list[np.ndarray] = []
    for flip_count in (1, 2, 3):
        for flip_set in itertools.combinations(range(inner_area), flip_count):
            variant = base_inner.copy()
            for idx in flip_set:
                variant.flat[int(idx)] = sp if int(variant.flat[int(idx)]) == int(bg) else bg
            key = variant.tobytes()
            if key == base_inner.tobytes() or key in seen:
                continue
            if not _has_full_inner_coverage(variant, bg):
                continue
            candidates.append(variant)
            seen.add(key)
    if len(candidates) < n_rows:
        return None
    pick_idx = rng.choice(len(candidates), size=n_rows, replace=False)
    return [candidates[int(idx)].copy() for idx in pick_idx]


def _sample_odd_cols(rng: np.random.Generator, n_rows: int, n_cols: int) -> list[int]:
    for _ in range(50):
        odd_cols = [int(rng.integers(0, n_cols)) for _ in range(n_rows)]
        if len(set(odd_cols)) >= 2:
            return odd_cols
    return [int(i % n_cols) for i in range(n_rows)]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg, sp = map(int, rng.choice(10, size=2, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(5, 7))
    n_cols = int(rng.integers(5, 7))
    row_gap = int(rng.integers(0, 2))
    col_gap = int(rng.integers(0, 2))

    inner_size = int(cell_size - 2)
    base_inner = _make_base_inner(rng, inner_size, bg, sp)
    if base_inner is None:
        return None

    unique_inners = _make_unique_variants(base_inner, n_rows, bg, sp, rng)
    if unique_inners is None:
        return None

    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]
    out = _build_spaced([[cell] for cell in unique_cells], bg, row_gap, col_gap)

    base_cell = _make_cell(base_inner, bg)
    odd_cols = _sample_odd_cols(rng, n_rows, n_cols)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[int(odd_cols[row])] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_spaced(in_cells, bg, row_gap, col_gap)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
