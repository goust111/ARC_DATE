import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
WALL_COLOR = 5
REFERENCE_COLORS = (1, 2, 3)
GEN_ATTEMPTS = 400


def _triangular_positions(limit: int) -> list[int]:
    cols: list[int] = []
    acc = 0
    step = 1
    while acc < limit:
        cols.append(acc)
        acc += step
        step += 1
    return cols


def _infer_wall_rows(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    wall_rows = [r for r in range(g.shape[0]) if int(np.count_nonzero(g[r] == WALL_COLOR)) > g.shape[1] // 2]
    assert len(wall_rows) in {3, 4}, f"expected 3 or 4 wall rows, got {wall_rows}"
    return wall_rows


def _defect_columns(grid: np.ndarray, wall_rows: Iterable[int]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    rows = list(wall_rows)
    assert rows
    patterns = [tuple(map(int, np.flatnonzero(g[r] != WALL_COLOR))) for r in rows]
    first = patterns[0]
    assert len(first) >= 5, "need several defect columns"
    assert all(p == first for p in patterns[1:]), "all wall rows must share the same defect columns"
    cols = list(first)
    expected = _triangular_positions(g.shape[1])[: len(cols)]
    assert cols == expected, f"defect columns must be the triangular prefix, got {cols}, expected {expected}"
    return cols


def _extract_cycle(grid: np.ndarray, wall_rows: Iterable[int]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    wall_set = set(map(int, wall_rows))
    refs = [(r, int(g[r, 0])) for r in range(g.shape[0]) if r not in wall_set and int(g[r, 0]) in REFERENCE_COLORS]
    assert len(refs) == 3, f"expected three left-edge reference cells, got {refs}"
    cycle = [color for _r, color in refs]
    assert len(set(cycle)) == 3, f"reference colors must be distinct, got {cycle}"
    return cycle


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the triangular-position gaps in each gray wall row by indexing into the left-edge color cycle.

    Input structure:
    - The grid has a black background, 3 or 4 horizontal gray wall rows, and exactly three colored
      reference pixels on the left edge outside the wall rows.
    - Each wall row is almost entirely gray, except at the shared defect columns where cells are either
      black gaps or already-filled colored clues.
    - The defect columns form the triangular-number prefix 0, 1, 3, 6, 10, 15, ...

    Rule:
    - Identify the wall rows as the rows dominated by gray pixels.
    - Read the shared defect columns from any wall row and verify that their left-to-right gaps increase
      by 1 each time, confirming the triangular sequence.
    - Number those defect columns by order k = 0, 1, 2, ...
    - Read the three left-edge reference colors from top to bottom as the cycle C.
    - For every black defect cell in a wall row, replace it with C[k mod 3]; keep already-colored clues
      and every non-wall row unchanged.

    Key insight:
    - The colored clues are indexed by the spatial order of triangular-number defect columns, so the
      solver must couple a position sequence with a separate color cycle rather than treating the holes
      as arbitrary missing cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_rows = _infer_wall_rows(g)
    defect_cols = _defect_columns(g, wall_rows)
    cycle = _extract_cycle(g, wall_rows)

    out = g.copy()
    for k, c in enumerate(defect_cols):
        expected_color = int(cycle[k % len(cycle)])
        for r in wall_rows:
            cell = int(g[r, c])
            assert cell in {BACKGROUND, expected_color}, f"inconsistent clue at {(r, c)}: {cell} vs {expected_color}"
            if cell == BACKGROUND:
                out[r, c] = expected_color
    return out


def _make_prefill_mask(wall_count: int, gap_count: int, rng: random.Random) -> np.ndarray | None:
    residue_columns: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for k in range(gap_count):
        residue_columns[k % 3].append(k)
    if any(not cols for cols in residue_columns.values()):
        return None

    for _ in range(200):
        mask = np.zeros((wall_count, gap_count), dtype=bool)

        hidden = {rng.choice(residue_columns[res]) for res in range(3)}
        for k in range(gap_count):
            if k in hidden:
                continue
            colored_count = rng.randint(1, wall_count)
            rows = rng.sample(range(wall_count), colored_count)
            mask[rows, k] = True

        colored_ratio = float(mask.mean())
        if not 0.35 <= colored_ratio <= 0.65:
            continue
        if not all(mask[row].any() and (~mask[row]).any() for row in range(wall_count)):
            continue
        if len({tuple(map(int, row)) for row in mask}) < 2:
            continue

        hidden_residues = {k % 3 for k in range(gap_count) if not mask[:, k].any()}
        if hidden_residues != {0, 1, 2}:
            continue

        visible_residues = {k % 3 for k in range(gap_count) if mask[:, k].any()}
        if len(visible_residues) < 2:
            continue
        if not any(0 < int(mask[:, k].sum()) < wall_count for k in range(gap_count)):
            continue
        return mask
    return None


def _semantic_validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        wall_rows = _infer_wall_rows(g)
        defect_cols = _defect_columns(g, wall_rows)
        cycle = _extract_cycle(g, wall_rows)
        out = main(g)
    except AssertionError:
        return False

    if np.array_equal(out, g):
        return False

    wall_values = g[np.array(wall_rows)]
    black_columns = [k for k, c in enumerate(defect_cols) if np.all(wall_values[:, c] == BACKGROUND)]
    if {k % 3 for k in black_columns} != {0, 1, 2}:
        return False

    if not all(np.any(wall_values[i, defect_cols] == BACKGROUND) and np.any(wall_values[i, defect_cols] != BACKGROUND) for i in range(len(wall_rows))):
        return False

    changed = np.argwhere(out != g)
    if changed.size == 0:
        return False
    if any(int(r) not in wall_rows for r, _c in changed):
        return False

    changed_colors = {int(out[r, c]) for r, c in changed}
    if changed_colors != set(cycle):
        return False

    for k, c in enumerate(defect_cols):
        expected = cycle[k % 3]
        if not np.all(out[np.array(wall_rows), c] == expected):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 16)
    w = rng.randint(18, 29)
    defect_cols = _triangular_positions(w)
    if len(defect_cols) < 6:
        return None

    wall_count = rng.choice((3, 4))
    if h < wall_count + 3:
        return None

    wall_rows = sorted(rng.sample(range(h), wall_count))
    non_wall_rows = [r for r in range(h) if r not in wall_rows]
    if len(non_wall_rows) < 3:
        return None

    ref_rows = sorted(rng.sample(non_wall_rows, 3))
    cycle = rng.sample(list(REFERENCE_COLORS), 3)
    prefill_mask = _make_prefill_mask(wall_count, len(defect_cols), rng)
    if prefill_mask is None:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for r, color in zip(ref_rows, cycle):
        grid[r, 0] = int(color)

    for row_index, r in enumerate(wall_rows):
        grid[r, :] = WALL_COLOR
        for k, c in enumerate(defect_cols):
            if prefill_mask[row_index, k]:
                grid[r, c] = int(cycle[k % 3])
            else:
                grid[r, c] = BACKGROUND

    if not _semantic_validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid triangular-gap wall input")
