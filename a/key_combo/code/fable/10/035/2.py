from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import numpy as np


GREY = 5
GENERATION_RETRIES = 400


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _pattern_key(cell: np.ndarray) -> bytes:
    return np.asarray(cell, dtype=int).tobytes()


def _uniform_line_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if len(set(map(int, grid[r, :]))) == 1]
    return [c for c in range(grid.shape[1]) if len(set(map(int, grid[:, c]))) == 1]


def _extract_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    row_lines = _uniform_line_indices(g, axis=0)
    col_lines = _uniform_line_indices(g, axis=1)
    assert row_lines and col_lines, "expected lattice grid lines"

    line_colors = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(line_colors).most_common(1)[0][0])

    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected outer lattice frame"

    row_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
    col_slices = [(col_lines[j] + 1, col_lines[j + 1]) for j in range(len(col_lines) - 1)]
    row_slices = [(a, b) for a, b in row_slices if b > a]
    col_slices = [(a, b) for a, b in col_slices if b > a]
    assert row_slices and col_slices, "expected non-empty lattice cells"
    assert len(row_slices) == len(col_slices), "expected square N x N lattice"

    cell_shapes = {(rb - ra, cb - ca) for ra, rb in row_slices for ca, cb in col_slices}
    assert len(cell_shapes) == 1, "expected uniform cell size"

    cells = []
    for ra, rb in row_slices:
        row = []
        for ca, cb in col_slices:
            row.append(g[ra:rb, ca:cb].copy())
        cells.append(row)

    return lattice, row_slices, col_slices, cells


def _find_occluder(cells: list[list[np.ndarray]]) -> tuple[int, list[int], np.ndarray]:
    n = len(cells)
    by_key: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    patch_by_key: dict[bytes, np.ndarray] = {}
    for r in range(n):
        for c in range(n):
            cell = cells[r][c]
            if np.all(cell == int(cell[0, 0])):
                key = _pattern_key(cell)
                by_key[key].append((r, c))
                patch_by_key[key] = cell

    candidates = [(key, pos) for key, pos in by_key.items() if len(pos) == 2 and pos[0][0] == pos[1][0]]
    assert len(candidates) == 1, "expected one pair of same-row uniform occluders"
    key, positions = candidates[0]
    row = positions[0][0]
    cols = sorted(c for _r, c in positions)
    return int(row), cols, patch_by_key[key]


def _identify_invader(cells: list[list[np.ndarray]], occluder_patch: np.ndarray) -> np.ndarray:
    row_presence: dict[bytes, set[int]] = defaultdict(set)
    patch_by_key: dict[bytes, np.ndarray] = {}
    occluder_key = _pattern_key(occluder_patch)
    n = len(cells)
    for r in range(n):
        for c in range(n):
            key = _pattern_key(cells[r][c])
            if key == occluder_key:
                continue
            row_presence[key].add(r)
            patch_by_key[key] = cells[r][c]

    candidates = [key for key, rows in row_presence.items() if len(rows) > 1]
    assert len(candidates) == 1, "expected exactly one cross-row repeated invader pattern"
    return patch_by_key[candidates[0]]


def _hidden_row_base(cells: list[list[np.ndarray]], hidden_row: int, invader: np.ndarray, occluder_patch: np.ndarray) -> np.ndarray:
    invader_key = _pattern_key(invader)
    occluder_key = _pattern_key(occluder_patch)
    visible = [cell for cell in cells[hidden_row] if _pattern_key(cell) not in (invader_key, occluder_key)]
    assert visible, "expected at least one visible base cell in the occluded row"
    base_key = _pattern_key(visible[0])
    assert all(_pattern_key(cell) == base_key for cell in visible), "base cells in hidden row must agree"
    return visible[0]


def _reconstruct_cells(cells: list[list[np.ndarray]]) -> tuple[list[list[np.ndarray]], np.ndarray]:
    n = len(cells)
    hidden_row, grey_cols, occluder_patch = _find_occluder(cells)
    invader = _identify_invader(cells, occluder_patch)
    base = _hidden_row_base(cells, hidden_row, invader, occluder_patch)
    invader_key = _pattern_key(invader)

    visible_invader_cols = []
    for r in range(n):
        for c in range(n):
            if _pattern_key(cells[r][c]) == invader_key:
                visible_invader_cols.append(c)
    visible_invader_cols = sorted(set(visible_invader_cols))
    assert len(visible_invader_cols) in (n - 1, n), "unexpected visible invader column coverage"

    filled = [[cell.copy() for cell in row] for row in cells]
    if len(visible_invader_cols) == n:
        for c in grey_cols:
            filled[hidden_row][c] = base.copy()
    else:
        missing_col = next(c for c in range(n) if c not in visible_invader_cols)
        assert missing_col in grey_cols, "missing invader column must be one of the grey cells"
        for c in grey_cols:
            filled[hidden_row][c] = invader.copy() if c == missing_col else base.copy()

    return filled, invader


def _build_grid(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    n = len(cells)
    cell_h, cell_w = cells[0][0].shape
    out = np.full((n * (cell_h + 1) + 1, n * (cell_w + 1) + 1), int(bg), dtype=int)
    for r in range(n + 1):
        out[r * (cell_h + 1), :] = int(lattice)
    for c in range(n + 1):
        out[:, c * (cell_w + 1)] = int(lattice)
    for r in range(n):
        for c in range(n):
            _paste(out, cells[r][c], 1 + r * (cell_h + 1), 1 + c * (cell_w + 1))
    return out


def _cell_background_from(cells: list[list[np.ndarray]], lattice: int) -> int:
    vals = [int(x) for row in cells for cell in row for x in cell.flat if int(x) != lattice]
    assert vals
    return int(Counter(vals).most_common(1)[0][0])


def _validate_completed_cells(cells: list[list[np.ndarray]], invader: np.ndarray) -> None:
    invader_key = _pattern_key(invader)
    n = len(cells)
    for r in range(n):
        count = sum(_pattern_key(cells[r][c]) == invader_key for c in range(n))
        assert count == 1, "each row must contain the invader exactly once"
    for c in range(n):
        count = sum(_pattern_key(cells[r][c]) == invader_key for r in range(n))
        assert count == 1, "each column must contain the invader exactly once"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the two grey cells in one lattice row using the invader's row-and-column uniqueness.

    Input structure:
    - A square lattice of equal-sized cells is separated by one-pixel grid lines.
    - Each row has its own repeated base sprite, distinct from every other row's base sprite.
    - One global invader sprite appears exactly once in each row and exactly once in each column.
    - Exactly two cells in one row are replaced by the same solid grey occluder cell, and grey appears nowhere else.

    Rule:
    - Identify the invader as the only visible cell pattern that repeats across multiple rows.
    - In the occluded row, read the row base from the other visible non-invader cells.
    - If every column already shows a visible invader, both grey cells become the row base.
    - Otherwise exactly one column is missing a visible invader; the grey cell in that column becomes the invader and the other grey cell becomes the row base.

    Key insight:
    - The hidden cells are resolved by combining two constraints at once: each row has one invader, and each column has one invader.
    """
    g = np.asarray(grid, dtype=int)
    lattice, _row_slices, _col_slices, cells = _extract_lattice(g)
    filled, invader = _reconstruct_cells(cells)
    _validate_completed_cells(filled, invader)
    bg = _cell_background_from(filled, lattice)
    return _build_grid(filled, lattice, bg)


def _make_cell(inner_mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    inner_mask = np.asarray(inner_mask, dtype=bool)
    cell = np.full((inner_mask.shape[0] + 2, inner_mask.shape[1] + 2), int(bg), dtype=int)
    cell[1:-1, 1:-1][inner_mask] = int(fg)
    return cell


def _sample_mask(inner_size: int) -> np.ndarray:
    area = inner_size * inner_size
    lo = max(inner_size, math.ceil(area * 0.35))
    hi = min(area - inner_size, math.floor(area * 0.7))
    for _ in range(300):
        on = random.randint(lo, hi)
        mask = np.zeros((inner_size, inner_size), dtype=bool)
        idx = random.sample(range(area), on)
        mask.flat[idx] = True
        if any(mask[r].all() or (~mask[r]).all() for r in range(inner_size)):
            continue
        if any(mask[:, c].all() or (~mask[:, c]).all() for c in range(inner_size)):
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate sprite mask")


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a, dtype=bool) ^ np.asarray(b, dtype=bool)))


def _sample_pattern_family(inner_size: int, count: int) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    min_dist = 3 if inner_size >= 4 else 2
    for _ in range(1000):
        candidate = _sample_mask(inner_size)
        if any(np.array_equal(candidate, existing) for existing in patterns):
            continue
        if any(_hamming(candidate, existing) < min_dist for existing in patterns):
            continue
        patterns.append(candidate)
        if len(patterns) == count:
            return patterns
    raise RuntimeError("failed to sample sufficiently distinct sprite patterns")


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict]:
    n = 4
    cell_size = random.choice((5, 6))
    inner_size = cell_size - 2
    grey = GREY

    palette = random.sample([c for c in range(10) if c != grey], 3)
    lattice, bg, fg = map(int, palette)

    invader_mask, *base_masks = _sample_pattern_family(inner_size, n + 1)
    invader_cell = _make_cell(invader_mask, bg, fg)
    base_cells = [_make_cell(mask, bg, fg) for mask in base_masks]

    perm = list(range(n))
    random.shuffle(perm)
    hidden_row = random.randrange(n)
    branch_has_hidden_invader = random.random() < 0.5

    if branch_has_hidden_invader:
        other_col = random.choice([c for c in range(n) if c != perm[hidden_row]])
        grey_cols = sorted([perm[hidden_row], other_col])
    else:
        grey_cols = sorted(random.sample([c for c in range(n) if c != perm[hidden_row]], 2))

    full_cells: list[list[np.ndarray]] = []
    for r in range(n):
        row = [base_cells[r].copy() for _ in range(n)]
        row[perm[r]] = invader_cell.copy()
        full_cells.append(row)

    occluder = np.full((cell_size, cell_size), grey, dtype=int)
    inp_cells = [[cell.copy() for cell in row] for row in full_cells]
    for c in grey_cols:
        inp_cells[hidden_row][c] = occluder.copy()

    inp = _build_grid(inp_cells, lattice, bg)
    out = _build_grid(full_cells, lattice, bg)
    meta = {
        "n": n,
        "lattice": lattice,
        "bg": bg,
        "fg": fg,
        "perm": tuple(perm),
        "hidden_row": hidden_row,
        "grey_cols": tuple(grey_cols),
        "branch_has_hidden_invader": branch_has_hidden_invader,
        "full_cells": tuple(tuple(_pattern_key(cell) for cell in row) for row in full_cells),
        "invader_key": _pattern_key(invader_cell),
        "row_base_keys": tuple(_pattern_key(base) for base in base_cells),
    }
    return inp, out, meta


def _semantic_checks(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    try:
        lattice, _rs, _cs, in_cells = _extract_lattice(inp)
        _, _ors, _ocs, out_cells = _extract_lattice(out)
    except Exception:
        return False

    if lattice != meta["lattice"]:
        return False

    try:
        hidden_row, grey_cols, occluder_patch = _find_occluder(in_cells)
        invader = _identify_invader(in_cells, occluder_patch)
        reconstructed, inferred_invader = _reconstruct_cells(in_cells)
        _validate_completed_cells(reconstructed, inferred_invader)
    except Exception:
        return False

    if hidden_row != meta["hidden_row"] or tuple(grey_cols) != meta["grey_cols"]:
        return False
    if _pattern_key(invader) != meta["invader_key"]:
        return False

    visible_invader_cols = sorted(
        {c for r in range(meta["n"]) for c in range(meta["n"]) if _pattern_key(in_cells[r][c]) == meta["invader_key"]}
    )
    if meta["branch_has_hidden_invader"]:
        if len(visible_invader_cols) != meta["n"] - 1:
            return False
        missing = next(c for c in range(meta["n"]) if c not in visible_invader_cols)
        if missing != meta["perm"][meta["hidden_row"]]:
            return False
        wrong = out.copy()
        _, _wrs, _wcs, wrong_cells = _extract_lattice(wrong)
        wrong_base_key = meta["row_base_keys"][meta["hidden_row"]]
        base_patch = None
        for row in reconstructed:
            for cell in row:
                if _pattern_key(cell) == wrong_base_key:
                    base_patch = cell
                    break
            if base_patch is not None:
                break
        for c in meta["grey_cols"]:
            wrong_cells[meta["hidden_row"]][c] = base_patch.copy()
        if np.array_equal(_build_grid(wrong_cells, meta["lattice"], meta["bg"]), out):
            return False
    else:
        if len(visible_invader_cols) != meta["n"]:
            return False

    expected_keys = meta["full_cells"]
    actual_keys = tuple(tuple(_pattern_key(cell) for cell in row) for row in out_cells)
    if actual_keys != expected_keys:
        return False
    actual_reconstructed = tuple(tuple(_pattern_key(cell) for cell in row) for row in reconstructed)
    if actual_reconstructed != expected_keys:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp, out, meta = _build_instance()
        if _semantic_checks(inp, out, meta):
            return inp
    raise RuntimeError("failed to generate a valid row-column intruder instance")
