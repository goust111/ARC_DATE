from __future__ import annotations

import random
from typing import Iterable

import numpy as np


FG = 1
BG = 0
PERIOD_MIN = 3
PERIOD_MAX = 5
GEN_ATTEMPTS = 400


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _orth_count(mask: np.ndarray, r: int, c: int) -> int:
    return sum(1 for rr, cc in _neighbors4(mask.shape, r, c) if mask[rr, cc])


def _has_diag_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    return any(mask[rr, cc] for rr, cc in _neighbors8(mask.shape, r, c) if rr != r and cc != c)


def _diag_only_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if _orth_count(mask, rr, cc) == 0 and _has_diag_neighbor(mask, rr, cc):
            out[rr, cc] = True
    return out


def _line_run_indices(mask: np.ndarray, axis: int) -> list[int]:
    h, w = mask.shape
    indices: list[int] = []
    if axis == 0:
        for r in range(h):
            run = 0
            for c in range(w):
                if mask[r, c]:
                    run += 1
                    if run >= 3:
                        indices.append(r)
                        break
                else:
                    run = 0
    else:
        for c in range(w):
            run = 0
            for r in range(h):
                if mask[r, c]:
                    run += 1
                    if run >= 3:
                        indices.append(c)
                        break
                else:
                    run = 0
    return indices


def _infer_period(indices: list[int], strengths: np.ndarray, dim: int) -> tuple[int, int]:
    best_key: tuple[int, int, int, int] | None = None
    best: tuple[int, int] | None = None
    for period in range(PERIOD_MIN, min(PERIOD_MAX, dim - 1) + 1):
        for residue in range(period):
            matched = [idx for idx in indices if idx % period == residue]
            if len(matched) < 2:
                continue
            expected = list(range(residue, dim, period))
            key = (
                len(matched),
                -abs(len(expected) - len(matched)),
                int(sum(int(strengths[idx]) for idx in matched)),
                -period,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (period, residue)
    assert best is not None, "failed to infer a periodic residue class"
    return best


def _infer_periodic_lines(grid: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    mask = grid == FG
    diag = _diag_only_mask(mask)
    core = mask & ~diag
    row_candidates = _line_run_indices(core, axis=0)
    col_candidates = _line_run_indices(core, axis=1)
    assert len(row_candidates) >= 2 and len(col_candidates) >= 2
    row_counts = core.sum(axis=1)
    col_counts = core.sum(axis=0)
    row_period, row_residue = _infer_period(row_candidates, row_counts, grid.shape[0])
    col_period, col_residue = _infer_period(col_candidates, col_counts, grid.shape[1])
    periodic_rows = list(range(row_residue, grid.shape[0], row_period))
    periodic_cols = list(range(col_residue, grid.shape[1], col_period))
    return periodic_rows, periodic_cols, diag


def _fill_short_gaps(grid: np.ndarray, periodic_rows: list[int], periodic_cols: list[int]) -> np.ndarray:
    out = grid.copy()
    mask = out == FG
    h, w = out.shape

    for r in periodic_rows:
        c = 0
        while c < w:
            if mask[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and not mask[r, c]:
                c += 1
            c1 = c - 1
            if c0 > 0 and c < w and c1 - c0 + 1 <= 3 and mask[r, c0 - 1] and mask[r, c]:
                out[r, c0:c] = FG
                mask[r, c0:c] = True

    for c in periodic_cols:
        r = 0
        while r < h:
            if mask[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and not mask[r, c]:
                r += 1
            r1 = r - 1
            if r0 > 0 and r < h and r1 - r0 + 1 <= 3 and mask[r0 - 1, c] and mask[r, c]:
                out[r0:r, c] = FG
                mask[r0:r, c] = True

    return out


def _transform(
    grid: np.ndarray,
    *,
    erase_diag: bool = True,
    erase_outside: bool = True,
    fill_gaps: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    periodic_rows, periodic_cols, diag = _infer_periodic_lines(g)

    out = g.copy()
    if erase_diag:
        out[diag] = BG

    if erase_outside:
        row_mask = np.zeros(g.shape[0], dtype=bool)
        col_mask = np.zeros(g.shape[1], dtype=bool)
        row_mask[periodic_rows] = True
        col_mask[periodic_cols] = True
        for r, c in np.argwhere(out == FG):
            rr = int(r)
            cc = int(c)
            if not row_mask[rr] and not col_mask[cc]:
                out[rr, cc] = BG

    if fill_gaps:
        out = _fill_short_gaps(out, periodic_rows, periodic_cols)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase diagonal and period-breaking blue noise, then repair short gaps in the underlying periodic line grid.

    Input structure:
    - Black background with blue horizontal and vertical 1-pixel lines forming a periodic orthogonal grid.
    - Some line segments are erased into short black gaps, away from true line intersections.
    - Extra blue noise appears in two forms: diagonal-only pixels/chains and off-period pixels side-adjacent to real lines.

    Rule:
    - Detect reliable horizontal and vertical line runs of length at least 3, ignoring diagonal-only blue pixels with no 4-neighbor.
    - Infer the horizontal and vertical periods from those reliable runs, giving the rows and columns where true lines belong.
    - Delete all diagonal-only blue pixels, then delete every remaining blue pixel whose row is not a periodic line row and whose column is not a periodic line column.
    - On each inferred periodic row and column, fill any black gap of length at most 3 whose two ends are bounded by blue pixels on that same row or column.

    Key insight:
    - Local blue continuity is not enough: the same inferred global row/column periods are needed both to reject same-colored distractors and to decide which short gaps are genuine broken line segments.
    """
    return _transform(grid, erase_diag=True, erase_outside=True, fill_gaps=True)


def _period_options(dim: int) -> list[tuple[int, int, list[int]]]:
    opts: list[tuple[int, int, list[int]]] = []
    for period in range(PERIOD_MIN, PERIOD_MAX + 1):
        for residue in range(period):
            indices = list(range(residue, dim, period))
            if len(indices) >= 3:
                opts.append((period, residue, indices))
    return opts


def _build_pristine(h: int, w: int, rows: list[int], cols: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[rows, :] = FG
    grid[:, cols] = FG
    return grid


def _nonperiod_intervals(length: int, periodic_indices: set[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx not in periodic_indices:
            if start is None:
                start = idx
        elif start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, length - 1))
    return intervals


def _max_run_length(vec: np.ndarray) -> int:
    best = 0
    run = 0
    for value in vec:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _contiguous_true_len(vec: np.ndarray, start: int, step: int) -> int:
    count = 0
    idx = start
    while 0 <= idx < vec.shape[0] and bool(vec[idx]):
        count += 1
        idx += step
    return count


def _carve_horizontal_gaps(
    grid: np.ndarray,
    row: int,
    periodic_cols: set[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    intervals = _nonperiod_intervals(grid.shape[1], periodic_cols)
    candidates: list[tuple[int, int]] = []
    for a, b in intervals:
        for gap_len in (1, 2, 3):
            if b - a + 1 < gap_len:
                continue
            for start in range(a, b - gap_len + 2):
                if start == 0 or start + gap_len - 1 == grid.shape[1] - 1:
                    continue
                candidates.append((start, start + gap_len - 1))
    rng.shuffle(candidates)
    target = rng.randint(1, min(3, len(candidates)))
    picked: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for c0, c1 in candidates:
        if any(c in occupied for c in range(c0, c1 + 1)):
            continue
        row_mask = grid[row] == FG
        left_len = _contiguous_true_len(row_mask, c0 - 1, -1)
        right_len = _contiguous_true_len(row_mask, c1 + 1, 1)
        if left_len < 2 or right_len < 2:
            continue
        test = grid[row].copy()
        test[c0 : c1 + 1] = BG
        if _max_run_length(test == FG) < 3:
            continue
        picked.append((c0, c1))
        occupied.update(range(c0, c1 + 1))
        grid[row, c0 : c1 + 1] = BG
        if len(picked) >= target:
            break
    assert picked
    return picked


def _carve_vertical_gaps(
    grid: np.ndarray,
    col: int,
    periodic_rows: set[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    intervals = _nonperiod_intervals(grid.shape[0], periodic_rows)
    candidates: list[tuple[int, int]] = []
    for a, b in intervals:
        for gap_len in (1, 2, 3):
            if b - a + 1 < gap_len:
                continue
            for start in range(a, b - gap_len + 2):
                if start == 0 or start + gap_len - 1 == grid.shape[0] - 1:
                    continue
                candidates.append((start, start + gap_len - 1))
    rng.shuffle(candidates)
    target = rng.randint(1, min(3, len(candidates)))
    picked: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for r0, r1 in candidates:
        if any(r in occupied for r in range(r0, r1 + 1)):
            continue
        col_mask = grid[:, col] == FG
        up_len = _contiguous_true_len(col_mask, r0 - 1, -1)
        down_len = _contiguous_true_len(col_mask, r1 + 1, 1)
        if up_len < 2 or down_len < 2:
            continue
        test = grid[:, col].copy()
        test[r0 : r1 + 1] = BG
        if _max_run_length(test == FG) < 3:
            continue
        picked.append((r0, r1))
        occupied.update(range(r0, r1 + 1))
        grid[r0 : r1 + 1, col] = BG
        if len(picked) >= target:
            break
    assert picked
    return picked


def _valid_diag_chain(base_mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    test = base_mask.copy()
    for r, c in cells:
        if test[r, c]:
            return False
        test[r, c] = True
    for r, c in cells:
        if _orth_count(test, r, c) != 0:
            return False
        if not _has_diag_neighbor(test, r, c):
            return False
    return True


def _find_chain_in_horizontal_gap(
    grid: np.ndarray,
    gap: tuple[int, int, int],
    periodic_cols: set[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    row, c0, c1 = gap
    base = grid == FG
    candidates: list[list[tuple[int, int]]] = []
    for start in range(c0, c1 + 1):
        for dr in (-1, 1):
            for dc in (-1, 1):
                for length in (3, 2):
                    cells = [(row + i * dr, start + i * dc) for i in range(length)]
                    if any(
                        rr < 0
                        or rr >= grid.shape[0]
                        or cc < 0
                        or cc >= grid.shape[1]
                        or cc in periodic_cols
                        for rr, cc in cells[1:]
                    ):
                        continue
                    if not _valid_diag_chain(base, cells):
                        continue
                    candidates.append(cells)
    if not candidates:
        return None
    return rng.choice(candidates)


def _find_chain_in_vertical_gap(
    grid: np.ndarray,
    gap: tuple[int, int, int],
    periodic_rows: set[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    col, r0, r1 = gap
    base = grid == FG
    candidates: list[list[tuple[int, int]]] = []
    for start in range(r0, r1 + 1):
        for dr in (-1, 1):
            for dc in (-1, 1):
                for length in (3, 2):
                    cells = [(start + i * dr, col + i * dc) for i in range(length)]
                    if any(
                        rr < 0
                        or rr >= grid.shape[0]
                        or cc < 0
                        or cc >= grid.shape[1]
                        or rr in periodic_rows
                        for rr, cc in cells[1:]
                    ):
                        continue
                    if not _valid_diag_chain(base, cells):
                        continue
                    candidates.append(cells)
    if not candidates:
        return None
    return rng.choice(candidates)


def _generic_diag_candidates(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    base = grid == FG
    h, w = grid.shape
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if base[r, c]:
                continue
            for dr in (-1, 1):
                for dc in (-1, 1):
                    for length in (3, 2, 1):
                        cells = [(r + i * dr, c + i * dc) for i in range(length)]
                        if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in cells):
                            continue
                        if _valid_diag_chain(base, cells):
                            out.append(cells)
    return out


def _add_diag_noise(
    grid: np.ndarray,
    h_gaps: list[tuple[int, int, int]],
    v_gaps: list[tuple[int, int, int]],
    periodic_rows: set[int],
    periodic_cols: set[int],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    noise: list[tuple[int, int]] = []

    special_candidates: list[list[tuple[int, int]]] = []
    for gap in h_gaps:
        if gap[2] - gap[1] + 1 >= 2:
            chain = _find_chain_in_horizontal_gap(grid, gap, periodic_cols, rng)
            if chain is not None:
                special_candidates.append(chain)
    for gap in v_gaps:
        if gap[2] - gap[1] + 1 >= 2:
            chain = _find_chain_in_vertical_gap(grid, gap, periodic_rows, rng)
            if chain is not None:
                special_candidates.append(chain)
    if not special_candidates:
        return noise, []

    special = rng.choice(special_candidates)
    for cell in special:
        grid[cell] = FG
    noise.extend(special)

    target = rng.randint(3, 5)
    while len(noise) < target:
        base = grid == FG
        singletons: list[tuple[int, int]] = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != BG:
                    continue
                probe = base.copy()
                probe[r, c] = True
                if _orth_count(probe, r, c) == 0 and _has_diag_neighbor(probe, r, c):
                    singletons.append((r, c))
        if not singletons:
            break
        cell = rng.choice(singletons)
        grid[cell] = FG
        noise.append(cell)
    return noise, special


def _add_offperiod_distractors(
    grid: np.ndarray,
    periodic_rows: set[int],
    periodic_cols: set[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or r in periodic_rows or c in periodic_cols:
                continue
            if _orth_count(grid == FG, r, c) == 0:
                continue
            candidates.append((r, c))
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        probe = grid.copy()
        probe[r, c] = FG
        row_candidates = _line_run_indices(probe == FG, axis=0)
        col_candidates = _line_run_indices(probe == FG, axis=1)
        if any(idx not in periodic_rows and idx == r for idx in row_candidates):
            continue
        if any(idx not in periodic_cols and idx == c for idx in col_candidates):
            continue
        grid[r, c] = FG
        chosen.append((r, c))
        if len(chosen) >= rng.randint(2, 4):
            break
    return chosen


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    periodic_rows: list[int],
    periodic_cols: list[int],
    diag_noise: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    special_diag: list[tuple[int, int]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if len(diag_noise) < 3 or not (2 <= len(distractors) <= 4):
        return False
    if not special_diag:
        return False
    if not any(r in periodic_rows or c in periodic_cols for r, c in special_diag):
        return False

    inferred_rows, inferred_cols, diag = _infer_periodic_lines(inp)
    if inferred_rows != periodic_rows or inferred_cols != periodic_cols:
        return False

    diag_mask = _diag_only_mask(inp == FG)
    if int(diag_mask.sum()) < 3:
        return False

    if np.array_equal(_transform(inp, erase_diag=True, erase_outside=False, fill_gaps=True), expected):
        return False
    if np.array_equal(_transform(inp, erase_diag=True, erase_outside=True, fill_gaps=False), expected):
        return False

    core = (inp == FG) & ~diag
    row_candidates = set(_line_run_indices(core, axis=0))
    col_candidates = set(_line_run_indices(core, axis=1))
    if any(r not in periodic_rows for r in row_candidates):
        return False
    if any(c not in periodic_cols for c in col_candidates):
        return False

    return True


def _attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    row_period, _row_residue, rows = rng.choice(_period_options(h))
    col_period, _col_residue, cols = rng.choice(_period_options(w))
    if len(rows) < 3 or len(cols) < 3:
        return None

    pristine = _build_pristine(h, w, rows, cols)
    inp = pristine.copy()
    periodic_rows = set(rows)
    periodic_cols = set(cols)

    untouched_rows = set(rng.sample(rows, 2))
    untouched_cols = set(rng.sample(cols, 2))
    damaged_rows = [r for r in rows if r not in untouched_rows]
    damaged_cols = [c for c in cols if c not in untouched_cols]
    if not damaged_rows or not damaged_cols:
        return None

    h_gaps: list[tuple[int, int, int]] = []
    for row in damaged_rows:
        for c0, c1 in _carve_horizontal_gaps(inp, row, periodic_cols, rng):
            h_gaps.append((row, c0, c1))
    v_gaps: list[tuple[int, int, int]] = []
    for col in damaged_cols:
        for r0, r1 in _carve_vertical_gaps(inp, col, periodic_rows, rng):
            v_gaps.append((col, r0, r1))

    diag_noise, special_diag = _add_diag_noise(inp, h_gaps, v_gaps, periodic_rows, periodic_cols, rng)
    if len(diag_noise) < 3:
        return None
    if not any(r in periodic_rows or c in periodic_cols for r, c in diag_noise):
        return None

    distractors = _add_offperiod_distractors(inp, periodic_rows, periodic_cols, rng)
    if not (2 <= len(distractors) <= 4):
        return None

    if not _semantic_checks(inp, pristine, rows, cols, diag_noise, distractors, special_diag):
        return None

    if row_period == col_period and rows[0] % row_period == cols[0] % col_period and np.array_equal(inp, pristine):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
