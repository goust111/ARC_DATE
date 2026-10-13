from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DIFF_MARK = 1
SAME_MARK = 2
SEP = 5

PATTERN_COLORS = (3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _line_mask(h: int, w: int, row_lines: list[int], col_lines: list[int]) -> tuple[np.ndarray, np.ndarray]:
    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[row_lines] = True
    col_mask[col_lines] = True
    return row_mask, col_mask


def _find_layout(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], list[int]]] = []

    for n_rows in range(3, 6):
        if (h - 1) % n_rows != 0:
            continue
        step_r = (h - 1) // n_rows
        if step_r < 2:
            continue
        row_lines = [i * step_r for i in range(n_rows + 1)]

        for n_cols in range(3, 6):
            if (w - 1) % n_cols != 0:
                continue
            step_c = (w - 1) // n_cols
            if step_c < 2:
                continue
            col_lines = [j * step_c for j in range(n_cols + 1)]

            row_mask, col_mask = _line_mask(h, w, row_lines, col_lines)
            internal_col_mask = np.zeros(w, dtype=bool)
            internal_col_mask[col_lines[1:-1]] = True
            marker_mask = row_mask[:, None] & internal_col_mask[None, :]
            separator_mask = (row_mask[:, None] | col_mask[None, :]) & ~marker_mask

            if not np.all(grid[separator_mask] == SEP):
                continue
            if marker_mask.sum() == 0:
                continue
            if not np.all(np.isin(grid[marker_mask], [SAME_MARK, DIFF_MARK])):
                continue

            pattern_keys = set()
            for r in range(n_rows):
                for c in range(n_cols):
                    cell = grid[row_lines[r] + 1 : row_lines[r + 1], col_lines[c] + 1 : col_lines[c + 1]]
                    if cell.size == 0:
                        break
                    pattern_keys.add(cell.tobytes())
            if 2 <= len(pattern_keys) <= 3:
                candidates.append((row_lines, col_lines))

    assert len(candidates) == 1, "expected one regular separator layout"
    return candidates[0]


def _extract_cells(
    grid: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> tuple[list[np.ndarray], np.ndarray]:
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    patterns: list[np.ndarray] = []
    pattern_ids: dict[bytes, int] = {}
    observed = np.empty((n_rows, n_cols), dtype=int)

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[row_lines[r] + 1 : row_lines[r + 1], col_lines[c] + 1 : col_lines[c + 1]].copy()
            key = cell.tobytes()
            if key not in pattern_ids:
                pattern_ids[key] = len(patterns)
                patterns.append(cell)
            observed[r, c] = pattern_ids[key]

    assert 2 <= len(patterns) <= 3, "expected exactly 2-3 distinct cell sprites"
    return patterns, observed


def _relations_from_markers(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> list[bool]:
    relations: list[bool] = []
    for col in col_lines[1:-1]:
        vals = {int(grid[row, col]) for row in row_lines}
        assert len(vals) == 1, "expected one marker color per vertical boundary"
        mark = vals.pop()
        assert mark in (SAME_MARK, DIFF_MARK), "unexpected marker color"
        relations.append(mark == SAME_MARK)
    return relations


def _valid_row_assignments(n_patterns: int, n_cols: int, relations: list[bool]) -> list[tuple[int, ...]]:
    valid = []
    for seq in itertools.product(range(n_patterns), repeat=n_cols):
        ok = True
        for i, same in enumerate(relations):
            if same and seq[i] != seq[i + 1]:
                ok = False
                break
            if not same and seq[i] == seq[i + 1]:
                ok = False
                break
        if ok:
            valid.append(seq)
    assert valid, "expected at least one satisfying row assignment"
    return valid


def _solve_rows(observed: np.ndarray, valid_rows: list[tuple[int, ...]]) -> np.ndarray:
    solved = np.empty_like(observed)
    for r, row in enumerate(observed):
        best_score = -1
        winners: list[tuple[int, ...]] = []
        row_tuple = tuple(map(int, row))
        for seq in valid_rows:
            score = sum(int(a == b) for a, b in zip(row_tuple, seq))
            if score > best_score:
                best_score = score
                winners = [seq]
            elif score == best_score:
                winners.append(seq)
        assert len(winners) == 1, "expected a unique best row assignment"
        solved[r, :] = winners[0]
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """Reassign cell sprites to satisfy same/different markers hidden on separator intersections.

    The grid is partitioned by one-pixel separator lines. Markers on internal
    vertical-boundary intersections encode whether horizontally adjacent cells
    must use the same sprite or different sprites. Some cells are already
    correct; choose the unique sprite assignment that satisfies all constraints
    while matching the observed cells as much as possible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_lines, col_lines = _find_layout(g)
    patterns, observed = _extract_cells(g, row_lines, col_lines)
    relations = _relations_from_markers(g, row_lines, col_lines)
    valid_rows = _valid_row_assignments(len(patterns), observed.shape[1], relations)
    solved = _solve_rows(observed, valid_rows)

    out = g.copy()
    for r in range(observed.shape[0]):
        for c in range(observed.shape[1]):
            out[row_lines[r] + 1 : row_lines[r + 1], col_lines[c] + 1 : col_lines[c + 1]] = patterns[int(solved[r, c])]
    return out


def _diag_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        c = (r * (w - 1)) // max(1, h - 1)
        mask[r, c] = True
    return mask


def _anti_diag_mask(h: int, w: int) -> np.ndarray:
    return np.fliplr(_diag_mask(h, w))


def _plus_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _l_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _t_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w // 2] = True
    return mask


def _box_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


PATTERN_BUILDERS = (_diag_mask, _anti_diag_mask, _plus_mask, _l_mask, _t_mask, _box_mask)


def _build_patterns(cell_h: int, cell_w: int, n_patterns: int) -> list[np.ndarray] | None:
    builders = random.sample(PATTERN_BUILDERS, n_patterns)
    colors = random.sample(PATTERN_COLORS, n_patterns)
    patterns: list[np.ndarray] = []

    for builder, color in zip(builders, colors):
        mask = builder(cell_h, cell_w)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        cell = np.full((cell_h, cell_w), BG, dtype=int)
        cell[mask] = color
        patterns.append(cell)

    if len({pattern.tobytes() for pattern in patterns}) != n_patterns:
        return None
    return patterns


def _choose_target_rows(
    n_rows: int, valid_rows: list[tuple[int, ...]], n_patterns: int
) -> list[tuple[int, ...]] | None:
    required = set(range(n_patterns))
    for _ in range(200):
        rows = [random.choice(valid_rows) for _ in range(n_rows)]
        if {val for row in rows for val in row} == required:
            return rows
    return None


def _observed_candidates(
    target: tuple[int, ...], valid_rows: list[tuple[int, ...]], n_patterns: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    changed: list[tuple[int, ...]] = []
    unchanged: list[tuple[int, ...]] = []
    width = len(target)

    for obs in itertools.product(range(n_patterns), repeat=width):
        best = -1
        winners: list[tuple[int, ...]] = []
        for seq in valid_rows:
            score = sum(int(a == b) for a, b in zip(obs, seq))
            if score > best:
                best = score
                winners = [seq]
            elif score == best:
                winners.append(seq)
        if len(winners) != 1 or winners[0] != target:
            continue

        diff = sum(int(a != b) for a, b in zip(obs, target))
        if diff == 0:
            unchanged.append(obs)
        elif 0 < diff < width:
            changed.append(obs)

    return changed, unchanged


def _render_grid(
    patterns: list[np.ndarray], assignments: list[tuple[int, ...]], relations: list[bool]
) -> np.ndarray:
    n_rows = len(assignments)
    n_cols = len(assignments[0])
    cell_h, cell_w = patterns[0].shape
    h = n_rows * cell_h + (n_rows + 1)
    w = n_cols * cell_w + (n_cols + 1)
    row_lines = [i * (cell_h + 1) for i in range(n_rows + 1)]
    col_lines = [j * (cell_w + 1) for j in range(n_cols + 1)]

    grid = np.full((h, w), BG, dtype=int)
    grid[row_lines, :] = SEP
    grid[:, col_lines] = SEP
    for idx, same in enumerate(relations, start=1):
        grid[row_lines, col_lines[idx]] = SAME_MARK if same else DIFF_MARK

    for r in range(n_rows):
        for c in range(n_cols):
            grid[row_lines[r] + 1 : row_lines[r + 1], col_lines[c] + 1 : col_lines[c + 1]] = patterns[
                assignments[r][c]
            ]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    cell_h = random.randint(3, 5)
    cell_w = random.randint(3, 5)
    n_patterns = random.choice((2, 3))

    patterns = _build_patterns(cell_h, cell_w, n_patterns)
    if patterns is None:
        return None

    relations = [random.choice([False, True]) for _ in range(n_cols - 1)]
    valid_rows = _valid_row_assignments(n_patterns, n_cols, relations)
    if len(valid_rows) < 2:
        return None

    target_rows = _choose_target_rows(n_rows, valid_rows, n_patterns)
    if target_rows is None:
        return None

    change_mask = [random.random() < 0.75 for _ in range(n_rows)]
    if not any(change_mask):
        change_mask[random.randrange(n_rows)] = True

    observed_rows: list[tuple[int, ...]] = []
    changed_any = False
    for must_change, target in zip(change_mask, target_rows):
        changed, unchanged = _observed_candidates(target, valid_rows, n_patterns)
        if must_change and changed:
            observed = random.choice(changed)
            changed_any = True
        elif not must_change and unchanged:
            observed = random.choice(unchanged)
        elif changed:
            observed = random.choice(changed)
            changed_any = True
        elif unchanged:
            observed = random.choice(unchanged)
        else:
            return None
        observed_rows.append(observed)

    if not changed_any:
        return None
    if {val for row in observed_rows for val in row} != set(range(n_patterns)):
        return None

    inp = _render_grid(patterns, observed_rows, relations)
    expected = _render_grid(patterns, target_rows, relations)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid constraint-grid puzzle")
