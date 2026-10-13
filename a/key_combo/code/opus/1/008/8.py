from __future__ import annotations

import random

import numpy as np


Grid = np.ndarray

INSTRUCTION_COLORS = (1, 2, 3, 4)
PLACEHOLDER = 6
TILE_COLOR_POOL = (5, 7, 8, 9)

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 500
_GEN_PERIOD_CYCLE = ((3, 3), (3, 4), (4, 3), (4, 4))
_GEN_INCOMPLETE_CYCLE = (1, 2, 3)


def _minimal_periods(block: Grid) -> tuple[int, int]:
    h, w = map(int, block.shape)

    row_period = None
    for period in range(1, h + 1):
        if all(int(block[r, c]) == int(block[r % period, c]) for r in range(h) for c in range(w)):
            row_period = period
            break

    col_period = None
    for period in range(1, w + 1):
        if all(int(block[r, c]) == int(block[r, c % period]) for r in range(h) for c in range(w)):
            col_period = period
            break

    assert row_period is not None and col_period is not None
    return int(row_period), int(col_period)


def _enumerate_period_candidates(observed: Grid, placeholder: int = PLACEHOLDER) -> list[tuple[int, int, dict[tuple[int, int], int]]]:
    h, w = map(int, observed.shape)
    cells = [
        (r, c, int(observed[r, c]))
        for r in range(h)
        for c in range(w)
        if int(observed[r, c]) != placeholder
    ]
    assert cells, "expected at least one visible tile cell"

    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for period_h in range(1, h + 1):
        for period_w in range(1, w + 1):
            mapping: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, value in cells:
                key = (r % period_h, c % period_w)
                prev = mapping.get(key)
                if prev is None:
                    mapping[key] = value
                elif prev != value:
                    ok = False
                    break
            if ok and len(mapping) == period_h * period_w:
                candidates.append((int(period_h), int(period_w), mapping))
    return candidates


def _pattern_from_mapping(shape: tuple[int, int], period_h: int, period_w: int, mapping: dict[tuple[int, int], int]) -> Grid:
    tile = np.empty((period_h, period_w), dtype=int)
    for r in range(period_h):
        for c in range(period_w):
            tile[r, c] = mapping[(r, c)]

    rr = np.arange(shape[0])[:, None] % period_h
    cc = np.arange(shape[1])[None, :] % period_w
    return tile[rr, cc]


def _infer_completed_pattern(observed: Grid, placeholder: int = PLACEHOLDER) -> tuple[Grid, tuple[int, int]]:
    candidates = _enumerate_period_candidates(observed, placeholder=placeholder)
    assert candidates, "no periodic completion fits the visible cells"

    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    best_key = (candidates[0][0] * candidates[0][1], candidates[0][0], candidates[0][1])
    best = [
        item
        for item in candidates
        if (item[0] * item[1], item[0], item[1]) == best_key
    ]

    completed = [_pattern_from_mapping(observed.shape, ph, pw, mapping) for ph, pw, mapping in best]
    ref = completed[0]
    assert all(np.array_equal(ref, other) for other in completed[1:]), "minimal-period completion is ambiguous"
    return ref, (int(best[0][0]), int(best[0][1]))


def _shift_rows(pattern: Grid, shifts: np.ndarray) -> Grid:
    out = np.empty_like(pattern)
    for r in range(pattern.shape[0]):
        out[r] = np.roll(pattern[r], int(shifts[r]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden periodic tile, then warp it by row-wise cyclic shifts encoded in the left edge.

    Input structure:
    - Column 0 is an instruction column whose cells use exactly the four control colors 1, 2, 3, and 4.
    - Columns 1..end form a repeated color tile built from non-control colors, except that some rows end with a contiguous suffix of placeholder color 6.
    - At least two rows are fully visible, and the visible non-placeholder cells determine one minimal 2D repeating tile.

    Rule:
    - Ignore column 0 and infer the minimal repeating tile of the remaining area from all visible non-placeholder cells.
    - Extend that tile periodically to replace every placeholder suffix cell.
    - For each row, read its column-0 color as a fixed right-shift amount: 1->1, 2->2, 3->3, 4->4.
    - Cyclically shift the completed tile row to the right by that amount.
    - Return only the shifted tile area, with the instruction column removed.

    Key insight:
    - The placeholder rows cannot be shifted directly: they must first be completed by the global tile periodicity, and the left-edge colors then act as external per-row shift codes rather than tile colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w_total = map(int, g.shape)
    assert h > 0 and w_total >= 2, "expected an instruction column plus a tile area"

    shifts = g[:, 0]
    assert set(map(int, shifts)).issubset(set(INSTRUCTION_COLORS)), "invalid instruction colors"

    observed = g[:, 1:]
    completed, _period = _infer_completed_pattern(observed, placeholder=PLACEHOLDER)
    return _shift_rows(completed, shifts)


def _repeat_tile(tile: Grid, out_h: int, out_w: int) -> Grid:
    rr = np.arange(out_h)[:, None] % tile.shape[0]
    cc = np.arange(out_w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _sample_instruction_column(height: int) -> np.ndarray:
    shifts: list[int] = []
    while len(shifts) < height:
        block = list(INSTRUCTION_COLORS)
        random.shuffle(block)
        shifts.extend(block)
    return np.asarray(shifts[:height], dtype=int)


def _gen_tile(period_h: int, period_w: int, n_colors: int) -> Grid | None:
    for _ in range(300):
        colors = _GEN_RNG.choice(TILE_COLOR_POOL, size=n_colors, replace=False)
        tile = _GEN_RNG.choice(colors, size=(period_h, period_w)).astype(int)
        if len(set(map(int, tile.flat))) != n_colors:
            continue
        if _minimal_periods(tile) != (period_h, period_w):
            continue
        if not np.any(tile[:, 1:] != tile[:, :-1]):
            continue
        if not np.any(tile[1:, :] != tile[:-1, :]):
            continue
        return tile
    return None


def _prefix_ambiguity_len(full_pattern: Grid, period_h: int, row: int, max_len: int) -> int:
    row_types = [tuple(map(int, full_pattern[r])) for r in range(period_h)]
    for length in range(max_len, 0, -1):
        prefix = tuple(map(int, full_pattern[row, :length]))
        matches = sum(row_pattern[:length] == prefix for row_pattern in row_types)
        if matches >= 2:
            return int(length)
    return 0


def _select_incomplete_rows(height: int, period_h: int, count: int) -> list[int]:
    residues = list(range(period_h))
    _GEN_RNG.shuffle(residues)
    selected: list[int] = []
    for residue in residues:
        rows = [row for row in range(height) if row % period_h == residue]
        if len(rows) >= 2:
            selected.append(int(_GEN_RNG.choice(rows)))
        if len(selected) == count:
            break
    return selected


def _build_observed_from_truth(full_pattern: Grid, incomplete_rows: list[int], special_visible_len: int) -> tuple[Grid, dict[int, int]]:
    observed = full_pattern.copy()
    visible_lens: dict[int, int] = {}
    special_row = incomplete_rows[0]
    visible_lens[special_row] = int(special_visible_len)
    observed[special_row, special_visible_len:] = PLACEHOLDER

    width = full_pattern.shape[1]
    for row in incomplete_rows[1:]:
        hidden_len = int(_GEN_RNG.integers(1, min(width - 1, full_pattern.shape[1] // 2 + 2) + 1))
        visible_len = width - hidden_len
        visible_lens[row] = int(visible_len)
        observed[row, visible_len:] = PLACEHOLDER

    return observed, visible_lens


def _expected_from_truth(full_pattern: Grid, shifts: np.ndarray) -> Grid:
    return _shift_rows(full_pattern, shifts)


def _validate_instance(
    inp: Grid,
    expected: Grid,
    full_pattern: Grid,
    period_h: int,
    period_w: int,
    visible_lens: dict[int, int],
) -> bool:
    g = np.asarray(inp, dtype=int)
    assert g.ndim == 2
    height, width_total = map(int, g.shape)
    width = width_total - 1
    assert height <= 30 and width_total <= 30
    assert expected.shape == (height, width)
    assert set(map(int, g[:, 0])) == set(INSTRUCTION_COLORS)
    assert PLACEHOLDER not in set(map(int, g[:, 0]))

    observed = g[:, 1:]
    complete_rows = [row for row in range(height) if not np.any(observed[row] == PLACEHOLDER)]
    if len(complete_rows) < max(2, period_h):
        return False
    if {row % period_h for row in complete_rows} != set(range(period_h)):
        return False

    for row in range(height):
        placeholder_cols = np.flatnonzero(observed[row] == PLACEHOLDER)
        if len(placeholder_cols) == 0:
            continue
        first = int(placeholder_cols[0])
        if not np.array_equal(placeholder_cols, np.arange(first, width)):
            return False
        if first <= 0:
            return False

    completed, inferred_period = _infer_completed_pattern(observed, placeholder=PLACEHOLDER)
    if inferred_period != (period_h, period_w):
        return False
    if not np.array_equal(completed, full_pattern):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    ambiguous_row_ok = False
    for row, visible_len in visible_lens.items():
        if visible_len >= period_w:
            continue
        prefix = tuple(map(int, full_pattern[row, :visible_len]))
        matches = sum(
            tuple(map(int, full_pattern[row_type, :visible_len])) == prefix
            for row_type in range(period_h)
        )
        if matches >= 2:
            ambiguous_row_ok = True
            break
    if not ambiguous_row_ok:
        return False

    return True


def _scheduled_params() -> tuple[int, int, int, int]:
    global _GEN_CALL_COUNT
    idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    period_h, period_w = _GEN_PERIOD_CYCLE[idx % len(_GEN_PERIOD_CYCLE)]
    incomplete_count = _GEN_INCOMPLETE_CYCLE[idx % len(_GEN_INCOMPLETE_CYCLE)]
    n_colors = 2 + (idx % 2)
    return int(period_h), int(period_w), int(incomplete_count), int(n_colors)


def _gen_attempt() -> Grid | None:
    period_h, period_w, incomplete_count, n_colors = _scheduled_params()

    height = int(period_h * 2 + _GEN_RNG.integers(0, 3))
    width = int(period_w * 2 + _GEN_RNG.integers(1, period_w + 3))

    tile = _gen_tile(period_h, period_w, n_colors)
    if tile is None:
        return None
    full_pattern = _repeat_tile(tile, height, width)

    incomplete_rows = _select_incomplete_rows(height, period_h, min(incomplete_count, period_h))
    if len(incomplete_rows) != min(incomplete_count, period_h):
        return None

    special_row = incomplete_rows[0]
    special_visible_len = _prefix_ambiguity_len(full_pattern, period_h, special_row, max_len=min(period_w - 1, 3))
    if special_visible_len == 0:
        return None

    observed, visible_lens = _build_observed_from_truth(full_pattern, incomplete_rows, special_visible_len)
    shifts = _sample_instruction_column(height)
    inp = np.column_stack([shifts, observed]).astype(int)
    expected = _expected_from_truth(full_pattern, shifts)

    if not _validate_instance(inp, expected, full_pattern, period_h, period_w, visible_lens):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
