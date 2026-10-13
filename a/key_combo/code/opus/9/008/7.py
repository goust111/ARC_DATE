from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)

_RNG = np.random.default_rng()


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert rows.size == 1, "expected exactly one full-width grey separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into two non-empty parts"
    return sep


def _fillable_color(row: np.ndarray) -> int | None:
    left = int(row[0])
    right = int(row[-1])
    if left == right and left in PALETTE:
        return left
    return None


def _filled_sequence(top: np.ndarray) -> list[int]:
    seq: list[int] = []
    for row in top:
        color = _fillable_color(row)
        if color is not None:
            seq.append(color)
    return seq


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    out = g.copy()

    seq: list[int] = []
    for r in range(sep):
        color = _fillable_color(g[r])
        if color is None:
            continue
        out[r, :] = color
        seq.append(color)

    assert seq, "task requires at least one fillable row"
    for k in range(g.shape[0] - sep - 1):
        out[sep + 1 + k, :] = seq[k % len(seq)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill top rows whose boundary markers match, then tile that derived color sequence below the separator.

    Input structure:
    - The grid is mostly black, with one full-width grey separator row splitting an upper region and a lower region.
    - In the upper region, some rows may have one colored marker at column 0 and one at the last column.
    - Marker colors come from blue, red, green, and yellow; unmarked rows remain black.

    Rule:
    - Find the grey separator row.
    - For each row above it, if the leftmost and rightmost marker colors are the same, fill the entire row with that color; otherwise leave the row unchanged.
    - Read the colors of those newly filled rows from top to bottom to form a sequence.
    - Fill the rows below the separator with full-width stripes cycling through that sequence.

    Key insight:
    - The lower tiling template is not given directly; it must be inferred from which upper rows pass the same-color endpoint test and in what top-to-bottom order.
    """
    return _expected_output(grid)


def _sample_same_colors(count: int) -> list[int]:
    assert count >= 2
    if count == 2:
        return [int(c) for c in _RNG.choice(PALETTE, size=2, replace=False)]

    for _ in range(200):
        colors = [int(c) for c in _RNG.choice(PALETTE, size=count, replace=True)]
        if len(set(colors)) < 2:
            continue
        if colors == sorted(colors):
            continue
        if colors == list(reversed(colors)):
            continue
        if colors == colors[::-1]:
            continue
        return colors
    raise RuntimeError("failed to sample a non-degenerate same-color sequence")


def _sample_mismatch_pairs(count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        left, right = _RNG.choice(PALETTE, size=2, replace=False)
        pairs.append((int(left), int(right)))
    return pairs


def _role_layout(top_h: int, same_count: int, mismatch_count: int) -> list[str] | None:
    blank_count = top_h - same_count - mismatch_count
    roles = ["S"] * same_count + ["M"] * mismatch_count + ["B"] * blank_count
    for _ in range(200):
        perm = [roles[i] for i in _RNG.permutation(len(roles))]
        same_rows = [i for i, role in enumerate(perm) if role == "S"]
        if same_rows == list(range(same_count)):
            continue
        if not any(perm[i] != "S" for i in range(min(same_rows), max(same_rows) + 1)):
            continue
        if perm[0] == perm[-1] == "S":
            continue
        return perm
    return None


def _all_marked_left_sequence(grid: np.ndarray) -> list[int]:
    sep = _find_separator_row(grid)
    seq: list[int] = []
    for r in range(sep):
        left = int(grid[r, 0])
        right = int(grid[r, -1])
        if left in PALETTE and right in PALETTE:
            seq.append(left)
    return seq


def _tiled_output_from_sequence(grid: np.ndarray, seq: list[int]) -> np.ndarray:
    sep = _find_separator_row(grid)
    out = grid.copy()
    for k in range(grid.shape[0] - sep - 1):
        out[sep + 1 + k, :] = seq[k % len(seq)]
    return out


def _dedupe_in_order(seq: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for color in seq:
        if color not in seen:
            seen.add(color)
            out.append(color)
    return out


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    sep = _find_separator_row(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]

    assert 5 <= top.shape[0] <= 7
    assert 4 <= bottom.shape[0] <= 6
    assert np.all(bottom == BLACK), "generator must leave the lower area empty in the input"

    same_rows = 0
    mismatch_rows = 0
    blank_rows = 0
    seq: list[int] = []
    for row in top:
        assert np.all(row[1:-1] == BLACK), "only boundary marker cells may be used in the upper area"
        left = int(row[0])
        right = int(row[-1])
        if left == BLACK and right == BLACK:
            blank_rows += 1
            continue
        assert left in PALETTE and right in PALETTE
        if left == right:
            same_rows += 1
            seq.append(left)
        else:
            mismatch_rows += 1

    assert same_rows >= 2
    assert mismatch_rows >= 1
    assert len(set(seq)) >= 2
    assert bottom.shape[0] > len(seq), "bottom area must be tall enough to demonstrate cycling"
    assert blank_rows >= 1, "include unmarked rows to cover the keep-unchanged branch"

    out = main(grid)
    assert np.array_equal(out, _expected_output(grid))

    correct = out
    alternatives: list[np.ndarray] = []

    all_marked = _all_marked_left_sequence(grid)
    if all_marked != seq:
        alternatives.append(_tiled_output_from_sequence(grid, all_marked))

    rev = list(reversed(seq))
    if rev != seq:
        alternatives.append(_tiled_output_from_sequence(grid, rev))

    deduped = _dedupe_in_order(seq)
    if deduped != seq:
        alternatives.append(_tiled_output_from_sequence(grid, deduped))

    sorted_seq = sorted(seq)
    if sorted_seq != seq:
        alternatives.append(_tiled_output_from_sequence(grid, sorted_seq))

    no_cycle = grid.copy()
    for k, color in enumerate(seq):
        no_cycle[sep + 1 + k, :] = color
    alternatives.append(no_cycle)

    distinct = sum(not np.array_equal(correct, alt) for alt in alternatives)
    assert distinct >= 2, "generator must rule out multiple shortcut hypotheses"


def generate_input() -> np.ndarray:
    for _ in range(400):
        width = int(_RNG.integers(8, 15))
        top_h = int(_RNG.integers(5, 8))

        same_options = [n for n in (2, 3, 4) if n <= top_h - 2]
        if not same_options:
            same_options = [2]
        same_count = int(_RNG.choice(same_options))

        max_mismatch = top_h - same_count - 1
        if max_mismatch < 1:
            continue
        mismatch_count = int(_RNG.integers(1, max_mismatch + 1))

        layout = _role_layout(top_h, same_count, mismatch_count)
        if layout is None:
            continue

        bottom_min = max(4, same_count + 1)
        if bottom_min > 6:
            continue
        bottom_h = int(_RNG.integers(bottom_min, 7))

        same_colors = _sample_same_colors(same_count)
        mismatch_pairs = _sample_mismatch_pairs(mismatch_count)

        grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
        grid[top_h, :] = GREY

        same_i = 0
        mismatch_i = 0
        for r, role in enumerate(layout):
            if role == "S":
                color = same_colors[same_i]
                same_i += 1
                grid[r, 0] = color
                grid[r, -1] = color
            elif role == "M":
                left, right = mismatch_pairs[mismatch_i]
                mismatch_i += 1
                grid[r, 0] = left
                grid[r, -1] = right

        _validate_input(grid)
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
