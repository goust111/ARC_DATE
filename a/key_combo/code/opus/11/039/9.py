from __future__ import annotations

import random

import numpy as np


BLACK = 0
CONTROL = 5
FOREGROUND_COLORS = (1, 2, 3, 4)
INPUT_SIZES = (4, 5)
GEN_RETRIES = 500

CONTROL_PATTERNS: dict[int, list[tuple[tuple[int, ...], int]]] = {
    2: [((2,), 1)],
    3: [((2, 1), 4), ((3,), 1)],
    4: [((2, 1, 1), 4), ((2, 2), 3), ((3, 1), 2), ((4,), 1)],
}


def _source_pattern(grid: np.ndarray) -> np.ndarray:
    source = np.asarray(grid, dtype=int).copy()
    source[source == CONTROL] = BLACK
    return source


def _stamp_without_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    source = _source_pattern(g)
    out = np.zeros((n * n, n * n), dtype=int)
    for r, c in np.argwhere(g == CONTROL):
        rr = int(r) * n
        cc = int(c) * n
        out[rr : rr + n, cc : cc + n] = source
    return out


def _apply_column_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = np.zeros_like(g)
    for c in range(w):
        values = g[:, c]
        falling = values[values != BLACK]
        if falling.size:
            out[h - falling.size :, c] = falling
    return out


def _column_sequence(grid: np.ndarray, c: int) -> tuple[int, ...]:
    return tuple(int(v) for v in grid[:, c] if int(v) != BLACK)


def _control_column_counts(grid: np.ndarray) -> list[int]:
    return [int(np.sum(grid[:, c] == CONTROL)) for c in range(int(grid.shape[1]))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp grey-selected copies of the source pattern, then collapse every output column under gravity.

    Input structure:
    - The input is a square black-background grid.
    - Some cells are grey control pixels; all other non-black, non-grey cells are the source-pattern foreground.
    - Grey pixels are selectors only: they indicate which output blocks receive a copy of the whole input pattern.

    Rule:
    - Replace grey with black inside the source pattern, then build an N-by-N array of N-by-N blocks.
    - For each grey input pixel at (r, c), copy the whole grey-erased input pattern into output block (r, c); all non-grey block slots stay black.
    - After stamping, process each output column independently: keep its non-black pixels in top-to-bottom order, slide them downward, and pack them tightly against the bottom.

    Key insight:
    - Grey pixels first decide where full copies appear, and gravity then merges the vertically separated copies column by column into new stacked color sequences.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = int(g.shape[0])
    assert n >= 1 and n * n <= 30, "output must fit inside ARC size limits"
    assert np.any(g == CONTROL), "expected at least one control pixel"

    stamped = _stamp_without_gravity(g)
    return _apply_column_gravity(stamped)


def _weighted_choice(items: list[tuple[tuple[int, ...], int]]) -> tuple[int, ...]:
    patterns = [pattern for pattern, _ in items]
    weights = [weight for _, weight in items]
    return random.choices(patterns, weights=weights, k=1)[0]


def _random_controls(n: int, count: int) -> set[tuple[int, int]]:
    pattern = _weighted_choice(CONTROL_PATTERNS[count])
    cols = random.sample(range(n), len(pattern))
    controls: set[tuple[int, int]] = set()
    for col, amount in zip(cols, pattern):
        rows = random.sample(range(n), amount)
        for row in rows:
            controls.add((row, col))
    return controls


def _random_foreground_positions(
    n: int, controls: set[tuple[int, int]], count: int
) -> tuple[list[tuple[int, int]], int] | None:
    free_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in controls]
    free_by_col = {
        c: [r for r in range(n) if (r, c) not in controls]
        for c in range(n)
    }
    seq_cols = [c for c, rows in free_by_col.items() if len(rows) >= 2]
    if not seq_cols:
        return None

    seq_col = random.choice(seq_cols)
    seq_rows = random.sample(free_by_col[seq_col], 2)
    positions = [(seq_rows[0], seq_col), (seq_rows[1], seq_col)]

    remaining = [cell for cell in free_cells if cell not in positions]
    if count >= 3:
        outside = [cell for cell in remaining if cell[1] != seq_col]
        if not outside:
            return None
        chosen = random.choice(outside)
        positions.append(chosen)
        remaining.remove(chosen)

    while len(positions) < count:
        if not remaining:
            return None
        chosen = random.choice(remaining)
        positions.append(chosen)
        remaining.remove(chosen)

    return positions, seq_col


def _fill_foreground(
    grid: np.ndarray, positions: list[tuple[int, int]]
) -> None:
    distinct = random.randint(2, min(len(FOREGROUND_COLORS), len(positions)))
    palette = random.sample(FOREGROUND_COLORS, distinct)
    first_two = random.sample(palette, 2)
    tail = [random.choice(palette) for _ in range(len(positions) - 2)]
    random.shuffle(tail)
    colors = first_two + tail
    for (r, c), color in zip(positions, colors):
        grid[r, c] = int(color)


def _build_candidate() -> np.ndarray | None:
    n = random.choice(INPUT_SIZES)
    control_count = random.choices([2, 3, 4], weights=[1, 3, 3], k=1)[0]
    controls = _random_controls(n, control_count)

    fg_count = random.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    chosen = _random_foreground_positions(n, controls, fg_count)
    if chosen is None:
        return None
    positions, _seq_col = chosen

    grid = np.zeros((n, n), dtype=int)
    for r, c in controls:
        grid[r, c] = CONTROL
    _fill_foreground(grid, positions)
    return grid


def _validate_distribution(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    assert g.shape == (n, n)
    assert n in INPUT_SIZES

    control_mask = g == CONTROL
    fg_mask = (g != BLACK) & (g != CONTROL)
    fg_colors = set(int(v) for v in g[fg_mask])

    assert 2 <= int(control_mask.sum()) <= 4
    assert 2 <= int(fg_mask.sum()) <= 4
    assert fg_colors <= set(FOREGROUND_COLORS)
    assert len(fg_colors) >= 2

    control_rows = {int(r) for r, _ in np.argwhere(control_mask)}
    assert len(control_rows) >= 2
    counts = _control_column_counts(g)
    assert max(counts) >= 2

    source = _source_pattern(g)
    rich_source_cols = [
        c
        for c in range(n)
        if len(_column_sequence(source, c)) >= 2
        and len(set(_column_sequence(source, c))) >= 2
    ]
    assert rich_source_cols, "need a source column whose color order matters"


def _validate_non_degenerate(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    source = _source_pattern(g)
    stamped = _stamp_without_gravity(g)
    out = main(g)
    n = int(g.shape[0])

    assert np.any(source != BLACK), "source pattern must contain visible pixels"
    assert np.any(stamped != BLACK), "stamping must create visible output"
    assert not np.array_equal(stamped, out), "gravity must change the stamped layout"

    moved = np.argwhere((stamped != BLACK) & (out == BLACK))
    assert moved.size > 0, "some stamped pixels must vacate upper cells"

    rich_source_cols = [
        c
        for c in range(n)
        if len(_column_sequence(source, c)) >= 2
        and len(set(_column_sequence(source, c))) >= 2
    ]
    stacked_control_cols = [
        c for c, count in enumerate(_control_column_counts(g)) if count >= 2
    ]
    assert rich_source_cols and stacked_control_cols

    found_repeated_sequence = False
    for block_c in stacked_control_cols:
        copies = _control_column_counts(g)[block_c]
        for src_c in rich_source_cols:
            seq = _column_sequence(source, src_c)
            out_seq = _column_sequence(out, block_c * n + src_c)
            if out_seq == seq * copies:
                found_repeated_sequence = True
                break
        if found_repeated_sequence:
            break
    assert found_repeated_sequence, "need an actual stacked repeated color sequence"

    control_positions = np.argwhere(g == CONTROL)
    rr, cc = map(int, control_positions[0])
    changed_control = g.copy()
    changed_control[rr, cc] = BLACK
    assert not np.array_equal(main(changed_control), out), "output must depend on control pixels"

    fg_positions = np.argwhere((g != BLACK) & (g != CONTROL))
    rr, cc = map(int, fg_positions[0])
    replacement = next(color for color in FOREGROUND_COLORS if color != int(g[rr, cc]))
    changed_fg = g.copy()
    changed_fg[rr, cc] = replacement
    assert not np.array_equal(main(changed_fg), out), "output must depend on source colors"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            _validate_distribution(grid)
            _validate_non_degenerate(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
