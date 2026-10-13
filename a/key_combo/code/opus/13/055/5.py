from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


GRID_SIZE = 4
OUTPUT_SIZE = GRID_SIZE * GRID_SIZE
BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 600

_GEN_CALL_COUNT = 0
_SEEN_SETTLED_SIGNATURES: set[bytes] = set()


def _settle_columns(grid: np.ndarray) -> np.ndarray:
    """Apply per-column downward gravity while preserving top-to-bottom order."""
    settled = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for c in range(GRID_SIZE):
        values = [int(v) for v in grid[:, c] if int(v) != BACKGROUND]
        if values:
            settled[GRID_SIZE - len(values) :, c] = values
    return settled


def _self_stamp(sprite: np.ndarray) -> np.ndarray:
    """Paste the whole sprite into each block selected by a non-black sprite cell."""
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if int(sprite[r, c]) == BACKGROUND:
                continue
            r0 = r * GRID_SIZE
            c0 = c * GRID_SIZE
            out[r0 : r0 + GRID_SIZE, c0 : c0 + GRID_SIZE] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop a 4x4 sprite by column gravity, then self-stamp the settled sprite into a 16x16 block grid.

    Input structure:
    - The input is a 4x4 grid on black background with 3 to 6 non-black pixels and 1 to 3 non-black colors.
    - Non-black pixels form a small multicolor sprite; some columns may contain floating pixels above black gaps.
    - Each column is interpreted independently using the standard ARC (row, col) coordinate system.

    Rule:
    - For each column, collect its non-black pixels from top to bottom and slide them downward so they stack at the bottom of that same column without reordering.
    - Let the settled 4x4 result be S.
    - Create a 16x16 output divided into a 4x4 grid of 4x4 blocks.
    - For every non-black cell of S, copy the entire settled sprite S into the corresponding 4x4 output block; leave blocks selected by black cells empty.

    Key insight:
    - The gravity result is used twice at once: it determines both which output blocks are activated and the exact 4x4 pattern pasted inside each activated block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 4x4 input grid"
    settled = _settle_columns(g)
    return _self_stamp(settled)


def _all_count_patterns(total: int, occupied_cols: int) -> list[tuple[int, int, int, int]]:
    patterns = []
    for cols in itertools.product(range(GRID_SIZE + 1), repeat=GRID_SIZE):
        if sum(cols) != total:
            continue
        if sum(v > 0 for v in cols) != occupied_cols:
            continue
        if any(v > GRID_SIZE for v in cols):
            continue
        patterns.append(tuple(int(v) for v in cols))
    return patterns


_COUNT_PATTERNS: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {
    (total, occupied): _all_count_patterns(total, occupied)
    for total in range(3, 7)
    for occupied in range(2, min(GRID_SIZE, total) + 1)
}


def _choose_palette(rng: np.random.Generator, distinct_colors: int) -> list[int]:
    colors = list(rng.choice(PALETTE, size=distinct_colors, replace=False))
    rng.shuffle(colors)
    return [int(c) for c in colors]


def _assign_settled_colors(
    rng: np.random.Generator,
    counts: Iterable[int],
    distinct_colors: int,
) -> np.ndarray:
    counts = list(counts)
    total = sum(counts)
    palette = _choose_palette(rng, distinct_colors)
    values = palette.copy()
    while len(values) < total:
        values.append(int(rng.choice(palette)))
    rng.shuffle(values)

    settled = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    idx = 0
    for c, count in enumerate(counts):
        if count == 0:
            continue
        rows = range(GRID_SIZE - count, GRID_SIZE)
        for r in rows:
            settled[r, c] = int(values[idx])
            idx += 1
    return settled


def _lifted_rows_options(count: int) -> list[tuple[int, ...]]:
    return [tuple(rows) for rows in itertools.combinations(range(GRID_SIZE), count)]


_ROW_OPTIONS = {count: _lifted_rows_options(count) for count in range(1, GRID_SIZE + 1)}


def _derive_input_from_settled(rng: np.random.Generator, settled: np.ndarray) -> np.ndarray | None:
    lifted = np.zeros_like(settled)
    moved_columns: list[int] = []
    movable_columns: list[int] = []

    for c in range(GRID_SIZE):
        values = [int(v) for v in settled[:, c] if int(v) != BACKGROUND]
        count = len(values)
        if count == 0:
            continue

        bottom_rows = tuple(range(GRID_SIZE - count, GRID_SIZE))
        options = _ROW_OPTIONS[count]
        if count < GRID_SIZE:
            movable_columns.append(c)

        if count == GRID_SIZE:
            chosen_rows = bottom_rows
        else:
            non_bottom = [rows for rows in options if rows != bottom_rows]
            choose_float = rng.random() < 0.7 or not moved_columns
            if choose_float and non_bottom:
                chosen_rows = non_bottom[int(rng.integers(0, len(non_bottom)))]
            else:
                chosen_rows = bottom_rows

        if chosen_rows != bottom_rows:
            moved_columns.append(c)
        for r, color in zip(chosen_rows, values, strict=False):
            lifted[r, c] = int(color)

    if not moved_columns and movable_columns:
        c = movable_columns[int(rng.integers(0, len(movable_columns)))]
        values = [int(v) for v in settled[:, c] if int(v) != BACKGROUND]
        count = len(values)
        bottom_rows = tuple(range(GRID_SIZE - count, GRID_SIZE))
        alternatives = [rows for rows in _ROW_OPTIONS[count] if rows != bottom_rows]
        if not alternatives:
            return None
        chosen_rows = alternatives[int(rng.integers(0, len(alternatives)))]
        lifted[:, c] = BACKGROUND
        for r, color in zip(chosen_rows, values, strict=False):
            lifted[r, c] = int(color)

    if np.array_equal(lifted, settled):
        return None
    return lifted


def _grid_signature(grid: np.ndarray) -> bytes:
    g = np.asarray(grid, dtype=np.int8)
    return g.tobytes()


def _block_is(grid: np.ndarray, block_r: int, block_c: int, sprite: np.ndarray) -> bool:
    r0 = block_r * GRID_SIZE
    c0 = block_c * GRID_SIZE
    return np.array_equal(grid[r0 : r0 + GRID_SIZE, c0 : c0 + GRID_SIZE], sprite)


def _validate_semantics(inp: np.ndarray, settled: np.ndarray) -> bool:
    if inp.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if settled.shape != (GRID_SIZE, GRID_SIZE):
        return False

    if np.count_nonzero(inp) < 3 or np.count_nonzero(inp) > 6:
        return False
    if len({int(v) for v in inp.flatten() if int(v) != BACKGROUND}) not in {1, 2, 3}:
        return False
    if np.array_equal(inp, settled):
        return False
    if not np.array_equal(_settle_columns(inp), settled):
        return False

    true_out = main(inp)
    expected_out = _self_stamp(settled)
    if not np.array_equal(true_out, expected_out):
        return False

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            is_active = int(settled[r, c]) != BACKGROUND
            block = true_out[r * GRID_SIZE : (r + 1) * GRID_SIZE, c * GRID_SIZE : (c + 1) * GRID_SIZE]
            if is_active and not np.array_equal(block, settled):
                return False
            if not is_active and np.any(block != BACKGROUND):
                return False

    # Reject degeneracies where simpler wrong rules accidentally match.
    if np.array_equal(true_out, _self_stamp(inp)):
        return False

    settle_template_original_stamp = np.zeros_like(true_out)
    original_template_settled_stamp = np.zeros_like(true_out)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if int(settled[r, c]) != BACKGROUND:
                r0 = r * GRID_SIZE
                c0 = c * GRID_SIZE
                settle_template_original_stamp[r0 : r0 + GRID_SIZE, c0 : c0 + GRID_SIZE] = inp
            if int(inp[r, c]) != BACKGROUND:
                r0 = r * GRID_SIZE
                c0 = c * GRID_SIZE
                original_template_settled_stamp[r0 : r0 + GRID_SIZE, c0 : c0 + GRID_SIZE] = settled

    if np.array_equal(true_out, settle_template_original_stamp):
        return False
    if np.array_equal(true_out, original_template_settled_stamp):
        return False

    # Ensure gravity changes the template support, not just hidden internals.
    if np.array_equal(inp != BACKGROUND, settled != BACKGROUND):
        return False

    active_blocks = int(np.count_nonzero(settled))
    if active_blocks < 3:
        return False

    # Require the settled sprite to appear in at least two different block rows or columns.
    active_positions = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if int(settled[r, c]) != BACKGROUND]
    if len({r for r, _ in active_positions}) == 1 and len({c for _, c in active_positions}) == 1:
        return False

    # Confirm the activated blocks are exactly those selected by the settled sprite.
    for r, c in active_positions:
        if not _block_is(true_out, r, c, settled):
            return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    rng = np.random.default_rng()
    nonzero_target = 3 + ((_GEN_CALL_COUNT - 1) % 4)
    distinct_colors_target = 1 + ((_GEN_CALL_COUNT - 1) % 3)
    occupied_cols_target = 2 + ((_GEN_CALL_COUNT - 1) % 3)
    occupied_cols_target = min(occupied_cols_target, nonzero_target, GRID_SIZE)

    for attempt in range(MAX_ATTEMPTS):
        # Keep cyclical branch coverage, but allow local variation after early retries.
        total = nonzero_target if attempt < MAX_ATTEMPTS // 2 else int(rng.integers(3, 7))
        distinct_colors = min(distinct_colors_target, total)
        occupied_cols = occupied_cols_target
        if occupied_cols > total:
            occupied_cols = total
        if attempt >= MAX_ATTEMPTS // 3:
            occupied_cols = int(rng.integers(2, min(GRID_SIZE, total) + 1))

        patterns = _COUNT_PATTERNS[(total, occupied_cols)]
        if not patterns:
            continue
        counts = patterns[int(rng.integers(0, len(patterns)))]
        settled = _assign_settled_colors(rng, counts, distinct_colors)
        inp = _derive_input_from_settled(rng, settled)
        if inp is None:
            continue
        if not _validate_semantics(inp, settled):
            continue

        settled_sig = _grid_signature(settled)
        if settled_sig in _SEEN_SETTLED_SIGNATURES:
            continue
        _SEEN_SETTLED_SIGNATURES.add(settled_sig)
        return inp

    raise RuntimeError("failed to generate a non-degenerate gravity self-stamp sample")
