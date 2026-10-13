import random

import numpy as np


RED = 2
BLUE = 1
GREEN = 3
BODY_PALETTE = [4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (4, 6)
GENERATION_ATTEMPTS = 300


def _instruction_shift(color: int) -> int:
    if color == RED:
        return 0
    if color == BLUE:
        return 1
    if color == GREEN:
        return -1
    raise AssertionError(f"unexpected instruction color: {color}")


def _apply_row_ops(body: np.ndarray, row_instructions: np.ndarray) -> np.ndarray:
    out = np.empty_like(body)
    for r, color in enumerate(np.asarray(row_instructions, dtype=int)):
        out[r] = np.roll(body[r], _instruction_shift(int(color)))
    return out


def _apply_col_ops(body: np.ndarray, col_instructions: np.ndarray) -> np.ndarray:
    out = np.empty_like(body)
    for c, color in enumerate(np.asarray(col_instructions, dtype=int)):
        out[:, c] = np.roll(body[:, c], _instruction_shift(int(color)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the two edge instruction columns to cyclically shift a square stripe field by rows, then by columns.

    Input structure:
    - The input contains an N x N colored body between two one-cell-wide edge columns.
    - The body is made of one-cell-high horizontal stripes with distinct base colors and a small stripe motif inside each row, so every row remains recognizable after a cyclic shift.
    - The left edge and right edge are instruction columns whose cells use only red, blue, and green.

    Rule:
    - Read the left edge top-to-bottom: red keeps the matching body row, blue rotates that row one step right, and green rotates it one step left.
    - Apply all row rotations simultaneously to the N x N body.
    - Read the right edge top-to-bottom as instructions for body columns from left to right: red keeps a column, blue rotates it one step downward, and green rotates it one step upward.
    - Apply all column rotations simultaneously to the row-shifted body, then remove both instruction columns.

    Key insight:
    - The edge colors are not part of the picture; they decode into two ordered layers of cyclic motion, and the second layer acts on the result of the first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 3

    body = g[:, 1:-1]
    row_instructions = g[:, 0]
    col_instructions = g[:, -1]

    assert body.shape == (h, h), "the body must be square so the right edge can encode one instruction per column"

    after_rows = _apply_row_ops(body, row_instructions)
    return _apply_col_ops(after_rows, col_instructions)


def _balanced_instructions(length: int, rng: random.Random) -> np.ndarray:
    seq = [RED, BLUE, GREEN] * (length // 3)
    seq.extend(rng.sample([RED, BLUE, GREEN], length % 3))
    rng.shuffle(seq)
    return np.array(seq, dtype=int)


def _make_body(row_colors: list[int], rng: random.Random) -> np.ndarray:
    n = len(row_colors)
    body = np.empty((n, n), dtype=int)
    for r, color in enumerate(row_colors):
        body[r, :] = color

    mode = rng.choice(("single", "double", "block"))
    phase_a = rng.randrange(n)
    phase_b = rng.randrange(n)
    step = rng.choice((1, 2)) if n >= 5 else 1

    for r in range(n):
        c1 = (phase_a + r * step) % n
        body[r, c1] = row_colors[(r + 1) % n]

        if mode == "double":
            c2 = (phase_b - r) % n
            if c2 == c1:
                c2 = (c2 + 1) % n
            body[r, c2] = row_colors[(r + 2) % n]
        elif mode == "block":
            c2 = (c1 + 1) % n
            if c2 != c1:
                body[r, c2] = row_colors[(r + 2) % n]

    return body


def _is_non_degenerate(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if w != h + 2:
        return False

    body = grid[:, 1:-1]
    row_instructions = grid[:, 0]
    col_instructions = grid[:, -1]

    if set(int(v) for v in row_instructions) != {RED, BLUE, GREEN}:
        return False
    if set(int(v) for v in col_instructions) != {RED, BLUE, GREEN}:
        return False

    if any(len(set(int(v) for v in row)) < 2 for row in body):
        return False
    if any(len(set(int(v) for v in body[:, c])) < 2 for c in range(body.shape[1])):
        return False

    after_rows = _apply_row_ops(body, row_instructions)
    after_cols = _apply_col_ops(after_rows, col_instructions)

    if np.array_equal(after_rows, body):
        return False
    if np.array_equal(after_cols, after_rows):
        return False
    if np.array_equal(after_cols, body):
        return False

    for r, color in enumerate(row_instructions):
        if int(color) != RED and np.array_equal(after_rows[r], body[r]):
            return False
    for c, color in enumerate(col_instructions):
        if int(color) != RED and np.array_equal(after_cols[:, c], after_rows[:, c]):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.randint(*GRID_SIZE_RANGE)
    row_colors = rng.sample(BODY_PALETTE, size)
    body = _make_body(row_colors, rng)
    row_instructions = _balanced_instructions(size, rng)
    col_instructions = _balanced_instructions(size, rng)
    grid = np.concatenate(
        [row_instructions[:, None], body, col_instructions[:, None]],
        axis=1,
    )
    if not _is_non_degenerate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
