from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4)
PALETTE_SET = set(PALETTE)
SEPARATOR = 5
ORDER = 4
BLANK = -1
GENERATION_ATTEMPTS = 500


def _separator_blocks(line_mask: np.ndarray) -> list[tuple[int, int]]:
    indices = [int(i) for i in np.flatnonzero(line_mask)]
    assert len(indices) == ORDER - 1, "expected exactly three separator lines"
    blocks: list[tuple[int, int]] = []
    start = 0
    for idx in indices:
        assert start < idx, "separator lines must leave non-empty windows"
        blocks.append((start, idx))
        start = idx + 1
    assert start < len(line_mask), "separator lines must leave a final window"
    blocks.append((start, len(line_mask)))
    sizes = [end - start for start, end in blocks]
    assert len(set(sizes)) == 1, "all windows must have equal size along an axis"
    return blocks


def _window_table(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = np.all(g == SEPARATOR, axis=1)
    sep_cols = np.all(g == SEPARATOR, axis=0)
    row_blocks = _separator_blocks(sep_rows)
    col_blocks = _separator_blocks(sep_cols)

    table = np.full((ORDER, ORDER), BLANK, dtype=int)
    for r_idx, (r0, r1) in enumerate(row_blocks):
        for c_idx, (c0, c1) in enumerate(col_blocks):
            window = g[r0:r1, c0:c1]
            assert window.size > 0
            assert not np.any(window == SEPARATOR), "separator color may only appear on full divider lines"

            unique = set(map(int, np.unique(window)))
            if unique == {BACKGROUND}:
                continue

            assert BACKGROUND not in unique, "a filled window must not contain black holes"
            assert len(unique) == 1, "each filled window must be uniformly monochrome"
            color = next(iter(unique))
            assert color in PALETTE_SET, "window color must belong to the four puzzle colors"
            table[r_idx, c_idx] = color

    return table, row_blocks, col_blocks


def _validate_partial_latin(table: np.ndarray) -> None:
    for row in table:
        vals = [int(v) for v in row if int(v) != BLANK]
        assert len(vals) == len(set(vals)), "a row may not repeat a filled color"
    for col in table.T:
        vals = [int(v) for v in col if int(v) != BLANK]
        assert len(vals) == len(set(vals)), "a column may not repeat a filled color"


def _domain(table: np.ndarray, r: int, c: int) -> list[int]:
    used = {int(v) for v in table[r, :] if int(v) != BLANK}
    used.update(int(v) for v in table[:, c] if int(v) != BLANK)
    return [color for color in PALETTE if color not in used]


def _solve_unique(table: np.ndarray) -> np.ndarray:
    work = np.array(table, dtype=int, copy=True)
    _validate_partial_latin(work)
    solutions: list[np.ndarray] = []

    def search() -> None:
        if len(solutions) > 1:
            return

        blanks = np.argwhere(work == BLANK)
        if len(blanks) == 0:
            assert all(set(map(int, row)) == PALETTE_SET for row in work)
            assert all(set(map(int, col)) == PALETTE_SET for col in work.T)
            solutions.append(work.copy())
            return

        best_r = best_c = -1
        best_choices: list[int] | None = None
        for r_raw, c_raw in blanks:
            r, c = int(r_raw), int(c_raw)
            choices = _domain(work, r, c)
            if not choices:
                return
            if best_choices is None or len(choices) < len(best_choices):
                best_r, best_c, best_choices = r, c, choices
                if len(best_choices) == 1:
                    break

        assert best_choices is not None
        for color in best_choices:
            work[best_r, best_c] = color
            search()
            work[best_r, best_c] = BLANK

    search()
    assert len(solutions) == 1, "the window Latin square must have exactly one completion"
    return solutions[0]


def _forced_fill_rounds(table: np.ndarray) -> list[int]:
    work = np.array(table, dtype=int, copy=True)
    rounds: list[int] = []
    while True:
        fills: list[tuple[int, int, int]] = []
        for r_raw, c_raw in np.argwhere(work == BLANK):
            r, c = int(r_raw), int(c_raw)
            choices = _domain(work, r, c)
            if len(choices) == 1:
                fills.append((r, c, choices[0]))
        if not fills:
            return rounds
        rounds.append(len(fills))
        for r, c, color in fills:
            work[r, c] = color


def _paint_solution(
    grid: np.ndarray,
    solution: np.ndarray,
    row_blocks: list[tuple[int, int]],
    col_blocks: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r_idx, (r0, r1) in enumerate(row_blocks):
        for c_idx, (c0, c1) in enumerate(col_blocks):
            out[r0:r1, c0:c1] = int(solution[r_idx, c_idx])
    for sep_r in range(out.shape[0]):
        if np.all(grid[sep_r, :] == SEPARATOR):
            out[sep_r, :] = SEPARATOR
    for sep_c in range(out.shape[1]):
        if np.all(grid[:, sep_c] == SEPARATOR):
            out[:, sep_c] = SEPARATOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the blank grey-separated windows so the 4x4 window table becomes a complete Latin square.

    Input structure:
    - A black background is partitioned by three full grey horizontal lines and three full grey vertical lines.
    - The separators split the canvas into a 4x4 matrix of equal windows.
    - Each window is either entirely black or entirely filled with one of the four foreground colors {blue, red, green, yellow}.

    Rule:
    - Treat each window as one Latin-square cell whose value is its solid fill color.
    - Keep every already-filled window fixed and keep every grey separator line unchanged.
    - Infer the unique completion in which every window-row and every window-column uses each of the four foreground colors exactly once.
    - Fill every blank window solidly with its inferred color.

    Key insight:
    - The grey grid is not decoration: it defines a 4x4 table of windows, and the missing colors are determined by global row-and-column Latin constraints across windows rather than by any marker inside a window.
    """
    table, row_blocks, col_blocks = _window_table(grid)
    solution = _solve_unique(table)
    return _paint_solution(grid, solution, row_blocks, col_blocks)


def _latin_square() -> np.ndarray:
    base = (np.arange(ORDER)[:, None] + np.arange(ORDER)[None, :]) % ORDER
    square = np.array(PALETTE, dtype=int)[base]
    square = square[random.sample(range(ORDER), ORDER), :]
    square = square[:, random.sample(range(ORDER), ORDER)]
    symbol_perm = np.array(random.sample(PALETTE, ORDER), dtype=int)
    remap = {src: dst for src, dst in zip(PALETTE, symbol_perm)}
    square = np.vectorize(remap.__getitem__, otypes=[int])(square)
    if random.random() < 0.5:
        square = square.T
    return np.asarray(square, dtype=int)


def _random_blank_mask(blank_count: int) -> np.ndarray:
    mask = np.zeros((ORDER, ORDER), dtype=bool)
    cells = random.sample(range(ORDER * ORDER), blank_count)
    for idx in cells:
        mask[idx // ORDER, idx % ORDER] = True
    return mask


def _render_input(solution: np.ndarray, blank_mask: np.ndarray, cell_size: int) -> np.ndarray:
    side = ORDER * cell_size + (ORDER - 1)
    grid = np.full((side, side), BACKGROUND, dtype=int)
    for k in range(1, ORDER):
        idx = k * cell_size + (k - 1)
        grid[idx, :] = SEPARATOR
        grid[:, idx] = SEPARATOR

    for r in range(ORDER):
        for c in range(ORDER):
            if blank_mask[r, c]:
                continue
            r0 = r * (cell_size + 1)
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = int(solution[r, c])
    return grid


def _semantic_check(inp: np.ndarray, expected_solution: np.ndarray, require_deferred: bool) -> bool:
    try:
        table, row_blocks, col_blocks = _window_table(inp)
        _validate_partial_latin(table)
        solved = _solve_unique(table)
    except Exception:
        return False

    if not np.array_equal(solved, expected_solution):
        return False

    blank_count = int(np.count_nonzero(table == BLANK))
    filled_count = ORDER * ORDER - blank_count
    if blank_count not in {6, 7, 8}:
        return False
    if filled_count not in {8, 9, 10}:
        return False

    visible = set(map(int, table[table != BLANK]))
    if visible != PALETTE_SET:
        return False

    row_blanks = [int(np.count_nonzero(table[r, :] == BLANK)) for r in range(ORDER)]
    col_blanks = [int(np.count_nonzero(table[:, c] == BLANK)) for c in range(ORDER)]
    if max(row_blanks) > 3 or max(col_blanks) > 3:
        return False
    if not all(count >= 1 for count in row_blanks):
        return False
    if not all(count >= 1 for count in col_blanks):
        return False

    rounds = _forced_fill_rounds(table)
    if require_deferred:
        if sum(rounds) != blank_count or len(rounds) < 2 or rounds[0] >= blank_count:
            return False
    else:
        if rounds != [blank_count]:
            return False

    out = _paint_solution(inp, solved, row_blocks, col_blocks)
    if not np.array_equal(main(inp), out):
        return False

    out_table, _, _ = _window_table(out)
    if not all(set(map(int, row)) == PALETTE_SET for row in out_table):
        return False
    if not all(set(map(int, col)) == PALETTE_SET for col in out_table.T):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    solution = _latin_square()
    cell_size = random.randint(3, 6)
    require_deferred = random.random() < 0.7
    blank_count_choices = (7, 8) if require_deferred else (6, 7)

    for _ in range(150):
        blank_mask = _random_blank_mask(random.choice(blank_count_choices))
        inp = _render_input(solution, blank_mask, cell_size)
        if _semantic_check(inp, solution, require_deferred):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid uniquely completable separator-window Latin square")
