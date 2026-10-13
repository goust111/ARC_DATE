from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PIXEL_COLORS = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)


def _find_bar_columns(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    cols: list[int] = []
    for c in range(w):
        column = grid[:, c]
        non_bg = column[column != BLACK]
        if non_bg.size < 2:
            continue
        if set(map(int, np.unique(non_bg))) <= {RED, GREY}:
            cols.append(int(c))

    assert 2 <= len(cols) <= 3, "expected two or three one-column guide bars"
    return cols


def _simulate_destination(row_instr: np.ndarray, bar_cols: list[int], width: int) -> int:
    for idx, col in enumerate(bar_cols):
        cell = int(row_instr[idx])
        if cell == GREY:
            return int(col - 1)
        if cell == RED:
            for prev in range(idx - 1, -1, -1):
                if int(row_instr[prev]) == GREY:
                    return int(bar_cols[prev] + 1)
            return 0
    return int(width - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each scattered pixel through the vertical instruction bars and place it at the terminal cell implied by that row's full instruction sequence.

    Input structure:
    - The grid has a black background, plus two or three one-column vertical guide bars.
    - Every guide-bar cell is either a grey wall, a red bounce cell, or a black gap.
    - Scattered colored pixels appear only to the left of the first bar, with at most one such pixel per row.

    Rule:
    - Each scattered pixel moves right along its row and reads the guide bars from left to right.
    - A grey cell stops the pixel immediately to that bar's left, a black gap lets it pass, and a red cell reverses it to the left.
    - After a reversal, the pixel keeps reading the already-passed bars in reverse order: gaps stay passable, grey cells stop it immediately to that bar's right, and if nothing blocks it then it stops on the left edge; if no reversal or wall ever occurs, it stops on the right edge.

    Key insight:
    - A row's outcome is determined by composing the whole left-to-right instruction string, not by treating any single bar as the answer by itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    bar_cols = _find_bar_columns(g)
    leftmost = int(bar_cols[0])

    instr = g[:, bar_cols]
    assert np.all(np.isin(instr, [BLACK, RED, GREY])), "bars may only contain black, red, or grey"
    assert np.all(np.sum(instr == RED, axis=1) <= 1), "each row may contain at most one red cell across all bars"

    out = g.copy()
    moved_pixels: list[tuple[int, int, int]] = []
    for r in range(h):
        cells = [int(c) for c in np.flatnonzero(g[r] != BLACK) if int(c) not in bar_cols]
        assert len(cells) <= 1, "expected at most one scattered pixel per row"
        if not cells:
            continue
        c = cells[0]
        assert c < leftmost, "scattered pixels must start left of the first guide bar"
        moved_pixels.append((int(r), c, int(g[r, c])))
        out[r, c] = BLACK

    assert moved_pixels, "expected at least one scattered pixel"

    for r, _c, color in moved_pixels:
        dest = _simulate_destination(instr[r], bar_cols, w)
        assert dest not in bar_cols, "terminal cell must not overlap a guide bar"
        assert out[r, dest] == BLACK, "terminal cells must be unique and empty"
        out[r, dest] = color

    return out


def _required_specs(bar_count: int) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = [("edge", -1)]
    for idx in range(bar_count):
        specs.append(("wall", idx))
    for idx in range(bar_count):
        specs.append(("bounce", idx))
    return specs


def _make_pattern(rng: np.random.Generator, bar_count: int, mode: str, idx: int) -> np.ndarray:
    row = np.zeros(bar_count, dtype=int)
    if mode == "edge":
        return row

    row[idx] = GREY if mode == "wall" else RED
    for suffix in range(idx + 1, bar_count):
        row[suffix] = int(rng.choice([BLACK, GREY]))
    return row


def _random_pattern(rng: np.random.Generator, bar_count: int) -> np.ndarray:
    mode = rng.choice(["edge", "wall", "bounce"], p=[0.25, 0.45, 0.30])
    if mode == "edge":
        return np.zeros(bar_count, dtype=int)
    idx = int(rng.integers(0, bar_count))
    return _make_pattern(rng, bar_count, str(mode), idx)


def _build_bar_columns(
    rng: np.random.Generator, bar_count: int
) -> tuple[int, list[int]]:
    left_margin = int(rng.integers(4, 8))
    gaps = [int(rng.integers(3, 7)) for _ in range(bar_count - 1)]
    right_margin = int(rng.integers(2, 6))

    cols = [left_margin]
    for gap in gaps:
        cols.append(int(cols[-1] + 1 + gap))
    width = int(cols[-1] + 1 + right_margin)
    return width, cols


def _pick_start_col(
    rng: np.random.Generator, first_bar_col: int, mode: str
) -> int:
    if mode == "bounce":
        return int(rng.integers(1, first_bar_col - 1))
    return int(rng.integers(0, first_bar_col - 1))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bar_count = int(rng.integers(2, 4))
    height = int(rng.integers(10, 19))
    width, bar_cols = _build_bar_columns(rng, bar_count)
    if width > 30:
        return None

    specs = _required_specs(bar_count)
    if len(specs) > height:
        return None

    instructions = np.zeros((height, bar_count), dtype=int)
    row_order = np.arange(height)
    rng.shuffle(row_order)
    pixel_rows = row_order[: len(specs)]
    non_pixel_rows = row_order[len(specs) :]

    assigned_specs: list[tuple[int, str, int]] = []
    for row, (mode, idx) in zip(pixel_rows, specs):
        instructions[row] = _make_pattern(rng, bar_count, mode, idx)
        assigned_specs.append((int(row), mode, idx))

    for row in non_pixel_rows:
        instructions[row] = _random_pattern(rng, bar_count)

    for bar_idx in range(bar_count):
        visible = int(np.count_nonzero(instructions[:, bar_idx] != BLACK))
        if visible >= 2:
            continue
        patch_rows = [int(r) for r in non_pixel_rows if instructions[r, bar_idx] == BLACK]
        rng.shuffle(patch_rows)
        for row in patch_rows:
            instructions[row, bar_idx] = GREY
            visible += 1
            if visible >= 2:
                break
        if visible < 2:
            return None

    grid = np.zeros((height, width), dtype=int)
    for idx, col in enumerate(bar_cols):
        mask = instructions[:, idx] != BLACK
        grid[mask, col] = instructions[mask, idx]

    rng.shuffle(assigned_specs)
    color_pool = rng.choice(PIXEL_COLORS, size=len(assigned_specs), replace=True)
    for color, (row, mode, _idx) in zip(color_pool, assigned_specs):
        start_col = _pick_start_col(rng, bar_cols[0], mode)
        grid[row, start_col] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    destinations = []
    for row, mode, idx in assigned_specs:
        destinations.append(_simulate_destination(instructions[row], bar_cols, width))
        if mode == "bounce" and destinations[-1] != 0:
            return None
        if mode == "edge" and destinations[-1] != width - 1:
            return None
        if mode == "wall" and destinations[-1] != bar_cols[idx] - 1:
            return None

    if len(set(destinations)) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-bar input")
