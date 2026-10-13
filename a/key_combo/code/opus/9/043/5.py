from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
CELL_SIZE = 3
GAP = 1
LEGEND_WIDTH = 4
MAX_GENERATION_ATTEMPTS = 400

BASE_PATTERNS = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 0, 0), (1, 0, 0), (1, 1, 1)),
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 1, 0), (0, 1, 1), (0, 0, 1)),
    ((0, 1, 1), (0, 1, 0), (1, 1, 0)),
    ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
)


def _build_pattern_library() -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for base in BASE_PATTERNS:
        arr = np.array(base, dtype=bool)
        for rot in range(4):
            rotated = np.rot90(arr, rot)
            for flip in (False, True):
                variant = np.fliplr(rotated) if flip else rotated
                key = tuple(int(v) for v in variant.reshape(-1))
                if key not in seen:
                    seen.add(key)
                    variants.append(variant.copy())
    return tuple(variants)


PATTERN_LIBRARY = _build_pattern_library()


def _grid_size_from_order(n: int) -> int:
    return n * CELL_SIZE + (n - 1) * GAP


def _validate_partial_latin(grid: np.ndarray, palette: tuple[int, ...]) -> None:
    palette_set = set(palette)
    for row in grid:
        values = [int(v) for v in row if int(v) != BLACK]
        assert len(values) == len(set(values)), "rows may not repeat a clue symbol"
        assert set(values) <= palette_set
    for col in grid.T:
        values = [int(v) for v in col if int(v) != BLACK]
        assert len(values) == len(set(values)), "columns may not repeat a clue symbol"
        assert set(values) <= palette_set


def _latin_completion_count(
    partial: np.ndarray, palette: tuple[int, ...], limit: int = 2
) -> tuple[int, np.ndarray | None]:
    work = partial.copy()
    palette_set = set(palette)
    solutions: list[np.ndarray] = []

    def domain(r: int, c: int) -> list[int]:
        row_used = {int(v) for v in work[r, :] if int(v) != BLACK}
        col_used = {int(v) for v in work[:, c] if int(v) != BLACK}
        return [color for color in palette if color not in row_used and color not in col_used]

    def search() -> None:
        if len(solutions) >= limit:
            return

        blanks = np.argwhere(work == BLACK)
        if len(blanks) == 0:
            if all({int(v) for v in row} == palette_set for row in work) and all(
                {int(v) for v in col} == palette_set for col in work.T
            ):
                solutions.append(work.copy())
            return

        best_choices: list[int] | None = None
        best_cell: tuple[int, int] | None = None
        for r_raw, c_raw in blanks:
            r, c = int(r_raw), int(c_raw)
            choices = domain(r, c)
            if not choices:
                return
            if best_choices is None or len(choices) < len(best_choices):
                best_choices = choices
                best_cell = (r, c)
                if len(best_choices) == 1:
                    break

        assert best_choices is not None and best_cell is not None
        r, c = best_cell
        for color in best_choices:
            work[r, c] = color
            search()
            work[r, c] = BLACK

    search()
    return len(solutions), (solutions[0] if solutions else None)


def _parse_legend(legend: np.ndarray, n: int) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    assert legend.shape == (_grid_size_from_order(n), LEGEND_WIDTH)
    palette: list[int] = []
    mapping: dict[int, np.ndarray] = {}

    for idx in range(n):
        r0 = idx * (CELL_SIZE + GAP)
        band = legend[r0 : r0 + CELL_SIZE, :]
        assert np.all(np.isin(band[:, 1:], (BLACK, TEAL))), "legend patterns must be black/teal"

        markers = np.argwhere((band[:, 0] != BLACK) & (band[:, 0] != TEAL) & (band[:, 0] != GREY))
        assert len(markers) == 1, "each legend entry needs one marker pixel"
        color = int(band[int(markers[0, 0]), 0])
        assert color in MARKER_COLORS, "unexpected legend color"

        pattern = (band[:, 1:] == TEAL)
        assert pattern.any(), "legend pattern may not be empty"
        key = tuple(int(v) for v in pattern.reshape(-1))
        assert color not in mapping, "legend colors must be unique"
        assert key not in {tuple(int(v) for v in m.reshape(-1)) for m in mapping.values()}, (
            "legend patterns must be unique"
        )

        palette.append(color)
        mapping[color] = pattern

        if idx < n - 1:
            assert np.all(legend[r0 + CELL_SIZE, :] == BLACK), "legend rows must be separated by black gaps"

    return tuple(palette), mapping


def _parse_right_grid(right: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    height, width = right.shape
    assert height == width and (height + 1) % (CELL_SIZE + GAP) == 0
    n = (height + 1) // (CELL_SIZE + GAP)
    allowed = np.array((BLACK, *palette), dtype=int)
    assert np.all(np.isin(right, allowed)), "right panel may only contain black and legend colors"

    for r in range(CELL_SIZE, height, CELL_SIZE + GAP):
        assert np.all(right[r, :] == BLACK), "row separators must stay black"
    for c in range(CELL_SIZE, width, CELL_SIZE + GAP):
        assert np.all(right[:, c] == BLACK), "column separators must stay black"

    partial = np.full((n, n), BLACK, dtype=int)
    for r in range(n):
        for c in range(n):
            r0 = r * (CELL_SIZE + GAP)
            c0 = c * (CELL_SIZE + GAP)
            block = right[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            clues = np.argwhere(block != BLACK)
            if len(clues) == 0:
                continue
            assert len(clues) == 1, "a clue cell must contain exactly one colored pixel"
            partial[r, c] = int(block[int(clues[0, 0]), int(clues[0, 1])])

    return partial


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, ...], dict[int, np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(separator_cols) == 1, "expected one grey separator column"
    sep = separator_cols[0]

    legend = g[:, :sep]
    right = g[:, sep + 1 :]
    assert right.shape[0] == g.shape[0], "right panel must span the full height"
    assert right.shape[0] == right.shape[1], "right panel must be square"
    assert (right.shape[0] + 1) % (CELL_SIZE + GAP) == 0, "right panel must match the 3x3+gap lattice"

    n = (right.shape[0] + 1) // (CELL_SIZE + GAP)
    assert n in (3, 4), "this task uses 3x3 or 4x4 Latin boards"
    palette, mapping = _parse_legend(legend, n)
    partial = _parse_right_grid(right, palette)
    _validate_partial_latin(partial, palette)
    return palette, mapping, partial


def _render_output(solution: np.ndarray, mapping: dict[int, np.ndarray]) -> np.ndarray:
    n = solution.shape[0]
    size = _grid_size_from_order(n)
    out = np.zeros((size, size), dtype=int)
    for r in range(n):
        for c in range(n):
            r0 = r * (CELL_SIZE + GAP)
            c0 = c * (CELL_SIZE + GAP)
            block = out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            block[mapping[int(solution[r, c])]] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend colors into 3x3 teal motifs and complete the right-hand Latin square with those motifs.

    Input structure:
    - A black grid is split by one full grey separator column into a left legend and a right board.
    - The left legend has 3 or 4 entries; each entry is a single colored marker pixel followed by a distinct 3x3 teal pattern, with one black spacer row between entries.
    - The right board is an NxN lattice of 3x3 cells separated by black gap rows and columns; some cells contain one colored clue pixel from the legend palette and the rest are blank.

    Rule:
    - Read the left legend as a one-to-one color-to-pattern codebook.
    - Interpret each colored clue on the right as the pattern named by that color.
    - Fill every blank board cell so each row and each column contains all N patterns exactly once, then replace every board cell by its full 3x3 teal pattern and keep the black gaps.

    Key insight:
    - The colors are only symbolic labels: the actual Latin-square vocabulary is the set of legend patterns, so solving requires both legend decoding and row/column uniqueness in pattern space.
    """
    palette, mapping, partial = _parse_input(grid)
    n_solutions, solution = _latin_completion_count(partial, palette, limit=2)
    assert n_solutions == 1 and solution is not None, "the board must have a unique Latin completion"
    return _render_output(solution, mapping)


def _generate_full_table(colors: tuple[int, ...]) -> np.ndarray:
    n = len(colors)
    base = np.array([[(r + c) % n for c in range(n)] for r in range(n)], dtype=int)
    row_order = random.sample(range(n), n)
    col_order = random.sample(range(n), n)
    symbol_order = random.sample(list(colors), n)
    table = base[row_order, :][:, col_order]
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = symbol_order[int(table[r, c])]
    if random.random() < 0.5:
        out = out.T.copy()
    return out


def _basic_partial_constraints(partial: np.ndarray, palette: tuple[int, ...]) -> bool:
    if np.any(np.sum(partial != BLACK, axis=1) == 0):
        return False
    if np.any(np.sum(partial != BLACK, axis=0) == 0):
        return False
    present = set(int(v) for v in partial[partial != BLACK])
    if present != set(palette):
        return False
    try:
        _validate_partial_latin(partial, palette)
    except AssertionError:
        return False
    return True


def _anti_shortcut_constraints(partial: np.ndarray, palette: tuple[int, ...]) -> bool:
    row_blanks = np.sum(partial == BLACK, axis=1)
    col_blanks = np.sum(partial == BLACK, axis=0)
    if not np.any(row_blanks >= 2):
        return False
    if not np.any(col_blanks >= 2):
        return False

    palette_set = set(palette)
    found_jointly_constrained_blank = False
    for r_raw, c_raw in np.argwhere(partial == BLACK):
        r, c = int(r_raw), int(c_raw)
        row_missing = palette_set - {int(v) for v in partial[r, :] if int(v) != BLACK}
        col_missing = palette_set - {int(v) for v in partial[:, c] if int(v) != BLACK}
        if len(row_missing) > 1 and len(col_missing) > 1:
            found_jointly_constrained_blank = True
            break
    return found_jointly_constrained_blank


def _blank_target_choices(n: int) -> tuple[int, ...]:
    return (3, 4, 5) if n == 3 else (6, 7, 8, 9)


def _make_partial_table(full: np.ndarray, palette: tuple[int, ...]) -> np.ndarray | None:
    n = full.shape[0]
    target_blanks = random.choice(_blank_target_choices(n))
    partial = full.copy()
    coords = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(coords)

    for r, c in coords:
        if int(np.sum(partial == BLACK)) >= target_blanks:
            break
        value = int(partial[r, c])
        partial[r, c] = BLACK
        if not _basic_partial_constraints(partial, palette):
            partial[r, c] = value
            continue
        n_solutions, _ = _latin_completion_count(partial, palette, limit=2)
        if n_solutions != 1:
            partial[r, c] = value

    if int(np.sum(partial == BLACK)) != target_blanks:
        return None
    if not _basic_partial_constraints(partial, palette):
        return None
    if not _anti_shortcut_constraints(partial, palette):
        return None
    return partial


def _choose_patterns(n: int) -> tuple[np.ndarray, ...]:
    chosen = random.sample(PATTERN_LIBRARY, n)
    return tuple(pattern.copy() for pattern in chosen)


def _assemble_input(
    palette: tuple[int, ...], mapping: dict[int, np.ndarray], partial: np.ndarray
) -> np.ndarray:
    n = len(palette)
    size = _grid_size_from_order(n)

    legend = np.zeros((size, LEGEND_WIDTH), dtype=int)
    for idx, color in enumerate(palette):
        r0 = idx * (CELL_SIZE + GAP)
        marker_row = random.randrange(CELL_SIZE)
        legend[r0 + marker_row, 0] = color
        legend[r0 : r0 + CELL_SIZE, 1:4][mapping[color]] = TEAL

    right = np.zeros((size, size), dtype=int)
    for r in range(n):
        for c in range(n):
            color = int(partial[r, c])
            if color == BLACK:
                continue
            r0 = r * (CELL_SIZE + GAP)
            c0 = c * (CELL_SIZE + GAP)
            rr = r0 + random.randrange(CELL_SIZE)
            cc = c0 + random.randrange(CELL_SIZE)
            right[rr, cc] = color

    out = np.zeros((size, LEGEND_WIDTH + 1 + size), dtype=int)
    out[:, :LEGEND_WIDTH] = legend
    out[:, LEGEND_WIDTH] = GREY
    out[:, LEGEND_WIDTH + 1 :] = right
    return out


def _generate_attempt() -> np.ndarray | None:
    n = random.choice((3, 4))
    palette = tuple(random.sample(MARKER_COLORS, n))
    patterns = _choose_patterns(n)
    mapping = {color: pattern for color, pattern in zip(palette, patterns)}

    full = _generate_full_table(palette)
    partial = _make_partial_table(full, palette)
    if partial is None:
        return None

    grid = _assemble_input(palette, mapping, partial)
    expected = _render_output(full, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely completable encoded Latin-pattern puzzle")
