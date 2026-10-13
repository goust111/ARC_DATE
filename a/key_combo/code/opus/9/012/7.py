import itertools
import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAROON = 9


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    row_bars = [r for r in range(h) if np.all(g[r, :] == Color.MAROON)]
    col_bars = [c for c in range(w) if np.all(g[:, c] == Color.MAROON)]
    assert len(row_bars) >= 5 and len(col_bars) >= 5, "expected a lattice with outer border and inner bars"
    assert row_bars[0] == 0 and row_bars[-1] == h - 1
    assert col_bars[0] == 0 and col_bars[-1] == w - 1

    row_steps = np.diff(row_bars)
    col_steps = np.diff(col_bars)
    assert len(set(map(int, row_steps))) == 1, "row bars must be equally spaced"
    assert len(set(map(int, col_steps))) == 1, "column bars must be equally spaced"

    cell_h = int(row_steps[0] - 1)
    cell_w = int(col_steps[0] - 1)
    assert cell_h == cell_w and cell_h in (2, 3), "cells must be uniform 2x2 or 3x3 squares"

    row_spans = [(row_bars[i] + 1, row_bars[i + 1] - 1) for i in range(len(row_bars) - 1)]
    col_spans = [(col_bars[i] + 1, col_bars[i + 1] - 1) for i in range(len(col_bars) - 1)]
    assert len(row_spans) >= 4 and len(col_spans) >= 4
    return row_spans, col_spans, cell_h


def _cell_color(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> int:
    r0, r1 = row_span
    c0, c1 = col_span
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals = colors_present(block)
    assert len(vals) == 1, "each lattice cell must be uniformly filled"
    return int(vals[0])


def _read_cell_colors(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[list[int]]:
    return [
        [_cell_color(grid, row_span, col_span) for col_span in col_spans]
        for row_span in row_spans
    ]


def _infer_sequence(cell_colors: list[list[int]]) -> tuple[int, tuple[int, ...]]:
    rows = len(cell_colors)
    cols = len(cell_colors[0])
    observations: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            color = int(cell_colors[r][c])
            if r in (0, rows - 1) or c in (0, cols - 1):
                assert color == Color.BLACK, "edge cells should be black in the input"
            elif color != Color.BLACK:
                assert color in PALETTE, "internal clues must use the pattern palette"
                observations.append((r, c, color))
    assert observations, "need colored internal clues to infer the pattern"

    solutions: list[tuple[int, tuple[int, ...]]] = []
    for k in (3, 4):
        for seq in itertools.permutations(PALETTE, k):
            if all(seq[(r + c) % k] == color for r, c, color in observations):
                solutions.append((k, tuple(map(int, seq))))

    unique = list(dict.fromkeys(solutions))
    assert len(unique) == 1, "pattern clues must determine a unique diagonal-period mapping"
    return unique[0]


def _fill_cell(out: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _build_output(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sequence: tuple[int, ...],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    rows = len(row_spans)
    cols = len(col_spans)
    k = len(sequence)
    for r, row_span in enumerate(row_spans):
        for c, col_span in enumerate(col_spans):
            color = Color.GREY if r in (0, rows - 1) or c in (0, cols - 1) else sequence[(r + c) % k]
            _fill_cell(out, row_span, col_span, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill interior lattice cells by diagonal-period color classes, then overwrite every edge cell grey.

    Input structure:
    - Maroon lattice bars form the full outer border and equally spaced interior separators.
    - The separators partition the canvas into a rectangular array of uniform 2x2 or 3x3 cells.
    - Some non-edge cells already show colors from a subset/permutation of blue, red, green, yellow; other cells are black.
    - Every edge cell of the lattice array is black in the input, and lattice-bar pixels stay maroon.

    Rule:
    - Recover the lattice cell coordinates `(r, c)` from the maroon separators.
    - Infer the unique period `K` in `{3, 4}` and the color sequence for residues `(r + c) mod K` from the colored internal clues.
    - For every internal cell, use that residue rule to fill black holes and preserve already correct colors.
    - Overwrite every edge cell of the lattice array with grey, leaving all maroon separator pixels unchanged.

    Key insight:
    - Two independent structures interact: internal cells follow a diagonal congruence class by lattice coordinates, but the whole outer ring ignores that pattern and is uniformly grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, _cell_size = _parse_lattice(g)
    cell_colors = _read_cell_colors(g, row_spans, col_spans)
    _k, sequence = _infer_sequence(cell_colors)
    return _build_output(g, row_spans, col_spans, sequence)


def _blank_lattice(rows: int, cols: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = rows * cell_size + rows + 1
    w = cols * cell_size + cols + 1
    grid = np.full((h, w), Color.BLACK, dtype=int)

    row_bars = [i * (cell_size + 1) for i in range(rows + 1)]
    col_bars = [i * (cell_size + 1) for i in range(cols + 1)]
    grid[row_bars, :] = Color.MAROON
    grid[:, col_bars] = Color.MAROON

    row_spans = [(row_bars[i] + 1, row_bars[i + 1] - 1) for i in range(rows)]
    col_spans = [(col_bars[i] + 1, col_bars[i + 1] - 1) for i in range(cols)]
    return grid, row_spans, col_spans


def _interior_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]


def _residue_buckets(rows: int, cols: int, k: int) -> dict[int, list[tuple[int, int]]]:
    buckets = {residue: [] for residue in range(k)}
    for r, c in _interior_cells(rows, cols):
        buckets[(r + c) % k].append((r, c))
    return buckets


def _dimension_options(cell_size: int, k: int) -> list[tuple[int, int]]:
    limit = (30 - 1) // (cell_size + 1)
    options: list[tuple[int, int]] = []
    min_size = 5 if k == 3 else 6
    for rows in range(min_size, limit + 1):
        for cols in range(min_size, limit + 1):
            counts = [len(v) for v in _residue_buckets(rows, cols, k).values()]
            if min(counts) >= 3:
                options.append((rows, cols))
    assert options
    return options


def _choose_pair(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_score = -1
    for a, b in itertools.combinations(cells, 2):
        score = 0
        if a[0] != b[0]:
            score += 1
        if a[1] != b[1]:
            score += 1
        if score > best_score:
            best = (a, b)
            best_score = score
            if score == 2:
                break
    assert best is not None
    return best


def _has_row_and_col_diversity(visible: set[tuple[int, int]], sequence: tuple[int, ...], rows: int, cols: int) -> bool:
    k = len(sequence)
    row_ok = any(
        len({sequence[(r + c) % k] for c in range(1, cols - 1) if (r, c) in visible}) >= 2
        for r in range(1, rows - 1)
    )
    col_ok = any(
        len({sequence[(r + c) % k] for r in range(1, rows - 1) if (r, c) in visible}) >= 2
        for c in range(1, cols - 1)
    )
    return row_ok and col_ok


def _choose_visible_cells(rows: int, cols: int, sequence: tuple[int, ...]) -> set[tuple[int, int]]:
    k = len(sequence)
    buckets = _residue_buckets(rows, cols, k)
    visible: set[tuple[int, int]] = set()
    for residue in range(k):
        pair = _choose_pair(buckets[residue])
        visible.update(pair)

    remaining = [cell for cell in _interior_cells(rows, cols) if cell not in visible]
    random.shuffle(remaining)
    extra_count = random.randint(0, max(0, len(remaining) - 2))
    visible.update(remaining[:extra_count])

    if len(visible) == len(_interior_cells(rows, cols)):
        visible.remove(random.choice(tuple(visible)))
    return visible


def _render_case(
    rows: int,
    cols: int,
    cell_size: int,
    sequence: tuple[int, ...],
    visible: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    base, row_spans, col_spans = _blank_lattice(rows, cols, cell_size)
    inp = base.copy()
    out = base.copy()
    k = len(sequence)

    for r, row_span in enumerate(row_spans):
        for c, col_span in enumerate(col_spans):
            if r in (0, rows - 1) or c in (0, cols - 1):
                _fill_cell(out, row_span, col_span, Color.GREY)
                _fill_cell(inp, row_span, col_span, Color.BLACK)
            else:
                color = sequence[(r + c) % k]
                _fill_cell(out, row_span, col_span, color)
                _fill_cell(inp, row_span, col_span, color if (r, c) in visible else Color.BLACK)
    return inp, out


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    rows: int,
    cols: int,
    sequence: tuple[int, ...],
    visible: set[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    row_spans, col_spans, _ = _parse_lattice(inp)
    cell_colors = _read_cell_colors(inp, row_spans, col_spans)
    inferred_k, inferred_sequence = _infer_sequence(cell_colors)
    if inferred_k != len(sequence) or inferred_sequence != tuple(sequence):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    internal = _interior_cells(rows, cols)
    hidden_internal = [cell for cell in internal if cell not in visible]
    if len(hidden_internal) < len(sequence):
        return False
    if not _has_row_and_col_diversity(visible, sequence, rows, cols):
        return False

    residue_counts = Counter((r + c) % len(sequence) for r, c in visible)
    if any(residue_counts[residue] < 2 for residue in range(len(sequence))):
        return False
    if set(sequence) != {expected[row_spans[r][0], col_spans[c][0]] for r, c in visible}:
        return False

    edge_sample_colors = {
        expected[row_spans[r][0], col_spans[c][0]]
        for r in range(rows)
        for c in range(cols)
        if r in (0, rows - 1) or c in (0, cols - 1)
    }
    if edge_sample_colors != {Color.GREY}:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.choice((3, 4))
    cell_size = random.choice((2, 3))
    rows, cols = random.choice(_dimension_options(cell_size, k))
    sequence = tuple(random.sample(PALETTE, k))

    for _ in range(80):
        visible = _choose_visible_cells(rows, cols, sequence)
        inp, expected = _render_case(rows, cols, cell_size, sequence, visible)
        if _validate_case(inp, expected, rows, cols, sequence, visible):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-diagonal input")
