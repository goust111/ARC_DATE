from __future__ import annotations

import random

import numpy as np


BG = 0
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_MIN_SIZE = 11
GRID_MAX_SIZE = 14
MIN_FOREGROUND = 20
MAX_FOREGROUND = 30
MAX_ATTEMPTS = 600


def _frequency_ranking(grid: np.ndarray) -> tuple[list[int], list[int]]:
    fg = np.asarray(grid, dtype=int)
    fg = fg[fg != BG]
    assert fg.size > 0, "expected non-background pixels"
    colors, counts = np.unique(fg, return_counts=True)
    pairs = sorted(((int(n), int(c)) for c, n in zip(colors, counts)), reverse=True)
    ranked_counts = [count for count, _ in pairs]
    assert len(set(ranked_counts)) == len(ranked_counts), "foreground color frequencies must be unique"
    ranked_colors = [color for _, color in pairs]
    return ranked_colors, ranked_counts


def _fill_positions_for_color(
    grid: np.ndarray, color: int, *, strict: bool = True
) -> tuple[set[tuple[int, int]], bool, bool]:
    g = np.asarray(grid, dtype=int)
    cells: set[tuple[int, int]] = set()
    used_row = False
    used_col = False

    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] == color)
        if strict:
            assert len(cols) <= 2, "line colors may have at most two pixels per row"
        if len(cols) == 2:
            used_row = True
            c0, c1 = map(int, cols)
            for c in range(c0 + 1, c1):
                if g[r, c] == BG:
                    cells.add((r, c))

    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] == color)
        if strict:
            assert len(rows) <= 2, "line colors may have at most two pixels per column"
        if len(rows) == 2:
            used_col = True
            r0, r1 = map(int, rows)
            for r in range(r0 + 1, r1):
                if g[r, c] == BG:
                    cells.add((r, c))

    return cells, used_row, used_col


def _apply_rule(
    grid: np.ndarray, connect_colors: list[int], repaint_colors: list[int], *, strict: bool = True
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for color in connect_colors:
        fill_cells, _, _ = _fill_positions_for_color(g, color, strict=strict)
        for r, c in fill_cells:
            out[r, c] = color
    if repaint_colors:
        out[np.isin(out, repaint_colors)] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve the two most frequent foreground colors, connect every lower-frequency color along aligned pairs, then repaint those line colors orange.

    Input structure:
    - Black (0) is the background and orange (7) is absent from the input.
    - The grid contains 4 or 5 non-background colors with strictly different pixel counts.
    - The two most frequent colors form structural shapes, while each lower-frequency color
      appears as sparse endpoints with at most two pixels in any row or column.

    Rule:
    - Count each non-background color and rank them by frequency.
    - Keep the top two colors unchanged.
    - For every remaining color, fill background cells strictly between same-color endpoint
      pairs in any row or column that contains exactly two such pixels.
    - Repaint every pixel of those lower-frequency colors, including the original endpoints
      and the newly filled connector cells, to orange (7).

    Key insight:
    - Color identity does not determine behavior: only frequency rank decides whether a color
      stays as structure or dissolves into orange connection segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == ORANGE), "orange must be absent from the input"

    ranked_colors, ranked_counts = _frequency_ranking(g)
    assert 4 <= len(ranked_colors) <= 5, "expected 4 or 5 non-background colors"
    assert 6 <= ranked_counts[0] <= 10 and 6 <= ranked_counts[1] <= 10, "structure colors must be frequent"
    assert all(2 <= count <= 4 for count in ranked_counts[2:]), "line colors must be sparse"

    structure_colors = ranked_colors[:2]
    line_colors = ranked_colors[2:]
    out = _apply_rule(g, line_colors, line_colors, strict=True)
    assert np.all(out[np.isin(g, structure_colors)] == g[np.isin(g, structure_colors)])
    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox_cells(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int = 0) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    out = set()
    for r in range(max(0, r0 - margin), min(h, r1 + margin + 1)):
        for c in range(max(0, c0 - margin), min(w, c1 + margin + 1)):
            out.add((r, c))
    return out


def _structure_pattern(count: int, orientation: str) -> set[tuple[int, int]]:
    candidates: list[set[tuple[int, int]]] = []
    for first_len in range(3, count - 2):
        second_len = count - first_len
        if second_len < 3:
            continue
        for gap in range(2, 5):
            cells: set[tuple[int, int]] = set()
            if orientation == "h":
                for c in range(first_len):
                    cells.add((0, c))
                for c in range(second_len):
                    cells.add((gap + 1, c))
            else:
                for r in range(first_len):
                    cells.add((r, 0))
                for r in range(second_len):
                    cells.add((r, gap + 1))
            candidates.append(cells)
    if not candidates:
        raise AssertionError(f"no structure pattern for count={count}")
    return random.choice(candidates)


def _line_pattern(count: int, mode: str) -> set[tuple[int, int]]:
    gap = lambda: random.randint(2, 5)

    if count == 2:
        g = gap()
        if mode == "row":
            return {(0, 0), (0, g + 1)}
        if mode == "col":
            return {(0, 0), (g + 1, 0)}
        raise AssertionError("count-2 patterns only support row or col")

    if count == 3:
        g = gap()
        if mode == "row":
            singleton_c = random.randint(1, g)
            return {(0, 0), (0, g + 1), (2, singleton_c)}
        if mode == "col":
            singleton_r = random.randint(1, g)
            return {(0, 0), (g + 1, 0), (singleton_r, 2)}
        raise AssertionError("count-3 patterns only support row or col")

    if count == 4 and mode == "both":
        return {(0, 0), (0, gap() + 1), (gap() + 1, 0), (gap() + 1, gap() + 1)}

    if count == 4 and mode == "row":
        while True:
            g0 = gap()
            g1 = gap()
            cells = {(0, 0), (0, g0 + 1), (2, 2), (2, g1 + 3)}
            cols = [c for _, c in cells]
            if len(cols) == len(set(cols)):
                return cells

    if count == 4 and mode == "col":
        while True:
            g0 = gap()
            g1 = gap()
            cells = {(0, 0), (g0 + 1, 0), (2, 2), (g1 + 3, 2)}
            rows = [r for r, _ in cells]
            if len(rows) == len(set(rows)):
                return cells

    raise AssertionError(f"unsupported line pattern count={count}, mode={mode}")


def _compatible_modes(count: int) -> list[str]:
    modes = ["row", "col"]
    if count == 4:
        modes.append("both")
    return modes


def _choose_line_modes(counts: list[int]) -> list[str]:
    modes: list[str | None] = [None] * len(counts)
    both_candidates = [i for i, count in enumerate(counts) if count == 4]
    if both_candidates and random.random() < 0.65:
        idx = random.choice(both_candidates)
        modes[idx] = "both"

    row_done = any(mode in ("row", "both") for mode in modes if mode is not None)
    col_done = any(mode in ("col", "both") for mode in modes if mode is not None)

    indices = list(range(len(counts)))
    random.shuffle(indices)
    if not row_done:
        idx = random.choice([i for i in indices if modes[i] is None and "row" in _compatible_modes(counts[i])])
        modes[idx] = "row"
    if not col_done:
        idx = random.choice([i for i in indices if modes[i] is None and "col" in _compatible_modes(counts[i])])
        modes[idx] = "col"

    for i, count in enumerate(counts):
        if modes[i] is None:
            modes[i] = random.choice(_compatible_modes(count))
    return [str(mode) for mode in modes]


def _place_pattern(
    shape: tuple[int, int],
    cells: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    *,
    margin: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = _bbox(cells)
    pattern_h = r1 - r0 + 1
    pattern_w = c1 - c0 + 1
    candidates = [(dr - r0, dc - c0) for dr in range(h - pattern_h + 1) for dc in range(w - pattern_w + 1)]
    random.shuffle(candidates)

    for dr, dc in candidates:
        moved = _translate(cells, dr, dc)
        box = _bbox(moved)
        blocked = _bbox_cells(box, shape, margin=margin)
        if moved & reserved:
            continue
        if blocked & reserved:
            continue
        return moved, blocked, dr, dc
    return None


def _sample_counts() -> tuple[list[int], list[int]]:
    n_colors = random.choice([4, 5])
    if n_colors == 5:
        line_counts = [4, 3, 2]
    else:
        line_counts = sorted(random.sample([2, 3, 4], 2), reverse=True)

    while True:
        first = random.randint(8, 10)
        second = random.randint(6, 9)
        if first <= second:
            continue
        if second <= max(line_counts):
            continue
        total = first + second + sum(line_counts)
        if MIN_FOREGROUND <= total <= MAX_FOREGROUND:
            return [first, second], line_counts


def _semantic_check(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert not np.any(g == ORANGE)

    ranked_colors, ranked_counts = _frequency_ranking(g)
    assert 4 <= len(ranked_colors) <= 5
    assert MIN_FOREGROUND <= int(np.count_nonzero(g)) <= MAX_FOREGROUND
    assert all(6 <= count <= 10 for count in ranked_counts[:2])
    assert all(2 <= count <= 4 for count in ranked_counts[2:])

    structure_colors = ranked_colors[:2]
    line_colors = ranked_colors[2:]
    expected = main(g)

    recolor_only = g.copy()
    recolor_only[np.isin(recolor_only, line_colors)] = ORANGE
    assert not np.array_equal(recolor_only, expected), "line drawing must matter"

    connect_all = _apply_rule(g, ranked_colors, line_colors, strict=False)
    assert not np.array_equal(connect_all, expected), "frequency ranking must matter"

    saw_row = False
    saw_col = False
    new_cells: set[tuple[int, int]] = set()
    for color in line_colors:
        fill_cells, row_used, col_used = _fill_positions_for_color(g, color)
        saw_row |= row_used
        saw_col |= col_used
        assert fill_cells, "each line color must create new connector cells"
        new_cells |= fill_cells

    assert saw_row and saw_col, "examples must exercise both row and column filling"
    assert new_cells, "expected new orange cells"
    assert int(np.sum(expected == ORANGE)) > sum(ranked_counts[2:]), "output must add orange cells"

    tempting_structure = False
    for color in structure_colors:
        fill_cells, _, _ = _fill_positions_for_color(g, color, strict=False)
        if fill_cells:
            tempting_structure = True
            break
    assert tempting_structure, "at least one structure color should tempt the connect-all shortcut"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    w = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    structure_counts, line_counts = _sample_counts()

    colors = random.sample(INPUT_COLORS, 2 + len(line_counts))
    structure_colors = colors[:2]
    line_colors = colors[2:]

    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()

    structure_orientations = random.choice(
        [["h", "v"], ["v", "h"], ["h", "h"], ["v", "v"]]
    )
    for color, count, orientation in zip(structure_colors, structure_counts, structure_orientations):
        pattern = _structure_pattern(count, orientation)
        placed = _place_pattern((h, w), pattern, reserved, margin=0)
        if placed is None:
            return None
        cells, blocked, _, _ = placed
        for r, c in cells:
            grid[r, c] = color
        reserved |= blocked

    line_modes = _choose_line_modes(line_counts)
    for color, count, mode in zip(line_colors, line_counts, line_modes):
        for _ in range(40):
            pattern = _line_pattern(count, mode)
            footprint = _pattern_footprint(pattern)
            placed = _place_pattern((h, w), footprint, reserved, margin=1)
            if placed is None:
                continue
            _, blocked, dr, dc = placed
            moved_pattern = _translate(pattern, dr, dc)
            for r, c in moved_pattern:
                if grid[r, c] != BG:
                    break
            else:
                for r, c in moved_pattern:
                    grid[r, c] = color
                reserved |= blocked
                break
        else:
            return None

    try:
        _semantic_check(grid)
    except Exception:
        return None
    return grid


def _render_relative_pattern(cells: set[tuple[int, int]], count: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        out[r - r0, c - c0] = 1
    assert int(np.sum(out == 1)) == count
    return out


def _pattern_footprint(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pattern_grid = _render_relative_pattern(cells, len(cells))
    fill_cells, _, _ = _fill_positions_for_color(pattern_grid, 1, strict=True)
    return cells | fill_cells


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
