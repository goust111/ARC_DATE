import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 600


def _active_columns(width: int, seed_col: int) -> list[int]:
    return list(range(seed_col, width, 2))


def _parse_input(grid: np.ndarray) -> tuple[int, int, dict[int, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    top_cols = np.flatnonzero(g[0] != BLACK)
    assert len(top_cols) == 1, "expected exactly one top-row seed pixel"
    seed_col = int(top_cols[0])
    seed_color = int(g[0, seed_col])
    assert seed_color not in (BLACK, GREY), "seed color must be a non-grey foreground color"

    segments: dict[int, tuple[int, int, int]] = {}
    for r in range(1, height):
        cols = np.flatnonzero(g[r] != BLACK)
        if len(cols) == 0:
            continue
        assert np.all(np.diff(cols) == 1), "each foreground row below the seed must be one contiguous segment"
        c0, c1 = int(cols[0]), int(cols[-1])
        color = int(g[r, c0])
        assert np.all(g[r, c0 : c1 + 1] == color), "segment rows must be monochromatic"
        assert c1 - c0 + 1 < width, "segments may not fill the whole row"
        segments[r] = (c0, c1, color)

    rows = sorted(segments)
    assert rows, "expected at least one horizontal segment"
    return seed_col, seed_color, segments


def _line_histories(
    shape: tuple[int, int],
    seed_col: int,
    seed_color: int,
    segments: dict[int, tuple[int, int, int]],
) -> dict[int, dict[str, object]]:
    _height, width = shape
    segment_rows = sorted(segments)
    histories: dict[int, dict[str, object]] = {}
    for col in _active_columns(width, seed_col):
        current = seed_color
        dyes: list[tuple[int, int, int]] = []
        terminator = None
        for row in segment_rows:
            c0, c1, color = segments[row]
            if c0 <= col <= c1:
                if color == GREY:
                    terminator = row
                    break
                dyes.append((row, current, color))
                current = color
        histories[col] = {"dyes": dyes, "terminator": terminator, "final_color": current}
    return histories


def _draw_verticals_and_collect_splashes(
    grid: np.ndarray,
    seed_col: int,
    seed_color: int,
    segments: dict[int, tuple[int, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()
    splashes: list[tuple[int, int, int]] = []

    for col in _active_columns(width, seed_col):
        current = seed_color
        out[0, col] = current
        for row in range(1, height):
            segment = segments.get(row)
            if segment is not None and segment[0] <= col <= segment[1]:
                if segment[2] == GREY:
                    splashes.append((row - 1, col, current))
                    break
                current = segment[2]
                continue
            out[row, col] = current

    return out, splashes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow periodic falling columns from the top seed, recolor them through dye bars, and splash them off grey terminators.

    Input structure:
    - The background is black.
    - Row 0 contains exactly one colored seed pixel.
    - Lower rows may contain at most one contiguous horizontal segment each; grey segments are terminators and every other segment is a dye bar.

    Rule:
    - Starting at the seed column, generate vertical columns in every second column to the right, beginning from row 0 with the seed color.
    - When a falling column passes through a dye bar, the bar itself stays unchanged and the column continues below it using that bar's color.
    - When a falling column reaches a grey terminator bar, it stops one cell above the bar and emits two diagonal rays upward-left and upward-right in its current color until they leave the grid; columns that never hit a terminator continue to the bottom.

    Key insight:
    - The output columns are absent from the input, so the seed fixes the periodic columns first, and each column's visible colors are determined by the ordered sequence of dye bars it crosses before any terminator.
    """
    g = np.asarray(grid, dtype=int)
    seed_col, seed_color, segments = _parse_input(g)
    out, splashes = _draw_verticals_and_collect_splashes(g, seed_col, seed_color, segments)
    height, width = g.shape

    for end_row, end_col, color in splashes:
        for dc in (-1, 1):
            row = end_row - 1
            col = end_col + dc
            while 0 <= row < height and 0 <= col < width:
                if g[row, col] == BLACK:
                    out[row, col] = color
                row -= 1
                col += dc
    return out


def _single_line_span(active_cols: list[int], line_idx: int, width: int) -> tuple[int, int]:
    col = active_cols[line_idx]
    choices = []
    if col > 0:
        choices.append((col - 1, col))
    if col < width - 1:
        choices.append((col, col + 1))
    if 0 < col < width - 1:
        choices.append((col - 1, col + 1))
    return random.choice(choices)


def _range_span(active_cols: list[int], left_idx: int, right_idx: int, width: int) -> tuple[int, int]:
    left = max(0, active_cols[left_idx] - random.randint(0, 1))
    right = min(width - 1, active_cols[right_idx] + random.randint(0, 1))
    if right - left + 1 < 2:
        if right < width - 1:
            right += 1
        else:
            left -= 1
    if left == 0 and right == width - 1:
        right -= 1
    return left, right


def _paint_segment(grid: np.ndarray, row: int, span: tuple[int, int], color: int) -> None:
    c0, c1 = span
    grid[row, c0 : c1 + 1] = color


def _splash_cells(height: int, width: int, end_row: int, end_col: int) -> set[tuple[int, int]]:
    cells = set()
    for dc in (-1, 1):
        row = end_row - 1
        col = end_col + dc
        while 0 <= row < height and 0 <= col < width:
            cells.add((row, col))
            row -= 1
            col += dc
    return cells


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        seed_col, seed_color, segments = _parse_input(grid)
        height, width = grid.shape
        active_cols = _active_columns(width, seed_col)
        histories = _line_histories(grid.shape, seed_col, seed_color, segments)

        if len(active_cols) < 4:
            return False

        rows = sorted(segments)
        if not all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1)):
            return False

        dye_rows = {row for row, (_c0, _c1, color) in segments.items() if color != GREY}
        term_rows = {row for row, (_c0, _c1, color) in segments.items() if color == GREY}
        if len(dye_rows) < 4 or len(term_rows) < 2:
            return False

        dye_used = {row: False for row in dye_rows}
        term_used = {row: False for row in term_rows}
        for info in histories.values():
            for row, prev_color, new_color in info["dyes"]:
                if prev_color != new_color:
                    dye_used[row] = True
            if info["terminator"] is not None:
                term_used[int(info["terminator"])] = True
        if not all(dye_used.values()) or not all(term_used.values()):
            return False

        terminated = [info for info in histories.values() if info["terminator"] is not None]
        open_lines = [info for info in histories.values() if info["terminator"] is None]
        if not terminated or not open_lines:
            return False
        if not any(len(info["dyes"]) >= 2 for info in histories.values()):
            return False
        if not any(info["terminator"] is not None and len(info["dyes"]) == 0 for info in histories.values()):
            return False
        if not any(info["terminator"] is not None and len(info["dyes"]) >= 1 for info in histories.values()):
            return False
        if not any(info["terminator"] is None and len(info["dyes"]) >= 1 for info in histories.values()):
            return False
        if len({int(info["final_color"]) for info in terminated}) < 2:
            return False

        out, splashes = _draw_verticals_and_collect_splashes(grid, seed_col, seed_color, segments)
        if np.array_equal(out, grid):
            return False
        splash_only_counts: dict[tuple[int, int], int] = defaultdict(int)
        splash_vertical_overlaps = 0
        for end_row, end_col, _color in splashes:
            for cell in _splash_cells(height, width, end_row, end_col):
                r, c = cell
                if grid[r, c] != BLACK:
                    return False
                splash_only_counts[cell] += 1
                if out[r, c] != BLACK:
                    splash_vertical_overlaps += 1
        if any(count > 1 for count in splash_only_counts.values()):
            return False
        if splash_vertical_overlaps > 5:
            return False

        final = main(grid)
        if not np.array_equal(final[grid != BLACK], grid[grid != BLACK]):
            return False
        if np.array_equal(final, grid):
            return False
        if np.count_nonzero(final != BLACK) <= np.count_nonzero(grid != BLACK):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    num_lines = random.randint(4, 6)
    seed_col = random.randint(0, 3)
    right_margin = random.randint(2, 5)
    width = seed_col + 2 * (num_lines - 1) + 1 + right_margin
    if width > 30:
        return None

    start_row = random.randint(1, 3)
    rows = [start_row]
    for _ in range(5):
        rows.append(rows[-1] + random.randint(2, 4))
    bottom_margin = random.randint(2, 5)
    height = rows[-1] + bottom_margin + 1
    if height > 30:
        return None

    colors = random.sample(PALETTE, 5)
    seed_color, deep_color_a, shared_color, dyed_color, deep_color_b = colors

    active_cols = _active_columns(width, seed_col)
    plain_idx = 0
    dyed_idx = num_lines - 1
    deep_idx = random.randint(1, num_lines - 2)

    grid = np.zeros((height, width), dtype=int)
    grid[0, seed_col] = seed_color

    spans = {
        rows[0]: (_single_line_span(active_cols, deep_idx, width), deep_color_a),
        rows[1]: (_single_line_span(active_cols, plain_idx, width), GREY),
        rows[2]: (_range_span(active_cols, min(deep_idx, dyed_idx), max(deep_idx, dyed_idx), width), shared_color),
        rows[3]: (_single_line_span(active_cols, dyed_idx, width), dyed_color),
        rows[4]: (_single_line_span(active_cols, deep_idx, width), deep_color_b),
        rows[5]: (_single_line_span(active_cols, dyed_idx, width), GREY),
    }

    for row, (span, color) in spans.items():
        _paint_segment(grid, row, span, color)

    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
