from __future__ import annotations

import random

import numpy as np


ALL_COLORS = tuple(range(10))
LINE_COUNT = 5
LINE_SPACING = 2
TOP_MARGIN_MIN = 1
SIDE_MARGIN_MIN = 1
LINE_LENGTH_MIN = 8
NOTE_COUNT_MIN = 2
NOTE_COUNT_MAX = 7
GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _staff_signature(grid: np.ndarray, color: int) -> tuple[list[int], int, int] | None:
    coords = np.argwhere(grid == int(color))
    if len(coords) == 0:
        return None

    rows = sorted({int(r) for r, _c in coords})
    if len(rows) < 4:
        return None

    gaps = np.diff(rows)
    if len(gaps) == 0 or np.any(gaps != gaps[0]):
        return None

    cols = [int(c) for _r, c in coords]
    c0, c1 = min(cols), max(cols)
    span = c1 - c0 + 1
    if span < 5:
        return None

    row_counts = [int(np.sum(grid[r, :] == int(color))) for r in rows]
    if min(row_counts) < max(2, span // 3):
        return None

    return rows, c0, c1


def _find_staff_structure(grid: np.ndarray, bg: int) -> tuple[int, list[int], int, int]:
    candidates: list[tuple[tuple[int, int, int], int, list[int], int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        signature = _staff_signature(grid, color)
        if signature is None:
            continue
        rows, c0, c1 = signature
        count = int(np.sum(grid == color))
        score = (count, len(rows), c1 - c0 + 1)
        candidates.append((score, color, rows, c0, c1))

    assert candidates, "expected a horizontal staff color"
    candidates.sort(reverse=True)
    _score, color, rows, c0, c1 = candidates[0]
    return color, rows, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode note positions on a horizontal staff into bottom-grown colored columns.

    Input structure:
    - The background is the dominant color.
    - The upper part contains several equally spaced horizontal staff lines of one color.
    - Single-pixel notes of other colors sit on staff lines or in the spaces between them, while the lower part is blank background.

    Rule:
    - Keep the staff lines and the note pixels in place.
    - For each note, draw a vertical bar in the same column from the bottom edge upward.
    - The bar height is (lowest staff line row - note row + 1), and the bar uses the note's color.

    Key insight:
    - The staff is a height code: a note's vertical offset from the lowest line determines how tall its output column must be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    staff_color, staff_rows, line_c0, line_c1 = _find_staff_structure(g, bg)
    lowest_line = int(staff_rows[-1])
    highest_line = int(staff_rows[0])

    note_coords = np.argwhere((g != bg) & (g != staff_color))
    assert len(note_coords) > 0, "expected at least one note"

    note_cols = [int(c) for _r, c in note_coords]
    assert len(note_cols) == len(set(note_cols)), "note columns must be unique"

    out = g.copy()
    for r, c in note_coords:
        r = int(r)
        c = int(c)
        assert line_c0 <= c <= line_c1, "notes must sit within the staff span"
        assert highest_line <= r <= lowest_line, "notes must stay between the top and bottom staff lines"

        height = lowest_line - r + 1
        assert height >= 1
        target_r0 = g.shape[0] - height
        assert target_r0 > lowest_line, "bottom blank region must fit every decoded bar"
        assert np.all(g[target_r0:, c] == bg), "decoded bars may only occupy blank background"
        out[target_r0:, c] = int(g[r, c])

    return out


def _pick_spaced_columns(start: int, end: int, count: int, rng: random.Random) -> list[int] | None:
    cols = list(range(start, end + 1))
    rng.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if all(abs(col - other) >= 2 for other in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(19, 30)
    top_max = min(6, (height - 1) // 2 - (LINE_COUNT - 1) * LINE_SPACING)
    if top_max < TOP_MARGIN_MIN:
        return None
    top_line = rng.randint(TOP_MARGIN_MIN, top_max)

    line_rows = [top_line + LINE_SPACING * i for i in range(LINE_COUNT)]
    bottom_line = line_rows[-1]
    max_bar_height = bottom_line - line_rows[0] + 1
    if height - (bottom_line + 1) < max_bar_height:
        return None

    width = rng.randint(12, 30)
    max_line_length = width - 2 * SIDE_MARGIN_MIN
    if max_line_length < LINE_LENGTH_MIN:
        return None
    line_length = rng.randint(LINE_LENGTH_MIN, max_line_length)
    left = rng.randint(SIDE_MARGIN_MIN, width - line_length - SIDE_MARGIN_MIN)
    right = left + line_length - 1

    max_notes = min(NOTE_COUNT_MAX, len(ALL_COLORS) - 2, (line_length + 1) // 2)
    if max_notes < NOTE_COUNT_MIN:
        return None
    note_count = rng.randint(NOTE_COUNT_MIN, max_notes)
    note_cols = _pick_spaced_columns(left, right, note_count, rng)
    if note_cols is None:
        return None

    palette = rng.sample(list(ALL_COLORS), note_count + 2)
    bg = int(palette[0])
    staff_color = int(palette[1])
    note_colors = [int(c) for c in palette[2:]]

    grid = np.full((height, width), bg, dtype=int)
    for row in line_rows:
        grid[row, left : right + 1] = staff_color

    staff_positions = list(line_rows)
    space_positions = [row + 1 for row in line_rows[:-1]]
    note_rows: list[int] = [rng.choice(staff_positions), rng.choice(space_positions)]
    all_positions = staff_positions + space_positions
    for _ in range(note_count - 2):
        note_rows.append(rng.choice(all_positions))
    rng.shuffle(note_rows)

    for col, row, color in zip(note_cols, note_rows, note_colors, strict=True):
        grid[row, col] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staff-decoding puzzle")
