from __future__ import annotations

import random

import numpy as np


GEN_BG = 0
GEN_STAFF = 5
GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _nearest_staff_row(row: int, staff_rows: list[int]) -> int:
    return min(staff_rows, key=lambda staff_row: (abs(int(row) - int(staff_row)), int(staff_row)))


def _find_staff(grid: np.ndarray) -> tuple[int, int, list[int]]:
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = grid.shape
    assert width >= 2, "staff rows need at least two columns"

    bg = _dominant_color(grid)
    candidates: list[tuple[int, int, int, list[int]]] = []

    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue

        rows = [r for r in range(height) if bool(np.all(grid[r, 1:] == color))]
        if not 3 <= len(rows) <= 5:
            continue

        diffs = np.diff(rows)
        if diffs.size == 0:
            continue
        if len(set(map(int, diffs.tolist()))) != 1 or int(diffs[0]) <= 0:
            continue

        key_colors = [int(grid[r, 0]) for r in rows]
        if any(key_color == color for key_color in key_colors):
            continue
        if len(set(key_colors)) != len(key_colors):
            continue

        candidates.append((len(rows), int(diffs[0]), color, [int(r) for r in rows]))

    assert candidates, "expected an equally spaced set of staff lines"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _count, _spacing, staff_color, staff_rows = candidates[-1]
    return int(bg), int(staff_color), staff_rows


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg, staff_color, staff_rows = _find_staff(g)
    key_by_row = {int(row): int(g[row, 0]) for row in staff_rows}
    staff_set = set(staff_rows)

    out = np.full((height, width), int(bg), dtype=int)
    for row in staff_rows:
        out[row, :] = int(staff_color)
        out[row, 0] = int(key_by_row[row])

    for r in range(height):
        for c in range(width):
            value = int(g[r, c])
            if value == bg:
                continue
            if c == 0 and r in staff_set:
                continue
            if c > 0 and r in staff_set and value == staff_color:
                continue

            target_row = _nearest_staff_row(int(r), staff_rows)
            out[int(target_row), int(c)] = int(key_by_row[int(target_row)])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Snap scattered note pixels onto the nearest staff line and recolor them using the line's key color.
    """
    return _transform(np.asarray(grid, dtype=int))


def _candidate_rows_by_target(height: int, staff_rows: list[int]) -> dict[int, list[int]]:
    staff_set = set(staff_rows)
    candidates: dict[int, list[int]] = {}
    for target_row in staff_rows:
        rows = [
            int(row)
            for row in range(height)
            if row not in staff_set
            and abs(int(row) - int(target_row)) <= 2
            and _nearest_staff_row(int(row), staff_rows) == int(target_row)
        ]
        candidates[int(target_row)] = rows
    return candidates


def _expected_output(shape: tuple[int, int], staff_rows: list[int], key_colors: list[int], notes: list[tuple[int, int]]) -> np.ndarray:
    height, width = shape
    out = np.full((height, width), GEN_BG, dtype=int)
    key_by_row = {int(row): int(color) for row, color in zip(staff_rows, key_colors)}

    for row in staff_rows:
        out[int(row), :] = GEN_STAFF
        out[int(row), 0] = int(key_by_row[int(row)])

    for row, col in notes:
        target_row = _nearest_staff_row(int(row), staff_rows)
        out[int(target_row), int(col)] = int(key_by_row[int(target_row)])

    return out


def _gen_attempt() -> np.ndarray | None:
    want_tie = random.random() < 0.5
    want_merge = random.random() < 0.5

    spacing_choices = (2, 4) if want_tie else (3, 5)
    spacing = random.choice(spacing_choices)
    line_count = random.randint(3, 5)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    width = random.randint(12, 24)

    staff_rows = [int(top_margin + i * spacing) for i in range(line_count)]
    height = int(staff_rows[-1] + bottom_margin + 1)
    if height < 7:
        return None

    key_colors = random.sample(GEN_COLOR_POOL, line_count)
    grid = np.full((height, width), GEN_BG, dtype=int)
    for row, key_color in zip(staff_rows, key_colors):
        grid[int(row), :] = GEN_STAFF
        grid[int(row), 0] = int(key_color)

    rows_by_target = _candidate_rows_by_target(height, staff_rows)
    if not any(rows_by_target.values()):
        return None

    notes: list[tuple[int, int]] = []
    used_input: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    note_colors = [int(color) for color in range(10) if color not in (GEN_BG, GEN_STAFF)]

    def place_note(src_row: int, col: int, *, allow_output_overlap: bool = False) -> bool:
        target_row = _nearest_staff_row(int(src_row), staff_rows)
        input_slot = (int(src_row), int(col))
        output_slot = (int(target_row), int(col))
        if input_slot in used_input:
            return False
        if output_slot in used_output and not allow_output_overlap:
            return False

        grid[int(src_row), int(col)] = int(random.choice(note_colors))
        used_input.add(input_slot)
        used_output.add(output_slot)
        notes.append(input_slot)
        return True

    if want_tie:
        tie_options = [(int(staff_rows[i] + spacing // 2), int(staff_rows[i])) for i in range(line_count - 1)]
        random.shuffle(tie_options)
        tie_placed = False
        for src_row, _target_row in tie_options:
            if src_row in staff_rows:
                continue
            for _ in range(20):
                col = random.randint(1, width - 1)
                if place_note(int(src_row), int(col)):
                    tie_placed = True
                    break
            if tie_placed:
                break
        if not tie_placed:
            return None

    if want_merge:
        merge_targets = [row for row, rows in rows_by_target.items() if len(rows) >= 2]
        random.shuffle(merge_targets)
        merge_placed = False
        for target_row in merge_targets:
            source_rows = rows_by_target[int(target_row)][:]
            random.shuffle(source_rows)
            src_a, src_b = int(source_rows[0]), int(source_rows[1])
            for _ in range(20):
                col = random.randint(1, width - 1)
                output_slot = (int(target_row), int(col))
                if output_slot in used_output:
                    continue
                if (src_a, col) in used_input or (src_b, col) in used_input:
                    continue
                if place_note(int(src_a), int(col)) and place_note(int(src_b), int(col), allow_output_overlap=True):
                    merge_placed = True
                    break
            if merge_placed:
                break
        if not merge_placed:
            return None

    target_note_count = random.randint(5, 10)
    attempts = 0
    populated_targets = [row for row, rows in rows_by_target.items() if rows]
    while len(notes) < target_note_count and attempts < 500:
        target_row = int(random.choice(populated_targets))
        src_row = int(random.choice(rows_by_target[target_row]))
        col = random.randint(1, width - 1)
        if place_note(src_row, col):
            attempts = 0
        else:
            attempts += 1

    if len(notes) < target_note_count:
        return None

    expected = _expected_output(grid.shape, staff_rows, key_colors, notes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
