from collections import Counter
import random

import numpy as np


def _unique_most_common_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).ravel()))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _separator_spec(grid: np.ndarray, bg: int) -> tuple[list[int], int, int, int]:
    height, width = map(int, grid.shape)
    assert width > 0

    rows: list[int] = []
    colors: list[int] = []
    for r in range(height):
        row = grid[r]
        first = int(row[0])
        if first != bg and np.all(row == first):
            rows.append(r)
            colors.append(first)

    assert rows, "expected at least one full-width separator row"
    assert len(set(colors)) == 1, "all separator rows must share one color"

    if len(rows) == 1:
        band_height = int(rows[0])
    else:
        step = int(rows[1] - rows[0])
        assert step >= 2
        assert all(int(b - a) == step for a, b in zip(rows, rows[1:]))
        band_height = step - 1

    band_count = len(rows) + 1
    assert band_height >= 1
    assert height == band_count * band_height + (band_count - 1)

    expected_rows = [band_height + i * (band_height + 1) for i in range(band_count - 1)]
    assert rows == expected_rows, "separator rows must split equal-height bands"
    return rows, int(colors[0]), int(band_height), int(band_count)


def _scan_note_colors(
    grid: np.ndarray, bg: int, sep_rows: list[int], sep_color: int
) -> list[int]:
    sep_set = set(sep_rows)
    order: list[int] = []
    seen: set[int] = set()

    for r in range(grid.shape[0]):
        if r in sep_set:
            continue
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (bg, sep_color):
                continue
            if color not in seen:
                seen.add(color)
                order.append(color)

    assert order, "expected at least one note color"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Merge multiple separated staff rows onto one shared time axis by color channel."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _unique_most_common_color(g)
    sep_rows, sep_color, band_height, band_count = _separator_spec(g, bg)
    note_colors = _scan_note_colors(g, bg, sep_rows, sep_color)
    assert len(note_colors) <= 4, "expected at most four note colors"

    color_to_row = {color: row for row, color in enumerate(note_colors)}
    out = np.full((len(note_colors), g.shape[1]), bg, dtype=int)

    for band in range(band_count):
        r0 = band * (band_height + 1)
        band_view = g[r0 : r0 + band_height]
        assert band_view.shape == (band_height, g.shape[1])
        assert not np.any(band_view == sep_color), "separator color should appear only on separator rows"

        for c in range(g.shape[1]):
            note_rows = np.flatnonzero(band_view[:, c] != bg)
            assert len(note_rows) <= 1, "each band/column may contain at most one note"
            if len(note_rows) == 0:
                continue
            color = int(band_view[int(note_rows[0]), c])
            assert color in color_to_row
            out[color_to_row[color], c] = color

    return out


# --------------- generator ---------------

ALL_COLORS = tuple(range(10))
BAND_COUNT_RANGE = (2, 5)
BAND_HEIGHT_RANGE = (2, 4)
WIDTH_RANGE = (8, 18)
NOTE_COLOR_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _gen_separator_rows(band_count: int, band_height: int) -> list[int]:
    return [band_height + i * (band_height + 1) for i in range(band_count - 1)]


def _gen_choose_palette() -> tuple[int, int, list[int]]:
    bg, sep_color = random.sample(ALL_COLORS, 2)
    note_count = random.randint(*NOTE_COLOR_COUNT_RANGE)
    note_colors = random.sample([c for c in ALL_COLORS if c not in (bg, sep_color)], note_count)
    return int(bg), int(sep_color), [int(color) for color in note_colors]


def _gen_expected_output(bg: int, width: int, notes: list[tuple[int, int, int]]) -> np.ndarray:
    scan_order = sorted(notes, key=lambda note: (note[0], note[1]))
    color_order: list[int] = []
    seen: set[int] = set()
    for _row, _col, color in scan_order:
        if color not in seen:
            seen.add(color)
            color_order.append(color)

    out = np.full((len(color_order), width), bg, dtype=int)
    color_to_row = {color: row for row, color in enumerate(color_order)}
    for _row, col, color in notes:
        out[color_to_row[color], col] = color
    return out


def _gen_build_assignments(
    band_count: int, width: int, note_colors: list[int]
) -> dict[tuple[int, int], int]:
    assignments: dict[tuple[int, int], int] = {}
    min_feature_cols = max(3, len(note_colors))
    feature_count = random.randint(min_feature_cols, width)
    feature_cols = random.sample(range(width), feature_count)

    layered_col = random.choice(feature_cols)
    layered_bands = random.sample(range(band_count), random.randint(2, band_count))
    layered_colors = [random.choice(note_colors) for _ in layered_bands]
    if len(note_colors) >= 2 and len(set(layered_colors)) == 1:
        layered_colors[-1] = random.choice([c for c in note_colors if c != layered_colors[0]])
    for band, color in zip(layered_bands, layered_colors):
        assignments[(band, layered_col)] = color

    if band_count >= 2 and len(feature_cols) >= 2 and random.random() < 0.5:
        duplicate_col = random.choice([c for c in feature_cols if c != layered_col])
        duplicate_bands = random.sample(range(band_count), random.randint(2, band_count))
        duplicate_color = random.choice(note_colors)
        for band in duplicate_bands:
            assignments[(band, duplicate_col)] = duplicate_color

    for col in feature_cols:
        target_occupancy = random.randint(1, min(3, band_count))
        occupied_bands = [band for band in range(band_count) if (band, col) in assignments]
        remaining_bands = [band for band in range(band_count) if (band, col) not in assignments]
        needed = max(0, target_occupancy - len(occupied_bands))
        if len(occupied_bands) == 0:
            needed = max(1, needed)
        needed = min(needed, len(remaining_bands))
        for band in random.sample(remaining_bands, needed):
            assignments[(band, col)] = random.choice(note_colors)

    for band in range(band_count):
        if any(slot_band == band for slot_band, _col in assignments):
            continue
        free_cols = [col for col in range(width) if (band, col) not in assignments]
        assignments[(band, random.choice(free_cols))] = random.choice(note_colors)

    used_colors = set(assignments.values())
    for color in note_colors:
        if color in used_colors:
            continue
        free_slots = [
            (band, col)
            for band in range(band_count)
            for col in range(width)
            if (band, col) not in assignments
        ]
        assert free_slots, "expected a free slot for every unused color"
        band, col = random.choice(free_slots)
        assignments[(band, col)] = color
        used_colors.add(color)

    return assignments


def _gen_assign_rows(
    assignments: dict[tuple[int, int], int], band_count: int, band_height: int
) -> dict[tuple[int, int], int] | None:
    row_offsets: dict[tuple[int, int], int] = {}
    for band in range(band_count):
        band_items = sorted(
            ((col, color) for (slot_band, col), color in assignments.items() if slot_band == band),
            key=lambda item: item[0],
        )
        chosen_rows: dict[int, int] = {}
        band_colors = {col: color for col, color in band_items}
        for col, color in band_items:
            forbidden: set[int] = set()
            if col - 1 in chosen_rows and band_colors[col - 1] == color:
                forbidden.add(chosen_rows[col - 1])
            choices = [row for row in range(band_height) if row not in forbidden]
            if not choices:
                return None
            row = random.choice(choices)
            chosen_rows[col] = row
            row_offsets[(band, col)] = row
    return row_offsets


def _gen_render_grid(
    bg: int,
    sep_color: int,
    band_count: int,
    band_height: int,
    width: int,
    assignments: dict[tuple[int, int], int],
    row_offsets: dict[tuple[int, int], int],
) -> tuple[np.ndarray, list[int], list[tuple[int, int, int]]]:
    height = band_count * band_height + (band_count - 1)
    grid = np.full((height, width), bg, dtype=int)
    sep_rows = _gen_separator_rows(band_count, band_height)
    for r in sep_rows:
        grid[r, :] = sep_color

    notes: list[tuple[int, int, int]] = []
    for (band, col), color in assignments.items():
        abs_row = band * (band_height + 1) + row_offsets[(band, col)]
        grid[abs_row, col] = color
        notes.append((abs_row, col, color))
    return grid, sep_rows, notes


def _gen_solid_non_bg_rows(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    colors: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        first = int(row[0])
        if first != bg and np.all(row == first):
            rows.append(r)
            colors.append(first)
    return rows, colors


def _gen_notes_are_clear(grid: np.ndarray, bg: int, sep_rows: list[int], sep_color: int) -> bool:
    sep_set = set(sep_rows)
    height, width = map(int, grid.shape)
    for r in range(height):
        if r in sep_set:
            continue
        for c in range(width):
            color = int(grid[r, c])
            if color in (bg, sep_color):
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and nr not in sep_set:
                    if int(grid[nr, nc]) == color:
                        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    band_count = random.randint(*BAND_COUNT_RANGE)
    band_height = random.randint(*BAND_HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    bg, sep_color, note_colors = _gen_choose_palette()

    assignments = _gen_build_assignments(band_count, width, note_colors)
    row_offsets = _gen_assign_rows(assignments, band_count, band_height)
    if row_offsets is None:
        return None

    grid, sep_rows, notes = _gen_render_grid(
        bg, sep_color, band_count, band_height, width, assignments, row_offsets
    )
    if not _gen_notes_are_clear(grid, bg, sep_rows, sep_color):
        return None

    solid_rows, solid_colors = _gen_solid_non_bg_rows(grid, bg)
    if solid_rows != sep_rows or any(color != sep_color for color in solid_colors):
        return None

    try:
        if _unique_most_common_color(grid) != bg:
            return None
    except AssertionError:
        return None

    expected = _gen_expected_output(bg, width, notes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
