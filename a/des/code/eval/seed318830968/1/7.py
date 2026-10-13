import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
MAROON = 9

NOTE_COLOR_POOL = (1, 2, 4, 6, 7, 8)
TRANSITION_REQUIREMENTS = (
    frozenset({"up", "down"}),
    frozenset({"up", "flat"}),
    frozenset({"down", "flat"}),
    frozenset({"up", "down", "flat"}),
)

MAX_GRID_HEIGHT = 30
MAX_GRID_WIDTH = 30
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _find_separator_col(grid: np.ndarray) -> int:
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(grey_cols) == 1, "expected exactly one full-height grey separator column"
    return int(grey_cols[0])


def _parse_legend(grid: np.ndarray, separator_col: int) -> dict[int, int]:
    row = np.asarray(grid[-1], dtype=int)
    width = row.shape[0]
    mapping: dict[int, int] = {}
    c = 0
    while c < width:
        if c == separator_col:
            c += 1
            continue
        color = int(row[c])
        if color in (BLACK, GREY, MAROON):
            c += 1
            continue

        j = c + 1
        beats = 0
        while j < width and j != separator_col and int(row[j]) == MAROON:
            beats += 1
            j += 1

        assert 1 <= beats <= 4, "legend entry must be followed by 1-4 maroon pixels"
        assert color not in mapping, "legend color must be unique"
        mapping[color] = beats
        c = j

    assert mapping, "legend must be non-empty"
    return mapping


def _extract_notes(grid: np.ndarray, separator_col: int, duration_by_color: dict[int, int]) -> list[tuple[int, int]]:
    notes: list[tuple[int, int]] = []
    for c in range(separator_col):
        column = np.asarray(grid[:-1, c], dtype=int)
        rows = np.flatnonzero((column != BLACK) & (column != GREY))
        if rows.size == 0:
            continue
        assert rows.size == 1, "each score column must contain exactly one note pixel"
        row = int(rows[0])
        color = int(column[row])
        assert color in duration_by_color, "note color missing from legend"
        notes.append((row, duration_by_color[color]))

    assert notes, "score must contain at least one note"
    return notes


def _draw_waveform(out: np.ndarray, separator_col: int, notes: list[tuple[int, int]]) -> None:
    cur_col = separator_col + 1
    height, width = out.shape
    for index, (row, beats) in enumerate(notes):
        assert 0 <= row < height - 1
        assert beats >= 1

        end_col = cur_col + beats - 1
        assert end_col < width, "waveform does not fit in the drawing area"
        out[row, cur_col : end_col + 1] = GREEN

        if index + 1 == len(notes):
            continue

        connector_col = end_col + 1
        assert connector_col < width, "connector does not fit in the drawing area"
        next_row = notes[index + 1][0]
        r0, r1 = sorted((row, next_row))
        out[r0 : r1 + 1, connector_col] = GREEN
        cur_col = connector_col + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the bottom-row color legend, read the note columns on the left, and
    draw the corresponding green step waveform in the right-hand region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "grid is too small"

    separator_col = _find_separator_col(g)
    duration_by_color = _parse_legend(g, separator_col)
    notes = _extract_notes(g, separator_col, duration_by_color)

    out = g.copy()
    _draw_waveform(out, separator_col, notes)
    return out


def _transition_tags(rows: list[int]) -> set[str]:
    tags: set[str] = set()
    for a, b in zip(rows, rows[1:]):
        if b < a:
            tags.add("up")
        elif b > a:
            tags.add("down")
        else:
            tags.add("flat")
    return tags


def _sample_note_rows(score_height: int, note_count: int) -> list[int] | None:
    required = TRANSITION_REQUIREMENTS[int(_RNG.integers(len(TRANSITION_REQUIREMENTS)))]
    for _ in range(200):
        rows = _RNG.integers(0, score_height, size=note_count).tolist()
        if required.issubset(_transition_tags(rows)):
            return [int(r) for r in rows]
    return None


def _sample_note_colors(note_colors: np.ndarray, note_count: int) -> list[int]:
    seq = note_colors.tolist()
    while len(seq) < note_count:
        seq.append(int(_RNG.choice(note_colors)))
    _RNG.shuffle(seq)
    return [int(c) for c in seq]


def _sample_note_columns(note_count: int) -> tuple[list[int], int]:
    lead_gap = int(_RNG.integers(0, 2))
    gaps = _RNG.integers(0, 2, size=max(0, note_count - 1))
    cols: list[int] = []
    cur = lead_gap
    for i in range(note_count):
        cols.append(cur)
        cur += 1
        if i < note_count - 1:
            cur += int(gaps[i])
    return cols, cur


def _legend_width(durations: list[int], gaps: list[int]) -> int:
    return sum(1 + d for d in durations) + sum(gaps)


def _build_candidate() -> np.ndarray | None:
    height = int(_RNG.integers(8, 13))
    score_height = height - 1
    note_count = int(_RNG.integers(4, 6))
    color_count = int(_RNG.integers(2, min(4, note_count) + 1))

    note_colors = _RNG.choice(np.array(NOTE_COLOR_POOL, dtype=int), size=color_count, replace=False)
    duration_values = _RNG.choice(np.array([1, 2, 3, 4], dtype=int), size=color_count, replace=False)
    duration_by_color = {int(color): int(duration) for color, duration in zip(note_colors, duration_values)}

    note_rows = _sample_note_rows(score_height, note_count)
    if note_rows is None:
        return None

    note_color_seq = _sample_note_colors(note_colors, note_count)
    note_durations = [duration_by_color[color] for color in note_color_seq]
    note_cols, left_width = _sample_note_columns(note_count)

    legend_colors = note_colors.tolist()
    _RNG.shuffle(legend_colors)
    legend_gaps = [int(_RNG.integers(1, 3)) for _ in range(max(0, color_count - 1))]
    legend_body_width = _legend_width([duration_by_color[color] for color in legend_colors], legend_gaps)

    draw_width = sum(note_durations) + (note_count - 1)
    right_width = max(draw_width, legend_body_width) + int(_RNG.integers(0, 3))
    width = left_width + 1 + right_width
    if height > MAX_GRID_HEIGHT or width > MAX_GRID_WIDTH:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    separator_col = left_width
    grid[:, separator_col] = GREY

    for row, col, color in zip(note_rows, note_cols, note_color_seq):
        grid[row, col] = int(color)

    legend_offset = int(_RNG.integers(0, right_width - legend_body_width + 1))
    cur = separator_col + 1 + legend_offset
    for i, color in enumerate(legend_colors):
        beats = duration_by_color[int(color)]
        grid[-1, cur] = int(color)
        grid[-1, cur + 1 : cur + 1 + beats] = MAROON
        cur += 1 + beats
        if i < len(legend_gaps):
            cur += legend_gaps[i]

    try:
        parsed_separator = _find_separator_col(grid)
        parsed_legend = _parse_legend(grid, parsed_separator)
        parsed_notes = _extract_notes(grid, parsed_separator, parsed_legend)
        expected_notes = [(int(r), int(duration_by_color[c])) for r, c in zip(note_rows, note_color_seq)]
        if parsed_separator != separator_col or parsed_notes != expected_notes:
            return None

        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out[:, : separator_col + 1], grid[:, : separator_col + 1]):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None
    if np.any(grid[:-1, separator_col + 1 :] != BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
