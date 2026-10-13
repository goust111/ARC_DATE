import numpy as np

from arc_utils import get_bg


OUTPUT_SHAPE = (3, 3)


def main(grid: np.ndarray) -> np.ndarray:
    """Pack scattered colored singleton pixels into a 3x3 snake summary ordered by input column.

    Input structure
    A grid has a dominant background color and up to nine non-background
    singleton pixels. The foreground markers occupy distinct columns.

    Rule
    Sort the foreground pixels from left to right by their input column. Write
    them into a 3x3 background canvas in groups of three, reversing every odd
    output row: row 0 left-to-right, row 1 right-to-left, row 2 left-to-right.

    Key insight
    The output is not a spatial crop; it is the column order of the scattered
    markers folded into a boustrophedon 3x3 reading pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker_coords = np.argwhere(g != bg)
    assert len(marker_coords) <= OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
    assert len({int(c) for _, c in marker_coords}) == len(marker_coords), "expected one marker per column"

    marker_coords = marker_coords[np.argsort(marker_coords[:, 1], kind="stable")]
    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    for index, (r, c) in enumerate(marker_coords):
        out_r = index // OUTPUT_SHAPE[1]
        offset = index % OUTPUT_SHAPE[1]
        out_c = offset if out_r % 2 == 0 else OUTPUT_SHAPE[1] - 1 - offset
        out[out_r, out_c] = g[r, c]

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GEN_INPUT_SHAPE = (10, 10)
GEN_BACKGROUND_COLOR = 0
GEN_COLOR_RANGE = (1, 10)

# Marker sampling and retry budget.
GEN_MARKER_COUNT_RANGE = (6, 10)
GEN_ADJACENT_COLUMN_ROW_GAP = 1
GENERATION_RETRIES = 300

# Fixed boustrophedon output order.
GEN_SNAKE_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2))


def _sample_marker_rows(columns: list[int]) -> list[int]:
    """Sample rows, keeping consecutive-column markers separated."""
    rows = []
    prev_col = prev_row = None
    gap = GEN_ADJACENT_COLUMN_ROW_GAP
    for c in columns:
        row_choices = list(range(GEN_INPUT_SHAPE[0]))
        if prev_col is not None and c == prev_col + 1:
            start, stop = max(0, prev_row - gap), min(GEN_INPUT_SHAPE[0], prev_row + gap + 1)
            for forbidden_row in range(start, stop):
                row_choices.remove(forbidden_row)
        r = int(np.random.choice(row_choices))
        rows.append(r)
        prev_col, prev_row = c, r
    return rows


def _gen_attempt() -> np.ndarray | None:
    n_markers = int(np.random.randint(*GEN_MARKER_COUNT_RANGE))

    out = np.full(OUTPUT_SHAPE, GEN_BACKGROUND_COLOR, dtype=int)
    colors = np.random.randint(*GEN_COLOR_RANGE, size=n_markers)
    for color, (r, c) in zip(colors, GEN_SNAKE_POSITIONS):
        out[r, c] = int(color)

    inp = np.full(GEN_INPUT_SHAPE, GEN_BACKGROUND_COLOR, dtype=int)
    columns = sorted(np.random.choice(GEN_INPUT_SHAPE[1], size=n_markers, replace=False).tolist())
    rows = _sample_marker_rows(columns)
    for color, r, c in zip(colors, rows, columns):
        inp[int(r), int(c)] = int(color)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cdecee7f input")
