import numpy as np

from arc_utils import Color, colors_present


def _boundary_background(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    colors = np.unique(border)
    assert len(colors) == 1
    return int(colors[0])


def _block_distance(i: int, start: int, end: int) -> int:
    if start <= i < end:
        return 0
    if i < start:
        return (start - i + 1) // 2
    return (i - end + 2) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor teal pixels yellow when their 2-pixel-block distance from the pink marker lies inside the marker-centered hourglass.

    Input structure
    A black-bordered grid contains one noisy teal field, one 2x2 pink marker,
    and full black separator lines along exactly one axis.

    Rule
    Remove the full black rows and columns to get compressed coordinates.  From
    the 2x2 pink marker, measure distances in two-pixel blocks along each axis.
    The separator-line axis is the narrow axis of the hourglass; the other axis
    is the opening axis.  A teal pixel becomes yellow exactly when both distances
    are nonzero and the narrow-axis distance is no greater than the opening-axis
    distance.

    Key insight
    The black separator lines identify which direction the hourglass opens, so
    the vertical-stripe and horizontal-stripe examples use the same distance
    predicate after compression.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_background(g)

    marker_colors = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if len(coords) != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2) and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            marker_colors.append(color)
    assert len(marker_colors) == 1
    marker = marker_colors[0]

    source_colors = [color for color in colors_present(g, exclude={bg, marker})]
    assert len(source_colors) == 1
    source = source_colors[0]

    all_bg_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == bg)]
    all_bg_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == bg)]
    internal_bg_rows = [r for r in all_bg_rows if 0 < r < g.shape[0] - 1]
    internal_bg_cols = [c for c in all_bg_cols if 0 < c < g.shape[1] - 1]
    assert (len(internal_bg_rows) > 0) != (len(internal_bg_cols) > 0)

    separator_axis = 0 if internal_bg_rows else 1
    opening_axis = 1 - separator_axis

    bg_row_set = set(all_bg_rows)
    bg_col_set = set(all_bg_cols)
    kept_rows = [r for r in range(g.shape[0]) if r not in bg_row_set]
    kept_cols = [c for c in range(g.shape[1]) if c not in bg_col_set]
    row_to_compressed = {r: i for i, r in enumerate(kept_rows)}
    col_to_compressed = {c: i for i, c in enumerate(kept_cols)}

    marker_coords = np.array([[row_to_compressed[r], col_to_compressed[c]] for r, c in np.argwhere(g == marker)])
    starts = marker_coords.min(axis=0)
    ends = marker_coords.max(axis=0) + 1
    assert tuple(ends - starts) == (2, 2)

    out = g.copy()
    for r, c in np.argwhere(g == source):
        cr = row_to_compressed[int(r)]
        cc = col_to_compressed[int(c)]
        distances = [
            _block_distance(cr, int(starts[0]), int(ends[0])),
            _block_distance(cc, int(starts[1]), int(ends[1])),
        ]
        opening_distance = distances[opening_axis]
        separator_distance = distances[separator_axis]
        if opening_distance > 0 and separator_distance > 0 and separator_distance <= opening_distance:
            out[r, c] = Color.YELLOW

    return out

# --------------- generator (o5x) ---------------

# Colors used by the task transformation.
BACKGROUND_COLOR = Color.BLACK
SOURCE_COLOR = Color.TEAL
MARKER_COLOR = Color.PINK
OUTPUT_COLOR = Color.YELLOW
# Canvas and separator structure.
BORDER_WIDTH = 1
BORDER_CELLS = 2 * BORDER_WIDTH
CONTENT_BLOCK = 2
AXIS_COUNT = 2
MIN_SEPARATOR_COUNT = 4
MAX_SEPARATOR_COUNT_EXCLUSIVE = 8
NON_SEPARATOR_CONTENT_OPTIONS = np.arange(10, 19, CONTENT_BLOCK)
# Marker placement and hole sampling.
MARKER_SIZE = CONTENT_BLOCK
MARKER_STEP = CONTENT_BLOCK
MARKER_EDGE_MARGIN = CONTENT_BLOCK
MIN_HOLE_RATE = 0.05
MAX_HOLE_RATE = 0.12
# Generation retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _content_axis_indices(compressed_size: int, has_separators: bool) -> np.ndarray:
    """Map compressed content coordinates to raw grid coordinates for one axis."""
    compressed = np.arange(compressed_size, dtype=int)
    if has_separators:
        return BORDER_WIDTH + compressed + compressed // CONTENT_BLOCK
    return BORDER_WIDTH + compressed


def _sample_layout():
    """Sample the separator axis, raw shape, compressed shape, and content indices."""
    separator_axis = int(np.random.randint(0, AXIS_COUNT))
    n_seps = int(np.random.randint(MIN_SEPARATOR_COUNT, MAX_SEPARATOR_COUNT_EXCLUSIVE))
    non_sep_content = int(np.random.choice(NON_SEPARATOR_CONTENT_OPTIONS))
    sep_content = CONTENT_BLOCK * (n_seps + 1)

    compressed_shape = [non_sep_content, non_sep_content]
    compressed_shape[separator_axis] = sep_content
    raw_shape = [compressed_shape[0] + BORDER_CELLS, compressed_shape[1] + BORDER_CELLS]
    raw_shape[separator_axis] += n_seps

    content_rows = _content_axis_indices(compressed_shape[0], separator_axis == 0)
    content_cols = _content_axis_indices(compressed_shape[1], separator_axis == 1)
    return separator_axis, tuple(raw_shape), tuple(compressed_shape), content_rows, content_cols


def _place_marker(grid: np.ndarray, content_rows: np.ndarray, content_cols: np.ndarray, compressed_shape: tuple[int, int]):
    comp_height, comp_width = compressed_shape
    marker_row = int(np.random.choice(np.arange(MARKER_EDGE_MARGIN, comp_height - MARKER_EDGE_MARGIN, MARKER_STEP)))
    marker_col = int(np.random.choice(np.arange(MARKER_EDGE_MARGIN, comp_width - MARKER_EDGE_MARGIN, MARKER_STEP)))
    marker_rows = content_rows[marker_row : marker_row + MARKER_SIZE]
    marker_cols = content_cols[marker_col : marker_col + MARKER_SIZE]
    grid[np.ix_(marker_rows, marker_cols)] = MARKER_COLOR
    return marker_row, marker_col


def _remove_holes(grid: np.ndarray, content_rows: np.ndarray, content_cols: np.ndarray) -> bool:
    """Apply random black holes while keeping every content row and column nonempty."""
    row_counts = {int(r): int(np.count_nonzero(grid[int(r), :] != BACKGROUND_COLOR)) for r in content_rows}
    col_counts = {int(c): int(np.count_nonzero(grid[:, int(c)] != BACKGROUND_COLOR)) for c in content_cols}
    hole_candidates = [(int(r), int(c)) for r in content_rows for c in content_cols if grid[int(r), int(c)] == SOURCE_COLOR]
    min_holes = int(np.ceil(MIN_HOLE_RATE * len(hole_candidates)))
    max_holes = int(np.floor(MAX_HOLE_RATE * len(hole_candidates)))
    hole_count = int(np.random.randint(min_holes, max_holes + 1))

    for _ in range(hole_count):
        eligible = [i for i, (r, c) in enumerate(hole_candidates) if row_counts[r] > 1 and col_counts[c] > 1]
        if not eligible:
            return False
        pick = int(np.random.choice(eligible))
        r, c = hole_candidates.pop(pick)
        grid[r, c] = BACKGROUND_COLOR
        row_counts[r] -= 1
        col_counts[c] -= 1
    return True


def _gen_expected_output(grid, separator_axis, content_rows, content_cols, marker_row, marker_col) -> np.ndarray:
    out = grid.copy()
    row_to_compressed = {int(r): i for i, r in enumerate(content_rows)}
    col_to_compressed = {int(c): i for i, c in enumerate(content_cols)}
    opening_axis = 1 - separator_axis
    starts = (marker_row, marker_col)
    ends = (marker_row + MARKER_SIZE, marker_col + MARKER_SIZE)

    for raw_r, raw_c in np.argwhere(grid == SOURCE_COLOR):
        cr = row_to_compressed[int(raw_r)]
        cc = col_to_compressed[int(raw_c)]
        distances = [
            _block_distance(cr, starts[0], ends[0]),
            _block_distance(cc, starts[1], ends[1]),
        ]
        if (
            distances[opening_axis] > 0
            and distances[separator_axis] > 0
            and distances[separator_axis] <= distances[opening_axis]
        ):
            out[int(raw_r), int(raw_c)] = OUTPUT_COLOR

    return out


def _passes_oracle(grid, separator_axis, content_rows, content_cols, marker_row, marker_col) -> bool:
    """Check that the solver and generator oracle agree for the sampled grid."""
    expected = _gen_expected_output(grid, separator_axis, content_rows, content_cols, marker_row, marker_col)

    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    separator_axis, raw_shape, compressed_shape, content_rows, content_cols = _sample_layout()
    grid = np.full(raw_shape, BACKGROUND_COLOR, dtype=int)
    grid[np.ix_(content_rows, content_cols)] = SOURCE_COLOR

    marker_row, marker_col = _place_marker(grid, content_rows, content_cols, compressed_shape)
    if not _remove_holes(grid, content_rows, content_cols):
        return None
    if not _passes_oracle(grid, separator_axis, content_rows, content_cols, marker_row, marker_col):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
