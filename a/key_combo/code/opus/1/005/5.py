import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIR_LEFT = "L"
DIR_RIGHT = "R"
DIR_DOWN = "D"
DIR_TO_DELTA = {
    DIR_LEFT: (0, -1),
    DIR_RIGHT: (0, 1),
    DIR_DOWN: (1, 0),
}

RNG = np.random.default_rng()

GEN_ATTEMPTS = 600
HEIGHT_MIN = 13
HEIGHT_MAX = 18
WIDTH_MAX = 19
ROW_MIN = 3
ROW_GAP = 2
COL_GAP = 2
NEED_BOTH_TURNS_PROB = 0.6
DIR_PROBS = (0.32, 0.32, 0.36)


def _glyph_columns(strip: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    c = 0
    while c < strip.shape[1]:
        if not np.any(strip[:, c] == RED):
            c += 1
            continue
        start = c
        while c + 1 < strip.shape[1] and np.any(strip[:, c + 1] == RED):
            c += 1
        segments.append((start, c))
        c += 1
    return segments


def _decode_strip(strip: np.ndarray) -> list[str]:
    directions = []
    for c0, c1 in _glyph_columns(strip):
        glyph = strip[:, c0 : c1 + 1]
        width = glyph.shape[1]
        if width == 1:
            assert int(glyph[0, 0]) == RED and int(glyph[1, 0]) == RED
            directions.append(DIR_DOWN)
            continue
        assert width == 2, "glyphs must be 1 or 2 columns wide"
        assert np.array_equal(glyph[1], np.array([RED, RED]))
        top = tuple(map(int, glyph[0]))
        if top == (BLACK, RED):
            directions.append(DIR_LEFT)
        elif top == (RED, BLACK):
            directions.append(DIR_RIGHT)
        else:
            raise AssertionError("invalid elbow glyph")
    return directions


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid == BLUE) if r >= 2]
    markers.sort()
    return markers


def _trace_rays(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    directions: list[str],
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = map(int, shape)
    out = np.full((height, width), BLACK, dtype=int)
    marker_set = set(markers)
    for r, c in markers:
        out[r, c] = BLUE

    paths = []
    for (r, c), direction in zip(markers, directions):
        dr, dc = DIR_TO_DELTA[direction]
        cur_r, cur_c = r, c
        path = []
        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                break
            if (nr, nc) in marker_set:
                break
            if out[nr, nc] in (GREEN, YELLOW):
                out[nr, nc] = YELLOW
                path.append((nr, nc))
                break
            out[nr, nc] = GREEN
            path.append((nr, nc))
            cur_r, cur_c = nr, nc
        paths.append(path)
    return out, paths


def _trace_independent(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    directions: list[str],
) -> np.ndarray:
    height, width = map(int, shape)
    counts = np.zeros((height, width), dtype=int)
    marker_set = set(markers)
    for (r, c), direction in zip(markers, directions):
        dr, dc = DIR_TO_DELTA[direction]
        cur_r, cur_c = r, c
        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                break
            if (nr, nc) in marker_set:
                break
            counts[nr, nc] += 1
            cur_r, cur_c = nr, nc

    out = np.full((height, width), BLACK, dtype=int)
    out[counts == 1] = GREEN
    out[counts >= 2] = YELLOW
    for r, c in markers:
        out[r, c] = BLUE
    return out


def _glyph_width(direction: str) -> int:
    return 1 if direction == DIR_DOWN else 2


def _strip_width(directions: list[str]) -> int:
    return sum(_glyph_width(direction) for direction in directions) + len(directions) - 1


def _render_strip(width: int, directions: list[str]) -> np.ndarray:
    strip = np.full((2, width), BLACK, dtype=int)
    total_width = _strip_width(directions)
    start = int(RNG.integers(0, width - total_width + 1))
    c = start
    for idx, direction in enumerate(directions):
        if direction == DIR_DOWN:
            strip[:, c] = RED
            c += 1
        elif direction == DIR_LEFT:
            strip[0, c + 1] = RED
            strip[1, c : c + 2] = RED
            c += 2
        else:
            strip[0, c] = RED
            strip[1, c : c + 2] = RED
            c += 2
        if idx + 1 < len(directions):
            c += 1
    return strip


def _sample_axis_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = np.arange(low, high + 1)
    if len(values) < count:
        return None
    for _ in range(120):
        sample = sorted(map(int, RNG.choice(values, size=count, replace=False)))
        if all(b - a >= gap for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _sample_directions(count: int) -> list[str]:
    need_both_turns = RNG.random() < NEED_BOTH_TURNS_PROB
    choices = np.array([DIR_LEFT, DIR_RIGHT, DIR_DOWN], dtype=object)
    while True:
        directions = list(map(str, RNG.choice(choices, size=count, p=DIR_PROBS)))
        if DIR_DOWN not in directions:
            continue
        if not any(direction in (DIR_LEFT, DIR_RIGHT) for direction in directions):
            continue
        if need_both_turns and not ({DIR_LEFT, DIR_RIGHT} <= set(directions)):
            continue
        return directions


def _assignment_by_column_order(
    markers: list[tuple[int, int]],
    directions: list[str],
) -> list[str] | None:
    markers_by_col = sorted(markers, key=lambda rc: (rc[1], rc[0]))
    if markers_by_col == markers:
        return None
    assigned = {marker: direction for marker, direction in zip(markers_by_col, directions)}
    return [assigned[marker] for marker in markers]


def _permutation_changes_output(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    directions: list[str],
    target: np.ndarray,
) -> bool:
    for perm in set(itertools.permutations(directions)):
        if list(perm) == directions:
            continue
        alt, _ = _trace_rays(shape, markers, list(perm))
        if not np.array_equal(alt, target):
            return True
    return False


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = int(RNG.integers(3, 6))
    directions = _sample_directions(count)
    min_height = max(HEIGHT_MIN, 2 * count + 3)
    height = int(RNG.integers(min_height, HEIGHT_MAX + 1))
    min_width = max(14, 2 * count + 4, _strip_width(directions))
    if min_width > WIDTH_MAX:
        return None
    width = int(RNG.integers(min_width, WIDTH_MAX + 1))

    rows = _sample_axis_positions(ROW_MIN, height - 2, count, ROW_GAP)
    cols = _sample_axis_positions(1, width - 2, count, COL_GAP)
    if rows is None or cols is None:
        return None
    markers = list(zip(rows, map(int, RNG.permutation(cols))))

    main_out, paths = _trace_rays((height, width), markers, directions)
    if np.count_nonzero(main_out == YELLOW) < 1:
        return None
    if not all(path for path in paths):
        return None
    if np.array_equal(main_out, _trace_independent((height, width), markers, directions)):
        return None
    if not _permutation_changes_output((height, width), markers, directions, main_out):
        return None

    col_order_dirs = _assignment_by_column_order(markers, directions)
    if col_order_dirs is None:
        return None
    col_out, _ = _trace_rays((height, width), markers, col_order_dirs)
    if np.array_equal(col_out, main_out):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:2] = _render_strip(width, directions)
    for r, c in markers:
        grid[r, c] = BLUE
    return grid, main_out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-row glyph strip into ray directions, assign them to blue markers, and draw the resulting crossing rays.

    Input structure:
    - A black grid has a red two-row glyph strip in the top two rows.
    - Each glyph is separated by at least one black column and is either a vertical 2-cell red column or a 2x2 elbow missing one top cell.
    - Below the strip, single blue marker pixels occupy distinct rows and distinct columns.

    Rule:
    - Read the glyphs from left to right: a vertical glyph means down, an elbow with only the top-right red cell means left, and an elbow with only the top-left red cell means right.
    - Sort the blue markers by row, then column, assign the decoded directions in that order, and launch the rays in that same sequence.
    - A ray paints green cells until its next step would leave the grid or hit a blue marker; if its next step hits an existing ray, recolor that meeting cell yellow and stop there.
    - Erase the entire top glyph strip to black and keep the blue markers unchanged.

    Key insight:
    - The marker layout alone does not determine the ray pattern; the top glyph strip dynamically decodes the direction assigned to each marker in sorted order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    directions = _decode_strip(g[:2])
    markers = _sorted_markers(g)
    assert len(markers) == len(directions), "glyph count must match marker count"

    out, _ = _trace_rays(g.shape, markers, directions)
    out[:2, :] = BLACK
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            grid, expected = built
            assert np.array_equal(main(grid), expected)
            return grid
    raise RuntimeError("failed to generate a valid glyph-ray input")
