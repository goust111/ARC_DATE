import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARM_OFFSETS = [
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
]

GEN_ATTEMPTS = 500
ROW_RANGE = (18, 24)
COL_RANGE = (18, 26)
MARKER_ROW_RANGE = (4, 6)
SEGMENT_WIDTH_RANGE = (8, 14)
DISTRACTOR_RANGE = (4, 10)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _detect_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(3, height - 3):
        for c in range(3, width - 3):
            if int(grid[r, c]) != BLACK:
                continue
            if any(int(grid[r + dr, c + dc]) != BLACK for dr, dc in GAP_OFFSETS):
                continue
            if all(int(grid[r + dr, c + dc]) == GREY for dr, dc in ARM_OFFSETS):
                centers.append((r, c))
    return centers


def _mark_centers(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in _detect_centers(out):
        out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect implied cross centers, color them yellow, then stack the row segments bracketed by blue and teal markers.

    Input structure:
    - The grid is mostly black background with grey two-cell micro-bars, each either horizontal or vertical.
    - Some micro-bars form implied crosses: two vertical bars above/below an empty center and two horizontal bars left/right of it, all with one black gap next to the center.
    - A subset of rows contains one blue marker on the left and one teal marker on the right; the pixels strictly between them define extractable row segments.

    Rule:
    - Find every black cell whose neighbors one step up, down, left, and right are black, and whose cells at offsets 2 and 3 in all four cardinal directions form the required grey two-cell bars.
    - Recolor each such missing cross center to yellow.
    - Read the blue/teal marker rows from top to bottom, copy the cells strictly between the markers from the center-marked grid, and stack those copied segments.

    Key insight:
    - The yellow information does not exist in the input and must be inferred geometrically first; only then can the bracket rows harvest the rows whose extracted spans happen to include those inferred centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    marked = _mark_centers(g)

    rows: list[np.ndarray] = []
    width = None
    for r in range(marked.shape[0]):
        blue_cols = np.flatnonzero(marked[r] == BLUE)
        teal_cols = np.flatnonzero(marked[r] == TEAL)
        if len(blue_cols) == 0 and len(teal_cols) == 0:
            continue

        assert len(blue_cols) == 1 and len(teal_cols) == 1, "each bracket row must have one blue and one teal marker"
        left = int(blue_cols[0])
        right = int(teal_cols[0])
        assert left + 1 < right, "markers must bracket at least one cell"

        segment = marked[r, left + 1 : right].copy()
        if width is None:
            width = int(segment.size)
        else:
            assert int(segment.size) == width, "all bracketed segments must have equal width"
        rows.append(segment)

    assert rows, "expected at least one bracket row"
    return np.vstack(rows)


def _plus_arm_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in ARM_OFFSETS]


def _plus_guard_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c)] + [(r + dr, c + dc) for dr, dc in GAP_OFFSETS] + _plus_arm_cells(r, c)


def _cells_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    return True


def _component_is_isolated(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BLACK:
                return False
    return True


def _can_place_plus(grid: np.ndarray, r: int, c: int) -> bool:
    if not (3 <= r < grid.shape[0] - 3 and 3 <= c < grid.shape[1] - 3):
        return False
    guard = _plus_guard_cells(r, c)
    arm_cells = _plus_arm_cells(r, c)
    return _cells_clear(grid, guard) and _component_is_isolated(grid, arm_cells)


def _place_plus(grid: np.ndarray, r: int, c: int) -> None:
    for rr, cc in _plus_arm_cells(r, c):
        grid[rr, cc] = GREY


def _can_place_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return _cells_clear(grid, cells) and _component_is_isolated(grid, cells)


def _place_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = GREY


def _sample_distinct_rows(rng: np.random.Generator, rows: np.ndarray, count: int) -> list[int]:
    if len(rows) < count:
        return []
    chosen = rng.choice(rows, size=count, replace=False)
    return sorted(int(v) for v in chosen)


def _try_place_centers_for_rows(
    grid: np.ndarray,
    rng: np.random.Generator,
    rows: list[int],
    col_candidates: list[int],
) -> list[tuple[int, int]] | None:
    centers: list[tuple[int, int]] = []
    for r in rows:
        cols = col_candidates.copy()
        rng.shuffle(cols)
        placed = False
        for c in cols:
            if _can_place_plus(grid, r, c):
                _place_plus(grid, r, c)
                centers.append((r, c))
                placed = True
                break
        if not placed:
            return None
    return centers


def _try_place_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    target_count: int,
    intended_centers: set[tuple[int, int]],
) -> bool:
    placed = 0
    attempts = target_count * 50
    height, width = grid.shape
    for _ in range(attempts):
        if placed >= target_count:
            break
        if int(rng.integers(0, 2)) == 0:
            r = int(rng.integers(0, height))
            c = int(rng.integers(0, width - 1))
            cells = [(r, c), (r, c + 1)]
        else:
            r = int(rng.integers(0, height - 1))
            c = int(rng.integers(0, width))
            cells = [(r, c), (r + 1, c)]

        if not _can_place_bar(grid, cells):
            continue

        _place_bar(grid, cells)
        if set(_detect_centers(grid)) != intended_centers:
            for rr, cc in cells:
                grid[rr, cc] = BLACK
            continue
        placed += 1
    return placed == target_count


def _build_expected_output(grid: np.ndarray, marker_rows: list[int], left: int, right: int) -> np.ndarray:
    marked = _mark_centers(grid)
    return np.vstack([marked[r, left + 1 : right].copy() for r in marker_rows])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(ROW_RANGE[0], ROW_RANGE[1] + 1))
    width = int(rng.integers(COL_RANGE[0], COL_RANGE[1] + 1))

    max_segment_width = min(SEGMENT_WIDTH_RANGE[1], width - 3)
    if max_segment_width < SEGMENT_WIDTH_RANGE[0]:
        return None
    segment_width = int(rng.integers(SEGMENT_WIDTH_RANGE[0], max_segment_width + 1))
    left = int(rng.integers(1, width - segment_width - 1))
    right = left + segment_width + 1

    marker_count = int(rng.integers(MARKER_ROW_RANGE[0], MARKER_ROW_RANGE[1] + 1))
    visible_count = 2 if marker_count == 4 or segment_width < 10 else int(rng.choice([2, 3]))
    hidden_count = int(rng.choice([1, 2]))

    interior_rows = np.arange(3, height - 3)
    visible_rows = _sample_distinct_rows(rng, interior_rows, visible_count)
    if not visible_rows:
        return None
    remaining_rows = np.array([r for r in interior_rows if r not in set(visible_rows)], dtype=int)
    hidden_rows = _sample_distinct_rows(rng, remaining_rows, hidden_count)
    if len(hidden_rows) != hidden_count:
        return None

    grid = np.zeros((height, width), dtype=int)

    visible_cols = list(range(left + 4, right - 3))
    hidden_cols = list(range(3, width - 3))
    if not visible_cols or not hidden_cols:
        return None

    visible_centers = _try_place_centers_for_rows(grid, rng, visible_rows, visible_cols)
    if visible_centers is None:
        return None
    hidden_centers = _try_place_centers_for_rows(grid, rng, hidden_rows, hidden_cols)
    if hidden_centers is None:
        return None

    intended_centers = set(visible_centers + hidden_centers)
    if set(_detect_centers(grid)) != intended_centers:
        return None

    distractor_count = int(rng.integers(DISTRACTOR_RANGE[0], DISTRACTOR_RANGE[1] + 1))
    if not _try_place_distractors(grid, rng, distractor_count, intended_centers):
        return None

    if any(int(grid[r, left]) != BLACK or int(grid[r, right]) != BLACK for r in visible_rows):
        return None

    forbidden_rows = set(hidden_rows)
    eligible_extra_rows = [
        r
        for r in range(height)
        if r not in forbidden_rows
        and r not in set(visible_rows)
        and int(grid[r, left]) == BLACK
        and int(grid[r, right]) == BLACK
    ]
    extra_needed = marker_count - visible_count
    if len(eligible_extra_rows) < extra_needed:
        return None
    extra_rows = sorted(int(v) for v in rng.choice(np.array(eligible_extra_rows), size=extra_needed, replace=False))
    marker_rows = sorted(visible_rows + extra_rows)

    for r in marker_rows:
        if int(grid[r, left]) != BLACK or int(grid[r, right]) != BLACK:
            return None
        grid[r, left] = BLUE
        grid[r, right] = TEAL

    if set(_detect_centers(grid)) != intended_centers:
        return None

    expected = _build_expected_output(grid, marker_rows, left, right)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    yellow_count = int((actual == YELLOW).sum())
    if yellow_count != visible_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
