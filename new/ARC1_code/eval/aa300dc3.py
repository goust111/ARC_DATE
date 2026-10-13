import numpy as np
import random
from arc_utils import find_components


def _maximal_diagonal_runs(g: np.ndarray, color: int) -> list[tuple[tuple[int, int], list[tuple[int, int]], int]]:
    h, w = g.shape
    runs = []
    for dr, dc in ((1, 1), (1, -1)):
        for r in range(h):
            for c in range(w):
                if g[r, c] != color:
                    continue
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < h and 0 <= prev_c < w and g[prev_r, prev_c] == color:
                    continue

                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and g[rr, cc] == color:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc

                if len(cells) >= 2:
                    diagonal_id = r - c if dc == 1 else r + c
                    runs.append(((dr, dc), cells, diagonal_id))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Paint the longest black diagonal segment teal, choosing the middle segment when the longest run is an odd-width parallel band.

    Input structure
    A 10x10 grid containing black pixels embedded in a grey field/frame.

    Rule
    Find every maximal contiguous black segment along both diagonal directions. Select the maximum-length segment; if the maximum-length candidates form adjacent parallel diagonals, select the central diagonal of that band. Recolor the selected black pixels to teal.

    Key insight
    The output marks the hidden straight diagonal stroke inside the black mask; the only ambiguous training case is a three-diagonal band, where the centered parallel run is the marked stroke.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g).tolist()) == {0, 5}

    runs = _maximal_diagonal_runs(g, 0)
    assert runs
    max_len = max(len(cells) for _, cells, _ in runs)
    candidates = [run for run in runs if len(run[1]) == max_len]

    orientations = {orientation for orientation, _, _ in candidates}
    assert len(orientations) == 1

    candidates.sort(key=lambda run: run[2])
    diagonal_ids = [diagonal_id for _, _, diagonal_id in candidates]
    assert diagonal_ids == list(range(diagonal_ids[0], diagonal_ids[-1] + 1))
    assert len(candidates) % 2 == 1

    _, cells, _ = candidates[len(candidates) // 2]
    out = g.copy()
    for r, c in cells:
        out[r, c] = 8
    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_SHAPE = (10, 10)
INTERIOR_MIN = 1
INTERIOR_MAX = 8
INTERIOR_COORDS = range(INTERIOR_MIN, INTERIOR_MAX + 1)
BLACK_COLOR = 0
FRAME_COLOR = 5
ANSWER_COLOR = 8

# Diagonal layout sampling.
ORIENTATIONS = [(1, 1), (1, -1)]
DIAGONAL_ID_RANGES = {(1, 1): range(-7, 8), (1, -1): range(2, 17)}
BAND_WIDTHS = (1, 3)
SINGLE_BAND_WIDTH = 1
BAND_FLANK_RADIUS = 1
RUN_LENGTH_MIN = 6
RUN_LENGTH_MAX = 8

# Blob size and near-run sampling.
BLACK_COUNT_MIN = 39
BLACK_COUNT_MAX = 48
NEAR_RUN_SHORT_PROB = 0.7
NEAR_RUN_SHORT_DELTA = 2
NEAR_RUN_LONG_DELTA = 1

# Retry limits.
GROW_ATTEMPTS = 200
TRIM_ATTEMPTS = 30
GENERATE_ATTEMPTS = 300

NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_diag_id(r: int, c: int, orientation: tuple[int, int]) -> int:
    return r - c if orientation == (1, 1) else r + c


def _gen_in_inner(r: int, c: int) -> bool:
    return INTERIOR_MIN <= r <= INTERIOR_MAX and INTERIOR_MIN <= c <= INTERIOR_MAX


def _gen_diagonal_cells(diagonal_id: int, orientation: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(
        (r, c)
        for r in INTERIOR_COORDS
        for c in INTERIOR_COORDS
        if _gen_diag_id(r, c, orientation) == diagonal_id
    )


def _gen_band_ids(diagonal_id: int, band_width: int) -> list[int]:
    return [diagonal_id] if band_width == SINGLE_BAND_WIDTH else list(
        range(diagonal_id - BAND_FLANK_RADIUS, diagonal_id + BAND_FLANK_RADIUS + 1)
    )


def _gen_valid_centers(orientation: tuple[int, int], band_width: int, length: int) -> list[int]:
    return [
        diagonal_id
        for diagonal_id in DIAGONAL_ID_RANGES[orientation]
        if all(len(_gen_diagonal_cells(band_id, orientation)) >= length for band_id in _gen_band_ids(diagonal_id, band_width))
    ]


def _gen_bad_runs(grid, allowed_runs, length):
    bad = []
    for orientation, cells, diagonal_id in _maximal_diagonal_runs(grid, BLACK_COLOR):
        key = (orientation, diagonal_id, tuple(cells))
        if len(cells) >= length and key not in allowed_runs:
            bad.append((orientation, cells, diagonal_id))
    return bad


def _gen_black_connected(grid):
    comps = find_components(grid, bg=FRAME_COLOR, connectivity=4, monochromatic=True, colors=[BLACK_COLOR], return_format="dict")
    return len(comps) == 1 and comps[0]["area"] == int((grid == BLACK_COLOR).sum())


def _gen_frontier(grid):
    frontier = set()
    for r, c in np.argwhere(grid == BLACK_COLOR):
        if not _gen_in_inner(r, c):
            continue
        for dr, dc in NEIGHBOR_OFFSETS_4:
            nr, nc = r + dr, c + dc
            if _gen_in_inner(nr, nc) and grid[nr, nc] == FRAME_COLOR:
                frontier.add((int(nr), int(nc)))
    return list(frontier)


def _gen_paint_cells(grid, cells, color):
    for cell in cells:
        grid[cell] = color


def _gen_trim_bad_runs(grid, allowed_runs, protected, length):
    for _ in range(TRIM_ATTEMPTS):
        bad_runs = _gen_bad_runs(grid, allowed_runs, length)
        if not bad_runs:
            return True

        _, cells, _ = random.choice(bad_runs)
        n = len(cells)
        lo = max(0, n - length)
        hi = min(n - 1, length - 1)
        blockers = [cells[i] for i in range(lo, hi + 1) if cells[i] not in protected]
        if not blockers:
            blockers = [cell for cell in cells if cell not in protected]
        random.shuffle(blockers)

        for cell in blockers:
            grid[cell] = FRAME_COLOR
            if _gen_black_connected(grid):
                break
            grid[cell] = BLACK_COLOR
        else:
            return False
    return False


def _gen_place_near_run(grid, allowed_runs, protected, length):
    near_delta = NEAR_RUN_SHORT_DELTA if random.random() < NEAR_RUN_SHORT_PROB else NEAR_RUN_LONG_DELTA
    near_length = length - near_delta
    choices = []
    for orientation in ORIENTATIONS:
        for diagonal_id in DIAGONAL_ID_RANGES[orientation]:
            cells = _gen_diagonal_cells(diagonal_id, orientation)
            for start in range(0, len(cells) - near_length + 1):
                segment = tuple(cells[start : start + near_length])
                if any(grid[cell] == BLACK_COLOR for cell in segment):
                    continue
                _gen_paint_cells(grid, segment, BLACK_COLOR)
                ok = not _gen_bad_runs(grid, allowed_runs, length)
                _gen_paint_cells(grid, segment, FRAME_COLOR)
                if ok:
                    choices.append(segment)

    if not choices:
        return False
    segment = random.choice(choices)
    _gen_paint_cells(grid, segment, BLACK_COLOR)
    protected.update(segment)
    return True


def _gen_choose_layout():
    """Sample the diagonal layout and target size in the original random-call order."""
    orientation = random.choice(ORIENTATIONS)
    length = random.randint(RUN_LENGTH_MIN, RUN_LENGTH_MAX)
    band_options = [band_width for band_width in BAND_WIDTHS if _gen_valid_centers(orientation, band_width, length)]
    band_width = random.choice(band_options)
    diagonal_id = random.choice(_gen_valid_centers(orientation, band_width, length))
    target_count = random.randint(BLACK_COUNT_MIN, BLACK_COUNT_MAX)
    return orientation, length, band_width, diagonal_id, target_count


def _gen_seed_answer_band(orientation, length, band_width, diagonal_id):
    """Draw the protected max-length diagonal band and return the central answer cells."""
    out = np.full(GRID_SHAPE, FRAME_COLOR, dtype=int)
    allowed_runs, protected = set(), set()
    answer_cells = []

    for band_id in _gen_band_ids(diagonal_id, band_width):
        cells = _gen_diagonal_cells(band_id, orientation)
        start = random.randint(0, len(cells) - length)
        segment = tuple(cells[start : start + length])
        _gen_paint_cells(out, segment, BLACK_COLOR)
        allowed_runs.add((orientation, band_id, segment))
        protected.update(segment)
        if band_id == diagonal_id:
            answer_cells = list(segment)
    return out, allowed_runs, protected, answer_cells


def _gen_grow_to_target(out, allowed_runs, length, target_count):
    """Flood-grow black cells without creating a competing max-length diagonal."""
    for _ in range(GROW_ATTEMPTS):
        if int((out == BLACK_COLOR).sum()) >= target_count:
            break

        safe_frontier = []
        frontier = _gen_frontier(out)
        random.shuffle(frontier)
        for cell in frontier:
            out[cell] = BLACK_COLOR
            if not _gen_bad_runs(out, allowed_runs, length):
                safe_frontier.append(cell)
            out[cell] = FRAME_COLOR

        if not safe_frontier:
            return False
        out[random.choice(safe_frontier)] = BLACK_COLOR
    return int((out == BLACK_COLOR).sum()) == target_count


def _gen_oracle_checked_input(out, allowed_runs, protected, answer_cells, length):
    """Validate invariants, mark the answer, and return the corresponding input grid."""
    if not _gen_black_connected(out):
        return None
    if not _gen_trim_bad_runs(out, allowed_runs, protected, length):
        return None
    if not (BLACK_COUNT_MIN <= int((out == BLACK_COLOR).sum()) <= BLACK_COUNT_MAX):
        return None
    if not _gen_black_connected(out):
        return None

    _gen_paint_cells(out, answer_cells, ANSWER_COLOR)

    inp = out.copy()
    inp[inp == ANSWER_COLOR] = BLACK_COLOR
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    orientation, length, band_width, diagonal_id, target_count = _gen_choose_layout()
    out, allowed_runs, protected, answer_cells = _gen_seed_answer_band(orientation, length, band_width, diagonal_id)
    if not _gen_place_near_run(out, allowed_runs, protected, length):
        return None
    if not _gen_grow_to_target(out, allowed_runs, length, target_count):
        return None
    return _gen_oracle_checked_input(out, allowed_runs, protected, answer_cells, length)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_ATTEMPTS} attempts")
