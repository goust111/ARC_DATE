import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GRID_MIN = 15
GRID_MAX = 21
MARKER_MIN = 4
MARKER_MAX = 6
MAX_RETRIES = 500


def _checkerboard(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, GREY, BLACK).astype(int)


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pos)) for pos in np.argwhere(grid == RED)]


def _trace_rays(
    grid: np.ndarray,
    *,
    swap_parity_rule: bool = False,
    ignore_blockers: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    markers = set(_marker_positions(grid))
    horizontal = np.zeros((h, w), dtype=bool)
    vertical = np.zeros((h, w), dtype=bool)

    for r, c in markers:
        emits_horizontally = (r + c) % 2 == 0
        if swap_parity_rule:
            emits_horizontally = not emits_horizontally
        directions = ((0, -1), (0, 1)) if emits_horizontally else ((-1, 0), (1, 0))
        target = horizontal if emits_horizontally else vertical
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if not ignore_blockers and (rr, cc) in markers:
                    break
                if (rr, cc) not in markers:
                    target[rr, cc] = True
                rr += dr
                cc += dc

    return horizontal, vertical


def _render(grid: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, *, crossings_yellow: bool = True) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    crossed = horizontal & vertical
    painted = horizontal | vertical
    if crossings_yellow:
        out[crossed] = YELLOW
        out[painted & ~crossed] = BLUE
    else:
        out[painted] = BLUE
    out[grid == RED] = RED
    return out


def _same_output(grid: np.ndarray, *, swap_parity_rule: bool = False, ignore_blockers: bool = False, crossings_yellow: bool = True) -> np.ndarray:
    horizontal, vertical = _trace_rays(
        grid,
        swap_parity_rule=swap_parity_rule,
        ignore_blockers=ignore_blockers,
    )
    return _render(grid, horizontal, vertical, crossings_yellow=crossings_yellow)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit parity-controlled rays from red markers and recolor horizontal/vertical crossings yellow.

    Input structure:
    - The grid is a grey/black checkerboard: cells with even `(row + col)` are grey (5), odd cells are black (0).
    - Four to six red (2) single-pixel markers overwrite checkerboard cells but are not on the outer border.
    - Markers may sit on either checkerboard phase, so identical-looking red pixels can encode different behaviors.

    Rule:
    - A red marker on an even-parity checkerboard cell emits horizontally, casting one ray left and one ray right.
    - A red marker on an odd-parity checkerboard cell emits vertically, casting one ray up and one ray down.
    - Each ray paints traversed background pixels blue (1) and stops before another red marker or the grid edge.
    - If a pixel is traversed by both a horizontal ray and a vertical ray, recolor that pixel yellow (4) instead of blue; preserve all red markers and untouched checkerboard cells.

    Key insight:
    - The checkerboard is not passive decoration: each marker's hidden role comes from the parity of its own location, and yellow cells appear only where independently emitted horizontal and vertical rays intersect.
    """
    g = np.asarray(grid, dtype=int)
    horizontal, vertical = _trace_rays(g)
    return _render(g, horizontal, vertical)


def _parity_values(start: int, stop: int, parity: int) -> list[int]:
    return [v for v in range(start, stop + 1) if v % 2 == parity]


def _build_core_positions(h: int, w: int) -> set[tuple[int, int]] | None:
    for _ in range(100):
        row = random.randint(4, h - 5)
        col = random.randint(4, w - 5)

        left_cols = _parity_values(2, col - 2, row % 2)
        right_cols = _parity_values(col + 2, w - 3, row % 2)
        top_rows = _parity_values(2, row - 2, 1 - (col % 2))
        bottom_rows = _parity_values(row + 2, h - 3, 1 - (col % 2))
        if not left_cols or not right_cols or not top_rows or not bottom_rows:
            continue

        left = random.choice(left_cols)
        right = random.choice(right_cols)
        top = random.choice(top_rows)
        bottom = random.choice(bottom_rows)
        return {(row, left), (row, right), (top, col), (bottom, col)}
    return None


def _chebyshev_separated(pos: tuple[int, int], used: set[tuple[int, int]], distance: int = 1) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > distance for r, c in used)


def _candidate_extras(h: int, w: int, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            pos = (r, c)
            if pos in used:
                continue
            if not _chebyshev_separated(pos, used):
                continue
            candidates.append(pos)
    random.shuffle(candidates)
    return candidates


def _has_blocker(markers: list[tuple[int, int]]) -> bool:
    marker_set = set(markers)
    for r, c in markers:
        if (r + c) % 2 == 0:
            for dc in (-1, 1):
                cc = c + dc
                while 0 <= cc and cc <= max(col for _, col in markers):
                    if (r, cc) in marker_set:
                        return True
                    cc += dc
        else:
            for dr in (-1, 1):
                rr = r + dr
                while 0 <= rr and rr <= max(row for row, _ in markers):
                    if (rr, c) in marker_set:
                        return True
                    rr += dr
    return False


def _validate_checkerboard_input(grid: np.ndarray) -> None:
    expected = _checkerboard(grid.shape)
    mask = grid != RED
    assert np.array_equal(grid[mask], expected[mask]), "non-marker cells must be the canonical checkerboard"


def _validate_semantics(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = map(int, grid.shape)
    assert GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX
    assert set(map(int, np.unique(grid))).issubset({BLACK, GREY, RED})

    _validate_checkerboard_input(grid)

    markers = _marker_positions(grid)
    assert MARKER_MIN <= len(markers) <= MARKER_MAX
    assert len(markers) == len(set(markers))
    assert all(0 < r < h - 1 and 0 < c < w - 1 for r, c in markers)
    assert any((r + c) % 2 == 0 for r, c in markers)
    assert any((r + c) % 2 == 1 for r, c in markers)
    assert _has_blocker(markers), "at least one ray must stop at another marker"

    horizontal, vertical = _trace_rays(grid)
    out = _render(grid, horizontal, vertical)
    crossed = horizontal & vertical
    painted = horizontal | vertical

    assert crossed.any(), "need at least one horizontal/vertical crossing"
    assert (painted & ~crossed).any(), "need some non-crossing ray cells"
    assert np.count_nonzero(painted) < (h * w) // 2, "ray coverage is too dense"
    assert np.count_nonzero(crossed) * 4 <= max(1, np.count_nonzero(painted)) * 3, "crossings dominate too much"

    expected = main(grid)
    assert np.array_equal(out, expected)
    assert np.array_equal(out[grid == RED], grid[grid == RED])
    assert np.array_equal(out[~painted & (grid != RED)], grid[~painted & (grid != RED)])
    assert np.any(out == YELLOW)
    assert np.any(out == BLUE)

    swapped = _same_output(grid, swap_parity_rule=True)
    unblocked = _same_output(grid, ignore_blockers=True)
    blue_only = _same_output(grid, crossings_yellow=False)
    assert not np.array_equal(swapped, out), "parity-to-direction mapping must matter"
    assert not np.array_equal(unblocked, out), "stop-at-marker behavior must matter"
    assert not np.array_equal(blue_only, out), "crossing recolor must matter"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    positions = _build_core_positions(h, w)
    if positions is None:
        return None

    target_count = random.randint(MARKER_MIN, MARKER_MAX)
    extras_needed = target_count - len(positions)
    for pos in _candidate_extras(h, w, positions):
        if extras_needed == 0:
            break
        positions.add(pos)
        extras_needed -= 1
    if extras_needed:
        return None

    grid = _checkerboard((h, w))
    for r, c in positions:
        grid[r, c] = RED

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-ray ARC input")
