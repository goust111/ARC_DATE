import numpy as np


BACKGROUND = 0
OBSTACLE = 1
MARKER = 2
PIVOT = 5

MIN_SIDE = 11
MAX_SIDE = 17
PIVOT_MARGIN = 3
MIN_OBSTACLES = 3
MAX_OBSTACLES = 6
MAX_ATTEMPTS = 500

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
QUADRANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXES = ("h", "v")

_RNG = np.random.default_rng()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _single_pos(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) == 1, f"expected exactly one cell of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _reflect_point(
    point: tuple[int, int], pivot: tuple[int, int]
) -> tuple[int, int]:
    r, c = point
    pr, pc = pivot
    return 2 * pr - r, 2 * pc - c


def _reflect_obstacles(
    obstacles: list[tuple[int, int]], pivot: tuple[int, int]
) -> list[tuple[int, int]]:
    return [_reflect_point(cell, pivot) for cell in obstacles]


def _all_clear(
    cells: list[tuple[int, int]],
    used: set[tuple[int, int]],
    *,
    min_gap: int,
) -> bool:
    for cell in cells:
        if cell in used:
            return False
        for other in used:
            if _chebyshev(cell, other) < min_gap:
                return False
    return True


def _ray_extent(
    marker: tuple[int, int],
    direction: tuple[int, int],
    blockers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, tuple[str, tuple[int, int] | None]]:
    dr, dc = direction
    step = 0
    r, c = marker
    while True:
        nr = r + dr * (step + 1)
        nc = c + dc * (step + 1)
        if not _in_bounds(shape, nr, nc):
            return step, ("boundary", None)
        if (nr, nc) in blockers:
            return step, ("obstacle", (nr, nc))
        step += 1


def _trace_ray_cells(
    marker: tuple[int, int],
    direction: tuple[int, int],
    blockers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    length, _ = _ray_extent(marker, direction, blockers, shape)
    dr, dc = direction
    r, c = marker
    return [(r + dr * step, c + dc * step) for step in range(length + 1)]


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BACKGROUND, OBSTACLE, MARKER, PIVOT})

    pivot = _single_pos(g, PIVOT)
    marker = _single_pos(g, MARKER)
    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)]
    assert MIN_OBSTACLES <= len(obstacles) <= MAX_OBSTACLES
    assert marker != pivot

    for i, a in enumerate(obstacles):
        assert _chebyshev(a, marker) >= 2, "marker too close to obstacle"
        assert _chebyshev(a, pivot) >= 2, "pivot too close to obstacle"
        for b in obstacles[i + 1 :]:
            assert _chebyshev(a, b) >= 2, "input obstacles must be isolated"

    reflected = _reflect_obstacles(obstacles, pivot)
    reflected_set = set(reflected)
    obstacle_set = set(obstacles)
    assert len(reflected_set) == len(obstacles)
    assert obstacle_set.isdisjoint(reflected_set)
    assert marker not in reflected_set and pivot not in reflected_set
    assert all(_in_bounds(g.shape, r, c) for r, c in reflected)

    final_obstacles = obstacle_set | reflected_set
    final_list = list(final_obstacles)
    for i, a in enumerate(final_list):
        for b in final_list[i + 1 :]:
            assert _chebyshev(a, b) >= 2, "final obstacles must stay visually separate"

    changed_dirs = []
    reflected_stop_dirs = []
    passes_pivot = []
    for name, direction in DIRS.items():
        full_len, (_, full_stop) = _ray_extent(marker, direction, final_obstacles, g.shape)
        orig_len, _ = _ray_extent(marker, direction, obstacle_set, g.shape)
        if full_len < orig_len:
            changed_dirs.append(name)
        if full_stop in reflected_set:
            reflected_stop_dirs.append(name)
        if pivot in _trace_ray_cells(marker, direction, final_obstacles, g.shape)[1:]:
            passes_pivot.append(name)

    assert changed_dirs, "reflection must shorten at least one ray"
    assert reflected_stop_dirs, "a reflected obstacle must block a ray"
    assert all(
        _trace_ray_cells(marker, direction, final_obstacles, g.shape)
        for direction in DIRS.values()
    )
    assert any(
        _ray_extent(marker, direction, final_obstacles, g.shape)[0] > 0
        for direction in DIRS.values()
    ), "output rays must extend beyond the marker"

    if marker[0] == pivot[0] or marker[1] == pivot[1]:
        assert passes_pivot, "aligned marker should produce a pivot-crossing ray"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the blue obstacles through the grey pivot, then extend red rays from the marker until the completed obstacle field blocks them.

    Input structure:
    - The grid is a black background with exactly one grey pivot pixel, one red marker pixel, and several isolated blue obstacle pixels.
    - Every blue obstacle has a valid 180-degree reflected partner through the pivot that stays in bounds and does not overlap any original object.

    Rule:
    - Copy each blue obstacle at its 180-degree reflection through the grey pivot, keeping the original blue obstacles as well.
    - From the red marker, extend a red ray in each of the four cardinal directions one cell at a time.
    - Each ray includes the marker cell itself and stops immediately before the first blue obstacle in that direction or, if none appears, at the grid boundary.
    - The grey pivot stays grey unless a red ray passes through it, in which case the ray overwrites it.

    Key insight:
    - The obstacle layout that matters is not the visible input alone: the hidden symmetric copies created around the pivot can shorten rays that would otherwise run to the boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(np.unique(g)).issubset({BACKGROUND, OBSTACLE, MARKER, PIVOT})

    pivot = _single_pos(g, PIVOT)
    marker = _single_pos(g, MARKER)
    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)]
    assert obstacles, "expected at least one obstacle"

    out = g.copy()
    reflected = _reflect_obstacles(obstacles, pivot)
    for rr, cc in reflected:
        assert _in_bounds(g.shape, rr, cc), "reflected obstacle must stay in bounds"
        assert (rr, cc) != marker and (rr, cc) != pivot, "reflected obstacle collision"
        out[rr, cc] = OBSTACLE

    blockers = {
        tuple(map(int, rc))
        for rc in np.argwhere(out == OBSTACLE)
    }

    for direction in DIRS.values():
        for r, c in _trace_ray_cells(marker, direction, blockers, out.shape):
            out[r, c] = MARKER

    return out


def _symmetric_limits(
    pivot: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    pr, pc = pivot
    return min(pr, shape[0] - 1 - pr), min(pc, shape[1] - 1 - pc)


def _line_candidates(
    target: tuple[int, int],
    pivot: tuple[int, int],
    shape: tuple[int, int],
    axis: str,
    cross_pivot: bool,
) -> list[tuple[int, int]]:
    tr, tc = target
    pr, pc = pivot
    candidates: list[tuple[int, int]] = []

    if cross_pivot:
        if axis == "h":
            cell = (tr, pc)
            return [cell] if cell != target and cell != pivot else []
        cell = (pr, tc)
        return [cell] if cell != target and cell != pivot else []

    if axis == "h":
        for c in range(1, shape[1] - 1):
            cell = (tr, c)
            if cell == target or cell == pivot:
                continue
            candidates.append(cell)
    else:
        for r in range(1, shape[0] - 1):
            cell = (r, tc)
            if cell == target or cell == pivot:
                continue
            candidates.append(cell)
    return candidates


def _pick_offset(
    pivot: tuple[int, int],
    shape: tuple[int, int],
    quadrant: tuple[int, int],
    axis: str,
) -> tuple[int, int] | None:
    row_limit, col_limit = _symmetric_limits(pivot, shape)
    if row_limit < 2 or col_limit < 2:
        return None

    min_dr = 2 if axis == "v" else 1
    min_dc = 2 if axis == "h" else 1
    row_choices = list(range(min_dr, row_limit + 1))
    col_choices = list(range(min_dc, col_limit + 1))
    if not row_choices or not col_choices:
        return None

    sr, sc = quadrant
    _ = sr, sc
    dr = int(_RNG.choice(row_choices))
    dc = int(_RNG.choice(col_choices))
    return dr, dc


def _valid_extra_offset(
    offset: tuple[int, int],
    used_offsets: list[tuple[int, int]],
    pivot: tuple[int, int],
    marker: tuple[int, int],
    quadrant: tuple[int, int],
    shape: tuple[int, int],
    forced_stop: tuple[int, int],
    forced_axis: str,
) -> bool:
    dr, dc = offset
    sr, sc = quadrant
    original = (pivot[0] + sr * dr, pivot[1] + sc * dc)
    reflected = _reflect_point(original, pivot)
    if not _in_bounds(shape, *original) or not _in_bounds(shape, *reflected):
        return False

    protected = {pivot, marker, forced_stop}
    if original in protected or reflected in protected:
        return False

    existing_originals = [(pivot[0] + sr * r, pivot[1] + sc * c) for r, c in used_offsets]
    existing_reflected = [_reflect_point(cell, pivot) for cell in existing_originals]
    all_existing = set(existing_originals) | set(existing_reflected) | protected
    if not _all_clear([original, reflected], all_existing, min_gap=2):
        return False

    full_blockers = set(existing_originals) | set(existing_reflected) | {original, reflected}
    forced_dir = (0, 1) if forced_axis == "h" and marker[1] < forced_stop[1] else None
    if forced_axis == "h" and marker[1] > forced_stop[1]:
        forced_dir = (0, -1)
    if forced_axis == "v" and marker[0] < forced_stop[0]:
        forced_dir = (1, 0)
    if forced_axis == "v" and marker[0] > forced_stop[0]:
        forced_dir = (-1, 0)
    assert forced_dir is not None
    _, (_, stop) = _ray_extent(marker, forced_dir, full_blockers, shape)
    return stop == forced_stop


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(MIN_SIDE, MAX_SIDE + 1))
    width = int(_RNG.integers(MIN_SIDE, MAX_SIDE + 1))
    grid = np.full((height, width), BACKGROUND, dtype=int)

    pr = int(_RNG.integers(PIVOT_MARGIN, height - PIVOT_MARGIN))
    pc = int(_RNG.integers(PIVOT_MARGIN, width - PIVOT_MARGIN))
    pivot = (pr, pc)
    quadrant = QUADRANTS[int(_RNG.integers(len(QUADRANTS)))]
    forced_axis = AXES[int(_RNG.integers(len(AXES)))]
    cross_pivot = bool(_RNG.integers(2))

    offset = _pick_offset(pivot, grid.shape, quadrant, forced_axis)
    if offset is None:
        return None

    sr, sc = quadrant
    dr, dc = offset
    forced_original = (pr + sr * dr, pc + sc * dc)
    forced_reflected = _reflect_point(forced_original, pivot)

    marker_candidates = _line_candidates(
        forced_reflected, pivot, grid.shape, forced_axis, cross_pivot
    )
    if not marker_candidates:
        return None
    _RNG.shuffle(marker_candidates)

    marker = None
    forced_direction = None
    for candidate in marker_candidates:
        if _chebyshev(candidate, forced_reflected) < 2:
            continue
        if candidate == forced_original:
            continue
        if forced_axis == "h":
            direction = (0, 1) if candidate[1] < forced_reflected[1] else (0, -1)
        else:
            direction = (1, 0) if candidate[0] < forced_reflected[0] else (-1, 0)
        if candidate == forced_reflected:
            continue
        marker = candidate
        forced_direction = direction
        break

    if marker is None or forced_direction is None:
        return None

    obstacle_count = int(_RNG.integers(MIN_OBSTACLES, MAX_OBSTACLES + 1))
    used_offsets = [offset]
    row_limit, col_limit = _symmetric_limits(pivot, grid.shape)
    candidates = [
        (r, c)
        for r in range(1, row_limit + 1)
        for c in range(1, col_limit + 1)
        if (r, c) != offset
    ]
    _RNG.shuffle(candidates)

    for candidate in candidates:
        if len(used_offsets) >= obstacle_count:
            break
        if _valid_extra_offset(
            candidate,
            used_offsets,
            pivot,
            marker,
            quadrant,
            grid.shape,
            forced_reflected,
            forced_axis,
        ):
            used_offsets.append(candidate)

    if len(used_offsets) != obstacle_count:
        return None

    originals = [(pr + sr * r, pc + sc * c) for r, c in used_offsets]
    reflected = _reflect_obstacles(originals, pivot)
    final_obstacles = set(originals) | set(reflected)

    full_len, (_, stop) = _ray_extent(marker, forced_direction, final_obstacles, grid.shape)
    orig_len, _ = _ray_extent(marker, forced_direction, set(originals), grid.shape)
    if stop != forced_reflected or full_len >= orig_len:
        return None

    grid[pivot] = PIVOT
    grid[marker] = MARKER
    for cell in originals:
        grid[cell] = OBSTACLE

    try:
        _semantic_checks(grid)
        out = main(grid)
    except AssertionError:
        return None

    if out[pivot] == PIVOT:
        if marker[0] == pivot[0] or marker[1] == pivot[1]:
            return None
    else:
        if marker[0] != pivot[0] and marker[1] != pivot[1]:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            out = main(grid)
            sig = out.tobytes() + str(out.shape).encode()
            if sig in _SEEN_OUTPUT_SIGS:
                continue
            _SEEN_OUTPUT_SIGS.add(sig)
            return grid
    raise RuntimeError("failed to generate a valid reflected-ray ARC input")
