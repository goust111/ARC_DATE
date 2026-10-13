import numpy as np


BG = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]

PAIR_COUNTS = (2, 4, 6)
WIDTH_RANGE = (7, 19)
HEIGHT_SLACK_RANGE = (0, 6)
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()

_BLOCKED_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((1, 0), (1, 1)),
    ((1, 0), (2, 0)),
    ((1, 0), (2, 1)),
)

_CLEAR_TEMPLATES = (
    ((0, 0), (0, 2)),
    ((0, 0), (0, 3)),
    ((0, 0), (2, 0)),
    ((0, 0), (2, 2)),
    ((0, 0), (1, 2)),
    ((1, 0), (1, 3)),
    ((0, 1), (2, 3)),
    ((0, 2), (2, 0)),
)


def _sorted_points(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == int(color))]
    pts.sort()
    return pts


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BG)


def _pair_midpoints(
    first: list[tuple[int, int]], second: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    mids = []
    for (r0, c0), (r1, c1) in zip(first, second):
        mids.append(((r0 + r1) // 2, (c0 + c1) // 2))
    return mids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair the two foreground colors by scan order and mark each empty midpoint in yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _foreground_colors(g)
    assert len(colors) == 2, "expected exactly two foreground colors"

    first = _sorted_points(g, colors[0])
    second = _sorted_points(g, colors[1])
    assert len(first) == len(second), "foreground colors must have equal counts"

    out = np.array(g, copy=True)
    for mr, mc in _pair_midpoints(first, second):
        if out[mr, mc] == BG:
            out[mr, mc] = YELLOW
    return out


def _sample_row_starts(height: int, pair_count: int) -> list[int]:
    extra = int(height - (3 * pair_count + 2))
    gaps = [0] * (pair_count + 1)
    for _ in range(extra):
        gaps[int(_RNG.integers(0, pair_count + 1))] += 1

    rows = []
    current = 1 + gaps[0]
    for idx in range(pair_count):
        rows.append(current)
        current += 3 + gaps[idx + 1]
    return rows


def _place_pair(
    band_row: int,
    width: int,
    blocked: bool,
    occupied: set[tuple[int, int]],
    used_midpoints: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    templates = _BLOCKED_TEMPLATES if blocked else _CLEAR_TEMPLATES
    order = _RNG.permutation(len(templates))

    for template_idx in order:
        left, right = templates[int(template_idx)]
        max_r = max(left[0], right[0])
        max_c = max(left[1], right[1])
        row_shift = int(_RNG.integers(0, 3 - max_r))
        col_shift = int(_RNG.integers(0, width - max_c))

        p0 = (int(band_row + row_shift + left[0]), int(col_shift + left[1]))
        p1 = (int(band_row + row_shift + right[0]), int(col_shift + right[1]))
        if p0 in occupied or p1 in occupied:
            continue

        midpoint = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        if midpoint in used_midpoints:
            continue
        if midpoint in occupied and midpoint not in {p0, p1}:
            continue

        is_blocked = midpoint in {p0, p1}
        if blocked != is_blocked:
            continue

        return p0, p1
    return None


def _build_grid(
    height: int,
    width: int,
    color_a: int,
    color_b: int,
    points_a: list[tuple[int, int]],
    points_b: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r, c in points_a:
        grid[r, c] = int(color_a)
    for r, c in points_b:
        grid[r, c] = int(color_b)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    colors = _foreground_colors(grid)
    if len(colors) != 2:
        return False

    first = _sorted_points(grid, colors[0])
    second = _sorted_points(grid, colors[1])
    if len(first) != len(second):
        return False
    if len(first) < 2 or len(first) % 2 != 0:
        return False

    occupied = set(first) | set(second)
    midpoints = _pair_midpoints(first, second)
    if len(set(midpoints)) != len(midpoints):
        return False

    blocked_count = sum(mid in occupied for mid in midpoints)
    clear_count = len(midpoints) - blocked_count
    if blocked_count != clear_count:
        return False

    out = main(grid)
    if int((out == YELLOW).sum()) != clear_count:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(PAIR_COUNTS))
    min_height = 3 * pair_count + 2
    height = int(min_height + _RNG.integers(HEIGHT_SLACK_RANGE[0], HEIGHT_SLACK_RANGE[1] + 1))
    width = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    color_a, color_b = map(int, _RNG.choice(COLOR_POOL, size=2, replace=False))

    row_starts = _sample_row_starts(height, pair_count)
    blocked_indices = set(map(int, _RNG.choice(pair_count, size=pair_count // 2, replace=False)))

    points_a: list[tuple[int, int]] = []
    points_b: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    used_midpoints: set[tuple[int, int]] = set()

    for pair_idx, band_row in enumerate(row_starts):
        placed = _place_pair(
            band_row=band_row,
            width=width,
            blocked=pair_idx in blocked_indices,
            occupied=occupied,
            used_midpoints=used_midpoints,
        )
        if placed is None:
            return None

        p0, p1 = placed
        midpoint = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        if _RNG.random() < 0.5:
            points_a.append(p0)
            points_b.append(p1)
        else:
            points_a.append(p1)
            points_b.append(p0)

        occupied.add(p0)
        occupied.add(p1)
        used_midpoints.add(midpoint)

    grid = _build_grid(height, width, color_a, color_b, points_a, points_b)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
