from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

Coord = tuple[int, int]

ARM_BARS: dict[str, tuple[Coord, Coord]] = {
    "up": ((-3, 0), (-2, 0)),
    "down": ((2, 0), (3, 0)),
    "left": ((0, -3), (0, -2)),
    "right": ((0, 2), (0, 3)),
}
GAP_OFFSETS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
BAR_NAMES: tuple[str, ...] = ("up", "down", "left", "right")

MIN_CENTER_COUNT = 3
MAX_CENTER_COUNT = 5
MARGIN = 3
CORNER_DEPTH = 3
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _bar_cells(center: Coord, arm_name: str) -> list[Coord]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in ARM_BARS[arm_name]]


def _component_coords(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _assert_blue_bars(grid: np.ndarray) -> None:
    blue_components = _component_coords(grid == BLUE)
    assert blue_components, "expected blue bars"
    for coords in blue_components:
        assert len(coords) == 2, "every blue object must be a two-pixel bar"
        (r0, c0), (r1, c1) = coords
        assert (r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1), (
            "every blue object must be horizontal or vertical"
        )


def _is_implied_center(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    for dr, dc in GAP_OFFSETS:
        if int(grid[r + dr, c + dc]) != BLACK:
            return False
    for offsets in ARM_BARS.values():
        for dr, dc in offsets:
            if int(grid[r + dr, c + dc]) != BLUE:
                return False
    return True


def _find_centers(grid: np.ndarray) -> list[Coord]:
    h, w = grid.shape
    centers = [
        (r, c)
        for r in range(MARGIN, h - MARGIN)
        for c in range(MARGIN, w - MARGIN)
        if _is_implied_center(grid, r, c)
    ]
    centers.sort()
    return centers


def _manhattan_segment(start: Coord, end: Coord) -> list[Coord]:
    sr, sc = start
    er, ec = end
    assert sr != er and sc != ec, "route segments must have both horizontal and vertical legs"

    coords: list[Coord] = []
    c_step = 1 if ec > sc else -1
    for cc in range(sc + c_step, ec + c_step, c_step):
        coords.append((sr, cc))

    r_step = 1 if er > sr else -1
    for rr in range(sr + r_step, er + r_step, r_step):
        coords.append((rr, ec))
    return coords


def _render_output(grid: np.ndarray, centers: list[Coord], start: Coord) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in centers:
        out[r, c] = YELLOW

    cursor = start
    for waypoint in centers:
        for r, c in _manhattan_segment(cursor, waypoint):
            if out[r, c] == BLACK:
                out[r, c] = GREEN
        cursor = waypoint
    return out


def _pairwise_manhattan_ok(points: list[Coord], threshold: int) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) <= threshold:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal implied cross centers and connect them from the red start using horizontal-then-vertical L paths.

    Input structure:
    - The grid is a black background with blue two-pixel bars and one red start marker.
    - Some groups of four blue bars form implied plus signs around an empty black center with one-cell black gaps.
    - True implied centers lie away from the border, occupy distinct rows and distinct columns, and the red marker is in a corner region on a different row and column from every center.

    Rule:
    - Scan every black cell and identify it as an implied center exactly when blue two-pixel bars appear at offsets 2-3 cells up, down, left, and right, while the four adjacent gap cells remain black.
    - Recolor every implied center yellow.
    - Sort the yellow centers in reading order, then draw green Manhattan L paths from the red start to the first center and between consecutive centers, always moving horizontally first and then vertically.
    - Paint green only onto black background cells; preserve the original red marker, the blue bars, and the yellow centers.

    Key insight:
    - The waypoints are not marked directly: they must be inferred as the missing centers of four-bar implied pluses before the ordered L-path can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected input colors"

    start_cells = np.argwhere(g == RED)
    assert start_cells.shape == (1, 2), "expected exactly one red start marker"
    start = tuple(map(int, start_cells[0]))

    _assert_blue_bars(g)
    centers = _find_centers(g)
    assert MIN_CENTER_COUNT <= len(centers) <= MAX_CENTER_COUNT, "expected 3-5 implied centers"
    assert len({r for r, _ in centers}) == len(centers), "centers must occupy distinct rows"
    assert len({c for _, c in centers}) == len(centers), "centers must occupy distinct columns"
    assert _pairwise_manhattan_ok(centers, threshold=8), "centers must be well separated"
    assert all(start[0] != r and start[1] != c for r, c in centers), "start must not share a row or column with any center"

    return _render_output(g, centers, start)


def _composition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(map(int, rng.multinomial(total, np.full(parts, 1.0 / parts))))


def _sample_spaced_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
    min_gap: int,
) -> list[int]:
    slack = high - low - min_gap * (count - 1)
    assert slack >= 0
    extras = _composition(rng, slack, count + 1)
    positions = [low + extras[0]]
    for idx in range(1, count):
        positions.append(positions[-1] + min_gap + extras[idx])
    return positions


def _sample_centers(
    rng: np.random.Generator,
    h: int,
    w: int,
    count: int,
    row_gap: int,
    col_gap: int,
) -> list[Coord]:
    min_h = 2 * MARGIN + 1 + row_gap * (count - 1)
    min_w = 2 * MARGIN + 1 + col_gap * (count - 1)
    assert h >= min_h and w >= min_w

    rows = _sample_spaced_positions(rng, MARGIN, h - MARGIN - 1, count, row_gap)
    cols = _sample_spaced_positions(rng, MARGIN, w - MARGIN - 1, count, col_gap)
    perm = list(map(int, rng.permutation(count)))
    centers = [(rows[idx], cols[perm[idx]]) for idx in range(count)]
    centers.sort()
    return centers


def _can_place_bar(grid: np.ndarray, cells: list[Coord]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BLACK:
                return False
    return True


def _place_bar(grid: np.ndarray, cells: list[Coord], color: int = BLUE) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_true_motifs(grid: np.ndarray, centers: list[Coord]) -> None:
    for center in centers:
        for arm_name in BAR_NAMES:
            cells = _bar_cells(center, arm_name)
            assert _can_place_bar(grid, cells)
            _place_bar(grid, cells)


def _candidate_start_cells(h: int, w: int) -> list[Coord]:
    candidates: list[Coord] = []
    row_sets = [range(0, CORNER_DEPTH), range(h - CORNER_DEPTH, h)]
    col_sets = [range(0, CORNER_DEPTH), range(w - CORNER_DEPTH, w)]
    for rows in row_sets:
        for cols in col_sets:
            for r in rows:
                for c in cols:
                    candidates.append((r, c))
    return candidates


def _place_start(grid: np.ndarray, rng: np.random.Generator, centers: list[Coord]) -> Coord:
    candidates = [
        cell
        for cell in _candidate_start_cells(*grid.shape)
        if int(grid[cell]) == BLACK and all(cell[0] != r and cell[1] != c for r, c in centers)
    ]
    assert candidates
    start = candidates[int(rng.integers(0, len(candidates)))]
    grid[start] = RED
    return start


def _scan_partial_centers(grid: np.ndarray, true_centers: set[Coord]) -> list[tuple[Coord, int]]:
    h, w = grid.shape
    partials: list[tuple[Coord, int]] = []
    for r in range(MARGIN, h - MARGIN):
        for c in range(MARGIN, w - MARGIN):
            if (r, c) in true_centers or int(grid[r, c]) != BLACK:
                continue
            if any(int(grid[r + dr, c + dc]) != BLACK for dr, dc in GAP_OFFSETS):
                continue
            arm_count = sum(
                all(int(grid[r + dr, c + dc]) == BLUE for dr, dc in ARM_BARS[name])
                for name in BAR_NAMES
            )
            if 0 < arm_count < 4:
                partials.append(((r, c), int(arm_count)))
    return partials


def _place_partial_decoys(
    grid: np.ndarray,
    rng: np.random.Generator,
    true_centers: list[Coord],
    required_bar_counts: list[int],
) -> int:
    placed_bars = 0
    true_center_set = set(true_centers)
    h, w = grid.shape

    for arm_count in required_bar_counts:
        placed = False
        for _ in range(250):
            center = (
                int(rng.integers(MARGIN, h - MARGIN)),
                int(rng.integers(MARGIN, w - MARGIN)),
            )
            if center in true_center_set:
                continue
            if min(abs(center[0] - r) + abs(center[1] - c) for r, c in true_centers) <= 6:
                continue
            if int(grid[center]) != BLACK:
                continue
            if any(int(grid[center[0] + dr, center[1] + dc]) != BLACK for dr, dc in GAP_OFFSETS):
                continue

            chosen = list(rng.choice(BAR_NAMES, size=arm_count, replace=False))
            bar_cells = [_bar_cells(center, arm_name) for arm_name in chosen]
            if not all(_can_place_bar(grid, cells) for cells in bar_cells):
                continue

            snapshot = grid.copy()
            for cells in bar_cells:
                _place_bar(grid, cells)
            if set(_find_centers(grid)) != true_center_set:
                grid[:, :] = snapshot
                continue

            placed_bars += arm_count
            placed = True
            break
        if not placed:
            return -1
    return placed_bars


def _random_bar_candidates(grid: np.ndarray, rng: np.random.Generator) -> list[list[Coord]]:
    h, w = grid.shape
    candidates: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            horiz = [(r, c), (r, c + 1)]
            vert = [(r, c), (r + 1, c)]
            if c + 1 < w and _can_place_bar(grid, horiz):
                candidates.append(horiz)
            if r + 1 < h and _can_place_bar(grid, vert):
                candidates.append(vert)
    rng.shuffle(candidates)
    return candidates


def _place_random_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    true_centers: list[Coord],
    target_count: int,
) -> int:
    true_center_set = set(true_centers)
    placed = 0
    for cells in _random_bar_candidates(grid, rng):
        if placed >= target_count:
            break
        if not _can_place_bar(grid, cells):
            continue
        snapshot = grid.copy()
        _place_bar(grid, cells)
        if set(_find_centers(grid)) != true_center_set:
            grid[:, :] = snapshot
            continue
        placed += 1
    return placed


def _route_has_both_horizontal_directions(start: Coord, centers: list[Coord]) -> bool:
    cols = [start[1], *(c for _, c in centers)]
    deltas = [cols[idx + 1] - cols[idx] for idx in range(len(cols) - 1)]
    return any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas)


def _is_corner_cell(cell: Coord, h: int, w: int) -> bool:
    r, c = cell
    top = r < CORNER_DEPTH
    bottom = r >= h - CORNER_DEPTH
    left = c < CORNER_DEPTH
    right = c >= w - CORNER_DEPTH
    return (top or bottom) and (left or right)


def _validate_generated_input(grid: np.ndarray, centers: list[Coord], start: Coord) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED}):
        return False

    try:
        _assert_blue_bars(grid)
    except AssertionError:
        return False

    found_centers = _find_centers(grid)
    if found_centers != centers:
        return False
    if not (MIN_CENTER_COUNT <= len(centers) <= MAX_CENTER_COUNT):
        return False
    if len({r for r, _ in centers}) != len(centers) or len({c for _, c in centers}) != len(centers):
        return False
    if not _pairwise_manhattan_ok(centers, threshold=8):
        return False
    if not all(start[0] != r and start[1] != c for r, c in centers):
        return False
    if not _is_corner_cell(start, *grid.shape):
        return False

    red_cells = np.argwhere(grid == RED)
    if red_cells.shape != (1, 2) or tuple(map(int, red_cells[0])) != start:
        return False

    partials = _scan_partial_centers(grid, set(centers))
    if not partials or max(arm_count for _center, arm_count in partials) < 3:
        return False

    distractor_bars = len(_component_coords(grid == BLUE)) - 4 * len(centers)
    if distractor_bars < len(centers):
        return False

    if not _route_has_both_horizontal_directions(start, centers):
        return False

    expected = _render_output(grid, centers, start)
    if np.count_nonzero(expected == GREEN) < 10:
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_once(rng: np.random.Generator) -> np.ndarray | None:
    center_count = int(rng.integers(MIN_CENTER_COUNT, MAX_CENTER_COUNT + 1))
    row_gap, col_gap = tuple(((4, 5), (5, 4), (5, 5))[int(rng.integers(0, 3))])
    min_h = max(20, 2 * MARGIN + 1 + row_gap * (center_count - 1))
    min_w = max(20, 2 * MARGIN + 1 + col_gap * (center_count - 1))
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    grid = np.zeros((h, w), dtype=int)

    centers = _sample_centers(rng, h, w, center_count, row_gap, col_gap)
    _place_true_motifs(grid, centers)
    start = _place_start(grid, rng, centers)

    required_bar_counts = [3]
    extra_decoys = int(rng.integers(1, 4))
    for _ in range(extra_decoys):
        required_bar_counts.append(int(rng.choice(np.array([1, 2, 3], dtype=int), p=[0.15, 0.35, 0.50])))

    decoy_bars = _place_partial_decoys(grid, rng, centers, required_bar_counts)
    if decoy_bars < 0:
        return None

    target_extra_bars = int(rng.integers(center_count + 2, center_count * 2 + 6))
    remaining = max(0, target_extra_bars - decoy_bars)
    if _place_random_distractors(grid, rng, centers, remaining) < remaining:
        return None

    if not _validate_generated_input(grid, centers, start):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
