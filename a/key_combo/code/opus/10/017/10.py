from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
HORIZONTAL_COLOR = 1
VERTICAL_COLOR = 2
ODD_SKIP_COLOR = 3
EVEN_SKIP_COLOR = 4
CENTER_COLOR = 5
OBSTACLE_COLOR = 7

PALETTE_CELLS = ((0, 0), (0, 1))
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # left, down, right, up
PLAN_TEMPLATES = (
    (0, 0, 1, 0, 2, 0),
    (0, 1, 0, 0, 2, 0),
    (0, 0, 2, 0, 1, 0),
    (0, 2, 0, 1, 0, 0),
    (0, 0, 1, 0, 2, 1),
    (0, 1, 0, 2, 0, 1),
)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _max_steps_to_boundary(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> int:
    h, w = shape
    if dr == 1:
        return h - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return w - 1 - c
    return c


def _segment_color(direction_index: int, skipped: int) -> int:
    if skipped == 0:
        return HORIZONTAL_COLOR if direction_index % 2 == 0 else VERTICAL_COLOR
    if skipped % 2 == 1:
        return ODD_SKIP_COLOR
    return EVEN_SKIP_COLOR


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _parse_grid(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"
    assert int(g[0, 0]) == HORIZONTAL_COLOR, "expected blue palette control at (0,0)"
    assert int(g[0, 1]) == VERTICAL_COLOR, "expected red palette control at (0,1)"
    assert np.count_nonzero(g == HORIZONTAL_COLOR) == 1, "blue should only appear in the palette"
    assert np.count_nonzero(g == VERTICAL_COLOR) == 1, "red should only appear in the palette"

    center_coords = np.argwhere(g == CENTER_COLOR)
    assert center_coords.shape == (1, 2), "expected exactly one grey center marker"
    center = tuple(map(int, center_coords[0]))
    assert center not in PALETTE_CELLS, "center marker cannot occupy the palette"

    allowed = {
        BACKGROUND,
        HORIZONTAL_COLOR,
        VERTICAL_COLOR,
        CENTER_COLOR,
        OBSTACLE_COLOR,
    }
    colors = set(map(int, np.unique(g)))
    assert colors <= allowed, "input contains unexpected colors"

    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE_COLOR)}
    assert 2 <= len(obstacles) <= 4, "expected two to four orange blockers"
    for obstacle in obstacles:
        assert obstacle not in PALETTE_CELLS, "obstacles cannot occupy palette cells"
        assert not _chebyshev_close(obstacle, center), "obstacles cannot touch the center marker"
    for obstacle in obstacles:
        for other in obstacles:
            if obstacle != other:
                assert not _chebyshev_close(obstacle, other), "blockers must be isolated"
    return center, obstacles


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    center, obstacles = _parse_grid(g)
    out = np.full_like(g, BACKGROUND)
    out[0, 0] = HORIZONTAL_COLOR
    out[0, 1] = VERTICAL_COLOR
    out[center] = CENTER_COLOR
    for r, c in obstacles:
        out[r, c] = OBSTACLE_COLOR

    r, c = center
    length = 2
    segments: list[dict[str, object]] = []

    while True:
        direction_index = (length - 2) % 4
        dr, dc = DIRECTIONS[direction_index]
        painted: list[tuple[int, int]] = []
        traversed: list[tuple[int, int]] = []
        skipped = 0
        terminated = False
        cur_r, cur_c = r, c

        while len(painted) < length:
            nr = cur_r + dr
            nc = cur_c + dc
            if not _in_bounds(g.shape, nr, nc):
                terminated = True
                break
            cur_r, cur_c = nr, nc
            traversed.append((nr, nc))
            if (nr, nc) in obstacles:
                skipped += 1
                continue
            painted.append((nr, nc))

        color = _segment_color(direction_index, skipped)
        for pr, pc in painted:
            out[pr, pc] = color

        segments.append(
            {
                "direction": direction_index,
                "length": length,
                "skipped": skipped,
                "painted": tuple(painted),
                "traversed": tuple(traversed),
                "terminated": terminated,
            }
        )
        r, c = cur_r, cur_c
        if terminated:
            break
        length += 1

    out[0, 0] = HORIZONTAL_COLOR
    out[0, 1] = VERTICAL_COLOR
    out[center] = CENTER_COLOR
    for obstacle in obstacles:
        out[obstacle] = OBSTACLE_COLOR
    return out, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the outward spiral from the grey center, letting orange blockers extend segments and recolor whole segments by skip parity.

    Input structure:
    - The grid is otherwise black, with a fixed palette at the upper-left: blue at (0,0) and red at (0,1).
    - Exactly one grey pixel marks the spiral center.
    - Two to four isolated orange pixels are blockers; they are not adjacent to the center marker.

    Rule:
    - Starting from the grey center, trace spiral segments left, down, right, up with lengths 2, 3, 4, 5, ...
    - Within one segment, advance cell by cell. Hitting an orange blocker skips that cell without painting it, keeps the blocker unchanged, and does not count toward the segment's required painted length.
    - After the segment finishes, color every painted cell in that segment uniformly: blue/red by direction when no blockers were skipped, green when the segment skipped an odd number of blockers, yellow when it skipped a positive even number.
    - If the next step would leave the grid, keep the already painted part of that segment and stop the entire spiral. Preserve the palette, center marker, and blockers.

    Key insight:
    - Each blocker has a coupled effect: it lengthens the physical span of its segment and simultaneously changes the segment's final color through the skip-count parity, so geometry and coloring must be simulated in the same segment-local reference frame.
    """
    out, _segments = _simulate(np.asarray(grid, dtype=int))
    return out


def _choose_center(n: int) -> tuple[int, int]:
    margin = random.randint(6, 7)
    rows = list(range(margin, n - margin))
    cols = list(range(margin, n - margin))
    random.shuffle(rows)
    random.shuffle(cols)
    return rows[0], cols[0]


def _valid_obstacle_coord(
    coord: tuple[int, int],
    center: tuple[int, int],
    existing: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    if coord in forbidden:
        return False
    if coord in existing:
        return False
    if _chebyshev_close(coord, center):
        return False
    return all(not _chebyshev_close(coord, other) for other in existing)


def _choose_offsets(
    start: tuple[int, int],
    direction: tuple[int, int],
    span_limit: int,
    skip_count: int,
    center: tuple[int, int],
    existing: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[int] | None:
    if skip_count == 0:
        return []

    dr, dc = direction
    candidate_offsets: list[int] = []
    for offset in range(1, span_limit):
        coord = (start[0] + dr * offset, start[1] + dc * offset)
        if _valid_obstacle_coord(coord, center, existing, forbidden):
            candidate_offsets.append(offset)

    random.shuffle(candidate_offsets)
    chosen: list[int] = []
    for offset in candidate_offsets:
        if all(abs(offset - prev) > 1 for prev in chosen):
            chosen.append(offset)
            if len(chosen) == skip_count:
                return sorted(chosen)
    return None


def _build_candidate() -> np.ndarray | None:
    n = random.randint(18, 22)
    center = _choose_center(n)
    plan = random.choice(PLAN_TEMPLATES)
    forbidden = set(PALETTE_CELLS) | {center}
    obstacles: set[tuple[int, int]] = set()
    r, c = center

    for segment_index, skip_count in enumerate(plan):
        length = segment_index + 2
        direction = DIRECTIONS[segment_index % 4]
        dr, dc = direction
        span = _max_steps_to_boundary((n, n), r, c, dr, dc)
        if span < length + skip_count:
            return None

        offsets = _choose_offsets(
            (r, c),
            direction,
            length + skip_count,
            skip_count,
            center,
            obstacles,
            forbidden,
        )
        if offsets is None:
            return None

        for offset in offsets:
            obstacles.add((r + dr * offset, c + dc * offset))

        r = r + dr * (length + skip_count)
        c = c + dc * (length + skip_count)

    if not (2 <= len(obstacles) <= 4):
        return None

    grid = np.full((n, n), BACKGROUND, dtype=int)
    grid[0, 0] = HORIZONTAL_COLOR
    grid[0, 1] = VERTICAL_COLOR
    grid[center] = CENTER_COLOR
    for obstacle in obstacles:
        grid[obstacle] = OBSTACLE_COLOR

    _, segments = _simulate(grid)
    actual_prefix = [int(seg["skipped"]) for seg in segments[: len(plan)]]
    if actual_prefix != list(plan):
        return None

    if len(segments) < len(plan):
        return None

    has_zero_horizontal = any(
        int(seg["skipped"]) == 0 and int(seg["direction"]) in (0, 2) for seg in segments
    )
    has_zero_vertical = any(
        int(seg["skipped"]) == 0 and int(seg["direction"]) in (1, 3) for seg in segments
    )
    has_odd = any(int(seg["skipped"]) % 2 == 1 for seg in segments)
    has_even_positive = any(
        int(seg["skipped"]) > 0 and int(seg["skipped"]) % 2 == 0 for seg in segments
    )
    if not (has_zero_horizontal and has_zero_vertical and has_odd and has_even_positive):
        return None

    traversed_seen: set[tuple[int, int]] = set()
    painted_seen: set[tuple[int, int]] = set()
    for seg in segments:
        traversed = list(seg["traversed"])
        painted = list(seg["painted"])
        if any(cell in PALETTE_CELLS or cell == center for cell in traversed):
            return None
        if any(cell in traversed_seen for cell in traversed):
            return None
        if any(cell in painted_seen for cell in painted):
            return None
        traversed_seen.update(traversed)
        painted_seen.update(painted)

    if np.count_nonzero(grid == OBSTACLE_COLOR) != len(obstacles):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-jump spiral input")
