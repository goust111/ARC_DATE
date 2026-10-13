from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# The puzzle gives these two colors visible roles rather than numeric meanings.
WALL_COLOR = 5       # grey pixels reflect the ray
CONTAINER_COLOR = 2  # red pixels form U-shaped containers


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of canonical (row, column) coords."""
    h, w = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        coords = [seed]
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
                    coords.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _find_arrow(grid: np.ndarray, bg: int) -> tuple[int, set[tuple[int, int]], tuple[int, int], int]:
    """Find the unique 3-cell L arrow and return color, cells, launch cell, dc."""
    candidates = []
    for color in map(int, np.unique(grid)):
        if color in (bg, WALL_COLOR, CONTAINER_COLOR):
            continue
        for coords in _components(grid == color):
            if len(coords) != 3:
                continue
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            if (r1 - r0, c1 - c0) != (1, 1):
                continue
            cells = {tuple(map(int, rc)) for rc in coords}
            corners = {(r, c) for r in (r0, r1) for c in (c0, c1)}
            missing = list(corners - cells)
            if len(missing) != 1:
                continue

            # As in the reference arrow sprite, the missing corner is the tail:
            # launch from the diagonally opposite filled corner, continuing away.
            mr, mc = missing[0]
            start = (r0 + r1 - mr, c0 + c1 - mc)
            dr = start[0] - mr
            dc = start[1] - mc
            if dr == 1 and dc in (-1, 1):
                candidates.append((color, cells, start, dc))

    assert len(candidates) == 1, "expected one downward-pointing 3-pixel L arrow"
    return candidates[0]


def _find_containers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Return exact red U shapes as inclusive (top, bottom, left, right) bboxes."""
    containers: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == CONTAINER_COLOR):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 <= r0 or c1 - c0 < 2:
            continue
        expected = {
            *((r, c0) for r in range(r0, r1 + 1)),
            *((r, c1) for r in range(r0, r1 + 1)),
            *((r1, c) for c in range(c0, c1 + 1)),
        }
        actual = {tuple(map(int, rc)) for rc in coords}
        if actual != expected:
            continue
        if not np.all(grid[r0:r1, c0 + 1:c1] == bg):
            continue
        containers.append((r0, r1, c0, c1))
    return sorted(containers)


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    initial_dc: int,
) -> tuple[int, list[tuple[str, int, int]], list[tuple[int, int]]]:
    """Trace the invisible downward ray; return landing column, events, and path."""
    h, w = grid.shape
    r, c = map(int, start)
    dc = int(initial_dc)
    events: list[tuple[str, int, int]] = []
    path: list[tuple[int, int]] = []

    while r < h - 1:
        nr = r + 1
        nc = c + dc
        if not (0 <= nc < w):
            events.append(("boundary", nr, nc))
            dc = -dc
            nc = c + dc
        elif int(grid[nr, nc]) == WALL_COLOR:
            events.append(("wall", nr, nc))
            dc = -dc
            nc = c + dc

        assert 0 <= nc < w, "reflection must return the ray to the grid"
        assert int(grid[nr, nc]) != WALL_COLOR, "ray may not be trapped between reflectors"
        r, c = nr, nc
        path.append((r, c))
        if int(grid[r, c]) == CONTAINER_COLOR or r == h - 1:
            return c, events, path

    return c, events, path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast an invisible bouncing ray from an L-arrow and fill the U-container selected by its landing column.

    Input structure:
    - A dominant background contains one monochrome 3-pixel L arrow in the upper area,
      separate grey wall pixels, and one or more exact red U-shaped containers below.
    - The arrow's missing corner and diagonally opposite filled corner encode one of
      the two downward diagonal launch directions; container interiors are background.

    Rule:
    - Launch from the arrow's right-angle cell away from its missing corner. Move down
      one row per step, reversing only the horizontal direction before a left/right
      boundary crossing or a diagonal move into a grey wall pixel.
    - Stop on the first red container pixel or on the bottom row. If that column lies
      in exactly one closed U footprint, fill its interior with the arrow color;
      otherwise color the entire bottom row with the arrow color.
    - Erase all three arrow pixels, preserve every wall and container pixel, and never
      draw any cell of the ray trajectory.

    Key insight:
    - The arrow's visible column is not the destination: the unique target is obtained
      only by simulating the unseen diagonal path in grid (row, column) coordinates,
      including both boundary and grey-wall horizontal reflections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid is too small"
    bg = _background(g)

    arrow_color, arrow_cells, start, dc = _find_arrow(g, bg)
    containers = _find_containers(g, bg)
    assert containers, "expected at least one closed red U-container"
    landing_col, _events, _path = _trace_ray(g, start, dc)

    selected = [box for box in containers if box[2] <= landing_col <= box[3]]
    assert len(selected) <= 1, "landing column must identify at most one container"

    out = g.copy()
    for r, c in arrow_cells:
        out[r, c] = bg

    if selected:
        top, bottom, left, right = selected[0]
        interior = out[top:bottom, left + 1:right]
        assert np.all(interior == bg), "selected container interior must be empty"
        interior[:] = arrow_color
    else:
        out[-1, :] = arrow_color
    return out


# ------------------------------ generator ---------------------------------

_GEN_BG = 0
_GEN_ARROW_COLORS = (1, 3, 4, 6, 7, 8, 9)
_GEN_CALL_COUNT = 0
_GEN_ATTEMPTS = 800


def _empty_trace(
    shape: tuple[int, int],
    start: tuple[int, int],
    dc: int,
    walls: set[tuple[int, int]],
    last_row: int,
) -> tuple[dict[int, int], list[tuple[str, int, int]]]:
    """Independent generation trace through only boundaries and a supplied wall set."""
    h, w = shape
    r, c = start
    cols: dict[int, int] = {r: c}
    events: list[tuple[str, int, int]] = []
    while r < min(last_row, h - 1):
        nr, nc = r + 1, c + dc
        if nc < 0 or nc >= w:
            events.append(("boundary", nr, nc))
            dc = -dc
            nc = c + dc
        elif (nr, nc) in walls:
            events.append(("wall", nr, nc))
            dc = -dc
            nc = c + dc
        if not (0 <= nc < w) or (nr, nc) in walls:
            return {}, []
        r, c = nr, nc
        cols[r] = c
    return cols, events


def _path_container_box(
    cols: dict[int, int],
    bottom: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    """Make a U bbox whose open interior contains the path until its bottom hit."""
    top = bottom - height
    segment = [cols[r] for r in range(top, bottom + 1)]
    lo, hi = min(segment), max(segment)
    spare = width - (hi - lo + 1) - 2
    if spare < 0:
        return None
    add_left = random.randint(0, spare)
    left = lo - 1 - add_left
    right = left + width - 1
    if left < 0:  # the right canvas bound is checked by the caller
        return None
    return top, bottom, left, right


def _draw_u(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    top, bottom, left, right = box
    grid[top:bottom + 1, left] = CONTAINER_COLOR
    grid[top:bottom + 1, right] = CONTAINER_COLOR
    grid[bottom, left:right + 1] = CONTAINER_COLOR


def _boxes_separated(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    _t, _b, left, right = candidate
    return all(right + 1 < old_left or old_right + 1 < left for _ot, _ob, old_left, old_right in boxes)


def _target_for_column(
    containers: list[tuple[int, int, int, int]], col: int
) -> tuple[int, int, int, int] | None:
    hits = [box for box in containers if box[2] <= col <= box[3]]
    return hits[0] if len(hits) == 1 else None


def _make_candidate(want_container: bool) -> np.ndarray | None:
    h = random.randint(18, 27)
    w = random.randint(20, 30)
    bottom = random.choice((h - 3, h - 2))
    arrow_top = random.randint(1, 3)
    dc = random.choice((-1, 1))
    edge_distance = random.randint(2, min(5, w // 4))
    start_c = edge_distance if dc == -1 else w - 1 - edge_distance
    start = (arrow_top + 1, start_c)

    # Plan a boundary-only route, then put one effective grey reflector after its
    # first edge reflection. This guarantees both bounce mechanisms are exercised.
    base_cols, base_events = _empty_trace((h, w), start, dc, set(), h - 1)
    boundary_rows = [row for kind, row, _ in base_events if kind == "boundary"]
    if not boundary_rows:
        return None
    first_boundary = boundary_rows[0]
    possible_wall_rows = list(range(first_boundary + 2, bottom - 6))
    random.shuffle(possible_wall_rows)

    active_wall = None
    actual_cols: dict[int, int] = {}
    actual_events: list[tuple[str, int, int]] = []
    for wall_row in possible_wall_rows:
        previous_col = base_cols.get(wall_row - 1)
        next_col = base_cols.get(wall_row)
        if previous_col is None or next_col is None:
            continue
        wall = (wall_row, next_col)
        trial_cols, trial_events = _empty_trace((h, w), start, dc, {wall}, h - 1)
        if not trial_cols:
            continue
        kinds = {event[0] for event in trial_events if event[1] <= bottom}
        if kinds != {"boundary", "wall"}:
            continue
        if abs(trial_cols[bottom] - base_cols[bottom]) < 9:
            continue
        active_wall = wall
        actual_cols, actual_events = trial_cols, trial_events
        break
    if active_wall is None:
        return None

    # Construct disjoint containers around the actual and wall-free trajectories.
    actual_height = random.randint(2, 4)
    base_height = random.randint(2, 4)
    actual_span = max(actual_cols[r] for r in range(bottom - actual_height, bottom + 1)) - min(
        actual_cols[r] for r in range(bottom - actual_height, bottom + 1)
    )
    base_span = max(base_cols[r] for r in range(bottom - base_height, bottom + 1)) - min(
        base_cols[r] for r in range(bottom - base_height, bottom + 1)
    )
    actual_width = actual_span + 3 + random.randint(0, 1)
    base_width = base_span + 3 + random.randint(0, 1)
    actual_box = _path_container_box(actual_cols, bottom, actual_height, actual_width)
    base_box = _path_container_box(base_cols, bottom, base_height, base_width)
    if actual_box is None or base_box is None:
        return None
    if actual_box[3] >= w or base_box[3] >= w or not _boxes_separated([actual_box], base_box):
        return None

    containers: list[tuple[int, int, int, int]] = []
    if want_container:
        containers.extend((actual_box, base_box))
    else:
        containers.append(base_box)

        # Deliberately put a container under the arrow's launch column. A solver
        # that substitutes a vertical drop for the specified ray will catch it,
        # whereas the real bounced ray must miss every container.
        vertical_options = []
        for height in range(2, 5):
            for width in range(4, 8):
                for left in range(max(0, start_c - width + 1), min(start_c, w - width) + 1):
                    box = (bottom - height, bottom, left, left + width - 1)
                    if _boxes_separated(containers, box):
                        vertical_options.append(box)
        if not vertical_options:
            return None
        containers.append(random.choice(vertical_options))

    # Both branches always have multiple visible U shapes; some scenes add a third.
    desired_count = random.choice((2, 2, 3))
    for _ in range(300):
        if len(containers) >= desired_count:
            break
        height = random.randint(2, 4)
        width = random.randint(4, 7)
        left = random.randint(0, w - width)
        box = (bottom - height, bottom, left, left + width - 1)
        if _boxes_separated(containers, box):
            containers.append(box)
    if len(containers) != desired_count:
        return None

    grid = np.full((h, w), _GEN_BG, dtype=int)
    for box in containers:
        _draw_u(grid, box)

    # Place the effective wall plus several off-path wall pixels. Excluding both
    # planned paths makes these extra pixels genuine distractors, not hidden bounces.
    grid[active_wall] = WALL_COLOR
    forbidden_path = {(r, c) for r, c in actual_cols.items()} | {(r, c) for r, c in base_cols.items()}
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != _GEN_BG)}
    distractor_goal = random.randint(2, 6)
    distractors = 0
    candidates = [
        (r, c)
        for r in range(1, bottom - 4)
        for c in range(1, w - 1)
        if (r, c) not in forbidden_path and (r, c) not in occupied
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if distractors >= distractor_goal:
            break
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in occupied):
            continue
        grid[r, c] = WALL_COLOR
        occupied.add((r, c))
        distractors += 1
    if distractors != distractor_goal:
        return None

    arrow_color = random.choice(_GEN_ARROW_COLORS)
    if dc == 1:
        arrow_c0 = start_c - 1
        missing = (arrow_top, arrow_c0)
    else:
        arrow_c0 = start_c
        missing = (arrow_top, arrow_c0 + 1)
    arrow_cells = {
        (r, c)
        for r in (arrow_top, arrow_top + 1)
        for c in (arrow_c0, arrow_c0 + 1)
        if (r, c) != missing
    }
    if any(grid[cell] != _GEN_BG for cell in arrow_cells):
        return None
    for cell in arrow_cells:
        grid[cell] = arrow_color

    # Semantic checks use the actual scene, not merely main(generate_input()).
    try:
        found_color, found_cells, found_start, found_dc = _find_arrow(grid, _GEN_BG)
        found_containers = _find_containers(grid, _GEN_BG)
        landing, events, path = _trace_ray(grid, found_start, found_dc)
    except (AssertionError, ValueError):
        return None
    if found_color != arrow_color or found_cells != arrow_cells or found_start != start or found_dc != dc:
        return None
    if set(found_containers) != set(containers):
        return None
    event_kinds = {kind for kind, _r, _c in events}
    if not {"boundary", "wall"}.issubset(event_kinds):
        return None
    if landing == start_c:
        return None

    actual_target = _target_for_column(found_containers, landing)
    intended_target = actual_box if want_container else None
    if actual_target != intended_target:
        return None

    vertical_target = _target_for_column(found_containers, start_c)
    if vertical_target == actual_target or (not want_container and vertical_target is None):
        return None

    # Counterexample against ignoring grey walls: after removing all grey pixels,
    # the ray must land in the other, explicitly constructed container.
    no_walls = grid.copy()
    no_walls[no_walls == WALL_COLOR] = _GEN_BG
    try:
        no_wall_landing, _no_wall_events, _no_wall_path = _trace_ray(no_walls, start, dc)
    except AssertionError:
        return None
    no_wall_target = _target_for_column(found_containers, no_wall_landing)
    if no_wall_target != base_box or no_wall_target == actual_target:
        return None

    expected = grid.copy()
    for cell in arrow_cells:
        expected[cell] = _GEN_BG
    if want_container:
        top, bot, left, right = actual_box
        expected[top:bot, left + 1:right] = arrow_color
    else:
        expected[-1, :] = arrow_color

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None

    # The invisible path must leave no trace outside the explicitly changed target.
    allowed_changes = set(arrow_cells)
    if want_container:
        top, bot, left, right = actual_box
        allowed_changes |= {(r, c) for r in range(top, bot) for c in range(left + 1, right)}
    else:
        allowed_changes |= {(h - 1, c) for c in range(w)}
    changed = {tuple(map(int, rc)) for rc in np.argwhere(actual != grid)}
    if changed != allowed_changes or any(actual[r, c] != grid[r, c] for r, c in path if (r, c) not in allowed_changes):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied valid scenes, alternating caught and floor branches."""
    global _GEN_CALL_COUNT
    want_container = (_GEN_CALL_COUNT % 2 == 0)
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_ATTEMPTS):
        candidate = _make_candidate(want_container)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid invisible-ray container puzzle")
