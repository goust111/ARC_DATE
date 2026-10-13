import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
CYAN = 8

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 600
PATH_RETRY_LIMIT = 200
CONTAINER_RETRY_LIMIT = 150


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    """Return 4-connected components for one color."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def _find_start(grid: np.ndarray) -> tuple[int, int, int]:
    blue_cells = np.argwhere(grid == BLUE)
    assert blue_cells.shape == (1, 2), "expected exactly one blue marker"
    r0, c0 = map(int, blue_cells[0])
    assert r0 == 0, "blue marker must be on the top row"

    h, w = grid.shape
    cyan_cols = []
    for dc in (-1, 1):
        nc = c0 + dc
        if 0 <= nc < w and int(grid[0, nc]) == CYAN:
            cyan_cols.append(nc)
    assert len(cyan_cols) == 1, "expected exactly one adjacent cyan direction pixel"
    cyan_c = int(cyan_cols[0])

    initial_dc = 1 if cyan_c < c0 else -1
    return c0, initial_dc, cyan_c


def _simulate_path(
    h: int,
    w: int,
    start_c: int,
    initial_dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]], list[str], list[tuple[int, int]]]:
    """
    Simulate the diagonal descent.

    Returns:
    - landing column
    - visited path cells including the start
    - bounce reason per step ("wall" or "obstacle") when a bounce happened
    - forward targets examined before any bounce
    """
    c = int(start_c)
    dc = int(initial_dc)
    path = [(0, c)]
    bounces: list[str] = []
    forward_targets: list[tuple[int, int]] = []

    for r in range(1, h):
        target_c = c + dc
        forward_targets.append((r, target_c))
        bounced = False
        if target_c < 0 or target_c >= w:
            dc = -dc
            target_c = c + dc
            bounces.append("wall")
            bounced = True
        elif (r, target_c) in blocked:
            dc = -dc
            target_c = c + dc
            bounces.append("obstacle")
            bounced = True

        if target_c < 0 or target_c >= w or (r, target_c) in blocked:
            raise AssertionError("invalid trajectory: post-bounce target is still blocked")

        c = int(target_c)
        path.append((r, c))
        if not bounced:
            continue

    return c, path, bounces, forward_targets


def _find_containers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect red open-up U containers.

    Each returned tuple is (top, bottom, left, right), inclusive.
    """
    containers: list[tuple[int, int, int, int]] = []
    for cells in _components_of_color(grid, RED):
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        if right - left < 2 or bottom - top < 1:
            continue

        cell_set = set(cells)
        expected = {(bottom, c) for c in range(left, right + 1)}
        expected.update((r, left) for r in range(top, bottom + 1))
        expected.update((r, right) for r in range(top, bottom + 1))
        if cell_set != expected:
            continue

        if np.any(grid[top:bottom, left + 1 : right] != BG):
            continue

        containers.append((top, bottom, left, right))

    containers.sort(key=lambda box: (box[2], box[0], box[3]))
    return containers


def _container_selected(
    containers: list[tuple[int, int, int, int]],
    landing_c: int,
) -> tuple[int, int, int, int] | None:
    matches = [box for box in containers if box[2] < landing_c < box[3]]
    assert len(matches) <= 1, "landing column must select at most one container cavity"
    return matches[0] if matches else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bounce the top blue marble down to choose a landing column and fill the selected red container.

    Input structure:
    - A black background contains exactly one blue marble pixel on the top row and one
      adjacent cyan direction pixel on its left or right.
    - Grey singleton pixels inside the grid act as reflective blockers.
    - The bottom region contains one or more red upward-open U containers whose interiors are empty.

    Rule:
    - Erase the blue and cyan top-row markers in the output.
    - Start from the blue pixel and move one row downward and one column sideways each step;
      the initial sideways direction is away from the cyan pixel.
    - If the next diagonal target would leave the grid horizontally or land on a grey pixel,
      reverse only the horizontal direction and then take that downward-diagonal step.
    - When the path reaches the bottom row, fill the entire interior of the unique container
      whose cavity columns contain the landing column; if no container cavity contains it,
      color only the landing bottom cell blue.

    Key insight:
    - The answer depends on the full reflected diagonal trajectory: side walls and grey
      blockers both redirect the same moving marble, and only the final landing column matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    start_c, initial_dc, cyan_c = _find_start(g)
    blocked = {tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)}
    landing_c, _path, _bounces, _targets = _simulate_path(h, w, start_c, initial_dc, blocked)
    containers = _find_containers(g)
    chosen = _container_selected(containers, landing_c)

    out = g.copy()
    out[0, start_c] = BG
    out[0, cyan_c] = BG

    if chosen is None:
        out[h - 1, landing_c] = BLUE
    else:
        top, bottom, left, right = chosen
        out[top:bottom, left + 1 : right] = BLUE
    return out


def _sample_shape(mode: str) -> tuple[int, int]:
    if "wall" in mode:
        return random.randint(14, 22), random.randint(8, 13)
    return random.randint(12, 21), random.randint(10, 18)


def _build_random_trajectory(
    h: int,
    w: int,
    require_obstacle: bool,
    require_wall: bool,
    forbid_obstacle: bool,
) -> tuple[int, int, set[tuple[int, int]], list[tuple[int, int]], list[str], list[tuple[int, int]]] | None:
    for _ in range(PATH_RETRY_LIMIT):
        start_c = random.randint(1, w - 2)
        initial_dc = random.choice((-1, 1))
        blocked: set[tuple[int, int]] = set()
        c = start_c
        dc = initial_dc
        path = [(0, c)]
        bounce_reasons: list[str] = []
        forward_targets: list[tuple[int, int]] = []

        for r in range(1, h):
            forward_c = c + dc
            forward_targets.append((r, forward_c))
            if forward_c < 0 or forward_c >= w:
                dc = -dc
                target_c = c + dc
                bounce_reasons.append("wall")
            else:
                can_place_obstacle = (
                    not forbid_obstacle
                    and 1 <= r < h - 1
                    and 0 <= c - dc < w
                    and (r, forward_c) not in blocked
                )
                place_obstacle = False
                if can_place_obstacle:
                    obstacle_bias = 0.35 if require_obstacle else 0.12
                    place_obstacle = random.random() < obstacle_bias
                if place_obstacle:
                    blocked.add((r, forward_c))
                    dc = -dc
                    target_c = c + dc
                    bounce_reasons.append("obstacle")
                else:
                    target_c = forward_c

            if target_c < 0 or target_c >= w or (r, target_c) in blocked:
                break

            c = target_c
            path.append((r, c))
        else:
            if require_obstacle and "obstacle" not in bounce_reasons:
                continue
            if require_wall and "wall" not in bounce_reasons:
                continue
            if forbid_obstacle and "obstacle" in bounce_reasons:
                continue
            if c == start_c:
                continue
            try:
                opposite_landing, _opp_path, _opp_bounces, _opp_targets = _simulate_path(
                    h, w, start_c, -initial_dc, blocked
                )
            except AssertionError:
                continue
            if opposite_landing == c:
                continue
            return start_c, initial_dc, blocked, path, bounce_reasons, forward_targets

    return None


def _candidate_containers(
    h: int,
    w: int,
    path_cols: list[int],
    landing_c: int,
    start_c: int,
    fill_container: bool,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    selected: list[tuple[int, int, int, int]] = []
    distractors: list[tuple[int, int, int, int]] = []

    for interior_h in range(2, 5):
        top = h - 1 - interior_h
        bottom = h - 1
        for interior_w in range(2, 6):
            span = interior_w + 2
            for left in range(0, w - span + 1):
                right = left + span - 1
                hits_wall = any(path_cols[r] in (left, right) for r in range(top, bottom + 1))
                if hits_wall:
                    continue

                box = (top, bottom, left, right)
                if left < landing_c < right and fill_container and not (left < start_c < right):
                    selected.append(box)
                if any(left <= path_cols[r] <= right for r in range(top, bottom + 1)):
                    continue
                if left < landing_c < right:
                    continue
                distractors.append(box)

    random.shuffle(selected)
    random.shuffle(distractors)
    return selected, distractors


def _non_overlapping(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return box_a[3] + 1 < box_b[2] or box_b[3] + 1 < box_a[2]


def _place_containers(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    landing_c: int,
    start_c: int,
    fill_container: bool,
) -> list[tuple[int, int, int, int]] | None:
    path_cols = [c for _r, c in path]
    hit_candidates, miss_candidates = _candidate_containers(
        h, w, path_cols, landing_c, start_c, fill_container
    )

    for _ in range(CONTAINER_RETRY_LIMIT):
        target_count = random.randint(2, 4)
        boxes: list[tuple[int, int, int, int]] = []

        if fill_container:
            if not hit_candidates:
                return None
            chosen = random.choice(hit_candidates)
            boxes.append(chosen)

        pool = miss_candidates[:]
        random.shuffle(pool)
        for box in pool:
            if all(_non_overlapping(box, other) for other in boxes):
                boxes.append(box)
            if len(boxes) == target_count:
                break

        if len(boxes) >= 2 and len(boxes) == target_count:
            boxes.sort(key=lambda box: box[2])
            return boxes

    return None


def _stamp_container(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    top, bottom, left, right = box
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED
    grid[bottom, left : right + 1] = RED


def _build_expected(
    grid: np.ndarray,
    landing_c: int,
    containers: list[tuple[int, int, int, int]],
    start_c: int,
    cyan_c: int,
) -> np.ndarray:
    out = grid.copy()
    out[0, start_c] = BG
    out[0, cyan_c] = BG
    chosen = _container_selected(containers, landing_c)
    if chosen is None:
        out[-1, landing_c] = BLUE
    else:
        top, bottom, left, right = chosen
        out[top:bottom, left + 1 : right] = BLUE
    return out


def _add_decoy_obstacles(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    forward_targets: list[tuple[int, int]],
    containers: list[tuple[int, int, int, int]],
    blocked: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    forbidden = set(path)
    forbidden.update(blocked)
    forbidden.update((r, c) for r, c in forward_targets if 0 <= c < w)
    for top, bottom, left, right in containers:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                forbidden.add((r, c))

    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(w)
        if (r, c) not in forbidden and int(grid[r, c]) == BG
    ]
    random.shuffle(candidates)
    count = random.randint(0, min(6, len(candidates)))
    for r, c in candidates[:count]:
        grid[r, c] = GRAY


def _validate_sample(
    grid: np.ndarray,
    containers: list[tuple[int, int, int, int]],
    blocked: set[tuple[int, int]],
    fill_container: bool,
    bounce_reasons: list[str],
) -> bool:
    h, w = grid.shape
    start_c, initial_dc, cyan_c = _find_start(grid)
    actual_blocked = {tuple(map(int, rc)) for rc in np.argwhere(grid == GRAY)}
    if not blocked.issubset(actual_blocked):
        return False

    landing_c, path, _solver_bounces, _targets = _simulate_path(h, w, start_c, initial_dc, actual_blocked)
    expected = _build_expected(grid, landing_c, containers, start_c, cyan_c)

    if not np.array_equal(main(grid), expected):
        return False

    if _find_containers(grid) != containers:
        return False

    chosen = _container_selected(containers, landing_c)
    if fill_container != (chosen is not None):
        return False

    try:
        opposite_landing, _opp_path, _opp_bounces, _opp_targets = _simulate_path(
            h, w, start_c, -initial_dc, actual_blocked
        )
        opposite_choice = _container_selected(containers, opposite_landing)
        if opposite_landing == landing_c and opposite_choice == chosen:
            return False
    except AssertionError:
        pass

    if "obstacle" in bounce_reasons:
        try:
            no_obstacle_landing, _p2, _b2, _t2 = _simulate_path(h, w, start_c, initial_dc, set())
            no_obstacle_choice = _container_selected(containers, no_obstacle_landing)
            if no_obstacle_landing == landing_c and no_obstacle_choice == chosen:
                return False
        except AssertionError:
            pass

    if fill_container:
        top, bottom, left, right = chosen
        if left < start_c < right:
            return False
        if not np.all(expected[top:bottom, left + 1 : right] == BLUE):
            return False
    else:
        if not (int(expected[-1, landing_c]) == BLUE and np.count_nonzero(expected == BLUE) == 1):
            return False

    if landing_c == start_c:
        return False
    if len(containers) < 2:
        return False
    if any(int(grid[r, c]) != GRAY for r, c in actual_blocked):
        return False
    if np.any(grid[0, 1:-1] == RED):
        return False
    if any(path[r][1] in (box[2], box[3]) for box in containers for r in range(box[0], box[1] + 1)):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(("hit_obstacle", "hit_wall", "miss_obstacle", "miss_wall"))
    fill_container = mode.startswith("hit")
    require_obstacle = "obstacle" in mode
    require_wall = "wall" in mode
    forbid_obstacle = mode.endswith("wall")

    h, w = _sample_shape(mode)
    trajectory = _build_random_trajectory(h, w, require_obstacle, require_wall, forbid_obstacle)
    if trajectory is None:
        return None

    start_c, initial_dc, blocked, path, bounce_reasons, forward_targets = trajectory
    landing_c = path[-1][1]
    cyan_c = start_c - initial_dc
    if not (0 <= cyan_c < w):
        return None

    containers = _place_containers(h, w, path, landing_c, start_c, fill_container)
    if containers is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in blocked:
        grid[r, c] = GRAY
    for box in containers:
        _stamp_container(grid, box)
    grid[0, start_c] = BLUE
    grid[0, cyan_c] = CYAN
    _add_decoy_obstacles(grid, path, forward_targets, containers, blocked)

    if not _validate_sample(grid, containers, blocked, fill_container, bounce_reasons):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
