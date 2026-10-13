from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r, c) lies inside a grid of shape (rows, cols)."""
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _distance_color(distance: int) -> int:
    mod = distance % 3
    if mod == 1:
        return BLUE
    if mod == 2:
        return GREEN
    return YELLOW


def _track_cells(marker: tuple[int, int], track_dir: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = marker
    dr, dc = track_dir
    return [(r + dr * step, c + dc * step) for step in range(1, 4)]


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED))


def _track_direction(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    dirs = []
    for dr, dc in DIRS4:
        cells = _track_cells((r, c), (dr, dc))
        if all(in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY for rr, cc in cells):
            dirs.append((dr, dc))
    assert len(dirs) == 1, "each marker must have exactly one 3-cell grey track"
    return dirs[0]


def _trace_current_state(
    state: np.ndarray,
    marker: tuple[int, int],
    ray_dir: tuple[int, int],
    *,
    erase_obstacle: bool,
) -> dict:
    r, c = marker
    dr, dc = ray_dir
    rr = r + dr
    cc = c + dc
    distance = 1
    path: list[tuple[int, int]] = []

    while in_bounds(state.shape, rr, cc):
        value = int(state[rr, cc])
        if value == BLACK:
            path.append((rr, cc))
            state[rr, cc] = _distance_color(distance)
            rr += dr
            cc += dc
            distance += 1
            continue
        if value == ORANGE:
            if erase_obstacle:
                state[rr, cc] = BLACK
            return {
                "marker": marker,
                "ray_dir": ray_dir,
                "path": path,
                "stop_kind": "obstacle",
                "stop_cell": (rr, cc),
            }
        return {
            "marker": marker,
            "ray_dir": ray_dir,
            "path": path,
            "stop_kind": "blocker",
            "stop_cell": (rr, cc),
        }

    return {
        "marker": marker,
        "ray_dir": ray_dir,
        "path": path,
        "stop_kind": "boundary",
        "stop_cell": None,
    }


def _simulate(grid: np.ndarray, *, collect: bool) -> tuple[np.ndarray, list[dict]]:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    events: list[dict] = []

    for marker in _marker_positions(base):
        track_dir = _track_direction(base, *marker)
        ray_dir = (-track_dir[0], -track_dir[1])
        event = _trace_current_state(out, marker, ray_dir, erase_obstacle=True)
        if collect:
            event["track_dir"] = track_dir
            events.append(event)

    return out, events


def _single_trace(grid: np.ndarray, marker: tuple[int, int]) -> dict:
    state = np.asarray(grid, dtype=int).copy()
    track_dir = _track_direction(state, *marker)
    ray_dir = (-track_dir[0], -track_dir[1])
    event = _trace_current_state(state, marker, ray_dir, erase_obstacle=False)
    event["track_dir"] = track_dir
    return event


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot reverse rays from each tracked red marker, cycle-color the traversed cells, and erase the first obstacle they hit.

    Input structure:
    - A black background contains 2 to 4 red single-pixel markers.
    - Each marker has exactly one straight 3-pixel grey track attached in one cardinal direction.
    - Orange single-pixel obstacles are scattered elsewhere; markers, tracks, and obstacles do not overlap.

    Rule:
    - Process the red markers in row-major order.
    - For each marker, read the grey track direction and shoot a ray in the opposite direction.
    - Paint each black ray cell by distance from the marker: distance 1 -> blue, 2 -> green, 3 -> yellow, then repeat every 3 cells.
    - Stop at the first non-black cell or at the grid edge.
    - If the first blocker is an orange obstacle, erase that obstacle to black without coloring that obstacle cell during the same ray.
    - If the blocker is a marker, a grey track, or an already-painted ray cell, leave it unchanged and stop immediately before it.

    Key insight:
    - The grey 3-pixel stub only encodes direction; the actual action happens on the reverse ray, and earlier rays can erase obstacles that extend later rays.
    """
    return _simulate(np.asarray(grid, dtype=int), collect=False)[0]


def _place_marker(grid: np.ndarray, marker: tuple[int, int], track_dir: tuple[int, int]) -> bool:
    cells = [marker, *_track_cells(marker, track_dir)]
    if not all(in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK for r, c in cells):
        return False
    r, c = marker
    grid[r, c] = RED
    for rr, cc in _track_cells(marker, track_dir):
        grid[rr, cc] = GREY
    return True


def _boundary_clearance(shape: tuple[int, int], start: tuple[int, int], ray_dir: tuple[int, int]) -> int:
    r, c = start
    dr, dc = ray_dir
    steps = 0
    rr = r + dr
    cc = c + dc
    while in_bounds(shape, rr, cc):
        steps += 1
        rr += dr
        cc += dc
    return steps


def _cells_distinct(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == len(set(cells))


def _build_base_scene(height: int, width: int) -> tuple[np.ndarray, dict] | None:
    for _ in range(300):
        grid = np.zeros((height, width), dtype=int)
        shared = (random.randrange(height), random.randrange(width))
        earlier_ray: tuple[int, int]
        later_ray: tuple[int, int]
        pattern = random.choice(["horizontal_then_up", "down_then_horizontal"])
        if pattern == "horizontal_then_up":
            earlier_ray = random.choice([(0, -1), (0, 1)])
            later_ray = (-1, 0)
        else:
            earlier_ray = (1, 0)
            later_ray = random.choice([(0, -1), (0, 1)])

        earlier_dist = random.randint(2, 5)
        later_dist = random.randint(2, 5)
        later_stop_mode = random.choice(["obstacle", "boundary"])
        stopper_dist = random.randint(2, 5)

        earlier_marker = (
            shared[0] - earlier_ray[0] * earlier_dist,
            shared[1] - earlier_ray[1] * earlier_dist,
        )
        later_marker = (
            shared[0] - later_ray[0] * later_dist,
            shared[1] - later_ray[1] * later_dist,
        )
        earlier_track = (-earlier_ray[0], -earlier_ray[1])
        later_track = (-later_ray[0], -later_ray[1])

        if not _place_marker(grid, earlier_marker, earlier_track):
            continue
        if not _place_marker(grid, later_marker, later_track):
            continue

        essential_obstacles = [shared]
        if later_stop_mode == "obstacle":
            stopper = (
                shared[0] + later_ray[0] * stopper_dist,
                shared[1] + later_ray[1] * stopper_dist,
            )
            if not in_bounds(grid.shape, *stopper) or int(grid[stopper]) != BLACK:
                continue
            essential_obstacles.append(stopper)
        else:
            stopper = None
            if _boundary_clearance(grid.shape, shared, later_ray) < 2:
                continue

        all_fixed = [shared, earlier_marker, later_marker]
        all_fixed.extend(_track_cells(earlier_marker, earlier_track))
        all_fixed.extend(_track_cells(later_marker, later_track))
        if stopper is not None:
            all_fixed.append(stopper)
        if not _cells_distinct(all_fixed):
            continue
        if not all(in_bounds(grid.shape, r, c) for r, c in all_fixed):
            continue

        for obstacle in essential_obstacles:
            if int(grid[obstacle]) != BLACK:
                break
            grid[obstacle] = ORANGE
        else:
            order = _marker_positions(grid)
            if order.index(earlier_marker) >= order.index(later_marker):
                continue
            out, events = _simulate(grid, collect=True)
            event_by_marker = {event["marker"]: event for event in events}
            earlier_event = event_by_marker[earlier_marker]
            later_event = event_by_marker[later_marker]
            later_before = _single_trace(grid, later_marker)
            if earlier_event["stop_kind"] != "obstacle" or earlier_event["stop_cell"] != shared:
                continue
            if later_before["stop_kind"] != "obstacle" or later_before["stop_cell"] != shared:
                continue
            if shared not in later_event["path"]:
                continue
            if len(later_event["path"]) <= len(later_before["path"]):
                continue
            if stopper is None:
                if later_event["stop_kind"] != "boundary":
                    continue
            else:
                if later_event["stop_kind"] != "obstacle" or later_event["stop_cell"] != stopper:
                    continue

            horizontal_owner = earlier_marker if earlier_track[0] == 0 else later_marker
            return grid, {
                "shared": shared,
                "stopper": stopper,
                "earlier_marker": earlier_marker,
                "later_marker": later_marker,
                "horizontal_owner": horizontal_owner,
                "out": out,
            }
    return None


def _can_use_cells(grid: np.ndarray, future: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(
        in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK and int(future[r, c]) == BLACK
        for r, c in cells
    )


def _try_add_track_blocker(grid: np.ndarray, info: dict) -> tuple[np.ndarray, dict] | None:
    future = info["out"]
    owner = info["horizontal_owner"]
    owner_track = _track_direction(grid, *owner)
    targets = _track_cells(owner, owner_track)
    random.shuffle(targets)

    for target in targets:
        for dist in random.sample([2, 3, 4], k=3):
            marker = (target[0] + dist, target[1])
            track_dir = (1, 0)
            if not _can_use_cells(grid, future, [marker, *_track_cells(marker, track_dir)]):
                continue
            corridor = [(rr, target[1]) for rr in range(target[0] + 1, marker[0])]
            if not _can_use_cells(grid, future, corridor):
                continue

            temp = grid.copy()
            if not _place_marker(temp, marker, track_dir):
                continue
            try:
                _out, events = _simulate(temp, collect=True)
            except AssertionError:
                continue
            event = next(item for item in events if item["marker"] == marker)
            if event["stop_kind"] == "blocker" and event["stop_cell"] == target and len(event["path"]) >= 1:
                return temp, {"marker": marker, "target": target}
    return None


def _find_boundary_candidate(grid: np.ndarray, future: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    height, width = grid.shape
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    for marker in cells:
        if int(grid[marker]) != BLACK or int(future[marker]) != BLACK:
            continue
        for ray_dir in random.sample(DIRS4, k=len(DIRS4)):
            track_dir = (-ray_dir[0], -ray_dir[1])
            track_cells = _track_cells(marker, track_dir)
            if not _can_use_cells(grid, future, [marker, *track_cells]):
                continue

            rr = marker[0] + ray_dir[0]
            cc = marker[1] + ray_dir[1]
            path: list[tuple[int, int]] = []
            while in_bounds(grid.shape, rr, cc):
                if int(grid[rr, cc]) != BLACK or int(future[rr, cc]) != BLACK:
                    path = []
                    break
                path.append((rr, cc))
                rr += ray_dir[0]
                cc += ray_dir[1]
            if len(path) >= 2:
                candidates.append((marker, track_dir))
    return candidates


def _try_add_boundary_marker(grid: np.ndarray) -> tuple[np.ndarray, dict] | None:
    future = main(grid)
    candidates = _find_boundary_candidate(grid, future)
    random.shuffle(candidates)

    for marker, track_dir in candidates:
        temp = grid.copy()
        if not _place_marker(temp, marker, track_dir):
            continue
        try:
            _out, events = _simulate(temp, collect=True)
        except AssertionError:
            continue
        event = next(item for item in events if item["marker"] == marker)
        if event["stop_kind"] == "boundary" and len(event["path"]) >= 2:
            return temp, {"marker": marker}
    return None


def _isolated_black_candidates(grid: np.ndarray, future: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    candidates = []
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) != BLACK or int(future[r, c]) != BLACK:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(height, r + 2)):
                for cc in range(max(0, c - 1), min(width, c + 2)):
                    if (rr, cc) != (r, c) and int(grid[rr, cc]) != BLACK:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r, c))
    return candidates


def _add_distractors(grid: np.ndarray) -> np.ndarray | None:
    current_obstacles = int(np.count_nonzero(grid == ORANGE))
    target_obstacles = random.randint(max(2, current_obstacles), 5)
    needed = target_obstacles - current_obstacles
    if needed <= 0:
        return grid

    future = main(grid)
    candidates = _isolated_black_candidates(grid, future)
    if len(candidates) < needed:
        candidates = [(int(r), int(c)) for r, c in np.argwhere((grid == BLACK) & (future == BLACK))]
    if len(candidates) < needed:
        return None

    temp = grid.copy()
    for r, c in random.sample(candidates, needed):
        temp[r, c] = ORANGE
    return temp


def _validate_scene(grid: np.ndarray, info: dict, extras: dict) -> bool:
    markers = _marker_positions(grid)
    if not (2 <= len(markers) <= 4):
        return False
    obstacle_count = int(np.count_nonzero(grid == ORANGE))
    if not (2 <= obstacle_count <= 5):
        return False

    try:
        out, events = _simulate(grid, collect=True)
    except AssertionError:
        return False

    event_by_marker = {event["marker"]: event for event in events}
    earlier = event_by_marker[info["earlier_marker"]]
    later = event_by_marker[info["later_marker"]]
    later_before = _single_trace(grid, info["later_marker"])

    if earlier["stop_kind"] != "obstacle" or earlier["stop_cell"] != info["shared"]:
        return False
    if later_before["stop_kind"] != "obstacle" or later_before["stop_cell"] != info["shared"]:
        return False
    if info["shared"] not in later["path"]:
        return False
    if len(later["path"]) <= len(later_before["path"]):
        return False

    stopper = info["stopper"]
    if stopper is None:
        if later["stop_kind"] != "boundary":
            return False
    else:
        if later["stop_kind"] != "obstacle" or later["stop_cell"] != stopper:
            return False

    if "track_blocker" in extras:
        marker = extras["track_blocker"]["marker"]
        target = extras["track_blocker"]["target"]
        event = event_by_marker.get(marker)
        if event is None or event["stop_kind"] != "blocker" or event["stop_cell"] != target:
            return False

    if "boundary_marker" in extras:
        marker = extras["boundary_marker"]["marker"]
        event = event_by_marker.get(marker)
        if event is None or event["stop_kind"] != "boundary":
            return False

    return np.array_equal(main(grid), out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(16, 22)
        width = random.randint(16, 22)
        built = _build_base_scene(height, width)
        if built is None:
            continue
        grid, info = built
        extras: dict[str, dict] = {}

        if random.random() < 0.7 and len(_marker_positions(grid)) < 4:
            added = _try_add_track_blocker(grid, info)
            if added is not None:
                grid, extras["track_blocker"] = added
                info["out"] = main(grid)

        if random.random() < 0.5 and len(_marker_positions(grid)) < 4:
            added = _try_add_boundary_marker(grid)
            if added is not None:
                grid, extras["boundary_marker"] = added
                info["out"] = main(grid)

        grid = _add_distractors(grid)
        if grid is None:
            continue

        if _validate_scene(grid, info, extras):
            return grid

    raise RuntimeError("failed to generate a valid cascade reverse-ray input")
