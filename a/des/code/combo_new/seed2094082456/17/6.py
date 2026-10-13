import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
ANTENNA_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

GEN_ATTEMPTS = 400
TOWER_ATTEMPTS = 300
BLOCKER_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_tower_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    if not (2 <= len(comp) <= 6):
        raise AssertionError("tower base length must be between 2 and 6")

    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        orientation = "horizontal"
        ordered = sorted(comp, key=lambda rc: rc[1])
        row = ordered[0][0]
        expected = [(row, ordered[0][1] + i) for i in range(len(ordered))]
        if ordered != expected:
            raise AssertionError("horizontal tower base must be contiguous")
        antenna_checks = [
            ((row - 1, ordered[0][1]), (-1, 0)),
            ((row + 1, ordered[0][1]), (1, 0)),
            ((row - 1, ordered[-1][1]), (-1, 0)),
            ((row + 1, ordered[-1][1]), (1, 0)),
        ]
    elif len(cols) == 1:
        orientation = "vertical"
        ordered = sorted(comp, key=lambda rc: rc[0])
        col = ordered[0][1]
        expected = [(ordered[0][0] + i, col) for i in range(len(ordered))]
        if ordered != expected:
            raise AssertionError("vertical tower base must be contiguous")
        antenna_checks = [
            ((ordered[0][0], col - 1), (0, -1)),
            ((ordered[0][0], col + 1), (0, 1)),
            ((ordered[-1][0], col - 1), (0, -1)),
            ((ordered[-1][0], col + 1), (0, 1)),
        ]
    else:
        raise AssertionError("grey multi-cell component must be a straight base")

    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for (ar, ac), direction in antenna_checks:
        if not _in_bounds(grid.shape, ar, ac):
            continue
        color = int(grid[ar, ac])
        if color not in (BLACK, GREY):
            candidates.append(((ar, ac), direction, color))
    if len(candidates) != 1:
        raise AssertionError("each tower base must have exactly one antenna")

    antenna, direction, color = candidates[0]
    tower = {
        "base": tuple(ordered),
        "antenna": antenna,
        "direction": direction,
        "length": len(ordered),
        "color": color,
        "orientation": orientation,
    }
    return tower


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_components = _components4(g == GREY)
    towers: list[dict] = []
    receivers: set[tuple[int, int]] = set()
    for comp in grey_components:
        if len(comp) == 1:
            receivers.add(comp[0])
        else:
            towers.append(_parse_tower_component(g, comp))

    towers.sort(key=lambda t: (t["antenna"][0], t["antenna"][1], t["color"]))
    return towers, receivers


def _tower_path(tower: dict) -> list[tuple[int, int]]:
    ar, ac = tower["antenna"]
    dr, dc = tower["direction"]
    return [(ar + dr * step, ac + dc * step) for step in range(1, tower["length"] + 1)]


def _visible_ray_cells(grid: np.ndarray, tower: dict) -> list[tuple[int, int]]:
    visible: list[tuple[int, int]] = []
    for r, c in _tower_path(tower):
        if not _in_bounds(grid.shape, r, c):
            break
        cell = int(grid[r, c])
        if cell not in (BLACK, GREY):
            break
        visible.append((r, c))
    return visible


def _scene_stats(grid: np.ndarray) -> dict[str, bool]:
    towers, receivers = _extract_scene(grid)
    stats = {
        "blocked": False,
        "endpoint_hit": False,
        "pass_hit": False,
    }
    for tower in towers:
        full_path = _tower_path(tower)
        visible = _visible_ray_cells(grid, tower)
        if len(visible) < len(full_path):
            next_index = len(visible)
            if next_index < len(full_path):
                blocker = full_path[next_index]
                if _in_bounds(grid.shape, blocker[0], blocker[1]) and int(grid[blocker]) not in (BLACK, GREY):
                    stats["blocked"] = True
        if visible and visible[-1] in receivers:
            stats["endpoint_hit"] = True
        if any(cell in receivers for cell in visible[:-1]):
            stats["pass_hit"] = True
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire antenna-colored rays from grey signal towers and recolor any receivers they reach.

    Input structure:
    - The background is black.
    - Each signal tower has a straight grey base of length 2 to 6 and a single colored antenna pixel attached orthogonally to one base endpoint.
    - Additional isolated grey singletons are receivers, and they do not touch any tower.

    Rule:
    - For every tower, continue from the base endpoint through the antenna in the same direction and trace a ray whose length equals the base length.
    - Paint black cells on that ray with the antenna color, recolor any receiver cell the ray visits to that antenna color, and leave grey tower bases unchanged.
    - If the ray meets a non-black, non-grey pixel, stop before that blocker; the tower itself always stays unchanged.

    Key insight:
    - The grey base is both the object to detect and the hidden number source: its length gives the ray distance, while the antenna position at an endpoint reveals the firing direction.
    """
    g = np.asarray(grid, dtype=int)
    towers, receivers = _extract_scene(g)
    assert towers, "expected at least one signal tower"

    out = g.copy()
    for tower in towers:
        for r, c in _visible_ray_cells(g, tower):
            if (r, c) in receivers:
                out[r, c] = tower["color"]
            elif int(g[r, c]) == BLACK:
                out[r, c] = tower["color"]
    return out


def _build_tower(
    shape: tuple[int, int],
    antenna: tuple[int, int],
    direction: tuple[int, int],
    perp: tuple[int, int],
    length: int,
    color: int,
) -> dict | None:
    ar, ac = antenna
    dr, dc = direction
    pr, pc = perp

    endpoint = (ar - dr, ac - dc)
    if not _in_bounds(shape, endpoint[0], endpoint[1]):
        return None

    base = []
    for i in range(length):
        cell = (endpoint[0] + pr * i, endpoint[1] + pc * i)
        if not _in_bounds(shape, cell[0], cell[1]):
            return None
        base.append(cell)

    path = []
    for step in range(1, length + 1):
        cell = (ar + dr * step, ac + dc * step)
        if not _in_bounds(shape, cell[0], cell[1]):
            return None
        path.append(cell)

    return {
        "base": tuple(base),
        "antenna": antenna,
        "direction": direction,
        "length": length,
        "color": color,
        "path": tuple(path),
    }


def _tower_cells(tower: dict) -> set[tuple[int, int]]:
    return set(tower["base"]) | {tower["antenna"]}


def _cell_clear_with_buffer(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    return all(not _chebyshev_close(cell, other) for other in occupied)


def _tower_fits(
    tower: dict,
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    allow_antenna_on_reserved: bool = False,
) -> bool:
    cells = _tower_cells(tower)
    antenna = tower["antenna"]

    for cell in cells:
        if cell in occupied:
            return False
        if cell in reserved_paths and not (allow_antenna_on_reserved and cell == antenna):
            return False
        if not _cell_clear_with_buffer(cell, occupied):
            return False

    for cell in tower["path"]:
        if cell in occupied or cell in reserved_paths or cell in cells:
            return False
    return True


def _sample_random_tower(
    shape: tuple[int, int],
    color: int,
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(TOWER_ATTEMPTS):
        direction = random.choice(DIRS4)
        perp = random.choice(PERPS[direction])
        length = random.randint(2, 6)
        antenna = (random.randrange(h), random.randrange(w))
        tower = _build_tower(shape, antenna, direction, perp, length, color)
        if tower is None:
            continue
        if _tower_fits(tower, occupied, reserved_paths):
            return tower
    return None


def _sample_blocker_tower(
    shape: tuple[int, int],
    color: int,
    blocked_antenna: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict | None:
    options: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for direction in DIRS4:
        for perp in PERPS[direction]:
            for length in range(2, 7):
                options.append((direction, perp, length))
    random.shuffle(options)

    for direction, perp, length in options[:BLOCKER_ATTEMPTS]:
        tower = _build_tower(shape, blocked_antenna, direction, perp, length, color)
        if tower is None:
            continue
        if _tower_fits(tower, occupied, reserved_paths, allow_antenna_on_reserved=True):
            return tower
    return None


def _reachable_steps(grid: np.ndarray, tower: dict) -> list[tuple[int, tuple[int, int]]]:
    return [(i + 1, cell) for i, cell in enumerate(_visible_ray_cells(grid, tower))]


def _receiver_ok(
    cell: tuple[int, int],
    tower_cells: set[tuple[int, int]],
    receivers: set[tuple[int, int]],
) -> bool:
    if cell in tower_cells or cell in receivers:
        return False
    if any(_chebyshev_close(cell, other) for other in tower_cells):
        return False
    if any(_chebyshev_close(cell, other) for other in receivers):
        return False
    return True


def _choose_showcase_tower(
    reachable: list[list[tuple[int, tuple[int, int]]]],
    tower_cells: set[tuple[int, int]],
    receivers: set[tuple[int, int]],
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    candidates = []
    for idx, steps in enumerate(reachable):
        if len(steps) < 4:
            continue
        endpoint = steps[-1][1]
        mids = [cell for step, cell in steps[1:-1] if step <= len(steps) - 2]
        mids = [cell for cell in mids if _receiver_ok(cell, tower_cells, receivers)]
        if not mids or not _receiver_ok(endpoint, tower_cells, receivers):
            continue
        candidates.append((idx, mids, endpoint))

    if not candidates:
        return None

    idx, mids, endpoint = random.choice(candidates)
    return idx, random.choice(mids), endpoint


def _place_required_receivers(
    grid: np.ndarray,
    towers: list[dict],
    need_pass_hit: bool,
    need_endpoint_hit: bool,
) -> set[tuple[int, int]] | None:
    reachable = [_reachable_steps(grid, tower) for tower in towers]
    tower_cells = set().union(*(_tower_cells(tower) for tower in towers))
    receivers: set[tuple[int, int]] = set()

    if need_pass_hit and need_endpoint_hit:
        showcase = _choose_showcase_tower(reachable, tower_cells, receivers)
        if showcase is not None:
            _idx, middle, endpoint = showcase
            receivers.add(middle)
            receivers.add(endpoint)
            return receivers

    if need_pass_hit:
        pass_candidates = []
        for idx, steps in enumerate(reachable):
            for step, cell in steps[1:-1]:
                if step >= 2 and _receiver_ok(cell, tower_cells, receivers):
                    pass_candidates.append((idx, step, cell))
        if not pass_candidates:
            return None
        _idx, _step, cell = random.choice(pass_candidates)
        receivers.add(cell)

    if need_endpoint_hit:
        end_candidates = []
        for idx, steps in enumerate(reachable):
            if len(steps) < 2:
                continue
            cell = steps[-1][1]
            if _receiver_ok(cell, tower_cells, receivers):
                end_candidates.append((idx, cell))
        if not end_candidates:
            return None
        _idx, cell = random.choice(end_candidates)
        receivers.add(cell)

    return receivers


def _add_extra_hit_receivers(
    grid: np.ndarray,
    towers: list[dict],
    receivers: set[tuple[int, int]],
    extra_count: int,
) -> None:
    reachable = [_reachable_steps(grid, tower) for tower in towers]
    tower_cells = set().union(*(_tower_cells(tower) for tower in towers))
    candidates: list[tuple[int, int]] = []
    for steps in reachable:
        for step, cell in steps:
            if step < 2:
                continue
            if _receiver_ok(cell, tower_cells, receivers):
                candidates.append(cell)
    random.shuffle(candidates)
    for cell in candidates:
        if extra_count <= 0:
            break
        if _receiver_ok(cell, tower_cells, receivers):
            receivers.add(cell)
            extra_count -= 1


def _add_decoy_receivers(
    shape: tuple[int, int],
    tower_cells: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    receivers: set[tuple[int, int]],
    count: int,
) -> bool:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for r, c in cells:
        if count <= 0:
            return True
        cell = (r, c)
        if cell in tower_cells or cell in reserved_paths:
            continue
        if not _receiver_ok(cell, tower_cells, receivers):
            continue
        receivers.add(cell)
        count -= 1
    return count == 0


def _paint_scene(shape: tuple[int, int], towers: list[dict], receivers: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for tower in towers:
        for r, c in tower["base"]:
            grid[r, c] = GREY
        ar, ac = tower["antenna"]
        grid[ar, ac] = tower["color"]
    for r, c in receivers:
        assert grid[r, c] == BLACK
        grid[r, c] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    shape = (h, w)

    need_blocked = random.random() < 0.5
    need_pass_hit = random.random() < 0.5
    need_endpoint_hit = random.random() < 0.5
    if not need_pass_hit and not need_endpoint_hit:
        if random.random() < 0.5:
            need_pass_hit = True
        else:
            need_endpoint_hit = True

    total_towers = random.randint(4 if need_blocked else 3, 6)
    blocker_count = 1 if need_blocked else 0
    base_tower_count = total_towers - blocker_count
    colors = random.sample(ANTENNA_COLORS, total_towers)

    towers: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()

    for i in range(base_tower_count):
        tower = _sample_random_tower(shape, colors[i], occupied, reserved_paths)
        if tower is None:
            return None
        towers.append(tower)
        occupied.update(_tower_cells(tower))
        reserved_paths.update(tower["path"])

    if need_blocked:
        blocker_color = colors[-1]
        blocked_targets = [idx for idx, tower in enumerate(towers) if tower["length"] >= 3]
        random.shuffle(blocked_targets)
        blocker = None
        for target_idx in blocked_targets:
            path = list(towers[target_idx]["path"])
            step_choices = list(range(2, len(path) + 1))
            random.shuffle(step_choices)
            for step in step_choices:
                blocker_antenna = path[step - 1]
                blocker = _sample_blocker_tower(
                    shape,
                    blocker_color,
                    blocker_antenna,
                    occupied,
                    reserved_paths,
                )
                if blocker is not None:
                    break
            if blocker is not None:
                break
        if blocker is None:
            return None
        towers.append(blocker)
        occupied.update(_tower_cells(blocker))
        reserved_paths.update(blocker["path"])

    grid = _paint_scene(shape, towers, set())

    receivers = _place_required_receivers(grid, towers, need_pass_hit, need_endpoint_hit)
    if receivers is None:
        return None

    _add_extra_hit_receivers(grid, towers, receivers, extra_count=random.randint(0, 2))

    tower_cells = set().union(*(_tower_cells(tower) for tower in towers))
    decoy_count = random.randint(0, 2)
    if len(receivers) == 0:
        decoy_count = max(decoy_count, 1)
    if not _add_decoy_receivers(shape, tower_cells, reserved_paths, receivers, decoy_count):
        return None

    grid = _paint_scene(shape, towers, receivers)

    parsed_towers, parsed_receivers = _extract_scene(grid)
    if len(parsed_towers) != len(towers):
        return None
    if parsed_receivers != receivers:
        return None
    if len({tower["color"] for tower in parsed_towers}) != len(parsed_towers):
        return None

    stats = _scene_stats(grid)
    if need_blocked and not stats["blocked"]:
        return None
    if need_pass_hit and not stats["pass_hit"]:
        return None
    if need_endpoint_hit and not stats["endpoint_hit"]:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid signal-tower puzzle after {GEN_ATTEMPTS} attempts")
