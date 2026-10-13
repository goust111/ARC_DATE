from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
MARKER_COLORS = (2, 3, 4, 6)
CORNER_ORDER = ("tl", "tr", "bl", "br")

CORNER_SPECS = {
    "tl": {
        "direction": (1, 1),
        "shape": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "anchor_rel": (1, 1),
    },
    "tr": {
        "direction": (1, -1),
        "shape": ((0, 0), (1, 0), (2, 0)),
        "anchor_rel": (2, 0),
    },
    "bl": {
        "direction": (-1, 1),
        "shape": ((0, 0), (0, 1), (0, 2)),
        "anchor_rel": (0, 2),
    },
    "br": {
        "direction": (-1, -1),
        "shape": ((0, 0), (1, 0), (1, 1)),
        "anchor_rel": (0, 0),
    },
}

GEN_HEIGHT_RANGE = (11, 15)
GEN_WIDTH_RANGE = (11, 15)
GEN_BLOCKER_COUNT_RANGE = (3, 5)
GEN_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _corner_position(shape: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = shape
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, w - 1
    if name == "bl":
        return h - 1, 0
    if name == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner: {name}")


def _ray_cells(shape: tuple[int, int], name: str) -> list[tuple[int, int]]:
    r, c = _corner_position(shape, name)
    dr, dc = CORNER_SPECS[name]["direction"]
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _stamp_cells(name: str, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    off_r, off_c = CORNER_SPECS[name]["anchor_rel"]
    top = ar - off_r
    left = ac - off_c
    return [(top + dr, left + dc) for dr, dc in CORNER_SPECS[name]["shape"]]


def _trace_anchor(grid: np.ndarray, name: str) -> tuple[int, int]:
    shape = tuple(map(int, grid.shape))
    r, c = _corner_position(shape, name)
    dr, dc = CORNER_SPECS[name]["direction"]
    last_empty: tuple[int, int] | None = None
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            raise AssertionError("expected a blue obstacle on every active ray")
        if int(grid[r, c]) == WALL:
            assert last_empty is not None, "obstacle cannot touch the corner marker"
            return last_empty
        last_empty = (r, c)


def _active_markers(grid: np.ndarray) -> list[tuple[str, int]]:
    g = np.asarray(grid, dtype=int)
    seen_colors: set[int] = set()
    active: list[tuple[str, int]] = []
    for name in CORNER_ORDER:
        r, c = _corner_position(g.shape, name)
        color = int(g[r, c])
        if color == BG:
            continue
        assert color in MARKER_COLORS, "corner markers must use the allowed palette"
        assert color not in seen_colors, "corner marker colors must be distinct"
        seen_colors.add(color)
        active.append((name, color))
    assert 2 <= len(active) <= 3, "expected 2 or 3 active corner markers"
    return active


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(comp)
    return components


def _fixed_offset_output(grid: np.ndarray) -> np.ndarray:
    """Guard against a degenerate generator that always places stamps at a constant offset."""
    out = np.asarray(grid, dtype=int).copy()
    for name, color in _active_markers(out):
        r0, c0 = _corner_position(out.shape, name)
        dr, dc = CORNER_SPECS[name]["direction"]
        out[r0, c0] = BG
        anchor = (r0 + 3 * dr, c0 + 3 * dc)
        for r, c in _stamp_cells(name, anchor):
            if _in_bounds(out.shape, r, c):
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace corner markers by corner-specific colored stamps whose anchor is found by a diagonal obstacle-seeking ray.

    Input structure:
    - The background is black, the interior contains 3 to 5 blue obstacle components, and 2 or 3 grid corners contain distinct marker pixels.
    - Each active corner marker sits exactly on one corner cell and uses one of the colors {2,3,4,6}.
    - For every active corner, its 45-degree inward diagonal ray encounters at least one blue obstacle pixel before leaving the grid.

    Rule:
    - Read each active corner marker; the corner position determines both the template shape and the inward ray direction.
    - March along that diagonal ray until the first blue obstacle pixel; the anchor cell is the last non-blue ray cell before the hit.
    - Erase the original corner marker, then stamp the corner's template in the marker color with the corner-specific anchor alignment.
    - Preserve all blue obstacles, and apply multiple stamps in reading order from top-left to bottom-right so later stamps would overwrite earlier ones.

    Key insight:
    - The corner does double duty: it chooses which template to stamp and which diagonal to probe, while the obstacle layout decides where that template finally lands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for name, color in _active_markers(g):
        r0, c0 = _corner_position(g.shape, name)
        out[r0, c0] = BG
        anchor = _trace_anchor(g, name)
        for r, c in _stamp_cells(name, anchor):
            assert _in_bounds(g.shape, r, c), "stamp must stay in bounds"
            assert int(g[r, c]) != WALL, "stamp must not cover blue obstacles"
            out[r, c] = color
    return out


def _is_central(shape: tuple[int, int], cell: tuple[int, int], margin: int = 2) -> bool:
    h, w = shape
    r, c = cell
    return margin <= r < h - margin and margin <= c < w - margin


def _far_from_cells(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    margin: int,
) -> bool:
    for r, c in cells:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _seed_candidates(
    shape: tuple[int, int],
    active_corners: list[str],
) -> dict[str, list[dict[str, object]]]:
    ray_sets = {name: set(_ray_cells(shape, name)) for name in active_corners}
    candidates: dict[str, list[dict[str, object]]] = {}

    for name in active_corners:
        own_ray = _ray_cells(shape, name)
        other_rays = set().union(*(ray_sets[other] for other in active_corners if other != name))
        options: list[dict[str, object]] = []
        for idx in range(3, len(own_ray) - 1):
            obstacle = own_ray[idx]
            anchor = own_ray[idx - 1]
            stamp = _stamp_cells(name, anchor)
            if obstacle in other_rays or not _is_central(shape, obstacle):
                continue
            if not all(_in_bounds(shape, r, c) for r, c in stamp):
                continue
            options.append(
                {
                    "obstacle": obstacle,
                    "anchor": anchor,
                    "stamp": tuple(stamp),
                    "distance": idx + 1,
                }
            )
        random.shuffle(options)
        candidates[name] = options
    return candidates


def _choose_seed_plan(
    shape: tuple[int, int],
    active_corners: list[str],
) -> dict[str, dict[str, object]] | None:
    candidates = _seed_candidates(shape, active_corners)
    if any(not candidates[name] for name in active_corners):
        return None

    order = sorted(active_corners, key=lambda name: len(candidates[name]))
    chosen: dict[str, dict[str, object]] = {}
    used_stamps: set[tuple[int, int]] = set()
    used_seeds: set[tuple[int, int]] = set()

    def search(i: int) -> bool:
        if i == len(order):
            distances = {int(chosen[name]["distance"]) for name in order}
            return len(distances) >= 2

        name = order[i]
        for cand in candidates[name]:
            seed = tuple(cand["obstacle"])
            stamp = set(cand["stamp"])
            if seed in used_stamps or stamp & used_stamps:
                continue
            if not _far_from_cells(stamp, used_stamps, margin=0):
                continue
            if not _far_from_cells([seed], used_seeds, margin=1):
                continue
            chosen[name] = cand
            used_stamps.update(stamp)
            used_seeds.add(seed)
            if search(i + 1):
                return True
            del chosen[name]
            used_seeds.remove(seed)
            used_stamps.difference_update(stamp)
        return False

    if not search(0):
        return None
    return chosen


def _forbidden_ray_cells(shape: tuple[int, int], active_corners: list[str]) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for name in active_corners:
        forbidden.update(_ray_cells(shape, name))
    return forbidden


def _grow_component(
    shape: tuple[int, int],
    seed: tuple[int, int],
    target_size: int,
    blocked: set[tuple[int, int]],
    all_obstacles: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    comp = [seed]
    comp_set = {seed}
    while len(comp) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in comp:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not _in_bounds(shape, nr, nc):
                    continue
                if (nr, nc) in blocked or (nr, nc) in all_obstacles or (nr, nc) in comp_set:
                    continue

                conflict = False
                for rr in range(nr - 1, nr + 2):
                    for cc in range(nc - 1, nc + 2):
                        if (rr, cc) in all_obstacles and (rr, cc) not in comp_set:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nxt = random.choice(frontier)
        comp.append(nxt)
        comp_set.add(nxt)
    return comp


def _place_distractor_components(
    shape: tuple[int, int],
    count: int,
    blocked: set[tuple[int, int]],
    all_obstacles: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    h, w = shape
    comps: list[list[tuple[int, int]]] = []
    attempts = 0
    while len(comps) < count and attempts < 300:
        attempts += 1
        seed = (random.randint(2, h - 3), random.randint(2, w - 3))
        if seed in blocked or seed in all_obstacles:
            continue

        isolated = True
        for rr in range(seed[0] - 1, seed[0] + 2):
            for cc in range(seed[1] - 1, seed[1] + 2):
                if (rr, cc) in all_obstacles:
                    isolated = False
                    break
            if not isolated:
                break
        if not isolated:
            continue

        size = random.randint(1, 3)
        comp = _grow_component(shape, seed, size, blocked, all_obstacles)
        if comp is None:
            continue
        comps.append(comp)
        all_obstacles.update(comp)
    return comps if len(comps) == count else None


def _validate_instance(
    grid: np.ndarray,
    active_info: dict[str, dict[str, object]],
    distractor_count: int,
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    active = _active_markers(grid)
    if len(active) != len(active_info):
        return False

    wall_components = _components_of_color(grid, WALL)
    if not (GEN_BLOCKER_COUNT_RANGE[0] <= len(wall_components) <= GEN_BLOCKER_COUNT_RANGE[1]):
        return False
    if any(not (1 <= len(comp) <= 3) for comp in wall_components):
        return False

    if len(wall_components) - len(active_info) != distractor_count:
        return False

    for name, color in active:
        spec = active_info[name]
        anchor = _trace_anchor(grid, name)
        if anchor != tuple(spec["anchor"]):
            return False
        seed = tuple(spec["obstacle"])
        dr, dc = CORNER_SPECS[name]["direction"]
        if seed != (anchor[0] + dr, anchor[1] + dc):
            return False
        for r, c in spec["stamp"]:
            if int(out[r, c]) != color:
                return False

    if np.array_equal(_fixed_offset_output(grid), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    shape = (h, w)

    active_count = random.choice((2, 3))
    active_corners = random.sample(list(CORNER_ORDER), active_count)
    seed_plan = _choose_seed_plan(shape, active_corners)
    if seed_plan is None:
        return None

    colors = random.sample(MARKER_COLORS, active_count)
    color_by_corner = dict(zip(active_corners, colors))

    stamp_cells: set[tuple[int, int]] = set()
    for spec in seed_plan.values():
        stamp_cells.update(spec["stamp"])

    forbidden = _forbidden_ray_cells(shape, active_corners)
    blocked = (
        set(stamp_cells)
        | {_corner_position(shape, name) for name in CORNER_ORDER}
        | forbidden
    )

    all_obstacles: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for name in active_corners:
        seed = tuple(seed_plan[name]["obstacle"])
        size = random.randint(1, 3)
        comp = _grow_component(shape, seed, size, blocked - {seed}, all_obstacles)
        if comp is None:
            return None
        components.append(comp)
        all_obstacles.update(comp)

    min_total = max(GEN_BLOCKER_COUNT_RANGE[0], active_count + 1)
    blocker_count = random.randint(min_total, GEN_BLOCKER_COUNT_RANGE[1])
    distractor_count = blocker_count - active_count
    distractors = _place_distractor_components(shape, distractor_count, blocked, all_obstacles)
    if distractors is None:
        return None
    components.extend(distractors)

    grid = np.zeros(shape, dtype=int)
    for comp in components:
        for r, c in comp:
            grid[r, c] = WALL
    for name, color in color_by_corner.items():
        r, c = _corner_position(shape, name)
        grid[r, c] = color

    if not _validate_instance(grid, seed_plan, distractor_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
