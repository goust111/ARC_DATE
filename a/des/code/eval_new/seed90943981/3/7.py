import random
from collections import defaultdict

import numpy as np


BG = 0
CHANNEL = 5
EMITTER = 4
AZURE = 8
RECEIVER = 9
PORTAL_POOL = [1, 2, 3, 6, 7]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _adjacent_dirs(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(h, w, rr, cc) and bool(mask[rr, cc]):
            out.append((int(dr), int(dc)))
    return out


def _direction_score(candidate: tuple[int, int], incoming: tuple[int, int]) -> int:
    return int(candidate[0] * incoming[0] + candidate[1] * incoming[1])


def _best_direction(
    candidates: list[tuple[int, int]],
    incoming: tuple[int, int],
) -> tuple[int, int] | None:
    if not candidates:
        return None
    best = candidates[0]
    best_key = (_direction_score(best, incoming), -DIRS.index(best))
    for cand in candidates[1:]:
        key = (_direction_score(cand, incoming), -DIRS.index(cand))
        if key > best_key:
            best = cand
            best_key = key
    return best


def _next_direction(
    mask: np.ndarray,
    r: int,
    c: int,
    incoming: tuple[int, int],
) -> tuple[int, int] | None:
    nbrs = _adjacent_dirs(mask, r, c)
    if not nbrs:
        return None

    back = (-incoming[0], -incoming[1])
    if incoming in nbrs:
        return incoming

    forward_choices = [d for d in nbrs if d != back]
    if len(forward_choices) == 1:
        return forward_choices[0]
    if not forward_choices:
        return None
    return _best_direction(forward_choices, incoming)


def _portal_exit_direction(
    mask: np.ndarray,
    r: int,
    c: int,
    incoming: tuple[int, int],
) -> tuple[int, int] | None:
    nbrs = _adjacent_dirs(mask, r, c)
    if not nbrs:
        return None
    if incoming in nbrs:
        return incoming

    back = (-incoming[0], -incoming[1])
    same_axis = [d for d in nbrs if d == back]
    if same_axis:
        return same_axis[0]
    if len(nbrs) == 1:
        return nbrs[0]
    return _best_direction(nbrs, incoming)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != BG
    out = g.copy()

    portal_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere((g != BG) & (g != CHANNEL) & (g != EMITTER) & (g != RECEIVER)):
        portal_cells[int(g[int(r), int(c)])].append((int(r), int(c)))

    portal_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for color, cells in portal_cells.items():
        assert len(cells) == 2, f"portal color {color} must appear exactly twice"
        a, b = cells
        portal_pairs[a] = b
        portal_pairs[b] = a

    emitters = [tuple(map(int, rc)) for rc in np.argwhere(g == EMITTER)]
    assert emitters, "expected at least one emitter"

    for start_r, start_c in emitters:
        start_dirs = _adjacent_dirs(mask, start_r, start_c)
        assert len(start_dirs) == 1, "emitter must sit on a dead end"

        pos = (int(start_r), int(start_c))
        direction = start_dirs[0]
        color = int(AZURE)
        seen: set[tuple[int, int, int, int, int]] = set()

        while True:
            state = (int(pos[0]), int(pos[1]), int(direction[0]), int(direction[1]), int(color))
            if state in seen:
                break
            seen.add(state)

            nr = int(pos[0] + direction[0])
            nc = int(pos[1] + direction[1])
            if not _in_bounds(h, w, nr, nc) or not bool(mask[nr, nc]):
                break

            pos = (int(nr), int(nc))
            cell = int(g[int(nr), int(nc)])

            if cell == CHANNEL:
                prev = int(out[int(nr), int(nc)])
                assert prev in (CHANNEL, color), "signals should not conflict on grey path cells"
                out[int(nr), int(nc)] = int(color)
            elif cell == RECEIVER:
                break
            elif cell == EMITTER:
                break

            if pos in portal_pairs:
                color = int(cell)
                mate = portal_pairs[pos]
                exit_dir = _portal_exit_direction(mask, int(mate[0]), int(mate[1]), direction)
                if exit_dir is None:
                    break
                pos = (int(mate[0]), int(mate[1]))
                direction = exit_dir
                continue

            next_dir = _next_direction(mask, int(pos[0]), int(pos[1]), direction)
            if next_dir is None:
                break
            direction = next_dir

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each launched signal through grey corridors, recoloring the traversed path and jumping across matched portals.

    Input structure:
    - The background is black.
    - Grey pixels form thin orthogonal corridors; some corridor cells are replaced by single-pixel markers.
    - Yellow markers are launch points on corridor dead ends, maroon markers are receiving endpoints on corridor dead ends, and every other non-black non-grey color appears exactly twice as a portal pair.

    Rule:
    - Launch one signal from each yellow endpoint toward its unique adjacent corridor cell, with initial color azure.
    - The signal follows the corridor: at a straight run or cross it keeps going straight, while at a bend it follows the only non-backtracking continuation.
    - When the signal lands on a portal, it instantly moves to the matching portal, changes to that portal color, and continues from there; if the exit portal sits on a cross, it preserves its incoming direction.
    - Grey corridor cells visited by the signal are recolored with the signal's current color. Unvisited grey cells stay grey, while portal, launch, and receiving markers keep their original colors.

    Key insight:
    - Solve the puzzle by building the corridor graph, pairing portals by color, and then simulating directional movement so that each teleport both changes position and updates the active paint color.
    """
    return _solve(grid)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return [(int(r - min_r), int(c - min_c)) for r, c in coords]


def _transform_point(r: int, c: int, rot: int, mirror: bool) -> tuple[int, int]:
    rr, cc = int(r), int(c)
    if mirror:
        cc = -cc
    for _ in range(rot):
        rr, cc = cc, -rr
    return (int(rr), int(cc))


def _transform_route(route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rot = random.randrange(4)
    mirror = random.random() < 0.5
    transformed = [_transform_point(r, c, rot, mirror) for r, c in route]
    if random.random() < 0.5:
        transformed = list(reversed(transformed))
    return _normalize(transformed)


def _base_route() -> list[tuple[int, int]]:
    runs = random.randint(3, 5)
    route = [(0, 0)]
    r = 0
    c = 0

    for i in range(runs):
        hlen = random.randint(4, 7)
        step_c = 1 if i % 2 == 0 else -1
        for _ in range(hlen):
            c += step_c
            route.append((int(r), int(c)))
        if i == runs - 1:
            break
        vlen = random.randint(2, 4)
        for _ in range(vlen):
            r += 1
            route.append((int(r), int(c)))

    return _normalize(route)


def _delta(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(b[0] - a[0]), int(b[1] - a[1]))


def _is_straight(route: list[tuple[int, int]], idx: int) -> bool:
    if not (0 < idx < len(route) - 1):
        return False
    return _delta(route[idx - 1], route[idx]) == _delta(route[idx], route[idx + 1])


def _can_add_cross(center: tuple[int, int], direction: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if direction[0] == 0:
        arms = [(-1, 0), (1, 0)]
    else:
        arms = [(0, -1), (0, 1)]

    new_cells = {(int(center[0] + dr), int(center[1] + dc)) for dr, dc in arms}
    if new_cells & occupied:
        return False

    for nr, nc in new_cells:
        for dr, dc in DIRS:
            rr = int(nr + dr)
            cc = int(nc + dc)
            if (rr, cc) == center:
                continue
            if (rr, cc) in occupied:
                return False
    return True


def _add_cross(
    route: list[tuple[int, int]],
    idx: int,
    occupied: set[tuple[int, int]],
    arms: set[tuple[int, int]],
    cross_indices: set[int],
) -> bool:
    if not _is_straight(route, idx):
        return False
    center = route[idx]
    direction = _delta(route[idx - 1], route[idx])
    if not _can_add_cross(center, direction, occupied):
        return False

    if direction[0] == 0:
        additions = {(int(center[0] - 1), int(center[1])), (int(center[0] + 1), int(center[1]))}
    else:
        additions = {(int(center[0]), int(center[1] - 1)), (int(center[0]), int(center[1] + 1))}

    arms |= additions
    occupied |= additions
    cross_indices.add(int(idx))
    return True


def _select_pairs(route: list[tuple[int, int]], max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    blocked: set[int] = set()
    first_a = 1

    for _ in range(max_pairs):
        options: list[tuple[int, list[int]]] = []
        for a in range(first_a, len(route) - 2):
            if any(k in blocked for k in (a - 1, a, a + 1)):
                continue
            enter = _delta(route[a - 1], route[a])
            bs: list[int] = []
            for b in range(a + 2, len(route) - 1):
                if any(k in blocked for k in (b - 1, b, b + 1)):
                    continue
                if _delta(route[b], route[b + 1]) != enter:
                    continue
                manhattan = abs(route[a][0] - route[b][0]) + abs(route[a][1] - route[b][1])
                if manhattan < 4:
                    continue
                bs.append(int(b))
            if bs:
                options.append((int(a), bs))

        if not options:
            break

        a, bs = random.choice(options)
        b = random.choice(bs)
        pairs.append((int(a), int(b)))
        blocked.update({a - 1, a, a + 1, b - 1, b, b + 1})
        first_a = int(b + 2)

    return pairs


def _shift_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _component_bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return (int(max(rows) + 1), int(max(cols) + 1))


def _component_spec() -> dict | None:
    route = _transform_route(_base_route())
    occupied = set(route)
    arms: set[tuple[int, int]] = set()
    cross_indices: set[int] = set()

    max_pairs = min(2, max(0, (len(route) - 8) // 7))
    target_pairs = random.randint(0, max_pairs) if max_pairs > 0 else 0
    pairs = _select_pairs(route, target_pairs)

    for _a, b in pairs:
        if random.random() < 0.5:
            _add_cross(route, b, occupied, arms, cross_indices)

    extra_cross_target = random.randint(0, 2)
    candidates = [idx for idx in range(1, len(route) - 1) if idx not in cross_indices]
    random.shuffle(candidates)
    for idx in candidates:
        if extra_cross_target <= 0:
            break
        if _add_cross(route, idx, occupied, arms, cross_indices):
            extra_cross_target -= 1

    all_cells = list(occupied)
    route = _normalize(route)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    arms = {(int(r - min_r), int(c - min_c)) for r, c in arms}
    route = [(int(r - min_r), int(c - min_c)) for r, c in route]

    return {
        "route": route,
        "arms": arms,
        "pairs": pairs,
        "receiver": bool(random.getrandbits(1)),
    }


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(int(r - 1), int(r + 2)):
            for cc in range(int(c - 1), int(c + 2)):
                if (int(rr), int(cc)) in occupied:
                    return True
    return False


def _place_component(
    comp: dict,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    route: list[tuple[int, int]] = comp["route"]
    arms: set[tuple[int, int]] = comp["arms"]
    local_cells = list(set(route) | arms)
    box_h, box_w = _component_bbox(local_cells)
    if box_h > h or box_w > w:
        return None

    row_min = 1 if h - box_h >= 2 else 0
    col_min = 1 if w - box_w >= 2 else 0
    row_max = h - box_h - 1 if h - box_h >= 2 else h - box_h
    col_max = w - box_w - 1 if w - box_w >= 2 else w - box_w
    if row_max < row_min or col_max < col_min:
        return None

    for _ in range(120):
        top = random.randint(row_min, row_max)
        left = random.randint(col_min, col_max)
        shifted = _shift_cells(local_cells, top, left)
        if _touches_existing(shifted, occupied):
            continue
        occupied.update(shifted)
        return (int(top), int(left))
    return None


def _paint_component(grid: np.ndarray, comp: dict, top: int, left: int, portal_colors: list[int]) -> int:
    route: list[tuple[int, int]] = comp["route"]
    arms: set[tuple[int, int]] = comp["arms"]
    pairs: list[tuple[int, int]] = comp["pairs"]
    receiver: bool = comp["receiver"]

    for r, c in set(route) | arms:
        grid[int(r + top), int(c + left)] = int(CHANNEL)

    grid[int(route[0][0] + top), int(route[0][1] + left)] = int(EMITTER)
    if receiver:
        grid[int(route[-1][0] + top), int(route[-1][1] + left)] = int(RECEIVER)

    used = 0
    for a, b in pairs:
        color = int(portal_colors[used])
        used += 1
        ar, ac = route[a]
        br, bc = route[b]
        grid[int(ar + top), int(ac + left)] = color
        grid[int(br + top), int(bc + left)] = color

    return int(used)


def _build_scene() -> np.ndarray | None:
    h = random.randint(16, 29)
    w = random.randint(16, 29)
    grid = np.full((h, w), int(BG), dtype=int)

    component_count = random.choices([1, 2, 3], weights=[3, 4, 2])[0]
    comps: list[dict] = []
    total_pairs = 0
    for _ in range(component_count):
        comp = _component_spec()
        if comp is None:
            return None
        comps.append(comp)
        total_pairs += len(comp["pairs"])

    if total_pairs > len(PORTAL_POOL):
        return None

    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[int, int]] = []
    for comp in comps:
        placed = _place_component(comp, h, w, occupied)
        if placed is None:
            return None
        placements.append(placed)

    colors = PORTAL_POOL.copy()
    random.shuffle(colors)
    used_colors = 0
    for comp, (top, left) in zip(comps, placements):
        need = len(comp["pairs"])
        used_colors += _paint_component(grid, comp, top, left, colors[used_colors : used_colors + need])

    if used_colors != total_pairs:
        return None
    if not bool((grid == EMITTER).any()):
        return None
    if not bool((grid == CHANNEL).any()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is None:
            continue
        try:
            _ = _solve(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid portal-path input")
