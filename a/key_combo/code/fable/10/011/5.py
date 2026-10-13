import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

MAX_ATTEMPTS = 400
GRID_MAX = 30
STRUCTURE_HEIGHT = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _detect_socket_and_direction(
    grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                continue
            exit_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                advanced = False
                while r0 <= rr <= r1 and c0 <= cc <= c1 and int(grid[rr, cc]) == bg:
                    advanced = True
                    rr += dr
                    cc += dc
                if advanced and not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    exit_dirs.append((dr, dc))
            if len(exit_dirs) == 1:
                candidates.append(((r, c), exit_dirs[0]))
    assert len(candidates) == 1, "each structure must contain exactly one detectable socket"
    return candidates[0]


def _nearest_visible_red(
    grid: np.ndarray,
    bg: int,
    red_at: dict[tuple[int, int], int],
    remaining_reds: set[int],
    socket: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, tuple[int, int], int] | None:
    dr, dc = direction
    r, c = socket
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not _in_bounds(grid.shape, r, c):
            return None
        val = int(grid[r, c])
        if val == RED:
            rid = red_at[(r, c)]
            if rid in remaining_reds:
                return rid, (r, c), dist
            continue
        _ = bg
        continue


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively capture the nearest visible hallway red into each structure socket and leave black trails.

    Input structure:
    - The most common color is the background; non-red connected structures each contain one interior socket.
    - Every socket opens onto the same straight hallway, and all red movable components lie on that hallway.
    - Different structures may face opposite directions along the hallway.

    Rule:
    - Repeatedly, for every unfilled socket, cast a ray along its opening direction to the nearest remaining red component.
    - Among those current socket-target pairs, choose the unique globally shortest distance, slide that red into the socket,
      and paint every hallway cell it traversed black.
    - Docked reds no longer block later rays; after all sockets are filled, erase every unused red and keep the structures unchanged.

    Key insight:
    - The matching is a dynamic nearest-first chain on one hallway, so later sockets must be recomputed after earlier captures.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    red_components = _components_from_mask(g == RED)
    assert red_components, "expected at least one red component"
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_components):
        for cell in coords:
            red_at[cell] = rid

    structure_mask = (g != bg) & (g != RED)
    structures = _components_from_mask(structure_mask)
    assert structures, "expected at least one non-red structure"

    sockets: list[tuple[tuple[int, int], tuple[int, int]]] = []
    structure_bbox_mask = np.zeros_like(g, dtype=bool)
    for coords in structures:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        structure_bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        sockets.append(_detect_socket_and_direction(g, bg, box))

    unresolved = set(range(len(sockets)))
    remaining_reds = set(range(len(red_components)))
    assignments: list[tuple[int, int, tuple[int, int]]] = []

    while unresolved:
        candidates: list[tuple[int, int, int, tuple[int, int]]] = []
        for sid in sorted(unresolved):
            socket, direction = sockets[sid]
            hit = _nearest_visible_red(g, bg, red_at, remaining_reds, socket, direction)
            assert hit is not None, "every unresolved socket must currently see a red component"
            rid, hit_cell, dist = hit
            candidates.append((dist, sid, rid, hit_cell))

        dists = [dist for dist, _sid, _rid, _hit in candidates]
        assert len(set(dists)) == len(dists), "visible nearest distances must stay unique"
        dist, sid, rid, hit_cell = min(candidates, key=lambda item: item[0])
        _ = dist
        unresolved.remove(sid)
        remaining_reds.remove(rid)
        assignments.append((sid, rid, hit_cell))

    out = g.copy()
    out[out == RED] = bg

    for sid, rid, hit_cell in assignments:
        socket, _direction = sockets[sid]
        shift_r = socket[0] - hit_cell[0]
        shift_c = socket[1] - hit_cell[1]
        step_r = _sign(shift_r)
        step_c = _sign(shift_c)
        steps = abs(shift_r) + abs(shift_c)

        for step in range(steps):
            off_r = step * step_r
            off_c = step * step_c
            for r, c in red_components[rid]:
                rr = r + off_r
                cc = c + off_c
                if _in_bounds(g.shape, rr, cc) and not structure_bbox_mask[rr, cc]:
                    out[rr, cc] = BLACK

        for r, c in red_components[rid]:
            rr = r + shift_r
            cc = c + shift_c
            if _in_bounds(g.shape, rr, cc):
                out[rr, cc] = RED

    return out


def _render_structure_patch(depth: int, direction: int, bg: int, sc: int) -> np.ndarray:
    width = depth + 2
    patch = np.full((STRUCTURE_HEIGHT, width), sc, dtype=int)
    mid = STRUCTURE_HEIGHT // 2
    if direction == 1:
        patch[mid, depth : depth + 2] = bg
    else:
        patch[mid, 0:2] = bg
    return patch


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k).copy()


def _simulate_line(
    sockets: list[tuple[int, int]],
    reds: list[int],
) -> tuple[list[tuple[int, int, int]], dict[int, int]] | None:
    remaining = set(reds)
    unresolved = set(range(len(sockets)))
    history: list[tuple[int, int, int]] = []
    initial_targets: dict[int, int] = {}

    while unresolved:
        step_candidates: list[tuple[int, int, int]] = []
        for sid in sorted(unresolved):
            pos, direction = sockets[sid]
            if direction == 1:
                visible = [r for r in remaining if r > pos]
                if not visible:
                    return None
                target = min(visible)
            else:
                visible = [r for r in remaining if r < pos]
                if not visible:
                    return None
                target = max(visible)
            if sid not in initial_targets:
                initial_targets[sid] = target
            step_candidates.append((abs(target - pos), sid, target))

        dists = [dist for dist, _sid, _target in step_candidates]
        if len(set(dists)) != len(dists):
            return None

        dist, sid, target = min(step_candidates, key=lambda item: item[0])
        history.append((sid, target, dist))
        unresolved.remove(sid)
        remaining.remove(target)

    return history, initial_targets


def _render_canonical_input(
    height: int,
    width: int,
    hall_row: int,
    structures: list[dict[str, int]],
    reds: list[int],
    bg: int,
    sc: int,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    top = hall_row - STRUCTURE_HEIGHT // 2
    for spec in structures:
        patch = _render_structure_patch(spec["depth"], spec["direction"], bg, sc)
        x0 = spec["x0"]
        grid[top : top + STRUCTURE_HEIGHT, x0 : x0 + patch.shape[1]] = patch
    for x in reds:
        grid[hall_row, x] = RED
    return grid


def _render_canonical_output(
    inp: np.ndarray,
    hall_row: int,
    structures: list[dict[str, int]],
    history: list[tuple[int, int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    out[out == RED] = bg
    structure_bbox_mask = np.zeros_like(inp, dtype=bool)
    top = hall_row - STRUCTURE_HEIGHT // 2
    for spec in structures:
        width = spec["depth"] + 2
        x0 = spec["x0"]
        structure_bbox_mask[top : top + STRUCTURE_HEIGHT, x0 : x0 + width] = True
    for sid, red_x, _dist in history:
        spec = structures[sid]
        socket_x = spec["socket_x"]
        if spec["direction"] == 1:
            for x in range(socket_x + 2, red_x + 1):
                if not structure_bbox_mask[hall_row, x]:
                    out[hall_row, x] = BLACK
        else:
            for x in range(red_x, socket_x - 1):
                if not structure_bbox_mask[hall_row, x]:
                    out[hall_row, x] = BLACK
        out[hall_row, socket_x] = RED
    return out


def _choose_counts() -> tuple[int, int]:
    total = random.randint(3, 4)
    if random.random() < 0.82:
        left = random.randint(1, total - 1)
        right = total - left
        return left, right
    if random.random() < 0.5:
        return total, 0
    return 0, total


def _build_model() -> tuple[
    int,
    int,
    int,
    list[dict[str, int]],
    list[int],
    int,
    int,
    list[tuple[int, int]],
]:
    bg = random.choice(PALETTE)
    sc = random.choice([c for c in PALETTE if c != bg])
    left_count, right_count = _choose_counts()
    depths_left = [random.randint(1, 3) for _ in range(left_count)]
    depths_right = [random.randint(1, 3) for _ in range(right_count)]
    left_gaps = [random.randint(1, 2) for _ in range(max(0, left_count - 1))]
    right_gaps = [random.randint(1, 2) for _ in range(max(0, right_count - 1))]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    structures: list[dict[str, int]] = []
    x = left_margin
    for i, depth in enumerate(depths_left):
        structures.append(
            {
                "x0": x,
                "depth": depth,
                "direction": 1,
                "socket_x": x + depth,
            }
        )
        x += depth + 2
        if i < len(left_gaps):
            x += left_gaps[i]
    corridor_start = x

    slot_count = left_count + right_count
    red_count = slot_count + random.randint(0, 1)
    min_corridor = max(red_count + 1, 2 * red_count - 1, 6)
    corridor_len = random.randint(min_corridor, 11)
    x = corridor_start + corridor_len

    for i, depth in enumerate(depths_right):
        x0 = x
        structures.append(
            {
                "x0": x0,
                "depth": depth,
                "direction": -1,
                "socket_x": x0 + 1,
            }
        )
        x += depth + 2
        if i < len(right_gaps):
            x += right_gaps[i]

    width = x + right_margin
    height = top_margin + STRUCTURE_HEIGHT + bottom_margin
    hall_row = top_margin + STRUCTURE_HEIGHT // 2
    if max(height, width) > GRID_MAX:
        raise ValueError("grid too large")

    corridor_cells = list(range(corridor_start, corridor_start + corridor_len))
    if len(corridor_cells) < red_count:
        raise ValueError("corridor too short")

    sockets_1d = [(spec["socket_x"], spec["direction"]) for spec in structures]
    for _ in range(200):
        reds = sorted(random.sample(corridor_cells, red_count))
        if any(b - a <= 1 for a, b in zip(reds, reds[1:])):
            continue
        sim = _simulate_line(sockets_1d, reds)
        if sim is None:
            continue
        history, initial_targets = sim
        initial_values = list(initial_targets.values())
        if len(set(initial_values)) == len(initial_values):
            continue
        if not any(target != initial_targets[sid] for sid, target, _dist in history):
            continue
        return height, width, hall_row, structures, reds, bg, sc, sockets_1d
    raise ValueError("could not find a chain-sensitive hallway layout")


def _gen_attempt() -> np.ndarray | None:
    try:
        height, width, hall_row, structures, reds, bg, sc, sockets_1d = _build_model()
    except ValueError:
        return None

    sim = _simulate_line(sockets_1d, reds)
    if sim is None:
        return None
    history, initial_targets = sim

    inp = _render_canonical_input(height, width, hall_row, structures, reds, bg, sc)
    expected = _render_canonical_output(inp, hall_row, structures, history, bg)

    rotation = random.choice((0, 2))
    inp = _rotate_grid(inp, rotation)
    expected = _rotate_grid(expected, rotation)

    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > GRID_MAX or max(expected.shape) > GRID_MAX:
        return None

    if not np.array_equal(main(inp), expected):
        return None

    rotated_bg = _get_bg(inp)
    if int(np.sum(inp == RED)) != len(reds):
        return None
    if rotated_bg == RED or rotated_bg == BLACK:
        return None
    if len(set(initial_targets.values())) == len(initial_targets):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained hallway-capture input")
