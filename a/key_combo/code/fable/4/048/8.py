from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
OBSTACLE_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
MAX_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _classify_obstacle(mask: np.ndarray, cells: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(cells)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if np.all(sub):
        return "solid"
    if (
        sub.shape[0] >= 3
        and sub.shape[1] >= 3
        and np.all(sub[0, :])
        and np.all(sub[-1, :])
        and np.all(sub[:, 0])
        and np.all(sub[:, -1])
        and not np.any(sub[1:-1, 1:-1])
    ):
        return "frame"
    raise AssertionError("every obstacle must be a solid rectangle or a one-cell-thick hollow frame")


def _emitter_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    emitters = np.argwhere(boundary & (grid == GREEN))
    assert emitters.shape[0] == 1, "expected exactly one green emitter on the boundary"
    r, c = map(int, emitters[0])
    on_edges = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
    assert on_edges == 1, "the emitter must lie on exactly one boundary side"
    if r == 0:
        return (r, c), 2
    if r == h - 1:
        return (r, c), 0
    if c == 0:
        return (r, c), 1
    return (r, c), 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a green beam that turns by obstacle topology until it exits or meets its own path.

    Input structure:
    - The grid has a black background and exactly one green emitter pixel on the boundary.
    - The interior contains several axis-aligned obstacle rectangles in arbitrary colors.
    - Each obstacle is either a solid filled block or a one-cell-thick hollow rectangular frame.

    Rule:
    - Start from the emitter and move one cell per step in the direction perpendicular to the boundary.
    - When the next cell is a solid obstacle, turn clockwise in place; when it is a hollow frame, turn counterclockwise in place.
    - Otherwise paint the next background cell green and continue; if the next cell is already green, stop.
    - Obstacles keep their original colors and the beam also stops once it has walked onto a boundary cell and would leave the grid next.

    Key insight:
    - Turn direction is encoded by obstacle topology rather than obstacle color, so the solver must classify solid blocks versus hollow frames before tracing the beam.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.any(g == BLACK)

    (r, c), d = _emitter_and_direction(g)

    obstacle_mask = g != BLACK
    obstacle_mask[r, c] = False
    comps = _components(obstacle_mask)
    assert comps, "expected at least one obstacle"

    obstacle_ids = np.full(g.shape, -1, dtype=int)
    kinds: dict[int, str] = {}
    for idx, cells in enumerate(comps):
        kind = _classify_obstacle(obstacle_mask, cells)
        kinds[idx] = kind
        for cr, cc in cells:
            obstacle_ids[cr, cc] = idx

    out = g.copy()
    painted = np.zeros(g.shape, dtype=bool)
    painted[r, c] = True
    seen_states: set[tuple[int, int, int]] = set()

    while True:
        state = (r, c, d)
        if state in seen_states:
            break
        seen_states.add(state)

        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        obstacle_id = int(obstacle_ids[nr, nc])
        if obstacle_id != -1:
            d = _rot_cw(d) if kinds[obstacle_id] == "solid" else _rot_ccw(d)
            continue

        if painted[nr, nc] or out[nr, nc] == GREEN:
            break

        assert out[nr, nc] == BLACK, "the beam may only advance through background"
        out[nr, nc] = GREEN
        painted[nr, nc] = True
        r, c = nr, nc

    return out


def _paint_shape(grid: np.ndarray, bbox: tuple[int, int, int, int], kind: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    if kind == "solid":
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_turn_obstacle(
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    hit: tuple[int, int],
    direction: int,
    kind: str,
    color: int,
) -> bool:
    h, w = grid.shape
    if kind == "solid":
        height_range = range(2, 5)
        width_range = range(2, 6)
    else:
        height_range = range(3, 7)
        width_range = range(3, 7)

    dims = [(hh, ww) for hh in height_range for ww in width_range]
    random.shuffle(dims)

    hit_r, hit_c = hit
    for hh, ww in dims:
        if direction == 1:
            c0 = hit_c
            if c0 + ww > w:
                continue
            r_lo = max(0, hit_r - hh + 1)
            r_hi = min(hit_r, h - hh)
            if r_lo > r_hi:
                continue
            r0 = random.randint(r_lo, r_hi)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        elif direction == 3:
            c0 = hit_c - ww + 1
            if c0 < 0:
                continue
            r_lo = max(0, hit_r - hh + 1)
            r_hi = min(hit_r, h - hh)
            if r_lo > r_hi:
                continue
            r0 = random.randint(r_lo, r_hi)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        elif direction == 2:
            r0 = hit_r
            if r0 + hh > h:
                continue
            c_lo = max(0, hit_c - ww + 1)
            c_hi = min(hit_c, w - ww)
            if c_lo > c_hi:
                continue
            c0 = random.randint(c_lo, c_hi)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        else:
            r0 = hit_r - hh + 1
            if r0 < 0:
                continue
            c_lo = max(0, hit_c - ww + 1)
            c_hi = min(hit_c, w - ww)
            if c_lo > c_hi:
                continue
            c0 = random.randint(c_lo, c_hi)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)

        r0, c0, r1, c1 = bbox
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        _paint_shape(grid, bbox, kind, color)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _place_decoy_obstacle(
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
) -> bool:
    h, w = grid.shape
    kind = random.choice(["solid", "frame"])
    color = random.choice(OBSTACLE_COLORS)
    if kind == "solid":
        hh = random.randint(2, 4)
        ww = random.randint(2, 5)
    else:
        hh = random.randint(3, 6)
        ww = random.randint(3, 6)

    if h - hh - 2 < 1 or w - ww - 2 < 1:
        return False

    for _ in range(60):
        r0 = random.randint(1, h - hh - 1)
        c0 = random.randint(1, w - ww - 1)
        r1, c1 = r0 + hh - 1, c0 + ww - 1
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        _paint_shape(grid, (r0, c0, r1, c1), kind, color)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _build_self_collision_path(h: int, w: int) -> dict | None:
    for _ in range(80):
        r0 = random.randint(2, h - 8)
        max_a = min(8, w - 8)
        if max_a < 4:
            return None
        a = random.randint(4, max_a)

        max_b = min(9, h - r0 - 4)
        if max_b < 5:
            continue
        b = random.randint(5, max_b)

        max_c = min(7, w - a - 4)
        if max_c < 3:
            continue
        c = random.randint(3, max_c)
        d = random.randint(2, b - 2)

        y = r0 + b - d
        path = [(r0, 0)]
        path.extend((r0, x) for x in range(1, a + 1))
        path.extend((r0 + i, a) for i in range(1, b + 1))
        path.extend((r0 + b, a + j) for j in range(1, c + 1))
        path.extend((r0 + b - i, a + c) for i in range(1, d + 1))
        path.extend((y, a + c - j) for j in range(1, c))
        if len(path) != len(set(path)):
            continue

        turns = [
            {"endpoint": (r0, a), "direction": 1, "kind": "solid"},
            {"endpoint": (r0 + b, a), "direction": 2, "kind": "frame"},
            {"endpoint": (r0 + b, a + c), "direction": 1, "kind": "frame"},
            {"endpoint": (y, a + c), "direction": 0, "kind": "frame"},
        ]
        return {"path": path, "turns": turns, "collision": True}
    return None


def _build_exit_path(h: int, w: int) -> dict | None:
    for _ in range(80):
        r0 = random.randint(2, h - 10)
        max_a = min(8, w - 7)
        if max_a < 4:
            return None
        a = random.randint(4, max_a)

        max_b = min(7, h - r0 - 5)
        if max_b < 3:
            continue
        b = random.randint(3, max_b)

        max_c = min(7, w - a - 4)
        if max_c < 3:
            continue
        c = random.randint(3, max_c)

        path = [(r0, 0)]
        path.extend((r0, x) for x in range(1, a + 1))
        path.extend((r0 + i, a) for i in range(1, b + 1))
        path.extend((r0 + b, a + j) for j in range(1, c + 1))
        path.extend((r0 + b + i, a + c) for i in range(1, h - (r0 + b)))
        if len(path) != len(set(path)):
            continue

        turns = [
            {"endpoint": (r0, a), "direction": 1, "kind": "solid"},
            {"endpoint": (r0 + b, a), "direction": 2, "kind": "frame"},
            {"endpoint": (r0 + b, a + c), "direction": 1, "kind": "solid"},
        ]
        return {"path": path, "turns": turns, "collision": False}
    return None


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    template = "self" if random.random() < 0.6 else "exit"
    plan = _build_self_collision_path(h, w) if template == "self" else _build_exit_path(h, w)
    if plan is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    for r, c in plan["path"]:
        reserved[r, c] = True
    blocked = np.zeros((h, w), dtype=bool)

    for turn in plan["turns"]:
        er, ec = turn["endpoint"]
        dr, dc = DIRS[turn["direction"]]
        hit = (er + dr, ec + dc)
        if not _place_turn_obstacle(
            grid,
            blocked,
            reserved,
            hit,
            turn["direction"],
            turn["kind"],
            random.choice(OBSTACLE_COLORS),
        ):
            return None

    for _ in range(random.randint(1, 3)):
        _place_decoy_obstacle(grid, blocked, reserved)

    emitter_r, emitter_c = plan["path"][0]
    grid[emitter_r, emitter_c] = GREEN

    rot = random.randint(0, 3)
    if rot:
        grid = np.rot90(grid, k=rot)

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.count_nonzero(grid == GREEN) != 1:
        return None
    if np.count_nonzero(out == GREEN) < 10:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
