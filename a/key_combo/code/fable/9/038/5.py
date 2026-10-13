import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RING_COLOR = GREEN
TAIL_COLOR = GREY
RAY_COLOR = BLUE
MIRROR_BSLASH = RED
MIRROR_SLASH = YELLOW

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GAP_TO_DIR = {
    (-1, 0): (-1, 0),
    (0, 1): (0, 1),
    (1, 0): (1, 0),
    (0, -1): (0, -1),
}
LEFT_TURN = {
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}
REFLECT = {
    MIRROR_BSLASH: {
        (-1, 0): (0, -1),
        (0, -1): (-1, 0),
        (0, 1): (1, 0),
        (1, 0): (0, 1),
    },
    MIRROR_SLASH: {
        (-1, 0): (0, 1),
        (0, 1): (-1, 0),
        (1, 0): (0, -1),
        (0, -1): (1, 0),
    },
}
TURN_TO_MIRROR = {}
for mirror_color, mapping in REFLECT.items():
    for incoming, outgoing in mapping.items():
        TURN_TO_MIRROR[(incoming, outgoing)] = mirror_color


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ring_cells(center: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _parse_scene(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK

    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            corners = [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
            if any(int(g[rr, cc]) != RING_COLOR for rr, cc in corners):
                continue
            side_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            side_vals = [int(g[r + dr, c + dc]) for dr, dc in side_dirs]
            if side_vals.count(RING_COLOR) != 3 or side_vals.count(BLACK) != 1:
                continue
            gap_dir = side_dirs[side_vals.index(BLACK)]
            back_dir = (-gap_dir[0], -gap_dir[1])
            ammo = 0
            tr, tc = r + 2 * back_dir[0], c + 2 * back_dir[1]
            while _in_bounds(g.shape, tr, tc) and int(g[tr, tc]) == TAIL_COLOR:
                ammo += 1
                tr += back_dir[0]
                tc += back_dir[1]
            if ammo <= 0:
                continue
            candidates.append({"center": (r, c), "dir": gap_dir, "ammo": ammo})

    assert len(candidates) == 1, "expected exactly one ring cannon"
    scene = candidates[0]
    center = scene["center"]
    direction = scene["dir"]
    ammo = int(scene["ammo"])
    back_dir = (-direction[0], -direction[1])
    tail = [(center[0] + k * back_dir[0], center[1] + k * back_dir[1]) for k in range(2, ammo + 2)]
    assert all(_in_bounds(g.shape, r, c) and int(g[r, c]) == TAIL_COLOR for r, c in tail)
    assert int(np.sum(g == TAIL_COLOR)) == ammo, "gray pixels must be exactly the ammo tail"

    mirrors: Dict[Tuple[int, int], int] = {}
    for rr, cc in np.argwhere((g == MIRROR_BSLASH) | (g == MIRROR_SLASH)):
        mirrors[(int(rr), int(cc))] = int(g[rr, cc])
    assert mirrors, "expected at least one mirror strip cell"

    return {
        "grid": g,
        "center": center,
        "direction": direction,
        "ammo": ammo,
        "tail": tail,
        "mirrors": mirrors,
    }


def _simulate_from_scene(
    scene: Dict[str, object],
    *,
    ammo_override: Optional[int] = None,
    direction_override: Optional[Tuple[int, int]] = None,
    mirrors_override: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    g = np.asarray(scene["grid"], dtype=int)
    out = g.copy()
    for r, c in scene["tail"]:
        out[r, c] = BLACK

    ammo = int(scene["ammo"] if ammo_override is None else ammo_override)
    direction = tuple(scene["direction"] if direction_override is None else direction_override)
    mirrors = dict(scene["mirrors"] if mirrors_override is None else mirrors_override)
    center = tuple(scene["center"])

    pos = (center[0] + 2 * direction[0], center[1] + 2 * direction[1])
    path: List[Tuple[int, int]] = []
    mirror_hits = 0
    for _ in range(ammo):
        if not _in_bounds(g.shape, pos[0], pos[1]):
            break
        path.append(pos)
        cell = int(g[pos])
        if cell == BLACK:
            out[pos] = RAY_COLOR
        if pos in mirrors:
            direction = REFLECT[mirrors[pos]][direction]
            mirror_hits += 1
        pos = (pos[0] + direction[0], pos[1] + direction[1])
    return out, path, mirror_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a blue ray from the ring cannon gap, reflecting on diagonal mirror strips until its ammo runs out.

    Input structure:
    - The background is black.
    - Exactly one green 3x3 ring cannon has a single missing side cell; that gap points outward and defines the firing direction.
    - A straight gray ammo tail sits behind the cannon on the opposite side, and red/yellow diagonal mirror strips are scattered elsewhere.

    Rule:
    - Remove the gray ammo tail in the output.
    - Start the ray at the first cell outside the gap and advance one cell per ammo unit, painting only black cells blue.
    - When the ray enters a red "\" mirror cell or a yellow "/" mirror cell, keep that cell unchanged and immediately turn according to that diagonal orientation; stop when ammo is exhausted or the ray would leave the grid.

    Key insight:
    - The gap chooses the starting direction, the tail length chooses exactly how many step-by-step updates happen, and the mirror orientations decide every turn along the polyline.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    out, _path, _mirror_hits = _simulate_from_scene(scene)
    return out


def _valid_center(shape: Tuple[int, int], center: Tuple[int, int], direction: Tuple[int, int], ammo: int) -> bool:
    r, c = center
    if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
        return False
    start = (r + 2 * direction[0], c + 2 * direction[1])
    if not _in_bounds(shape, start[0], start[1]):
        return False
    back_dir = (-direction[0], -direction[1])
    for k in range(2, ammo + 2):
        tr = r + k * back_dir[0]
        tc = c + k * back_dir[1]
        if not _in_bounds(shape, tr, tc):
            return False
    return True


def _forbidden_for_cannon(center: Tuple[int, int], ammo: int, direction: Tuple[int, int]) -> set:
    forbidden = set(_ring_cells(center))
    back_dir = (-direction[0], -direction[1])
    for k in range(2, ammo + 2):
        forbidden.add((center[0] + k * back_dir[0], center[1] + k * back_dir[1]))
    return forbidden


def _plan_path(
    shape: Tuple[int, int],
    start: Tuple[int, int],
    direction: Tuple[int, int],
    length: int,
    forbidden: set,
    target_turns: int,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    visited: set = set()
    cells: List[Tuple[int, int]] = []
    incoming_dirs: List[Tuple[int, int]] = []

    def rec(
        pos: Tuple[int, int],
        incoming: Tuple[int, int],
        steps_left: int,
        turns_used: int,
        can_turn: bool,
    ) -> bool:
        if not _in_bounds(shape, pos[0], pos[1]) or pos in forbidden or pos in visited:
            return False

        remaining_future = steps_left - 1
        remaining_turns = target_turns - turns_used
        if remaining_turns < 0:
            return False
        if remaining_turns > remaining_future:
            return False
        if remaining_turns * 2 - (1 if can_turn else 0) > remaining_future:
            return False

        visited.add(pos)
        cells.append(pos)
        incoming_dirs.append(incoming)

        if steps_left == 1:
            ok = turns_used == target_turns
            if ok:
                return True
        else:
            options = [incoming]
            if can_turn and turns_used < target_turns:
                options.extend([LEFT_TURN[incoming], RIGHT_TURN[incoming]])
            random.shuffle(options)
            for outgoing in options:
                next_pos = (pos[0] + outgoing[0], pos[1] + outgoing[1])
                next_can_turn = outgoing == incoming
                if rec(next_pos, outgoing, steps_left - 1, turns_used + int(outgoing != incoming), next_can_turn):
                    return True

        visited.remove(pos)
        cells.pop()
        incoming_dirs.pop()
        return False

    if rec(start, direction, length, 0, True):
        return cells[:], incoming_dirs[:]
    return None


def _paint_cannon(grid: np.ndarray, center: Tuple[int, int], direction: Tuple[int, int], ammo: int) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = BLACK
    for rr, cc in _ring_cells(center):
        grid[rr, cc] = RING_COLOR
    grid[r, c] = BLACK
    gap = (r + direction[0], c + direction[1])
    grid[gap] = BLACK
    back_dir = (-direction[0], -direction[1])
    for k in range(2, ammo + 2):
        grid[r + k * back_dir[0], c + k * back_dir[1]] = TAIL_COLOR


def _mirror_anchors_from_path(
    cells: List[Tuple[int, int]], dirs: List[Tuple[int, int]]
) -> List[Tuple[Tuple[int, int], int]]:
    anchors = []
    for i in range(len(cells) - 1):
        if dirs[i + 1] != dirs[i]:
            anchors.append((cells[i], TURN_TO_MIRROR[(dirs[i], dirs[i + 1])]))
    return anchors


def _diag_vector(mirror_color: int) -> Tuple[int, int]:
    return (1, 1) if mirror_color == MIRROR_BSLASH else (1, -1)


def _place_anchor_strip(
    grid: np.ndarray,
    anchor: Tuple[int, int],
    mirror_color: int,
    occupied: set,
    blocked_path: set,
) -> bool:
    vr, vc = _diag_vector(mirror_color)
    candidates = []
    for left_len in range(0, 3):
        for right_len in range(0, 3):
            if left_len + right_len + 1 < 2:
                continue
            cells = [(anchor[0] + t * vr, anchor[1] + t * vc) for t in range(-left_len, right_len + 1)]
            candidates.append(cells)
    random.shuffle(candidates)

    for cells in candidates:
        ok = True
        for cell in cells:
            if not _in_bounds(grid.shape, cell[0], cell[1]):
                ok = False
                break
            if cell != anchor and cell in blocked_path:
                ok = False
                break
            if cell in occupied and cell != anchor:
                ok = False
                break
        if not ok:
            continue
        for rr, cc in cells:
            grid[rr, cc] = mirror_color
            occupied.add((rr, cc))
        return True
    return False


def _place_extra_strip(grid: np.ndarray, mirror_color: int, occupied: set, blocked_path: set) -> bool:
    vr, vc = _diag_vector(mirror_color)
    h, w = grid.shape
    attempts = 120
    for _ in range(attempts):
        length = random.randint(2, 5)
        anchor = (random.randrange(h), random.randrange(w))
        side = random.randint(0, length - 1)
        start = (anchor[0] - side * vr, anchor[1] - side * vc)
        cells = [(start[0] + t * vr, start[1] + t * vc) for t in range(length)]
        if any(not _in_bounds(grid.shape, r, c) for r, c in cells):
            continue
        if any((r, c) in occupied or (r, c) in blocked_path for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = mirror_color
            occupied.add((r, c))
        return True
    return False


def _build_scene() -> Optional[np.ndarray]:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    grid = np.full((h, w), BLACK, dtype=int)

    target_turns = random.choice([1, 2, 2, 3])
    ammo = random.randint(target_turns + 4, min(10, max(h, w) - 3))
    direction = random.choice(DIRS)

    centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if _valid_center((h, w), (r, c), direction, ammo)]
    random.shuffle(centers)
    if not centers:
        return None

    for center in centers[:40]:
        forbidden = _forbidden_for_cannon(center, ammo, direction)
        start = (center[0] + 2 * direction[0], center[1] + 2 * direction[1])
        plan = _plan_path((h, w), start, direction, ammo, forbidden, target_turns)
        if plan is None:
            continue
        path_cells, incoming_dirs = plan
        mirror_anchors = _mirror_anchors_from_path(path_cells, incoming_dirs)
        if len(mirror_anchors) != target_turns:
            continue
        if not mirror_anchors:
            continue

        temp = np.full((h, w), BLACK, dtype=int)
        _paint_cannon(temp, center, direction, ammo)
        occupied = {(int(r), int(c)) for r, c in np.argwhere(temp != BLACK)}
        blocked_path = set(path_cells)

        ok = True
        for anchor, mirror_color in mirror_anchors:
            if not _place_anchor_strip(temp, anchor, mirror_color, occupied, blocked_path - {anchor}):
                ok = False
                break
        if not ok:
            continue

        orientations = {mirror_color for _, mirror_color in mirror_anchors}
        needed = []
        if MIRROR_BSLASH not in orientations:
            needed.append(MIRROR_BSLASH)
        if MIRROR_SLASH not in orientations:
            needed.append(MIRROR_SLASH)
        extra_count = random.randint(1, 3)
        extra_colors = needed[:]
        while len(extra_colors) < extra_count:
            extra_colors.append(random.choice([MIRROR_BSLASH, MIRROR_SLASH]))
        random.shuffle(extra_colors)

        for mirror_color in extra_colors:
            if not _place_extra_strip(temp, mirror_color, occupied, blocked_path):
                ok = False
                break
        if not ok:
            continue

        if _validate_generated_scene(temp):
            return temp

    return None


def _validate_generated_scene(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        out, path, mirror_hits = _simulate_from_scene(scene)
    except Exception:
        return False

    if mirror_hits <= 0:
        return False
    if len(path) != int(scene["ammo"]):
        return False
    if int(np.sum(out == RAY_COLOR)) <= 0:
        return False
    if not np.array_equal(main(grid), out):
        return False

    short_grid = np.array(grid, copy=True)
    center = tuple(scene["center"])
    direction = tuple(scene["direction"])
    ammo = int(scene["ammo"])
    back_dir = (-direction[0], -direction[1])
    far_tail = (center[0] + (ammo + 1) * back_dir[0], center[1] + (ammo + 1) * back_dir[1])
    short_grid[far_tail] = BLACK
    try:
        short_out = main(short_grid)
    except Exception:
        return False
    if np.array_equal(short_out, out):
        return False

    no_mirror_out, _path2, _hits2 = _simulate_from_scene(scene, mirrors_override={})
    if np.array_equal(no_mirror_out, out):
        return False

    mirror_cells = set(scene["mirrors"].keys())
    if len(mirror_cells - set(path)) <= 0:
        return False
    if MIRROR_BSLASH not in scene["mirrors"].values() or MIRROR_SLASH not in scene["mirrors"].values():
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-cannon mirror scene")
