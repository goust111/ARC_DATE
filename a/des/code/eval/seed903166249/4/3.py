from __future__ import annotations

import random
import numpy as np


# ARC colors used by this task.
BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 8)

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# The 3-pixel L occupies a 2x2 box with one missing corner.
# Missing corner TL/TR/BR/BL encodes up/right/down/left respectively.
DIR_TO_MISSING = {
    "up": (0, 0),
    "right": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
}
MISSING_TO_DIR = {v: k for k, v in DIR_TO_MISSING.items()}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _room_from_grey_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    comp_set = set(comp)
    border = _border_cells(r0, c0, r1, c1)
    if len(comp_set) != len(border) - 1:
        return None

    door: tuple[int, int] | None = None
    for cell in border:
        r, c = cell
        if cell in comp_set:
            if int(grid[r, c]) != GREY:
                return None
            continue
        if door is not None or int(grid[r, c]) != BLACK:
            return None
        if (r in (r0, r1)) and (c in (c0, c1)):
            return None
        door = cell

    if door is None:
        return None

    # A room contains no grey pixels in its interior.
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or bool(np.any(interior == GREY)):
        return None

    if door[0] == r0:
        side = "up"
    elif door[0] == r1:
        side = "down"
    elif door[1] == c0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "door": door,
        "door_side": side,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components(grid == GREY):
        room = _room_from_grey_component(grid, comp)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _pointer_cells_from_origin(direction: str, r0: int, c0: int) -> list[tuple[int, int]]:
    missing = DIR_TO_MISSING[direction]
    cells = []
    for dr in (0, 1):
        for dc in (0, 1):
            if (dr, dc) != missing:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _pointer_direction(grid: np.ndarray, room: dict) -> str:
    r0, c0, r1, c1 = room["bbox"]
    coords = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == ORANGE)
    assert int(coords.shape[0]) == 3, "each room must contain one 3-pixel orange pointer"

    pr0 = int(coords[:, 0].min())
    pc0 = int(coords[:, 1].min())
    pr1 = int(coords[:, 0].max())
    pc1 = int(coords[:, 1].max())
    assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (2, 2), "pointer must fit a 2x2 box"

    present = {(int(r - pr0), int(c - pc0)) for r, c in coords}
    missing = tuple(({(0, 0), (0, 1), (1, 0), (1, 1)} - present).pop())
    return MISSING_TO_DIR[missing]


def _payloads_in_room(grid: np.ndarray, room: dict) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    payloads: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color not in (BLACK, ORANGE):
                payloads.append((r, c, color))
    return payloads


def _sort_key(direction: str, payload: tuple[int, int, int]) -> tuple[int, int]:
    r, c, _ = payload
    if direction == "up":
        return (r, c)
    if direction == "down":
        return (-r, c)
    if direction == "left":
        return (c, r)
    return (-c, r)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each room's payload out through its door according to the orange L pointer.

    The direction is read from the pointer's missing corner:
    - missing top-left -> up
    - missing top-right -> right
    - missing bottom-right -> down
    - missing bottom-left -> left

    Non-pointer payload pixels are processed from the front toward the back along the
    chosen direction, so they stack against the first blocking wall or the grid edge.
    Their original cells become maroon markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _find_rooms(g)
    assert rooms, "expected at least one grey room"

    out = g.copy()
    h, w = g.shape

    for room in rooms:
        direction = _pointer_direction(g, room)
        payloads = _payloads_in_room(g, room)
        assert payloads, "each room must contain at least one payload pixel"

        dr, dc = DIRS[direction]
        moved: list[tuple[int, int, int]] = []
        occupied: set[tuple[int, int]] = set()

        for r0, c0, color in sorted(payloads, key=lambda item: _sort_key(direction, item)):
            r = r0
            c = c0
            while True:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if int(g[nr, nc]) == GREY:
                    break
                if (nr, nc) in occupied:
                    break
                r = nr
                c = nc
            assert (r, c) != (r0, c0), "payloads must be able to move"
            occupied.add((r, c))
            moved.append((r, c, color))

        for r, c, _color in payloads:
            out[r, c] = MAROON
        for r, c, color in moved:
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
ROOM_COUNT_RANGE = (2, 5)
INTERIOR_H_RANGE = (5, 8)
INTERIOR_W_RANGE = (5, 8)
ROOM_SEPARATION = 2
MAX_ATTEMPTS = 400

_GEN_COUNTER = 0


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_room_boxes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            ih = random.randint(*INTERIOR_H_RANGE)
            iw = random.randint(*INTERIOR_W_RANGE)
            rh = ih + 2
            rw = iw + 2
            if rh + 2 > h or rw + 2 > w:
                continue
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(_boxes_separated(box, other, ROOM_SEPARATION) for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _wall_owner(boxes: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], int]:
    owner: dict[tuple[int, int], int] = {}
    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        for cell in _border_cells(r0, c0, r1, c1):
            owner[cell] = idx
    return owner


def _pointer_origins(box: tuple[int, int, int, int], direction: str, line_value: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    origins: list[tuple[int, int]] = []
    for pr in range(r0 + 1, r1):
        for pc in range(c0 + 1, c1):
            if pr + 1 >= r1 or pc + 1 >= c1:
                continue
            cells = {(pr, pc), (pr + 1, pc), (pr, pc + 1), (pr + 1, pc + 1)}
            if direction in ("up", "down"):
                if any(c == line_value for _, c in cells):
                    continue
            else:
                if any(r == line_value for r, _ in cells):
                    continue
            origins.append((pr, pc))
    return origins


def _trace_corridor(
    h: int,
    w: int,
    room_idx: int,
    door: tuple[int, int],
    direction: str,
    owner: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], str, int | None, tuple[int, int] | None]:
    dr, dc = DIRS[direction]
    r = door[0] + dr
    c = door[1] + dc
    cells: list[tuple[int, int]] = []
    blocker_room: int | None = None
    blocker_cell: tuple[int, int] | None = None

    while 0 <= r < h and 0 <= c < w:
        hit = owner.get((r, c))
        if hit is not None and hit != room_idx:
            blocker_room = hit
            blocker_cell = (r, c)
            break
        cells.append((r, c))
        r += dr
        c += dc

    branch = "wall" if blocker_room is not None else "edge"
    return cells, branch, blocker_room, blocker_cell


def _room_configs(
    room_idx: int,
    box: tuple[int, int, int, int],
    h: int,
    w: int,
    owner: dict[tuple[int, int], int],
) -> list[dict]:
    r0, c0, r1, c1 = box
    configs: list[dict] = []

    for direction in ("up", "right", "down", "left"):
        if direction in ("up", "down"):
            for door_c in range(c0 + 1, c1):
                door = (r0, door_c) if direction == "up" else (r1, door_c)
                line_cells = [(r, door_c) for r in range(r0 + 1, r1)]
                pointer_origins = _pointer_origins(box, direction, door_c)
                corridor, branch, blocker_room, blocker_cell = _trace_corridor(h, w, room_idx, door, direction, owner)
                if pointer_origins and corridor:
                    configs.append(
                        {
                            "direction": direction,
                            "door": door,
                            "line_cells": line_cells,
                            "pointer_origins": pointer_origins,
                            "corridor": corridor,
                            "branch": branch,
                            "blocker_room": blocker_room,
                            "blocker_cell": blocker_cell,
                        }
                    )
        else:
            for door_r in range(r0 + 1, r1):
                door = (door_r, c0) if direction == "left" else (door_r, c1)
                line_cells = [(door_r, c) for c in range(c0 + 1, c1)]
                pointer_origins = _pointer_origins(box, direction, door_r)
                corridor, branch, blocker_room, blocker_cell = _trace_corridor(h, w, room_idx, door, direction, owner)
                if pointer_origins and corridor:
                    configs.append(
                        {
                            "direction": direction,
                            "door": door,
                            "line_cells": line_cells,
                            "pointer_origins": pointer_origins,
                            "corridor": corridor,
                            "branch": branch,
                            "blocker_room": blocker_room,
                            "blocker_cell": blocker_cell,
                        }
                    )
    return configs


def _still_possible(
    order: list[int],
    pos: int,
    room_configs: list[list[dict]],
    forbidden_doors: list[set[tuple[int, int]]],
    need_both: bool,
    branch_counts: dict[str, int],
) -> bool:
    if not need_both:
        return True

    remaining = order[pos + 1 :]
    for branch in ("edge", "wall"):
        if branch_counts[branch] > 0:
            continue
        found = False
        for room in remaining:
            for cfg in room_configs[room]:
                if cfg["branch"] != branch:
                    continue
                if cfg["door"] not in forbidden_doors[room]:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _select_configs(room_configs: list[list[dict]]) -> list[dict] | None:
    n = len(room_configs)
    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda idx: len(room_configs[idx]))

    any_edge = any(any(cfg["branch"] == "edge" for cfg in cfgs) for cfgs in room_configs)
    any_wall = any(any(cfg["branch"] == "wall" for cfg in cfgs) for cfgs in room_configs)
    need_both = n >= 2 and any_edge and any_wall

    selected: list[dict | None] = [None] * n
    used_corridor: set[tuple[int, int]] = set()
    forbidden_doors: list[set[tuple[int, int]]] = [set() for _ in range(n)]
    branch_counts = {"edge": 0, "wall": 0}

    def branch_score(cfg: dict) -> tuple[int, int]:
        # Prefer the currently missing branch, then rotate directions over calls for variety.
        want = 0 if branch_counts[cfg["branch"]] == 0 else 1
        dir_rank = ("up", "right", "down", "left").index(cfg["direction"])
        return (want, (dir_rank - (_GEN_COUNTER % 4)) % 4)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return (not need_both) or (branch_counts["edge"] > 0 and branch_counts["wall"] > 0)

        room = order[pos]
        candidates = room_configs[room][:]
        random.shuffle(candidates)
        candidates.sort(key=branch_score)

        for cfg in candidates:
            if cfg["door"] in forbidden_doors[room]:
                continue
            if any(cell in used_corridor for cell in cfg["corridor"]):
                continue

            blocker_room = cfg["blocker_room"]
            blocker_cell = cfg["blocker_cell"]
            if blocker_room is not None and selected[blocker_room] is not None:
                if selected[blocker_room]["door"] == blocker_cell:
                    continue

            added_forbidden: list[tuple[int, tuple[int, int]]] = []
            if blocker_room is not None and selected[blocker_room] is None:
                if blocker_cell not in forbidden_doors[blocker_room]:
                    forbidden_doors[blocker_room].add(blocker_cell)
                    added_forbidden.append((blocker_room, blocker_cell))

            selected[room] = cfg
            for cell in cfg["corridor"]:
                used_corridor.add(cell)
            branch_counts[cfg["branch"]] += 1

            if _still_possible(order, pos, room_configs, forbidden_doors, need_both, branch_counts) and backtrack(pos + 1):
                return True

            branch_counts[cfg["branch"]] -= 1
            for cell in cfg["corridor"]:
                used_corridor.remove(cell)
            selected[room] = None
            for room_idx, cell in added_forbidden:
                forbidden_doors[room_idx].remove(cell)

        return False

    if backtrack(0):
        return [cfg for cfg in selected if cfg is not None]
    return None


def _draw_room(grid: np.ndarray, box: tuple[int, int, int, int], door: tuple[int, int]) -> None:
    r0, c0, r1, c1 = box
    for r, c in _border_cells(r0, c0, r1, c1):
        grid[r, c] = GREY
    grid[door] = BLACK


def _pick_payload_positions(line_cells: list[tuple[int, int]], count: int, direction: str) -> list[tuple[int, int]]:
    if direction in ("up", "down"):
        ordered = sorted(line_cells, key=lambda cell: cell[0])
    else:
        ordered = sorted(line_cells, key=lambda cell: cell[1])

    attempts = ordered[:]
    random.shuffle(attempts)
    picked: list[tuple[int, int]] = []
    for cell in attempts:
        axis = cell[0] if direction in ("up", "down") else cell[1]
        if all(abs(axis - (p[0] if direction in ("up", "down") else p[1])) >= 2 for p in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    return random.sample(line_cells, count)


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    room_count = random.randint(*ROOM_COUNT_RANGE)

    boxes = _place_room_boxes(h, w, room_count)
    if boxes is None:
        return None

    owner = _wall_owner(boxes)
    room_configs = [_room_configs(idx, box, h, w, owner) for idx, box in enumerate(boxes)]
    if any(not cfgs for cfgs in room_configs):
        return None

    selected = _select_configs(room_configs)
    if selected is None or len(selected) != room_count:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for box, cfg in zip(boxes, selected):
        _draw_room(grid, box, cfg["door"])

    for box, cfg in zip(boxes, selected):
        pointer_origin = random.choice(cfg["pointer_origins"])
        pointer_cells = _pointer_cells_from_origin(cfg["direction"], *pointer_origin)
        if any(int(grid[r, c]) != BLACK for r, c in pointer_cells):
            return None
        for r, c in pointer_cells:
            grid[r, c] = ORANGE

        max_count = min(3, len(cfg["line_cells"]), len(cfg["corridor"]))
        if max_count < 1:
            return None
        payload_count = random.randint(1, max_count)
        payload_positions = _pick_payload_positions(cfg["line_cells"], payload_count, cfg["direction"])
        payload_colors = random.sample(list(PAYLOAD_COLORS), payload_count)
        for (r, c), color in zip(payload_positions, payload_colors):
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = int(color)

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    _GEN_COUNTER += 1

    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid room-push sample")
