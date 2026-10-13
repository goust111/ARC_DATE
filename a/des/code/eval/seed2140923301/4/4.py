from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)

LEFT_TURN = {
    UP: LEFT,
    LEFT: DOWN,
    DOWN: RIGHT,
    RIGHT: UP,
}

RIGHT_TURN = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}

REFLECT = {
    ("\\", UP): LEFT,
    ("\\", LEFT): UP,
    ("\\", DOWN): RIGHT,
    ("\\", RIGHT): DOWN,
    ("/", UP): RIGHT,
    ("/", RIGHT): UP,
    ("/", DOWN): LEFT,
    ("/", LEFT): DOWN,
}

SIDE_TO_DIR = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_mirror_kinds(grid: np.ndarray) -> dict[tuple[int, int], str]:
    red_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == RED)}
    kinds: dict[tuple[int, int], str] = {}

    for cell in red_cells:
        r, c = cell
        partners = [
            (r + dr, c + dc)
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))
            if (r + dr, c + dc) in red_cells
        ]
        assert len(partners) == 1, "every red mirror pixel must have exactly one diagonal partner"
        partner = partners[0]
        if partner < cell:
            continue
        kind = "\\" if partner[0] - cell[0] == partner[1] - cell[1] else "/"
        kinds[cell] = kind
        kinds[partner] = kind

    assert len(kinds) == len(red_cells), "every red mirror pixel must belong to a valid diagonal pair"
    return kinds


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    mirror_kinds = _find_mirror_kinds(g)

    closed = g.copy()
    closed[closed == YELLOW] = GREY
    room_mask = closed != GREY

    rooms = 0
    for comp in _components(room_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        rooms += 1
        comp_set = set(comp)
        windows = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == YELLOW:
                    windows.add((int(nr), int(nc)))

        assert len(windows) == 1, "each room must have exactly one window"
        window = next(iter(windows))
        entry_neighbors = [nbr for nbr in _neighbors4(*window) if nbr in comp_set]
        assert len(entry_neighbors) == 1, "window must face exactly one room interior direction"

        direction = (
            int(entry_neighbors[0][0] - window[0]),
            int(entry_neighbors[0][1] - window[1]),
        )
        pos = window

        for _ in range(h * w * 4):
            nr = pos[0] + direction[0]
            nc = pos[1] + direction[1]
            assert 0 <= nr < h and 0 <= nc < w, "light left the grid"

            cell = int(g[nr, nc])
            if cell == BLACK:
                out[nr, nc] = YELLOW
                pos = (nr, nc)
                continue
            if cell == RED:
                pos = (nr, nc)
                direction = REFLECT[(mirror_kinds[(nr, nc)], direction)]
                continue
            if cell == GREY:
                out[nr, nc] = ORANGE
                break
            if cell == YELLOW:
                break
            raise AssertionError(f"unexpected color along light path: {cell}")
        else:
            raise AssertionError("light path did not terminate")

    assert rooms > 0, "expected at least one enclosed room"
    return out


GRID_SIDE_RANGE = (22, 32)
ROOM_COUNT_RANGE = (1, 4)
ROOM_OUTER_BASE = (6, 11)
MAX_TURNS = 3
PLACEMENT_MARGIN = 2
ROOM_BUILD_ATTEMPTS = 80
SCENE_ATTEMPTS = 400


def _inside_interior(cell: tuple[int, int], room_h: int, room_w: int) -> bool:
    r, c = cell
    return 1 <= r < room_h - 1 and 1 <= c < room_w - 1


def _king_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _mirror_kind_for_turn(dir_in: tuple[int, int], dir_out: tuple[int, int]) -> str:
    for kind in ("\\", "/"):
        if REFLECT[(kind, dir_in)] == dir_out:
            return kind
    raise AssertionError("invalid 90-degree turn")


def _ray_free_cells(
    current: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
    room_h: int,
    room_w: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    dr, dc = direction
    r, c = current
    cells: list[tuple[int, int]] = []

    while True:
        r += dr
        c += dc
        if r == 0 or r == room_h - 1 or c == 0 or c == room_w - 1:
            return cells, (int(r), int(c))
        if (r, c) in blocked:
            return None
        cells.append((int(r), int(c)))


def _can_place_red_pair(
    a: tuple[int, int],
    b: tuple[int, int],
    room_h: int,
    room_w: int,
    used_black: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
) -> bool:
    if not _inside_interior(a, room_h, room_w) or not _inside_interior(b, room_h, room_w):
        return False
    if a == b or a in used_black or b in used_black or a in red_cells or b in red_cells:
        return False
    if abs(a[0] - b[0]) != 1 or abs(a[1] - b[1]) != 1:
        return False

    for cell, mate in ((a, b), (b, a)):
        for nbr in _king_neighbors(cell):
            if nbr in red_cells and nbr != mate:
                return False
    return True


def _search_room_path(
    room_h: int,
    room_w: int,
    current: tuple[int, int],
    direction: tuple[int, int],
    remaining_turns: int,
    used_black: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    mirrors: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> dict[str, object] | None:
    ray = _ray_free_cells(current, direction, used_black | red_cells, room_h, room_w)
    if ray is None:
        return None
    free_cells, wall_hit = ray

    if remaining_turns == 0:
        return {
            "used_black": used_black | set(free_cells),
            "red_cells": red_cells,
            "mirrors": mirrors,
            "wall_hit": wall_hit,
        }

    if not free_cells:
        return None

    indices = list(range(len(free_cells)))
    random.shuffle(indices)
    turn_dirs = [LEFT_TURN[direction], RIGHT_TURN[direction]]

    for idx in indices:
        turn_cell = free_cells[idx]
        trail = set(free_cells[:idx])
        next_used_black = used_black | trail

        random.shuffle(turn_dirs)
        for new_dir in turn_dirs:
            kind = _mirror_kind_for_turn(direction, new_dir)
            offsets = [(-1, -1), (1, 1)] if kind == "\\" else [(-1, 1), (1, -1)]
            random.shuffle(offsets)
            for dr, dc in offsets:
                partner = (turn_cell[0] + dr, turn_cell[1] + dc)
                if not _can_place_red_pair(turn_cell, partner, room_h, room_w, next_used_black, red_cells):
                    continue
                result = _search_room_path(
                    room_h=room_h,
                    room_w=room_w,
                    current=turn_cell,
                    direction=new_dir,
                    remaining_turns=remaining_turns - 1,
                    used_black=next_used_black,
                    red_cells=red_cells | {turn_cell, partner},
                    mirrors=mirrors + [(turn_cell, partner, kind)],
                )
                if result is not None:
                    return result

    return None


def _window_cell(room_h: int, room_w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, room_w - 2)
    if side == "bottom":
        return room_h - 1, random.randint(1, room_w - 2)
    if side == "left":
        return random.randint(1, room_h - 2), 0
    if side == "right":
        return random.randint(1, room_h - 2), room_w - 1
    raise AssertionError("unknown side")


def _distance_from_path_ok(cell: tuple[int, int], used_black: set[tuple[int, int]]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in used_black)


def _add_decoy_mirrors(
    room: np.ndarray,
    used_black: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
) -> list[str]:
    room_h, room_w = room.shape
    decoy_kinds: list[str] = []

    for _ in range(random.choice((0, 0, 1, 2))):
        candidates: list[tuple[tuple[int, int], tuple[int, int], str]] = []
        for r in range(1, room_h - 1):
            for c in range(1, room_w - 1):
                a = (r, c)
                if a in used_black or a in red_cells or not _distance_from_path_ok(a, used_black):
                    continue
                for kind, offsets in (("\\", [(-1, -1), (1, 1)]), ("/", [(-1, 1), (1, -1)])):
                    for dr, dc in offsets:
                        b = (r + dr, c + dc)
                        if not _can_place_red_pair(a, b, room_h, room_w, used_black, red_cells):
                            continue
                        if not _distance_from_path_ok(b, used_black):
                            continue
                        candidates.append((a, b, kind))

        if not candidates:
            break

        a, b, kind = random.choice(candidates)
        room[a] = RED
        room[b] = RED
        red_cells |= {a, b}
        decoy_kinds.append(kind)

    return decoy_kinds


def _build_room(
    room_h: int,
    room_w: int,
    side: str,
    turn_count: int,
) -> tuple[np.ndarray, dict[str, object]] | None:
    for _ in range(ROOM_BUILD_ATTEMPTS):
        room = np.full((room_h, room_w), BLACK, dtype=int)
        room[0, :] = GREY
        room[-1, :] = GREY
        room[:, 0] = GREY
        room[:, -1] = GREY

        window = _window_cell(room_h, room_w, side)
        room[window] = YELLOW
        direction = SIDE_TO_DIR[side]

        result = _search_room_path(
            room_h=room_h,
            room_w=room_w,
            current=window,
            direction=direction,
            remaining_turns=turn_count,
            used_black=set(),
            red_cells=set(),
            mirrors=[],
        )
        if result is None:
            continue

        used_black = set(result["used_black"])
        red_cells = set(result["red_cells"])
        mirror_kinds = [kind for _cell, _partner, kind in result["mirrors"]]

        for cell, partner, _kind in result["mirrors"]:
            room[cell] = RED
            room[partner] = RED

        mirror_kinds.extend(_add_decoy_mirrors(room, used_black, red_cells))
        return room, {"mirror_kinds": mirror_kinds, "turn_count": turn_count, "side": side}

    return None


def _choose_turn_targets(room_count: int) -> list[int]:
    if room_count == 1:
        return [random.randint(0, MAX_TURNS)]

    targets = [0, random.randint(1, MAX_TURNS)]
    if room_count >= 3:
        targets.append(random.randint(1, MAX_TURNS))
    if room_count >= 4:
        targets.append(random.randint(0, MAX_TURNS))

    while len(targets) < room_count:
        targets.append(random.randint(0, MAX_TURNS))

    random.shuffle(targets)
    return targets[:room_count]


def _sample_room_size(turn_count: int) -> tuple[int, int]:
    min_outer = max(ROOM_OUTER_BASE[0], 6 + turn_count)
    max_outer = ROOM_OUTER_BASE[1]
    return (
        random.randint(min_outer, max_outer),
        random.randint(min_outer, max_outer),
    )


def _rect_conflicts(
    top: int,
    left: int,
    room_h: int,
    room_w: int,
    placed: list[tuple[int, int, int, int]],
) -> bool:
    r0 = top - PLACEMENT_MARGIN
    c0 = left - PLACEMENT_MARGIN
    r1 = top + room_h - 1 + PLACEMENT_MARGIN
    c1 = left + room_w - 1 + PLACEMENT_MARGIN

    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
            return True
    return False


def _place_rooms(
    grid_h: int,
    grid_w: int,
    rooms: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(rooms)), key=lambda i: rooms[i].shape[0] * rooms[i].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(rooms)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        room_h, room_w = rooms[idx].shape
        if room_h + 2 > grid_h or room_w + 2 > grid_w:
            return None

        success = False
        for _ in range(120):
            top = random.randint(1, grid_h - room_h - 1)
            left = random.randint(1, grid_w - room_w - 1)
            if _rect_conflicts(top, left, room_h, room_w, placed_rects):
                continue
            placements[idx] = (top, left)
            placed_rects.append((top, left, top + room_h - 1, left + room_w - 1))
            success = True
            break
        if not success:
            return None

    assert all(p is not None for p in placements)
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    room_count = random.randint(*ROOM_COUNT_RANGE)

    turn_targets = _choose_turn_targets(room_count)
    side_order = random.sample(SIDES, len(SIDES))

    rooms: list[np.ndarray] = []
    mirror_kinds_seen: list[str] = []
    for idx in range(room_count):
        turn_count = turn_targets[idx]
        side = side_order[idx % len(side_order)]
        room_h, room_w = _sample_room_size(turn_count)
        built = _build_room(room_h, room_w, side, turn_count)
        if built is None:
            return None
        room, info = built
        rooms.append(room)
        mirror_kinds_seen.extend(info["mirror_kinds"])

    placements = _place_rooms(grid_h, grid_w, rooms)
    if placements is None:
        return None

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    for room, (top, left) in zip(rooms, placements, strict=False):
        room_h, room_w = room.shape
        inp[top : top + room_h, left : left + room_w] = room

    if sum(turn_targets) >= 2 and len(set(mirror_kinds_seen)) < 2:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(out, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
