from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
ALL_SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 30
ROOM_OUTER_RANGE = (4, 7)
AXIS_COUNT_CHOICES = (2, 3)
GAP_RANGE = (2, 5)
ROOM_COUNT_RANGE = (3, 6)
GEN_RETRIES = 400


@dataclass
class DoorSpec:
    side: str
    pos: int
    kind: str
    target: int | None = None


@dataclass
class RoomSpec:
    r0: int
    c0: int
    r1: int
    c1: int
    marker_color: int
    marker_pos: tuple[int, int]
    doors: list[DoorSpec] = field(default_factory=list)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_room_for_marker(
    grid: np.ndarray,
    bg: int,
    border: int,
    marker_comp: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    rows = [r for r, _ in marker_comp]
    cols = [c for _, c in marker_comp]
    mr0, mr1 = min(rows), max(rows)
    mc0, mc1 = min(cols), max(cols)
    marker_color = int(grid[marker_comp[0]])
    marker_size = len(marker_comp)

    best_key: tuple[int, int, int, int, int, int] | None = None
    best_rect: tuple[int, int, int, int] | None = None

    for r0 in range(mr0):
        for r1 in range(mr1 + 1, h):
            if r1 - r0 + 1 < 3:
                continue
            for c0 in range(mc0):
                for c1 in range(mc1 + 1, w):
                    if c1 - c0 + 1 < 3:
                        continue

                    if int(grid[r0, c0]) != border or int(grid[r0, c1]) != border:
                        continue
                    if int(grid[r1, c0]) != border or int(grid[r1, c1]) != border:
                        continue

                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if np.any((interior != bg) & (interior != marker_color)):
                        continue
                    if int(np.count_nonzero(interior == marker_color)) != marker_size:
                        continue

                    door_count = 0
                    border_count = 4
                    ok = True

                    for c in range(c0 + 1, c1):
                        top = int(grid[r0, c])
                        bottom = int(grid[r1, c])
                        if top == bg:
                            if int(grid[r0, c - 1]) != border or int(grid[r0, c + 1]) != border:
                                ok = False
                                break
                            door_count += 1
                        elif top == border:
                            border_count += 1
                        else:
                            ok = False
                            break

                        if bottom == bg:
                            if int(grid[r1, c - 1]) != border or int(grid[r1, c + 1]) != border:
                                ok = False
                                break
                            door_count += 1
                        elif bottom == border:
                            border_count += 1
                        else:
                            ok = False
                            break

                    if not ok:
                        continue

                    for r in range(r0 + 1, r1):
                        left = int(grid[r, c0])
                        right = int(grid[r, c1])
                        if left == bg:
                            if int(grid[r - 1, c0]) != border or int(grid[r + 1, c0]) != border:
                                ok = False
                                break
                            door_count += 1
                        elif left == border:
                            border_count += 1
                        else:
                            ok = False
                            break

                        if right == bg:
                            if int(grid[r - 1, c1]) != border or int(grid[r + 1, c1]) != border:
                                ok = False
                                break
                            door_count += 1
                        elif right == border:
                            border_count += 1
                        else:
                            ok = False
                            break

                    if not ok or door_count == 0:
                        continue

                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    key = (area, -border_count, r0, c0, r1, c1)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_rect = (r0, c0, r1, c1)

    assert best_rect is not None, "failed to locate a valid room rectangle"
    return best_rect


def _doors_for_room(
    grid: np.ndarray,
    bg: int,
    border: int,
    rect: tuple[int, int, int, int],
) -> list[tuple[int, int, str]]:
    r0, c0, r1, c1 = rect
    doors: list[tuple[int, int, str]] = []

    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) == bg:
            assert int(grid[r0, c - 1]) == border and int(grid[r0, c + 1]) == border
            doors.append((r0, c, "top"))
        if int(grid[r1, c]) == bg:
            assert int(grid[r1, c - 1]) == border and int(grid[r1, c + 1]) == border
            doors.append((r1, c, "bottom"))

    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == bg:
            assert int(grid[r - 1, c0]) == border and int(grid[r + 1, c0]) == border
            doors.append((r, c0, "left"))
        if int(grid[r, c1]) == bg:
            assert int(grid[r - 1, c1]) == border and int(grid[r + 1, c1]) == border
            doors.append((r, c1, "right"))

    return doors


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)

    counts = Counter(map(int, g.flatten()))
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected at least one foreground color"
    border = max(non_bg, key=lambda color: counts[color])

    marker_mask = (g != bg) & (g != border)
    marker_comps = _components(marker_mask)
    assert marker_comps, "expected at least one room marker"

    out = g.copy()
    seen_rects: set[tuple[int, int, int, int]] = set()
    for comp in marker_comps:
        marker_colors = {int(g[r, c]) for r, c in comp}
        assert len(marker_colors) == 1, "each marker component must be monochrome"
        marker_color = next(iter(marker_colors))
        rect = _find_room_for_marker(g, bg, border, comp)
        seen_rects.add(rect)
        for door_r, door_c, side in _doors_for_room(g, bg, border, rect):
            dr, dc = SIDE_TO_DIR[side]
            path = [(door_r, door_c)]
            cr = door_r + dr
            cc = door_c + dc
            hit_wall = False
            while 0 <= cr < h and 0 <= cc < w:
                cell = int(g[cr, cc])
                if cell == border:
                    hit_wall = True
                    break
                if cell != bg:
                    hit_wall = False
                    break
                path.append((cr, cc))
                cr += dr
                cc += dc
            if not hit_wall:
                continue
            for pr, pc in path:
                out[pr, pc] = marker_color

    assert seen_rects, "expected at least one detected room"
    return out


def _sample_axis(rng: random.Random, count: int) -> tuple[int, list[int], list[int]] | None:
    for _ in range(200):
        sizes = [rng.randint(*ROOM_OUTER_RANGE) for _ in range(count)]
        gaps = [rng.randint(*GAP_RANGE) for _ in range(count - 1)]
        base = sum(sizes) + sum(gaps)
        low = max(GRID_MIN, base + 2)
        if low > GRID_MAX:
            continue
        total = rng.randint(low, GRID_MAX)
        extra = total - base
        before = rng.randint(1, extra - 1)
        after = extra - before
        starts: list[int] = []
        cursor = before
        for idx, size in enumerate(sizes):
            starts.append(cursor)
            cursor += size
            if idx < count - 1:
                cursor += gaps[idx]
        assert cursor + after == total
        return total, starts, sizes
    return None


def _choose_occupied_slots(
    rng: random.Random, row_count: int, col_count: int
) -> set[tuple[int, int]] | None:
    slots = [(r, c) for r in range(row_count) for c in range(col_count)]
    max_rooms = min(ROOM_COUNT_RANGE[1], len(slots))
    if max_rooms < ROOM_COUNT_RANGE[0]:
        return None
    for _ in range(200):
        room_count = rng.randint(ROOM_COUNT_RANGE[0], max_rooms)
        occupied = set(rng.sample(slots, room_count))
        has_alignment = False
        for r in range(row_count):
            if sum((r, c) in occupied for c in range(col_count)) >= 2:
                has_alignment = True
                break
        if not has_alignment:
            for c in range(col_count):
                if sum((r, c) in occupied for r in range(row_count)) >= 2:
                    has_alignment = True
                    break
        if has_alignment:
            return occupied
    return None


def _nearest_room_in_direction(
    occupied: set[tuple[int, int]],
    row: int,
    col: int,
    side: str,
) -> tuple[int, int] | None:
    if side == "left":
        cols = [c for r, c in occupied if r == row and c < col]
        return None if not cols else (row, max(cols))
    if side == "right":
        cols = [c for r, c in occupied if r == row and c > col]
        return None if not cols else (row, min(cols))
    if side == "top":
        rows = [r for r, c in occupied if c == col and r < row]
        return None if not rows else (max(rows), col)
    rows = [r for r, c in occupied if c == col and r > row]
    return None if not rows else (min(rows), col)


def _door_positions(room: RoomSpec, side: str) -> list[int]:
    if side in ("left", "right"):
        return list(range(room.r0 + 1, room.r1))
    return list(range(room.c0 + 1, room.c1))


def _build_room_requests(
    rng: random.Random,
    rooms: list[RoomSpec],
    slots: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    room_index_by_slot: dict[tuple[int, int], int],
) -> list[tuple[int, str, str, int | None]] | None:
    requests: list[tuple[int, str, str, int | None]] = []

    for room_idx, slot in enumerate(slots):
        row, col = slot
        valid_items: list[tuple[str, str, int]] = []
        invalid_items: list[tuple[str, str, None]] = []
        for side in ALL_SIDES:
            target_slot = _nearest_room_in_direction(occupied, row, col, side)
            if target_slot is None:
                invalid_items.append(("invalid", side, None))
            else:
                valid_items.append(("valid", side, room_index_by_slot[target_slot]))

        all_items = valid_items + invalid_items
        if not all_items:
            return None

        side_count = len({item[1] for item in all_items})
        want_two = side_count >= 2 and rng.random() < 0.55

        chosen: list[tuple[str, str, int | None]] = []
        if want_two:
            mixed_pairs = [
                (v_item, i_item)
                for v_item in valid_items
                for i_item in invalid_items
                if v_item[1] != i_item[1]
            ]
            if mixed_pairs:
                first, second = rng.choice(mixed_pairs)
                chosen = [first, second]
            else:
                all_pairs = [
                    (a, b)
                    for i, a in enumerate(all_items)
                    for b in all_items[i + 1 :]
                    if a[1] != b[1]
                ]
                if all_pairs:
                    first, second = rng.choice(all_pairs)
                    chosen = [first, second]

        if not chosen:
            if valid_items and invalid_items:
                pool = valid_items if rng.random() < 0.5 else invalid_items
            else:
                pool = all_items
            chosen = [rng.choice(pool)]

        for kind, side, target in chosen:
            requests.append((room_idx, kind, side, target))

    valid_count = sum(kind == "valid" for _, kind, _, _ in requests)
    invalid_count = sum(kind == "invalid" for _, kind, _, _ in requests)
    if valid_count == 0 or invalid_count == 0:
        return None
    return requests


def _assign_doors(
    rng: random.Random,
    rooms: list[RoomSpec],
    requests: list[tuple[int, str, str, int | None]],
) -> bool:
    used = [{side: set() for side in ALL_SIDES} for _ in rooms]
    reserved = [{side: set() for side in ALL_SIDES} for _ in rooms]

    valid_requests = [req for req in requests if req[1] == "valid"]
    invalid_requests = [req for req in requests if req[1] == "invalid"]
    rng.shuffle(valid_requests)
    rng.shuffle(invalid_requests)

    for room_idx, _kind, side, target_idx in valid_requests:
        assert target_idx is not None
        target_side = OPPOSITE_SIDE[side]
        source_positions = set(_door_positions(rooms[room_idx], side))
        target_positions = set(_door_positions(rooms[target_idx], target_side))
        candidates = (
            source_positions
            & target_positions
            - used[room_idx][side]
            - reserved[room_idx][side]
            - used[target_idx][target_side]
            - reserved[target_idx][target_side]
        )
        if not candidates:
            return False
        pos = rng.choice(sorted(candidates))
        used[room_idx][side].add(pos)
        reserved[target_idx][target_side].add(pos)
        rooms[room_idx].doors.append(DoorSpec(side=side, pos=pos, kind="valid", target=target_idx))

    for room_idx, _kind, side, _target_idx in invalid_requests:
        candidates = (
            set(_door_positions(rooms[room_idx], side))
            - used[room_idx][side]
            - reserved[room_idx][side]
        )
        if not candidates:
            return False
        pos = rng.choice(sorted(candidates))
        used[room_idx][side].add(pos)
        rooms[room_idx].doors.append(DoorSpec(side=side, pos=pos, kind="invalid"))

    return True


def _door_cell(room: RoomSpec, door: DoorSpec) -> tuple[int, int]:
    if door.side == "top":
        return room.r0, door.pos
    if door.side == "bottom":
        return room.r1, door.pos
    if door.side == "left":
        return door.pos, room.c0
    return door.pos, room.c1


def _render_case(shape: tuple[int, int], bg: int, border: int, rooms: list[RoomSpec]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(shape, bg, dtype=int)
    for room in rooms:
        grid[room.r0, room.c0 : room.c1 + 1] = border
        grid[room.r1, room.c0 : room.c1 + 1] = border
        grid[room.r0 : room.r1 + 1, room.c0] = border
        grid[room.r0 : room.r1 + 1, room.c1] = border

    for room in rooms:
        for door in room.doors:
            dr, dc = _door_cell(room, door)
            grid[dr, dc] = bg

    for room in rooms:
        mr, mc = room.marker_pos
        grid[mr, mc] = room.marker_color

    out = grid.copy()
    for room in rooms:
        for door in room.doors:
            if door.kind != "valid":
                continue
            color = room.marker_color
            target = rooms[door.target]
            if door.side == "right":
                out[door.pos, room.c1 : target.c0] = color
            elif door.side == "left":
                out[door.pos, target.c1 + 1 : room.c0 + 1] = color
            elif door.side == "bottom":
                out[room.r1 : target.r0, door.pos] = color
            else:
                out[target.r1 + 1 : room.r0 + 1, door.pos] = color

    return grid, out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    row_count = rng.choice(AXIS_COUNT_CHOICES)
    col_count = rng.choice(AXIS_COUNT_CHOICES)

    row_axis = _sample_axis(rng, row_count)
    col_axis = _sample_axis(rng, col_count)
    if row_axis is None or col_axis is None:
        return None
    height, row_starts, row_sizes = row_axis
    width, col_starts, col_sizes = col_axis

    occupied = _choose_occupied_slots(rng, row_count, col_count)
    if occupied is None:
        return None

    slots = sorted(occupied)
    bg = rng.randint(0, 9)
    border_choices = [color for color in range(10) if color != bg]
    border = rng.choice(border_choices)
    marker_choices = [color for color in range(10) if color not in {bg, border}]
    if len(marker_choices) < len(slots):
        return None
    marker_colors = rng.sample(marker_choices, len(slots))

    rooms: list[RoomSpec] = []
    room_index_by_slot: dict[tuple[int, int], int] = {}
    for idx, (row, col) in enumerate(slots):
        r0 = row_starts[row]
        c0 = col_starts[col]
        r1 = r0 + row_sizes[row] - 1
        c1 = c0 + col_sizes[col] - 1
        marker_r = rng.randint(r0 + 1, r1 - 1)
        marker_c = rng.randint(c0 + 1, c1 - 1)
        room = RoomSpec(
            r0=r0,
            c0=c0,
            r1=r1,
            c1=c1,
            marker_color=marker_colors[idx],
            marker_pos=(marker_r, marker_c),
        )
        room_index_by_slot[(row, col)] = idx
        rooms.append(room)

    requests = _build_room_requests(rng, rooms, slots, occupied, room_index_by_slot)
    if requests is None:
        return None
    if not _assign_doors(rng, rooms, requests):
        return None

    grid, out = _render_case((height, width), bg, border, rooms)
    if Counter(map(int, grid.flatten())).most_common(1)[0][0] != bg:
        return None
    if max((count for color, count in Counter(map(int, grid.flatten())).items() if color != bg), default=0) != Counter(
        map(int, grid.flatten())
    )[border]:
        return None
    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            grid, _out = case
            return grid
    raise RuntimeError("generate_input: exhausted retries")
