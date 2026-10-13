from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 800


@dataclass(frozen=True)
class Room:
    bbox: tuple[int, int, int, int]
    door: tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _is_corner(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return (r in (r0, r1)) and (c in (c0, c1))


def _room_area(room: Room) -> int:
    r0, c0, r1, c1 = room.bbox
    return (r1 - r0 - 1) * (c1 - c0 - 1)


def _find_wall_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g[g != BG], return_counts=True)
    assert colors.size > 0, "expected walls and markers"
    wall = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) >= 10, "wall color must form extended room borders"
    return wall


def _find_rooms(g: np.ndarray, wall: int) -> list[Room]:
    rooms: list[Room] = []
    for comp in _components4(g == wall):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "rooms need a nontrivial interior"

        perim = set(_perimeter_cells(box))
        comp_set = {tuple(map(int, rc)) for rc in comp}
        assert comp_set <= perim, "wall components must be one-cell-thick rectangles"

        missing = [cell for cell in perim if int(g[cell]) != wall]
        assert len(missing) == 1, "each room must have exactly one one-cell door gap"
        door = missing[0]
        assert not _is_corner(door, box), "door gaps cannot be corners"
        assert int(g[door]) == BG, "door gap must be background"

        expected = perim - {door}
        assert comp_set == expected, "room border must be complete except for its door"
        rooms.append(Room(box, door))

    assert rooms, "expected at least one room"
    rooms.sort(key=lambda room: room.bbox)
    return rooms


def _find_markers(g: np.ndarray, wall: int) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    color_to_pos: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere((g != BG) & (g != wall)):
        color = int(g[r, c])
        pos = (int(r), int(c))
        assert color not in color_to_pos, "marker colors must be unique singleton pixels"
        color_to_pos[color] = pos
        markers[pos] = color
    assert markers, "expected colored marker pixels"
    return markers


def _interior_index(rooms: list[Room]) -> dict[tuple[int, int], list[int]]:
    index: dict[tuple[int, int], list[int]] = {}
    for i, room in enumerate(rooms):
        r0, c0, r1, c1 = room.bbox
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                index.setdefault((r, c), []).append(i)
    return index


def _ray_hits(
    g: np.ndarray,
    rooms: list[Room],
    markers: dict[tuple[int, int], int],
    wall: int,
) -> dict[int, set[tuple[int, tuple[int, int], tuple[int, int]]]]:
    h, w = map(int, g.shape)
    interior_at = _interior_index(rooms)
    hits: dict[int, set[tuple[int, tuple[int, int], tuple[int, int]]]] = {i: set() for i in range(len(rooms))}

    for (sr, sc), color in markers.items():
        for dr, dc in DIRS:
            r, c = sr + dr, sc + dc
            ray_id = (int(color), (sr, sc), (dr, dc))
            while 0 <= r < h and 0 <= c < w:
                if int(g[r, c]) == wall:
                    break
                for room_idx in interior_at.get((r, c), ()):
                    hits[room_idx].add(ray_id)
                r += dr
                c += dc
    return hits


def _room_colors_from_rays(g: np.ndarray, rooms: list[Room], markers: dict[tuple[int, int], int], wall: int) -> dict[int, int]:
    hits = _ray_hits(g, rooms, markers, wall)
    colors: dict[int, int] = {}
    for idx, rays in hits.items():
        assert len(rays) == 1, "each room must be crossed by exactly one marker ray"
        color, _pos, _direction = next(iter(rays))
        colors[idx] = int(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each door-lit rectangular room with the color of the marker ray that passes through it.

    Input structure:
    - The background is black; a single wall color forms one-cell-thick rectangular room borders.
    - Every room border has exactly one one-cell background door gap, and the room interior starts as background.
    - Colored singleton marker pixels sit outside the walls; marker colors are all distinct, and some markers are decoys.

    Rule:
    - From every marker, trace four cardinal rays through background cells, door gaps, and room interiors.
    - A ray stops only when it reaches a wall-color cell.
    - If exactly one marker ray crosses a room's interior, recolor all background interior cells of that room to that marker color.
    - Wall cells, door gaps, marker pixels, and all other background cells remain unchanged.

    Key insight:
    - The correct color is not the nearest marker or a contained label; it is the unique straight line of sight
      that enters through aligned door gaps, possibly continuing through nested rooms before it is blocked by a wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "expected ARC palette colors"

    wall = _find_wall_color(g)
    rooms = _find_rooms(g, wall)
    markers = _find_markers(g, wall)
    room_colors = _room_colors_from_rays(g, rooms, markers, wall)

    out = g.copy()
    for idx in sorted(range(len(rooms)), key=lambda i: _room_area(rooms[i]), reverse=True):
        color = room_colors[idx]
        r0, c0, r1, c1 = rooms[idx].bbox
        base_region = g[r0 + 1 : r1, c0 + 1 : c1]
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target[base_region == BG] = color
    return out


# --------------- generator ---------------


@dataclass
class GenRoom:
    bbox: tuple[int, int, int, int]
    door: tuple[int, int]
    color: int
    source: tuple[int, int]
    is_chain: bool = False


def _side_for_door(box: tuple[int, int, int, int], door: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    r, c = door
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    assert c == c1
    return "right"


def _inward_dir(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _door_on_side(box: tuple[int, int, int, int], side: str, line_coord: int | None = None) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        c = random.randint(c0 + 1, c1 - 1) if line_coord is None else int(line_coord)
        return r0, c
    if side == "bottom":
        c = random.randint(c0 + 1, c1 - 1) if line_coord is None else int(line_coord)
        return r1, c
    if side == "left":
        r = random.randint(r0 + 1, r1 - 1) if line_coord is None else int(line_coord)
        return r, c0
    r = random.randint(r0 + 1, r1 - 1) if line_coord is None else int(line_coord)
    return r, c1


def _draw_room(grid: np.ndarray, box: tuple[int, int, int, int], door: tuple[int, int]) -> None:
    for r, c in _perimeter_cells(box):
        if (r, c) != door:
            assert int(grid[r, c]) == BG, "room walls must not overlap existing objects"
            grid[r, c] = WALL


def _paint_marker(grid: np.ndarray, pos: tuple[int, int], color: int) -> bool:
    r, c = pos
    h, w = map(int, grid.shape)
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) == (r, c):
                continue
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) not in (BG, WALL):
                return False
    grid[r, c] = int(color)
    return True


def _expanded(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _source_position_for_door(
    box: tuple[int, int, int, int],
    door: tuple[int, int],
    shape: tuple[int, int],
    distance: int | None = None,
) -> tuple[int, int] | None:
    side = _side_for_door(box, door)
    dr, dc = _inward_dir(side)
    h, w = map(int, shape)
    if distance is None:
        distance = random.randint(2, 5)
    r = door[0] - dr * int(distance)
    c = door[1] - dc * int(distance)
    if 0 <= r < h and 0 <= c < w:
        return int(r), int(c)
    return None


def _clear_between(grid: np.ndarray, start: tuple[int, int], stop: tuple[int, int], step: tuple[int, int]) -> bool:
    r, c = start
    dr, dc = step
    while (r, c) != stop:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or int(grid[r, c]) != BG:
            return False
        r += dr
        c += dc
    return int(grid[stop]) == BG


def _make_horizontal_chain(
    shape: tuple[int, int],
    color: int,
    direction: str,
) -> tuple[list[GenRoom], tuple[int, int]] | None:
    h, w = map(int, shape)
    length = random.choice((2, 2, 2, 3))
    outer_h = random.randint(13 if length == 3 else 10, min(20, h - 4))
    outer_w = random.randint(13 if length == 3 else 10, min(20, w - 7))
    r0 = random.randint(2, h - outer_h - 2)
    if direction == "right":
        lo, hi = 6, w - outer_w - 2
    else:
        lo, hi = 2, w - outer_w - 7
    if lo > hi:
        return None
    c0 = random.randint(lo, hi)
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    line_r = random.randint(r0 + 3, r1 - 3)

    boxes = [(r0, c0, r1, c1)]
    cur = boxes[0]
    for _ in range(1, length):
        pr0, pc0, pr1, pc1 = cur
        top_room = line_r - pr0
        bottom_room = pr1 - line_r
        if top_room < 3 or bottom_room < 3:
            return None
        top_shrink = random.randint(1, min(3, top_room - 2))
        bottom_shrink = random.randint(1, min(3, bottom_room - 2))
        nr0 = pr0 + top_shrink
        nr1 = pr1 - bottom_shrink
        if direction == "right":
            nc0 = pc0 + random.randint(3, 5)
            nc1 = pc1 - random.randint(2, 4)
        else:
            nc0 = pc0 + random.randint(2, 4)
            nc1 = pc1 - random.randint(3, 5)
        if nr1 - nr0 + 1 < 5 or nc1 - nc0 + 1 < 5:
            return None
        cur = (nr0, nc0, nr1, nc1)
        boxes.append(cur)

    side = "left" if direction == "right" else "right"
    rooms = [GenRoom(box, _door_on_side(box, side, line_r), color, (-1, -1), True) for box in boxes]
    src = _source_position_for_door(boxes[0], rooms[0].door, shape, random.randint(2, 5))
    if src is None:
        return None
    rooms = [GenRoom(room.bbox, room.door, color, src, True) for room in rooms]
    return rooms, src


def _make_vertical_chain(
    shape: tuple[int, int],
    color: int,
    direction: str,
) -> tuple[list[GenRoom], tuple[int, int]] | None:
    h, w = map(int, shape)
    length = random.choice((2, 2, 2, 3))
    outer_h = random.randint(13 if length == 3 else 10, min(20, h - 7))
    outer_w = random.randint(13 if length == 3 else 10, min(20, w - 4))
    if direction == "down":
        lo, hi = 6, h - outer_h - 2
    else:
        lo, hi = 2, h - outer_h - 7
    if lo > hi:
        return None
    r0 = random.randint(lo, hi)
    c0 = random.randint(2, w - outer_w - 2)
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    line_c = random.randint(c0 + 3, c1 - 3)

    boxes = [(r0, c0, r1, c1)]
    cur = boxes[0]
    for _ in range(1, length):
        pr0, pc0, pr1, pc1 = cur
        left_room = line_c - pc0
        right_room = pc1 - line_c
        if left_room < 3 or right_room < 3:
            return None
        left_shrink = random.randint(1, min(3, left_room - 2))
        right_shrink = random.randint(1, min(3, right_room - 2))
        nc0 = pc0 + left_shrink
        nc1 = pc1 - right_shrink
        if direction == "down":
            nr0 = pr0 + random.randint(3, 5)
            nr1 = pr1 - random.randint(2, 4)
        else:
            nr0 = pr0 + random.randint(2, 4)
            nr1 = pr1 - random.randint(3, 5)
        if nr1 - nr0 + 1 < 5 or nc1 - nc0 + 1 < 5:
            return None
        cur = (nr0, nc0, nr1, nc1)
        boxes.append(cur)

    side = "top" if direction == "down" else "bottom"
    rooms = [GenRoom(box, _door_on_side(box, side, line_c), color, (-1, -1), True) for box in boxes]
    src = _source_position_for_door(boxes[0], rooms[0].door, shape, random.randint(2, 5))
    if src is None:
        return None
    rooms = [GenRoom(room.bbox, room.door, color, src, True) for room in rooms]
    return rooms, src


def _place_independent_room(
    grid: np.ndarray,
    existing_outer: list[tuple[int, int, int, int]],
    color: int,
) -> GenRoom | None:
    h, w = map(int, grid.shape)
    for _ in range(200):
        rh = random.randint(5, 9)
        rw = random.randint(5, 10)
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if any(_boxes_overlap(_expanded(box, 2, grid.shape), _expanded(other, 0, grid.shape)) for other in existing_outer):
            continue
        side = random.choice(SIDES)
        door = _door_on_side(box, side)
        src = _source_position_for_door(box, door, grid.shape)
        if src is None:
            continue
        dr, dc = _inward_dir(side)
        if not _clear_between(grid, (src[0] + dr, src[1] + dc), door, (dr, dc)):
            continue
        temp = grid.copy()
        try:
            _draw_room(temp, box, door)
        except AssertionError:
            continue
        if not _paint_marker(temp, src, color):
            continue
        grid[:, :] = temp
        return GenRoom(box, door, color, src, False)
    return None


def _outside_all_rooms(pos: tuple[int, int], rooms: list[GenRoom]) -> bool:
    r, c = pos
    for room in rooms:
        r0, c0, r1, c1 = room.bbox
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    return True


def _aligned_with_any_door(pos: tuple[int, int], rooms: list[GenRoom]) -> bool:
    r, c = pos
    return any(r == room.door[0] or c == room.door[1] for room in rooms)


def _dist_to_box(pos: tuple[int, int], box: tuple[int, int, int, int]) -> int:
    r, c = pos
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _place_decoys(grid: np.ndarray, rooms: list[GenRoom], colors: list[int]) -> list[tuple[tuple[int, int], int]]:
    decoys: list[tuple[tuple[int, int], int]] = []
    h, w = map(int, grid.shape)
    target_rooms = rooms[:]
    random.shuffle(target_rooms)
    for color in colors:
        candidates: list[tuple[int, int]] = []
        for room in target_rooms[:1] + rooms:
            r0, c0, r1, c1 = room.bbox
            for r in range(max(0, r0 - 3), min(h, r1 + 4)):
                for c in range(max(0, c0 - 3), min(w, c1 + 4)):
                    pos = (r, c)
                    if int(grid[r, c]) != BG:
                        continue
                    if not _outside_all_rooms(pos, rooms) or _aligned_with_any_door(pos, rooms):
                        continue
                    if _dist_to_box(pos, room.bbox) > 3:
                        continue
                    candidates.append(pos)
        random.shuffle(candidates)
        placed = False
        for pos in candidates:
            if _paint_marker(grid, pos, color):
                decoys.append((pos, color))
                placed = True
                break
        if placed:
            continue

        for _ in range(300):
            pos = (random.randrange(h), random.randrange(w))
            if int(grid[pos]) != BG:
                continue
            if not _outside_all_rooms(pos, rooms) or _aligned_with_any_door(pos, rooms):
                continue
            if _paint_marker(grid, pos, color):
                decoys.append((pos, color))
                break
    return decoys


def _expected_output(inp: np.ndarray, rooms: list[GenRoom]) -> np.ndarray:
    out = inp.copy()
    for room in sorted(rooms, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]), reverse=True):
        r0, c0, r1, c1 = room.bbox
        base = inp[r0 + 1 : r1, c0 + 1 : c1]
        dest = out[r0 + 1 : r1, c0 + 1 : c1]
        dest[base == BG] = int(room.color)
    return out


def _has_proximity_decoy(rooms: list[GenRoom], decoys: list[tuple[tuple[int, int], int]]) -> bool:
    if not decoys:
        return False
    for room in rooms:
        true_dist = _dist_to_box(room.source, room.bbox)
        if any(_dist_to_box(pos, room.bbox) < true_dist for pos, _color in decoys):
            return True
    return False


def _room_plans_from_solver(inp: np.ndarray) -> tuple[list[Room], dict[int, int]]:
    wall = _find_wall_color(inp)
    rooms = _find_rooms(inp, wall)
    markers = _find_markers(inp, wall)
    return rooms, _room_colors_from_rays(inp, rooms, markers, wall)


def _gen_attempt() -> np.ndarray | None:
    use_chain = random.random() < 0.7
    h = random.randint(24, 30) if use_chain else random.randint(20, 30)
    w = random.randint(24, 30) if use_chain else random.randint(20, 30)
    grid = np.full((h, w), BG, dtype=int)
    palette = list(MARKER_COLORS)
    random.shuffle(palette)
    rooms: list[GenRoom] = []
    occupied_outer: list[tuple[int, int, int, int]] = []

    if use_chain:
        chain_color = palette.pop()
        orientation = random.choice(("right", "left", "down", "up"))
        if orientation in ("right", "left"):
            chain = _make_horizontal_chain(grid.shape, chain_color, orientation)
        else:
            chain = _make_vertical_chain(grid.shape, chain_color, orientation)
        if chain is None:
            return None
        chain_rooms, src = chain
        try:
            for room in chain_rooms:
                _draw_room(grid, room.bbox, room.door)
        except AssertionError:
            return None
        if not _paint_marker(grid, src, chain_color):
            return None
        rooms.extend(chain_rooms)
        occupied_outer.append(chain_rooms[0].bbox)
        independent_count = 1
    else:
        independent_count = random.randint(2, 4)

    for _ in range(independent_count):
        if not palette:
            return None
        color = palette.pop()
        placed = _place_independent_room(grid, occupied_outer, color)
        if placed is None:
            return None
        rooms.append(placed)
        occupied_outer.append(placed.bbox)

    decoy_count = random.randint(1, min(3, len(palette)))
    decoy_colors = [palette.pop() for _ in range(decoy_count)]
    decoys = _place_decoys(grid, rooms, decoy_colors)
    if len(decoys) != decoy_count:
        return None

    if len(rooms) < 2 or len({room.color for room in rooms}) < 2:
        return None
    if any(_aligned_with_any_door(pos, rooms) for pos, _color in decoys):
        return None
    if not _has_proximity_decoy(rooms, decoys):
        return None

    expected = _expected_output(grid, rooms)
    try:
        actual = main(grid)
        detected_rooms, detected_colors = _room_plans_from_solver(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(detected_rooms) != len(rooms):
        return None
    if len(set(detected_colors.values())) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid door-gap ray room input after {GEN_ATTEMPTS} attempts")
