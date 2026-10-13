from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Room:
    bbox: tuple[int, int, int, int]
    border_color: int
    floor_color: int
    doors: tuple[tuple[int, int], ...]
    interior_area: int
    interior_bits: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _build_prefixes(grid: np.ndarray) -> dict[int, np.ndarray]:
    prefixes: dict[int, np.ndarray] = {}
    for color in np.unique(grid):
        mask = (grid == int(color)).astype(int)
        prefixes[int(color)] = mask.cumsum(axis=0).cumsum(axis=1)
    return prefixes


def _rect_count(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _interior_bits(h: int, w: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    bits = 0
    for r in range(r0 + 1, r1):
        base = r * w
        for c in range(c0 + 1, c1):
            bits |= 1 << (base + c)
    return bits


def _perimeter_bg_cells(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    doors: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) == bg:
            doors.append((r0, c))
        if int(grid[r1, c]) == bg:
            doors.append((r1, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == bg:
            doors.append((r, c0))
        if int(grid[r, c1]) == bg:
            doors.append((r, c1))
    return doors


def _valid_candidate(
    grid: np.ndarray,
    prefixes: dict[int, np.ndarray],
    bg: int,
    bbox: tuple[int, int, int, int],
) -> Room | None:
    r0, c0, r1, c1 = bbox
    border = int(grid[r0, c0])
    if border == bg:
        return None
    if (
        int(grid[r0, c1]) != border
        or int(grid[r1, c0]) != border
        or int(grid[r1, c1]) != border
    ):
        return None

    floor = int(grid[r0 + 1, c0 + 1])
    if floor == border:
        return None

    interior_area = (r1 - r0 - 1) * (c1 - c0 - 1)
    if _rect_count(prefixes[floor], r0 + 1, c0 + 1, r1 - 1, c1 - 1) != interior_area:
        return None

    perimeter_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    border_perim = _rect_count(prefixes[border], r0, c0, r1, c1) - _rect_count(
        prefixes[border], r0 + 1, c0 + 1, r1 - 1, c1 - 1
    )
    bg_perim = _rect_count(prefixes[bg], r0, c0, r1, c1) - _rect_count(
        prefixes[bg], r0 + 1, c0 + 1, r1 - 1, c1 - 1
    )
    if border_perim + bg_perim != perimeter_area:
        return None

    doors = _perimeter_bg_cells(grid, bbox, bg)
    if len(doors) != bg_perim or len(doors) > 2:
        return None
    if len(doors) == 2:
        (r_a, c_a), (r_b, c_b) = doors
        if abs(r_a - r_b) + abs(c_a - c_b) == 1:
            return None

    return Room(
        bbox=bbox,
        border_color=border,
        floor_color=floor,
        doors=tuple(sorted(doors)),
        interior_area=interior_area,
        interior_bits=_interior_bits(*grid.shape, bbox),
    )


def _find_candidates(grid: np.ndarray, bg: int) -> list[Room]:
    h, w = grid.shape
    prefixes = _build_prefixes(grid)
    rooms: list[Room] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) == bg:
                continue
            for r1 in range(r0 + 2, h):
                for c1 in range(c0 + 2, w):
                    room = _valid_candidate(grid, prefixes, bg, (r0, c0, r1, c1))
                    if room is not None:
                        rooms.append(room)
    door_occurrences = Counter(door for room in rooms for door in room.doors)
    return [
        room
        for room in rooms
        if all(door_occurrences[door] >= 2 for door in room.doors)
    ]


def _greedy_rooms(candidates: list[Room]) -> list[Room]:
    rooms: list[Room] = []
    occupied = 0
    for room in sorted(
        candidates,
        key=lambda r: (r.interior_area, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]),
    ):
        if room.interior_bits & occupied:
            continue
        rooms.append(room)
        occupied |= room.interior_bits
    return rooms


def _selected_rooms_valid(rooms: list[Room], bg: int) -> bool:
    if not rooms:
        return False

    door_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        for door in room.doors:
            door_map[door].append(idx)
    if any(len(indices) != 2 for indices in door_map.values()):
        return False

    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(rooms))}
    for indices in door_map.values():
        a, b = indices
        if a == b:
            return False
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbors in adjacency.values():
        if len(neighbors) > 2:
            return False
        if len(neighbors) == 2 and neighbors[0] == neighbors[1]:
            return False

    seen: set[int] = set()
    for start in range(len(rooms)):
        if start in seen:
            continue
        comp: list[int] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        degrees = [len(adjacency[idx]) for idx in comp]
        if len(comp) == 1:
            if degrees[0] != 0:
                return False
        else:
            if sum(deg == 1 for deg in degrees) != 2 or any(deg not in (1, 2) for deg in degrees):
                return False
        if all(rooms[idx].floor_color == bg for idx in comp):
            return False
    return True


def _search_rooms(candidates: list[Room], bg: int) -> list[Room]:
    ordered = sorted(
        candidates,
        key=lambda r: (r.interior_area, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]),
    )
    n = len(ordered)
    best_indices: list[int] = []
    best_area = -1

    def dfs(i: int, occupied: int, chosen: list[int], area: int) -> None:
        nonlocal best_indices, best_area
        if len(chosen) + (n - i) < len(best_indices):
            return
        if i == n:
            rooms = [ordered[idx] for idx in chosen]
            if not _selected_rooms_valid(rooms, bg):
                return
            if len(chosen) > len(best_indices) or (len(chosen) == len(best_indices) and area > best_area):
                best_indices = chosen.copy()
                best_area = area
            return

        room = ordered[i]
        if not (room.interior_bits & occupied):
            chosen.append(i)
            dfs(i + 1, occupied | room.interior_bits, chosen, area + room.interior_area)
            chosen.pop()
        dfs(i + 1, occupied, chosen, area)

    dfs(0, 0, [], 0)
    return [ordered[idx] for idx in best_indices]


def _detect_rooms(grid: np.ndarray, bg: int) -> list[Room]:
    candidates = _find_candidates(grid, bg)
    greedy = _greedy_rooms(candidates)
    if _selected_rooms_valid(greedy, bg):
        return greedy
    return _search_rooms(candidates, bg)


def _components(adjacency: dict[int, list[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in adjacency:
        if start in seen:
            continue
        comp: list[int] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def _ordered_chain(room_ids: list[int], adjacency: dict[int, list[int]], rooms: list[Room]) -> list[int]:
    if len(room_ids) == 1:
        return room_ids
    endpoints = [idx for idx in room_ids if len(adjacency[idx]) == 1]
    assert len(endpoints) == 2
    start = min(endpoints, key=lambda idx: (rooms[idx].bbox[1], rooms[idx].bbox[0]))
    ordered = [start]
    prev = -1
    cur = start
    while True:
        next_nodes = [idx for idx in adjacency[cur] if idx != prev]
        if not next_nodes:
            break
        nxt = next_nodes[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def _fill_room(out: np.ndarray, room: Room, color: int) -> None:
    r0, c0, r1, c1 = room.bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate room floor colors through door-linked chains of rectangular rooms.

    Input structure:
    - The background is the dominant color.
    - One or more axis-aligned rectangular rooms are enclosed by one-pixel-thick colored borders.
    - Each room interior is uniform: either all background (an empty room) or one non-background floor color distinct from the border.
    - Adjacent rooms may share a wall cell that is replaced by background, creating a one-pixel door; each room has at most two such doors.

    Rule:
    - Detect every rectangular room and connect rooms that share the same door pixel.
    - Each connected component of rooms forms a chain with at least one colored source room.
    - For every empty room, find the nearest colored room along its chain and fill that room's interior with the source floor color.
    - If two colored rooms are equally close, use the source that appears earlier when the chain is read from the left endpoint, or from the top endpoint when the chain is vertical.
    - Leave all borders and door pixels unchanged.

    Key insight:
    - The background gaps in otherwise solid rectangular borders are meaningful doors, so the task is to recover the room graph first and only then propagate floor colors by nearest-source distance along each path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rooms = _detect_rooms(g, bg)
    assert rooms, "expected at least one valid room"

    door_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        for door in room.doors:
            door_map[door].append(idx)

    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(rooms))}
    for holders in door_map.values():
        assert len(holders) == 2
        a, b = holders
        adjacency[a].append(b)
        adjacency[b].append(a)

    out = g.copy()
    for component in _components(adjacency):
        ordered = _ordered_chain(component, adjacency, rooms)
        source_positions = [i for i, idx in enumerate(ordered) if rooms[idx].floor_color != bg]
        assert source_positions, "every chain must contain a colored source room"
        for pos, room_idx in enumerate(ordered):
            room = rooms[room_idx]
            if room.floor_color != bg:
                continue
            best_source = min(source_positions, key=lambda src: (abs(src - pos), src))
            source_color = rooms[ordered[best_source]].floor_color
            _fill_room(out, room, source_color)
    return out


GRID_LIMIT = 30
CANVAS_SIZE = 30
CHAIN_COUNT_OPTIONS = (1, 1, 2)
CASE_TYPES = ("one_end", "both_ends_no_tie", "both_ends_tie")
ORIENTATIONS = ("horizontal", "vertical")
MARGIN_RANGE = (1, 8)
CHAIN_GAP = 2
GEN_RETRIES = 400


def _choose_case_length(case_type: str) -> int:
    if case_type == "one_end":
        return random.randint(2, 5)
    if case_type == "both_ends_no_tie":
        return random.choice((2, 4))
    return random.choice((3, 5))


def _available_colors(exclude: set[int]) -> list[int]:
    return [color for color in range(10) if color not in exclude]


def _make_chain_spec(bg: int) -> dict:
    case_type = random.choice(CASE_TYPES)
    orientation = random.choice(ORIENTATIONS)
    length = _choose_case_length(case_type)

    used = {bg}
    border = random.choice(_available_colors(used))
    used.add(border)

    if orientation == "horizontal":
        interior_h = random.randint(2, 4)
        interior_sizes = [random.randint(2, 4) for _ in range(length)]
        total_h = interior_h + 2
        total_w = sum(interior_sizes) + length + 1
        door_offsets = [random.randint(1, interior_h) for _ in range(length - 1)]
    else:
        interior_w = random.randint(2, 4)
        interior_sizes = [random.randint(2, 4) for _ in range(length)]
        total_h = sum(interior_sizes) + length + 1
        total_w = interior_w + 2
        door_offsets = [random.randint(1, interior_w) for _ in range(length - 1)]

    floors = [bg] * length
    if case_type == "one_end":
        src_idx = random.choice((0, length - 1))
        floors[src_idx] = random.choice(_available_colors(used))
    else:
        src_colors = random.sample(_available_colors(used), 2)
        floors[0], floors[-1] = src_colors

    return {
        "case_type": case_type,
        "orientation": orientation,
        "length": length,
        "border": border,
        "interior_sizes": interior_sizes,
        "door_offsets": door_offsets,
        "floors": floors,
        "shape": (total_h, total_w),
    }


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_chains(specs: list[dict]) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(specs)))
    random.shuffle(order)

    for idx in order:
        h, w = specs[idx]["shape"]
        positions = [
            (r, c)
            for r in range(CANVAS_SIZE - h + 1)
            for c in range(CANVAS_SIZE - w + 1)
        ]
        random.shuffle(positions)
        found = None
        for r, c in positions:
            bbox = (r, c, r + h - 1, c + w - 1)
            if any(_bbox_intersects(bbox, other, CHAIN_GAP) for other in bboxes):
                continue
            found = (r, c)
            bboxes.append(bbox)
            break
        if found is None:
            return None
        placed.append(found)

    anchors = [None] * len(specs)
    for idx, anchor in zip(order, placed, strict=False):
        anchors[idx] = anchor
    return anchors


def _draw_chain(
    grid: np.ndarray,
    spec: dict,
    anchor: tuple[int, int],
    bg: int,
) -> list[tuple[int, int, int, int]]:
    r0, c0 = anchor
    border = int(spec["border"])
    rooms: list[tuple[int, int, int, int]] = []

    if spec["orientation"] == "horizontal":
        interior_h = spec["shape"][0] - 2
        c = c0
        for i, width in enumerate(spec["interior_sizes"]):
            room = (r0, c, r0 + interior_h + 1, c + width + 1)
            rooms.append(room)
            rr0, cc0, rr1, cc1 = room
            grid[rr0 : rr1 + 1, cc0] = border
            grid[rr0 : rr1 + 1, cc1] = border
            grid[rr0, cc0 : cc1 + 1] = border
            grid[rr1, cc0 : cc1 + 1] = border
            floor = int(spec["floors"][i])
            if floor != bg:
                grid[rr0 + 1 : rr1, cc0 + 1 : cc1] = floor
            c = cc1
        for door_idx, door_offset in enumerate(spec["door_offsets"]):
            shared_c = rooms[door_idx][3]
            grid[r0 + door_offset, shared_c] = bg
    else:
        interior_w = spec["shape"][1] - 2
        r = r0
        for i, height in enumerate(spec["interior_sizes"]):
            room = (r, c0, r + height + 1, c0 + interior_w + 1)
            rooms.append(room)
            rr0, cc0, rr1, cc1 = room
            grid[rr0 : rr1 + 1, cc0] = border
            grid[rr0 : rr1 + 1, cc1] = border
            grid[rr0, cc0 : cc1 + 1] = border
            grid[rr1, cc0 : cc1 + 1] = border
            floor = int(spec["floors"][i])
            if floor != bg:
                grid[rr0 + 1 : rr1, cc0 + 1 : cc1] = floor
            r = rr1
        for door_idx, door_offset in enumerate(spec["door_offsets"]):
            shared_r = rooms[door_idx][2]
            grid[shared_r, c0 + door_offset] = bg

    return rooms


def _expected_fill_colors(floors: list[int], bg: int) -> list[int]:
    sources = [idx for idx, color in enumerate(floors) if color != bg]
    assert sources
    return [
        color if color != bg else floors[min(sources, key=lambda src: (abs(src - idx), src))]
        for idx, color in enumerate(floors)
    ]


def _crop_bounds(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    non_bg = np.argwhere(grid != bg)
    assert non_bg.size
    r0, c0 = map(int, non_bg.min(axis=0))
    r1, c1 = map(int, non_bg.max(axis=0))

    def sample_margin(limit: int) -> int:
        if limit <= 0:
            return 0
        low = min(MARGIN_RANGE[0], limit)
        high = min(MARGIN_RANGE[1], limit)
        if low > high:
            low = high
        return random.randint(low, high)

    for _ in range(40):
        top = sample_margin(r0)
        left = sample_margin(c0)
        bottom = sample_margin(grid.shape[0] - 1 - r1)
        right = sample_margin(grid.shape[1] - 1 - c1)
        rr0, cc0 = r0 - top, c0 - left
        rr1, cc1 = r1 + bottom, c1 + right
        crop = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if _get_bg(crop) == bg:
            return rr0, cc0, rr1, cc1

    return 0, 0, grid.shape[0] - 1, grid.shape[1] - 1


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randint(0, 9)
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), bg, dtype=int)
    specs = [_make_chain_spec(bg) for _ in range(random.choice(CHAIN_COUNT_OPTIONS))]
    anchors = _place_chains(specs)
    if anchors is None:
        return None

    expected = canvas.copy()
    changed = False
    for spec, anchor in zip(specs, anchors, strict=False):
        rooms = _draw_chain(canvas, spec, anchor, bg)
        _draw_chain(expected, spec, anchor, bg)
        filled = _expected_fill_colors(spec["floors"], bg)
        for room, color, original in zip(rooms, filled, spec["floors"], strict=False):
            if original == bg:
                _fill_room(expected, Room(room, 0, 0, (), 0, 0), color)
                changed = True

    if not changed:
        return None

    r0, c0, r1, c1 = _crop_bounds(canvas, bg)
    inp = canvas[r0 : r1 + 1, c0 : c1 + 1]
    out = expected[r0 : r1 + 1, c0 : c1 + 1]
    if _get_bg(inp) != bg or _get_bg(out) != bg:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        inp, out = built
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if max(inp.shape) > GRID_LIMIT or max(out.shape) > GRID_LIMIT:
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid room-chain puzzle")
