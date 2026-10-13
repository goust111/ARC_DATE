from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    YELLOW = 4
    GREY = 5
    ORANGE = 7
    TEAL = 8


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Segment:
    axis: str
    rect: BBox
    length: int


@dataclass(frozen=True)
class Room:
    segment_id: int
    walls: frozenset[Coord]
    interior: frozenset[Coord]


BLOCK_SIZE = 2


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _component_dict(coords: list[Coord], color: int) -> dict:
    arr = np.array(coords, dtype=int)
    return {
        "color": int(color),
        "coords": arr,
        "bbox": bbox_coords(arr),
        "area": int(arr.shape[0]),
    }


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(_component_dict(cells, color))
    return comps


def _blocks(grid: np.ndarray) -> tuple[dict, list[dict]]:
    starts = _components_of_color(grid, Color.RED)
    targets = _components_of_color(grid, Color.TEAL)
    assert len(starts) == 1, "expected exactly one red start block"
    assert 4 <= len(targets) <= 6, "expected four to six teal target blocks"
    for block in starts + targets:
        assert int(block["area"]) == 4, "expected 2x2 blocks"
        assert bbox_h(block["bbox"]) == 2 and bbox_w(block["bbox"]) == 2, "expected 2x2 blocks"
    return starts[0], targets


def _visible_step(grid: np.ndarray, a: dict, b: dict) -> tuple[str, int, BBox] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1 and not np.all(grid[ar0 : ar1 + 1, c0 : c1 + 1] == Color.BLACK):
            return None
        rect = (int(ar0), int(c0), int(ar1), int(c1))
        return "h", abs(int(bc0) - int(ac0)), rect

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac0 : ac1 + 1] == Color.BLACK):
            return None
        rect = (int(r0), int(ac0), int(r1), int(ac1))
        return "v", abs(int(br0) - int(ar0)), rect

    return None


def _segment_length(axis: str, rect: BBox) -> int:
    if axis == "h":
        return max(0, int(rect[3] - rect[1] + 1))
    return max(0, int(rect[2] - rect[0] + 1))


def _paint_segment(out: np.ndarray, segment_ids: np.ndarray, seg_id: int, rect: BBox) -> None:
    r0, c0, r1, c1 = rect
    if r0 > r1 or c0 > c1:
        return
    out[r0 : r1 + 1, c0 : c1 + 1] = Color.ORANGE
    current = segment_ids[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(current == -1), "corridor segments must not overlap"
    current[:] = int(seg_id)


def _greedy_corridors(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Segment]]:
    current, unvisited = _blocks(grid)
    out = np.asarray(grid, dtype=int).copy()
    segment_ids = np.full(out.shape, -1, dtype=int)
    segments: list[Segment] = []

    while True:
        candidates = []
        for block in unvisited:
            step = _visible_step(grid, current, block)
            if step is None:
                continue
            axis, distance, rect = step
            candidates.append((distance, axis, rect, block))

        if not candidates:
            break

        min_distance = min(distance for distance, _axis, _rect, _block in candidates)
        nearest = [item for item in candidates if item[0] == min_distance]
        assert len(nearest) == 1, "expected a unique nearest visible teal block"

        _distance, axis, rect, block = nearest[0]
        seg = Segment(axis=axis, rect=rect, length=_segment_length(axis, rect))
        seg_id = len(segments)
        _paint_segment(out, segment_ids, seg_id, rect)
        segments.append(seg)
        unvisited = [candidate for candidate in unvisited if candidate is not block]
        current = block

    return out, segment_ids, segments


def _region_segment_adjacency(grid: np.ndarray, segment_ids: np.ndarray, coords: np.ndarray) -> tuple[set[int], set[int]]:
    region = {tuple(map(int, rc)) for rc in coords.tolist()}
    neighbor_colors: set[int] = set()
    neighbor_segments: set[int] = set()
    h, w = grid.shape
    for rr, cc in region:
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region:
                continue
            color = int(grid[nr, nc])
            if color != Color.BLACK:
                neighbor_colors.add(color)
            seg_id = int(segment_ids[nr, nc])
            if seg_id >= 0:
                neighbor_segments.add(seg_id)
    return neighbor_colors, neighbor_segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the greedy orange corridors between 2x2 blocks, then recolor each adjacent enclosed chamber by corridor-length parity.

    Input structure:
    - The grid has a black background, one red 2x2 start block, four to six teal 2x2 target blocks, and thin grey wall segments.
    - Greedy movement is allowed only between blocks that share the same two-row or two-column band and have only black cells between them.
    - Some black chambers become enclosed once a corridor is painted along the open side left between grey walls.

    Rule:
    - Starting from the red block, repeatedly connect to the unique nearest visible unvisited teal block and paint the background gap orange.
    - Treat each straight orange gap as one corridor segment, using its span between block faces as the segment length.
    - In the intermediate grid after corridor painting, find every black connected region whose non-black boundary uses only grey walls and exactly one orange corridor segment.
    - Fill that region orange when the adjacent corridor length is odd, or yellow when it is even; keep the red, teal, grey, and corridor cells unchanged.

    Key insight:
    - The chamber color is not fixed by local wall shape alone: it depends on the emergent greedy traversal order, because that order determines the corridor segment lengths.
    """
    painted, segment_ids, segments = _greedy_corridors(np.asarray(grid, dtype=int))
    out = painted.copy()

    for region in _components_of_color(painted, Color.BLACK):
        neighbor_colors, neighbor_segments = _region_segment_adjacency(painted, segment_ids, region["coords"])
        if neighbor_colors and not neighbor_colors <= {Color.GREY, Color.ORANGE}:
            continue
        if len(neighbor_segments) != 1:
            continue
        seg_id = next(iter(neighbor_segments))
        fill = Color.ORANGE if segments[seg_id].length % 2 == 1 else Color.YELLOW
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(fill)

    return out


GEN_ROWS_RANGE = (20, 29)
GEN_COLS_RANGE = (20, 29)
GEN_TARGET_COUNT_RANGE = (4, 6)
GEN_EDGE_PADDING = 4
GEN_GAP_VALUES = tuple(range(4, 9))
GEN_START_TRIES = 120
GEN_MAX_ATTEMPTS = 500
GEN_ROOM_SPAN_RANGE = (2, 4)
GEN_ROOM_DEPTH_RANGE = (2, 4)


def _gen_bbox(pos: Coord) -> BBox:
    r, c = pos
    return r, c, r + BLOCK_SIZE - 1, c + BLOCK_SIZE - 1


def _gen_rects_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_expand_rect(box: BBox, margin: int) -> BBox:
    return box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin


def _gen_blocks_separated(a: Coord, b: Coord, margin: int = 1) -> bool:
    return not _gen_rects_overlap(_gen_expand_rect(_gen_bbox(a), margin), _gen_bbox(b))


def _gen_axis(a: Coord, b: Coord) -> str | None:
    if a[0] == b[0]:
        return "h"
    if a[1] == b[1]:
        return "v"
    return None


def _gen_corridor_rect(a: Coord, b: Coord) -> tuple[BBox | None, str | None]:
    axis = _gen_axis(a, b)
    if axis == "h":
        r = a[0]
        c0 = min(a[1], b[1]) + BLOCK_SIZE
        c1 = max(a[1], b[1]) - 1
        return (r, c0, r + BLOCK_SIZE - 1, c1), axis
    if axis == "v":
        c = a[1]
        r0 = min(a[0], b[0]) + BLOCK_SIZE
        r1 = max(a[0], b[0]) - 1
        return (r0, c, r1, c + BLOCK_SIZE - 1), axis
    return None, None


def _gen_distance(a: Coord, b: Coord) -> int | None:
    axis = _gen_axis(a, b)
    if axis == "h":
        return abs(int(a[1]) - int(b[1]))
    if axis == "v":
        return abs(int(a[0]) - int(b[0]))
    return None


def _gen_corridor_clear(a: Coord, b: Coord, positions: list[Coord]) -> bool:
    rect, axis = _gen_corridor_rect(a, b)
    if axis is None or rect is None:
        return False
    for pos in positions:
        if pos == a or pos == b:
            continue
        if _gen_rects_overlap(rect, _gen_bbox(pos)):
            return False
    return True


def _gen_visible_axis(a: Coord, b: Coord, positions: list[Coord]) -> str | None:
    if _gen_axis(a, b) is None:
        return None
    if not _gen_corridor_clear(a, b, positions):
        return None
    return _gen_axis(a, b)


def _gen_walk_is_forced(path: list[Coord]) -> bool:
    corridor_rects = []
    for a, b in zip(path, path[1:]):
        rect, axis = _gen_corridor_rect(a, b)
        if rect is None or axis is None:
            return False
        for prev in corridor_rects:
            if _gen_rects_overlap(rect, prev):
                return False
        corridor_rects.append(rect)

    positions = list(path)
    for i, current in enumerate(path[:-1]):
        candidates = []
        for block in path[i + 1 :]:
            axis = _gen_visible_axis(current, block, positions)
            if axis is not None:
                candidates.append((_gen_distance(current, block), block))
        if not candidates:
            return False
        nearest_distance = min(distance for distance, _block in candidates)
        nearest = [block for distance, block in candidates if distance == nearest_distance]
        if len(nearest) != 1 or nearest[0] != path[i + 1]:
            return False
    return True


def _gen_turn_count(path: list[Coord]) -> int:
    turns = 0
    prev_axis = None
    for a, b in zip(path, path[1:]):
        axis = _gen_axis(a, b)
        if prev_axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
    return turns


def _gen_in_bounds(pos: Coord, rows: int, cols: int) -> bool:
    r, c = pos
    return (
        GEN_EDGE_PADDING <= r <= rows - GEN_EDGE_PADDING - BLOCK_SIZE
        and GEN_EDGE_PADDING <= c <= cols - GEN_EDGE_PADDING - BLOCK_SIZE
    )


def _gen_shuffled(items: list, rng: random.Random) -> list:
    items = list(items)
    rng.shuffle(items)
    return items


def _gen_next_options(rows: int, cols: int, path: list[Coord], target_turns: int, total_steps: int, rng: random.Random):
    current = path[-1]
    prev_axis = _gen_axis(path[-2], path[-1]) if len(path) >= 2 else None
    turns_so_far = _gen_turn_count(path)
    moves_done = len(path) - 1
    moves_left_after = total_steps - (moves_done + 1)
    options = []

    for axis in ("h", "v"):
        new_turns = turns_so_far + int(prev_axis is not None and axis != prev_axis)
        if not (new_turns <= target_turns <= new_turns + moves_left_after):
            continue
        for sign in (-1, 1):
            for gap in _gen_shuffled(list(GEN_GAP_VALUES), rng):
                offset = sign * (gap + BLOCK_SIZE)
                candidate = (current[0], current[1] + offset) if axis == "h" else (current[0] + offset, current[1])
                if not _gen_in_bounds(candidate, rows, cols):
                    continue
                if not all(_gen_blocks_separated(candidate, other) for other in path):
                    continue
                candidate_path = path + [candidate]
                if not _gen_walk_is_forced(candidate_path):
                    continue
                options.append((axis, candidate))

    turn_options = [item for item in options if prev_axis is not None and item[0] != prev_axis]
    straight_options = [item for item in options if prev_axis is None or item[0] == prev_axis]
    if turn_options and turns_so_far < target_turns and rng.random() < 0.7:
        return _gen_shuffled(turn_options, rng) + _gen_shuffled(straight_options, rng)
    return _gen_shuffled(options, rng)


def _gen_extend_path(rows: int, cols: int, steps: int, target_turns: int, path: list[Coord], rng: random.Random) -> list[Coord] | None:
    if len(path) == steps + 1:
        if _gen_turn_count(path) == target_turns and _gen_walk_is_forced(path):
            return path
        return None

    for _axis, candidate in _gen_next_options(rows, cols, path, target_turns, steps, rng):
        found = _gen_extend_path(rows, cols, steps, target_turns, path + [candidate], rng)
        if found is not None:
            return found
    return None


def _gen_build_path(rows: int, cols: int, steps: int, rng: random.Random) -> list[Coord] | None:
    starts = [(r, c) for r in range(GEN_EDGE_PADDING, rows - GEN_EDGE_PADDING - BLOCK_SIZE + 1) for c in range(GEN_EDGE_PADDING, cols - GEN_EDGE_PADDING - BLOCK_SIZE + 1)]
    max_turns = min(steps - 1, 3)
    turn_choices = list(range(max_turns + 1))
    rng.shuffle(turn_choices)
    for target_turns in turn_choices:
        for start in _gen_shuffled(starts, rng)[:GEN_START_TRIES]:
            found = _gen_extend_path(rows, cols, steps, target_turns, [start], rng)
            if found is not None:
                return found
    return None


def _segment_cells(segment: Segment) -> set[Coord]:
    r0, c0, r1, c1 = segment.rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand4(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _room_candidates_for_segment(segment: Segment, seg_id: int, shape: tuple[int, int]) -> list[Room]:
    rows, cols = shape
    r0, c0, r1, c1 = segment.rect
    length = segment.length
    candidates: list[Room] = []
    max_span = min(GEN_ROOM_SPAN_RANGE[1], length - 2)
    if max_span < GEN_ROOM_SPAN_RANGE[0]:
        return candidates

    if segment.axis == "h":
        for span in range(GEN_ROOM_SPAN_RANGE[0], max_span + 1):
            for open_c0 in range(c0 + 1, c1 - span + 1):
                open_c1 = open_c0 + span - 1
                for depth in range(GEN_ROOM_DEPTH_RANGE[0], GEN_ROOM_DEPTH_RANGE[1] + 1):
                    top = r0 - depth - 1
                    if top >= 0:
                        walls = {(top, cc) for cc in range(open_c0 - 1, open_c1 + 2)}
                        walls |= {(rr, open_c0 - 1) for rr in range(top + 1, r0)}
                        walls |= {(rr, open_c1 + 1) for rr in range(top + 1, r0)}
                        interior = {(rr, cc) for rr in range(top + 1, r0) for cc in range(open_c0, open_c1 + 1)}
                        candidates.append(Room(seg_id, frozenset(walls), frozenset(interior)))

                    bottom = r1 + depth + 1
                    if bottom < rows:
                        walls = {(bottom, cc) for cc in range(open_c0 - 1, open_c1 + 2)}
                        walls |= {(rr, open_c0 - 1) for rr in range(r1 + 1, bottom)}
                        walls |= {(rr, open_c1 + 1) for rr in range(r1 + 1, bottom)}
                        interior = {(rr, cc) for rr in range(r1 + 1, bottom) for cc in range(open_c0, open_c1 + 1)}
                        candidates.append(Room(seg_id, frozenset(walls), frozenset(interior)))
    else:
        for span in range(GEN_ROOM_SPAN_RANGE[0], max_span + 1):
            for open_r0 in range(r0 + 1, r1 - span + 1):
                open_r1 = open_r0 + span - 1
                for depth in range(GEN_ROOM_DEPTH_RANGE[0], GEN_ROOM_DEPTH_RANGE[1] + 1):
                    left = c0 - depth - 1
                    if left >= 0:
                        walls = {(rr, left) for rr in range(open_r0 - 1, open_r1 + 2)}
                        walls |= {(open_r0 - 1, cc) for cc in range(left + 1, c0)}
                        walls |= {(open_r1 + 1, cc) for cc in range(left + 1, c0)}
                        interior = {(rr, cc) for rr in range(open_r0, open_r1 + 1) for cc in range(left + 1, c0)}
                        candidates.append(Room(seg_id, frozenset(walls), frozenset(interior)))

                    right = c1 + depth + 1
                    if right < cols:
                        walls = {(rr, right) for rr in range(open_r0 - 1, open_r1 + 2)}
                        walls |= {(open_r0 - 1, cc) for cc in range(c1 + 1, right)}
                        walls |= {(open_r1 + 1, cc) for cc in range(c1 + 1, right)}
                        interior = {(rr, cc) for rr in range(open_r0, open_r1 + 1) for cc in range(c1 + 1, right)}
                        candidates.append(Room(seg_id, frozenset(walls), frozenset(interior)))

    return candidates


def _room_is_valid(room: Room, segment_id_grid: np.ndarray, block_buffer: set[Coord], forbidden: set[Coord]) -> bool:
    footprint = set(room.walls) | set(room.interior)
    if not room.interior or footprint & block_buffer or footprint & forbidden:
        return False
    adjacent_segments: set[int] = set()
    h, w = segment_id_grid.shape
    for r, c in room.interior:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                seg_id = int(segment_id_grid[nr, nc])
                if seg_id >= 0:
                    adjacent_segments.add(seg_id)
    return adjacent_segments == {room.segment_id}


def _place_rooms(segments: list[Segment], shape: tuple[int, int], block_cells: set[Coord], rng: random.Random) -> list[Room] | None:
    segment_id_grid = np.full(shape, -1, dtype=int)
    for seg_id, segment in enumerate(segments):
        r0, c0, r1, c1 = segment.rect
        segment_id_grid[r0 : r1 + 1, c0 : c1 + 1] = int(seg_id)

    block_buffer = _expand4(block_cells, shape)
    candidates_by_seg = {
        seg_id: _gen_shuffled(_room_candidates_for_segment(segment, seg_id, shape), rng)
        for seg_id, segment in enumerate(segments)
    }

    def search(remaining: set[int], forbidden: set[Coord], chosen: list[Room]) -> list[Room] | None:
        if not remaining:
            return chosen

        best_seg = None
        best_options = None
        for seg_id in remaining:
            options = [
                room
                for room in candidates_by_seg[seg_id]
                if _room_is_valid(room, segment_id_grid, block_buffer, forbidden)
            ]
            if not options:
                return None
            if best_options is None or len(options) < len(best_options):
                best_seg = seg_id
                best_options = options

        assert best_seg is not None and best_options is not None
        for room in _gen_shuffled(best_options, rng):
            footprint = set(room.walls) | set(room.interior)
            result = search(
                remaining - {best_seg},
                forbidden | _expand4(footprint, shape),
                chosen + [room],
            )
            if result is not None:
                return result
        return None

    return search(set(range(len(segments))), set(), [])


def _render_input(rows: int, cols: int, path: list[Coord], rooms: list[Room]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for idx, (r, c) in enumerate(path):
        color = Color.RED if idx == 0 else Color.TEAL
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = int(color)
    for room in rooms:
        for r, c in room.walls:
            grid[r, c] = Color.GREY
    return grid


def _render_expected(inp: np.ndarray, path: list[Coord], rooms: list[Room]) -> np.ndarray:
    out = inp.copy()
    segments: list[Segment] = []
    for a, b in zip(path, path[1:]):
        rect, axis = _gen_corridor_rect(a, b)
        assert rect is not None and axis is not None
        seg = Segment(axis=axis, rect=rect, length=_segment_length(axis, rect))
        segments.append(seg)
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = Color.ORANGE

    for room in rooms:
        fill = Color.ORANGE if segments[room.segment_id].length % 2 == 1 else Color.YELLOW
        for r, c in room.interior:
            out[r, c] = int(fill)
    return out


def _gen_path_lengths(path: list[Coord]) -> list[int]:
    lengths = []
    for a, b in zip(path, path[1:]):
        rect, axis = _gen_corridor_rect(a, b)
        assert rect is not None and axis is not None
        lengths.append(_segment_length(axis, rect))
    return lengths


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(*GEN_ROWS_RANGE)
    cols = rng.randint(*GEN_COLS_RANGE)
    target_count = rng.randint(*GEN_TARGET_COUNT_RANGE)
    path = _gen_build_path(rows, cols, target_count, rng)
    if path is None:
        return None

    lengths = _gen_path_lengths(path)
    if len(set(lengths)) < 2:
        return None
    if {length % 2 for length in lengths} != {0, 1}:
        return None

    segments = [
        Segment(axis=axis, rect=rect, length=_segment_length(axis, rect))
        for rect, axis in (_gen_corridor_rect(a, b) for a, b in zip(path, path[1:]))
    ]
    assert all(seg[0] is not None for seg in [(s.rect, s.axis) for s in segments])

    block_cells = {
        (r, c)
        for br, bc in path
        for r in range(br, br + BLOCK_SIZE)
        for c in range(bc, bc + BLOCK_SIZE)
    }
    rooms = _place_rooms(segments, (rows, cols), block_cells, rng)
    if rooms is None:
        return None

    inp = _render_input(rows, cols, path, rooms)
    expected = _render_expected(inp, path, rooms)

    room_parities = {segments[room.segment_id].length % 2 for room in rooms}
    if room_parities != {0, 1}:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
