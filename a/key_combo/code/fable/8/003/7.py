from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
CHAIN_PITCH = 8
LATTICE_SIZE = 3
ROOM_SIDE_CHOICES = (5, 7)
SOURCE_COLOR_CHOICES = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = cell
    if r == r0:
        return "N"
    if r == r1:
        return "S"
    if c == c0:
        return "W"
    if c == c1:
        return "E"
    raise AssertionError("gap must lie on the ring perimeter")


def _gap_dir(side: str) -> tuple[int, int]:
    return DIRS[side]


def _rect_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(
        integral[r1 + 1, c1 + 1]
        - integral[r0, c1 + 1]
        - integral[r1 + 1, c0]
        + integral[r0, c0]
    )


def _find_rings(grid: np.ndarray, wall_color: int) -> list[dict]:
    wall = np.asarray(grid == wall_color, dtype=bool)
    h, w = wall.shape
    wall_cells = {tuple(map(int, cell)) for cell in np.argwhere(wall)}
    integral = np.pad(wall.astype(int).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    candidates: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not wall[r0, c0]:
                continue
            for r1 in range(r0 + 2, h):
                if not wall[r1, c0]:
                    continue
                for c1 in range(c0 + 2, w):
                    if not wall[r0, c1] or not wall[r1, c1]:
                        continue
                    bbox = (r0, c0, r1, c1)
                    perim_len = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    total_walls = _rect_sum(integral, r0, c0, r1, c1)
                    inner_walls = _rect_sum(integral, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if inner_walls != 0:
                        continue
                    wall_on_perim = total_walls - inner_walls
                    gap_count = perim_len - wall_on_perim
                    if gap_count not in (1, 2):
                        continue
                    perimeter = _perimeter_cells(bbox)
                    gaps = sorted(cell for cell in perimeter if not wall[cell])
                    if len(gaps) != gap_count:
                        continue
                    if any(gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)} for gap in gaps):
                        continue
                    wall_subset = perimeter - set(gaps)
                    candidates.append(
                        {
                            "bbox": bbox,
                            "gaps": [
                                {
                                    "cell": gap,
                                    "side": _gap_side(bbox, gap),
                                    "dir": _gap_dir(_gap_side(bbox, gap)),
                                }
                                for gap in gaps
                            ],
                            "interior": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
                            "wall_cells": frozenset(wall_subset),
                        }
                    )

    cell_to_candidates: dict[tuple[int, int], list[int]] = {cell: [] for cell in wall_cells}
    for idx, cand in enumerate(candidates):
        for cell in cand["wall_cells"]:
            if cell in cell_to_candidates:
                cell_to_candidates[cell].append(idx)

    def rec(remaining: frozenset[tuple[int, int]]):
        if not remaining:
            return []
        pivot = min(
            remaining,
            key=lambda cell: sum(candidates[idx]["wall_cells"] <= remaining for idx in cell_to_candidates[cell]),
        )
        options = [idx for idx in cell_to_candidates[pivot] if candidates[idx]["wall_cells"] <= remaining]
        if not options:
            return None
        options.sort(key=lambda idx: (-len(candidates[idx]["wall_cells"]), candidates[idx]["bbox"]))
        for idx in options:
            result = rec(remaining - candidates[idx]["wall_cells"])
            if result is not None:
                return [candidates[idx]] + result
        return None

    chosen = rec(frozenset(wall_cells))
    assert chosen is not None, "failed to partition wall pixels into rectangular rings"
    return chosen


def _analyze_rings(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert len(colors) == 2, "expected wall color plus source color"
    singleton_colors = [color for color in colors if int(np.count_nonzero(g == color)) == 1]
    assert len(singleton_colors) == 1, "expected exactly one source pixel color"
    source_color = int(singleton_colors[0])
    wall_color = int([color for color in colors if color != source_color][0])
    rings = _find_rings(g, wall_color)

    source_pos = tuple(map(int, np.argwhere(g == source_color)[0]))
    source_ring_ids = []
    for idx, ring in enumerate(rings):
        r0, c0, r1, c1 = ring["bbox"]
        if r0 < source_pos[0] < r1 and c0 < source_pos[1] < c1:
            source_ring_ids.append(idx)
    assert len(source_ring_ids) == 1, "expected the source pixel inside exactly one ring"
    source_ring = rings[source_ring_ids[0]]
    assert len(source_ring["gaps"]) == 1, "the source ring must have one gap"

    return {
        "bg": bg,
        "wall_color": wall_color,
        "source_color": source_color,
        "source_pos": source_pos,
        "rings": rings,
        "source_ring_id": source_ring_ids[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the ring interiors reached by the gap-aligned paint stream, and draw the stream itself.

    Input structure:
    - The background is black.
    - Several grey one-pixel rectangular rings are scattered on the grid; each ring wall has one or two one-pixel gaps.
    - Exactly one one-gap ring contains a single colored source pixel in its interior.

    Rule:
    - Fill the source ring interior with the source color.
    - From the source ring gap, shoot a straight ray outward normal to that gap, coloring the ray cells until the ray reaches the grid edge or lands on another ring gap.
    - When a ray reaches a ring gap, fill that ring interior too; if the entered ring has a second gap, continue the chain from that other gap, otherwise stop there.
    - Keep every grey wall cell unchanged; only the traversed ring interiors and ray cells become the source color.

    Key insight:
    - The gaps are ports, not damage: the filled rings are exactly the rings encountered by sequentially tracing the ray from port to port, so a ring is painted only if the chain actually enters it.
    """
    info = _analyze_rings(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    source_color = int(info["source_color"])
    wall_color = int(info["wall_color"])
    rings = info["rings"]

    gap_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for ring_id, ring in enumerate(rings):
        for gap_idx, gap in enumerate(ring["gaps"]):
            gap_lookup[tuple(gap["cell"])] = (ring_id, gap_idx)

    def fill_ring(ring_id: int) -> None:
        ring = rings[ring_id]
        interior = ring["interior"]
        out[interior] = source_color

    visited: set[int] = set()
    current_ring = int(info["source_ring_id"])
    current_gap_idx = 0
    fill_ring(current_ring)
    visited.add(current_ring)

    while True:
        gap = rings[current_ring]["gaps"][current_gap_idx]
        r, c = map(int, gap["cell"])
        dr, dc = map(int, gap["dir"])

        while True:
            out[r, c] = source_color
            nr, nc = r + dr, c + dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                return out
            r, c = nr, nc
            assert int(g[r, c]) != wall_color, "ray cannot pass through a wall cell"
            if (r, c) not in gap_lookup:
                continue

            next_ring, entry_gap_idx = gap_lookup[(r, c)]
            assert next_ring != current_ring, "ray cannot re-enter the same ring immediately"
            assert next_ring not in visited, "expected a simple non-cyclic chain"
            fill_ring(next_ring)
            visited.add(next_ring)
            if len(rings[next_ring]["gaps"]) == 1:
                out[r, c] = source_color
                return out
            current_ring = next_ring
            current_gap_idx = 1 - int(entry_gap_idx)
            break


def _room_bbox(center: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int]:
    cr, cc = center
    return cr - h // 2, cc - w // 2, cr + h // 2, cc + w // 2


def _room_gap_cell(room: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = room["bbox"]
    if side == "N":
        return r0, room["center"][1]
    if side == "S":
        return r1, room["center"][1]
    if side == "W":
        return room["center"][0], c0
    if side == "E":
        return room["center"][0], c1
    raise ValueError(side)


def _render_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    for side in room["gap_sides"]:
        rr, cc = _room_gap_cell(room, side)
        grid[rr, cc] = BG


def _fill_room_interior(grid: np.ndarray, room: dict, color: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def _segment_cells(start: tuple[int, int], direction: str, stop: tuple[int, int] | None, shape: tuple[int, int]):
    dr, dc = DIRS[direction]
    r, c = start
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        if stop is not None and (r, c) == stop:
            break
        r += dr
        c += dc
    return cells


def _render_expected(spec: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for room in spec["rooms"]:
        _render_room(out, room)
    source_room = spec["chain_rooms"][0]
    sr, sc = source_room["source_pos"]
    out[sr, sc] = spec["source_color"]

    for room in spec["chain_rooms"]:
        _fill_room_interior(out, room, spec["source_color"])

    for seg in spec["segments"]:
        for r, c in _segment_cells(seg["start"], seg["direction"], seg["stop"], shape):
            out[r, c] = spec["source_color"]
    return out


def _render_input(spec: dict, shape: tuple[int, int]) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    for room in spec["rooms"]:
        _render_room(inp, room)
    source_room = spec["chain_rooms"][0]
    sr, sc = source_room["source_pos"]
    inp[sr, sc] = spec["source_color"]
    return inp


def _choose_room_size() -> tuple[int, int]:
    return random.choice(ROOM_SIDE_CHOICES), random.choice(ROOM_SIDE_CHOICES)


def _room_from_node(node: tuple[int, int], gap_sides: list[str], role: str) -> dict:
    center = (node[0] * CHAIN_PITCH, node[1] * CHAIN_PITCH)
    h, w = _choose_room_size()
    bbox = _room_bbox(center, h, w)
    return {
        "node": node,
        "center": center,
        "bbox": bbox,
        "gap_sides": list(gap_sides),
        "role": role,
    }


def _path_dirs(nodes: list[tuple[int, int]]) -> list[str]:
    dirs = []
    for (r0, c0), (r1, c1) in zip(nodes, nodes[1:]):
        dr = r1 - r0
        dc = c1 - c0
        for side, (sdr, sdc) in DIRS.items():
            if (dr, dc) == (sdr, sdc):
                dirs.append(side)
                break
        else:
            raise AssertionError("nodes must be adjacent on the lattice")
    return dirs


def _sample_chain_nodes(room_count: int, require_turn: bool) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(LATTICE_SIZE) for c in range(LATTICE_SIZE)]
    random.shuffle(starts)

    def rec(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == room_count:
            dirs = _path_dirs(path)
            if require_turn and len(dirs) >= 2 and not any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))):
                return None
            return path[:]

        cur_r, cur_c = path[-1]
        options = []
        for side, (dr, dc) in DIRS.items():
            nr, nc = cur_r + dr, cur_c + dc
            nxt = (nr, nc)
            if 0 <= nr < LATTICE_SIZE and 0 <= nc < LATTICE_SIZE and nxt not in path:
                options.append((side, nxt))
        if not options:
            return None

        if len(path) >= 2:
            prev_dir = _path_dirs(path)[-1]
            turn_options = [item for item in options if item[0] != prev_dir]
            straight_options = [item for item in options if item[0] == prev_dir]
            weighted = []
            if require_turn and len(path) < room_count - 1 and turn_options:
                weighted.extend(turn_options * 3)
                weighted.extend(straight_options)
            else:
                weighted.extend(options)
            random.shuffle(weighted)
            options = weighted
        else:
            random.shuffle(options)

        seen = set()
        for side, nxt in options:
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            result = rec(path)
            path.pop()
            if result is not None:
                return result
        return None

    for start in starts:
        result = rec([start])
        if result is not None:
            return result
    return None


def _build_chain_rooms(nodes: list[tuple[int, int]], terminal_mode: str) -> list[dict] | None:
    dirs = _path_dirs(nodes)
    rooms = []
    for idx, node in enumerate(nodes):
        if idx == 0:
            gap_sides = [dirs[0]]
            role = "source"
        elif idx == len(nodes) - 1 and terminal_mode == "ring":
            gap_sides = [OPPOSITE[dirs[-1]]]
            role = "terminal"
        elif idx == len(nodes) - 1 and terminal_mode == "edge":
            entry = OPPOSITE[dirs[-1]]
            exit_options = [side for side in DIRS if side != entry]
            gap_sides = [entry, random.choice(exit_options)]
            role = "edge_last"
        else:
            gap_sides = [OPPOSITE[dirs[idx - 1]], dirs[idx]]
            role = "middle"
        room = _room_from_node(node, gap_sides, role)
        rooms.append(room)

    source_room = rooms[0]
    r0, c0, r1, c1 = source_room["bbox"]
    source_room["source_pos"] = (
        random.randint(r0 + 1, r1 - 1),
        random.randint(c0 + 1, c1 - 1),
    )
    return rooms


def _build_distractors(used_nodes: set[tuple[int, int]], max_count: int) -> list[dict]:
    free_nodes = [(r, c) for r in range(LATTICE_SIZE) for c in range(LATTICE_SIZE) if (r, c) not in used_nodes]
    random.shuffle(free_nodes)
    count = random.randint(1, min(max_count, len(free_nodes)))
    distractors = []
    for node in free_nodes[:count]:
        gap_count = 1 if random.random() < 0.45 else 2
        sides = random.sample(list(DIRS.keys()), gap_count)
        distractors.append(_room_from_node(node, sides, "distractor"))
    return distractors


def _all_room_cells(room: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _rooms_disjoint(rooms: list[dict]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for room in rooms:
        cells = _all_room_cells(room)
        if occupied & cells:
            return False
        occupied |= cells
    return True


def _build_segments(chain_rooms: list[dict], terminal_mode: str) -> list[dict]:
    segments = []
    for idx, room in enumerate(chain_rooms):
        if idx == len(chain_rooms) - 1 and terminal_mode == "ring":
            break
        if idx == 0:
            exit_side = room["gap_sides"][0]
        else:
            exit_side = room["gap_sides"][1]
        start = _room_gap_cell(room, exit_side)
        if idx + 1 < len(chain_rooms):
            next_room = chain_rooms[idx + 1]
            stop = _room_gap_cell(next_room, OPPOSITE[exit_side])
        else:
            stop = None
        segments.append({"start": start, "direction": exit_side, "stop": stop})
    return segments


def _raw_bbox(rooms: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(room["bbox"][0] for room in rooms)
    c0 = min(room["bbox"][1] for room in rooms)
    r1 = max(room["bbox"][2] for room in rooms)
    c1 = max(room["bbox"][3] for room in rooms)
    return r0, c0, r1, c1


def _shift_room(room: dict, dr: int, dc: int) -> dict:
    shifted = dict(room)
    r0, c0, r1, c1 = room["bbox"]
    shifted["center"] = (room["center"][0] + dr, room["center"][1] + dc)
    shifted["bbox"] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    if "source_pos" in room:
        shifted["source_pos"] = (room["source_pos"][0] + dr, room["source_pos"][1] + dc)
    return shifted


def _place_spec(chain_rooms: list[dict], distractors: list[dict], terminal_mode: str, source_color: int) -> dict | None:
    all_rooms = chain_rooms + distractors
    if not _rooms_disjoint(all_rooms):
        return None

    raw_r0, raw_c0, raw_r1, raw_c1 = _raw_bbox(all_rooms)
    raw_h = raw_r1 - raw_r0 + 1
    raw_w = raw_c1 - raw_c0 + 1
    if raw_h > 28 or raw_w > 28:
        return None

    final_exit_side = None
    if terminal_mode == "edge":
        final_exit_side = chain_rooms[-1]["gap_sides"][1]

    min_top = 2 if final_exit_side == "N" else 1
    min_bottom = 2 if final_exit_side == "S" else 1
    min_left = 2 if final_exit_side == "W" else 1
    min_right = 2 if final_exit_side == "E" else 1
    if raw_h + min_top + min_bottom > 30 or raw_w + min_left + min_right > 30:
        return None

    top = random.randint(min_top, 30 - raw_h - min_bottom)
    bottom = random.randint(min_bottom, 30 - raw_h - top)
    left = random.randint(min_left, 30 - raw_w - min_right)
    right = random.randint(min_right, 30 - raw_w - left)
    shape = (raw_h + top + bottom, raw_w + left + right)
    dr = top - raw_r0
    dc = left - raw_c0

    shifted_chain = [_shift_room(room, dr, dc) for room in chain_rooms]
    shifted_distractors = [_shift_room(room, dr, dc) for room in distractors]
    shifted_rooms = shifted_chain + shifted_distractors
    segments = _build_segments(shifted_chain, terminal_mode)

    spec = {
        "rooms": shifted_rooms,
        "chain_rooms": shifted_chain,
        "distractors": shifted_distractors,
        "segments": segments,
        "terminal_mode": terminal_mode,
        "source_color": int(source_color),
        "shape": shape,
    }
    return spec


def _trace_rooms_from_spec(spec: dict, shape: tuple[int, int]) -> set[int] | None:
    gap_lookup = {}
    for idx, room in enumerate(spec["rooms"]):
        for side in room["gap_sides"]:
            gap_lookup[_room_gap_cell(room, side)] = idx

    visited = {0}
    cur_idx = 0
    cur_gap_side = spec["chain_rooms"][0]["gap_sides"][0]
    while True:
        start = _room_gap_cell(spec["rooms"][cur_idx], cur_gap_side)
        dr, dc = DIRS[cur_gap_side]
        r, c = start
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            if (r, c) in gap_lookup and gap_lookup[(r, c)] != cur_idx:
                nxt = gap_lookup[(r, c)]
                if nxt in visited:
                    return None
                visited.add(nxt)
                next_room = spec["rooms"][nxt]
                if len(next_room["gap_sides"]) == 1:
                    return visited
                entry_side = _gap_side(next_room["bbox"], (r, c))
                cur_idx = nxt
                cur_gap_side = [side for side in next_room["gap_sides"] if side != entry_side][0]
                break
            nr, nc = r + dr, c + dc
            if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
                return visited
            r, c = nr, nc
        else:
            return visited


def _wrong_fill_all_rings(inp: np.ndarray, source_color: int) -> np.ndarray:
    wrong = inp.copy()
    for ring in _find_rings(inp, WALL):
        r0, c0, r1, c1 = ring["bbox"]
        wrong[r0 + 1 : r1, c0 + 1 : c1] = int(source_color)
    return wrong


def _wrong_source_only(spec: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for room in spec["rooms"]:
        _render_room(out, room)
    source_room = spec["chain_rooms"][0]
    sr, sc = source_room["source_pos"]
    out[sr, sc] = spec["source_color"]
    _fill_room_interior(out, source_room, spec["source_color"])
    first = spec["segments"][0]
    for r, c in _segment_cells(first["start"], first["direction"], None, shape):
        out[r, c] = spec["source_color"]
    return out


def _validate_spec(spec: dict, inp: np.ndarray, expected: np.ndarray, require_turn: bool) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if len(spec["distractors"]) < 1:
        return False
    if np.array_equal(_wrong_fill_all_rings(inp, spec["source_color"]), expected):
        return False
    if np.array_equal(_wrong_source_only(spec, inp.shape), expected):
        return False

    dirs = _path_dirs([room["node"] for room in spec["chain_rooms"]])
    has_turn = any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))
    if require_turn and len(dirs) >= 2 and not has_turn:
        return False

    traced = _trace_rooms_from_spec(spec, inp.shape)
    if traced is None:
        return False
    if traced != set(range(len(spec["chain_rooms"]))):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    source_color = random.choice(SOURCE_COLOR_CHOICES)
    terminal_mode = "ring" if random.random() < 0.5 else "edge"
    room_count = random.choices([2, 3, 4, 5], weights=[1, 3, 3, 2], k=1)[0]
    require_turn = random.random() < 0.7 and room_count >= 3

    chain_nodes = _sample_chain_nodes(room_count, require_turn)
    if chain_nodes is None:
        return None
    chain_rooms = _build_chain_rooms(chain_nodes, terminal_mode)
    if chain_rooms is None:
        return None

    distractors = _build_distractors(set(chain_nodes), max_count=3)
    spec = _place_spec(chain_rooms, distractors, terminal_mode, source_color)
    if spec is None:
        return None

    inp = _render_input(spec, spec["shape"])
    expected = _render_expected(spec, spec["shape"])

    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_spec(spec, inp, expected, require_turn):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
