from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BRANCH_TYPES = ("all_narrow", "mixed", "all_wide")
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_wall_color(grid: np.ndarray) -> int:
    if np.any(grid == GREY):
        return GREY
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _room_seed_mask(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    room = np.zeros((h, w), dtype=bool)
    for r in range(h - 2):
        for c in range(w - 2):
            if bool(passable[r : r + 3, c : c + 3].all()):
                room[r : r + 3, c : c + 3] = True
    return room


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall = _infer_wall_color(g)
    source_mask = (g != wall) & (g != BLACK)
    source_pts = np.argwhere(source_mask)
    assert len(source_pts) == 1, "expected exactly one signal pixel"
    sr, sc = map(int, source_pts[0])
    source_color = int(g[sr, sc])

    passable = g != wall
    room_mask = _room_seed_mask(passable)
    assert not bool(room_mask[sr, sc]), "source must be outside junction chambers"

    room_cells = _components(room_mask)
    assert room_cells, "expected at least one junction chamber"
    room_id = -np.ones((h, w), dtype=int)
    for rid, comp in enumerate(room_cells):
        for r, c in comp:
            room_id[r, c] = rid

    corridor_mask = passable & (~room_mask)
    corridor_cells = _components(corridor_mask)
    assert corridor_cells, "expected at least one corridor component"
    corridor_id = -np.ones((h, w), dtype=int)
    for cid, comp in enumerate(corridor_cells):
        for r, c in comp:
            corridor_id[r, c] = cid

    source_cid = int(corridor_id[sr, sc])
    assert source_cid >= 0, "source must lie on a corridor"

    corridor_width: list[int] = []
    corridor_rooms: list[tuple[int, ...]] = []
    room_to_corridors = [set() for _ in room_cells]

    for cid, comp in enumerate(corridor_cells):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        comp_width = max(1, min(height, width))
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                rid = int(room_id[nr, nc])
                if rid >= 0:
                    touched.add(rid)
        corridor_width.append(int(comp_width))
        corridor_rooms.append(tuple(sorted(touched)))
        for rid in touched:
            room_to_corridors[rid].add(cid)

    return {
        "source": (sr, sc),
        "source_color": source_color,
        "room_cells": room_cells,
        "corridor_cells": corridor_cells,
        "corridor_width": corridor_width,
        "corridor_rooms": corridor_rooms,
        "room_to_corridors": [tuple(sorted(ids)) for ids in room_to_corridors],
        "source_corridor": source_cid,
    }


def _paint_black(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        if int(out[r, c]) == BLACK:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the signal from the unique colored dead-end marker through the maze.

    The signal starts from the marker, moves through the corridor network, and
    only branches into width-1 outgoing corridors at junctions. Width-2
    corridors are skipped at branch points and remain black. Junction chambers
    reached by the signal are recolored as part of the traversed route.
    """
    g = np.asarray(grid, dtype=int)
    structure = _extract_structure(g)
    source_color = int(structure["source_color"])
    out = g.copy()

    corridor_cells = structure["corridor_cells"]
    corridor_width = structure["corridor_width"]
    corridor_rooms = structure["corridor_rooms"]
    room_cells = structure["room_cells"]
    room_to_corridors = structure["room_to_corridors"]

    visited_corridors: set[int] = set()
    visited_rooms: set[int] = set()

    def walk_room(rid: int, incoming_cid: int) -> None:
        if rid in visited_rooms:
            return
        visited_rooms.add(rid)
        _paint_black(out, room_cells[rid], source_color)

        exits = [
            cid
            for cid in room_to_corridors[rid]
            if cid != incoming_cid and cid not in visited_corridors
        ]
        if not exits:
            return
        if len(exits) == 1:
            walk_corridor(exits[0], rid)
            return

        chosen = [cid for cid in exits if int(corridor_width[cid]) == 1]
        for cid in chosen:
            walk_corridor(cid, rid)

    def walk_corridor(cid: int, from_room: int | None) -> None:
        if cid in visited_corridors:
            return
        visited_corridors.add(cid)
        if int(corridor_width[cid]) == 1:
            _paint_black(out, corridor_cells[cid], source_color)

        next_rooms = [rid for rid in corridor_rooms[cid] if rid != from_room]
        for rid in next_rooms:
            walk_room(rid, cid)

    walk_corridor(int(structure["source_corridor"]), None)
    return out


def _slot_neighbors(slot: tuple[int, int], rows: int, cols: int) -> list[tuple[str, tuple[int, int]]]:
    r, c = slot
    out: list[tuple[str, tuple[int, int]]] = []
    for name, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((name, (nr, nc)))
    return out


def _slot_top_left(
    slot: tuple[int, int], room_size: int, gap: int, margin: int
) -> tuple[int, int]:
    r, c = slot
    pitch = room_size + gap
    return margin + r * pitch, margin + c * pitch


def _random_tree(
    rows: int, cols: int, room_count: int, rng: random.Random
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    root = rng.choice(slots)
    used = {root}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(used) < room_count:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for slot in used:
            for _d, nb in _slot_neighbors(slot, rows, cols):
                if nb not in used:
                    frontier.append((slot, nb))
        if not frontier:
            return None
        parent, child = rng.choice(frontier)
        used.add(child)
        edges.append((parent, child))
    return used, edges


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    for name, (rr, cc) in DIRS.items():
        if (dr, dc) == (rr, cc):
            return name
    raise ValueError("non-adjacent slots")


def _choose_branch_widths(
    exit_links: list[tuple[str, int]], target_type: str, rng: random.Random
) -> dict[tuple[str, int], int]:
    assert len(exit_links) >= 2
    if target_type == "all_narrow":
        return {link: 1 for link in exit_links}
    if target_type == "all_wide":
        return {link: 2 for link in exit_links}

    widths = {link: 2 for link in exit_links}
    narrow_link = rng.choice(exit_links)
    widths[narrow_link] = 1
    wide_pool = [link for link in exit_links if link != narrow_link]
    if wide_pool:
        widths[rng.choice(wide_pool)] = 2
    for link in exit_links:
        if link == narrow_link:
            continue
        widths[link] = rng.choice([1, 2])
    if all(v == 1 for v in widths.values()):
        widths[rng.choice(exit_links)] = 2
    if all(v == 2 for v in widths.values()):
        widths[rng.choice(exit_links)] = 1
    return widths


def _paint_room(grid: np.ndarray, top: tuple[int, int], room_size: int) -> None:
    r0, c0 = top
    grid[r0 : r0 + room_size, c0 : c0 + room_size] = BLACK


def _paint_edge(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    room_size: int,
    gap: int,
    margin: int,
    width: int,
    offset: int,
) -> None:
    ta = _slot_top_left(a, room_size, gap, margin)
    tb = _slot_top_left(b, room_size, gap, margin)
    if ta[0] == tb[0]:
        left, right = (ta, tb) if ta[1] < tb[1] else (tb, ta)
        r0 = left[0] + offset
        c0 = left[1] + room_size
        c1 = right[1]
        grid[r0 : r0 + width, c0:c1] = BLACK
        return
    top, bottom = (ta, tb) if ta[0] < tb[0] else (tb, ta)
    r0 = top[0] + room_size
    r1 = bottom[0]
    c0 = top[1] + offset
    grid[r0:r1, c0 : c0 + width] = BLACK


def _paint_stub(
    grid: np.ndarray,
    room: tuple[int, int],
    side: str,
    room_size: int,
    gap: int,
    margin: int,
    width: int,
    offset: int,
    length: int,
    color: int | None,
    rng: random.Random,
) -> None:
    r0, c0 = _slot_top_left(room, room_size, gap, margin)
    if side == "N":
        rr0 = r0 - length
        cc0 = c0 + offset
        grid[rr0:r0, cc0 : cc0 + width] = BLACK
        if color is not None:
            grid[rr0, cc0 + rng.randrange(width)] = color
        return
    if side == "S":
        rr0 = r0 + room_size
        cc0 = c0 + offset
        grid[rr0 : rr0 + length, cc0 : cc0 + width] = BLACK
        if color is not None:
            grid[rr0 + length - 1, cc0 + rng.randrange(width)] = color
        return
    if side == "W":
        rr0 = r0 + offset
        cc0 = c0 - length
        grid[rr0 : rr0 + width, cc0:c0] = BLACK
        if color is not None:
            grid[rr0 + rng.randrange(width), cc0] = color
        return

    rr0 = r0 + offset
    cc0 = c0 + room_size
    grid[rr0 : rr0 + width, cc0 : cc0 + length] = BLACK
    if color is not None:
        grid[rr0 + rng.randrange(width), cc0 + length - 1] = color


def _branch_type_matches(widths: list[int], target_type: str) -> bool:
    if len(widths) < 2:
        return False
    if target_type == "all_narrow":
        return all(w == 1 for w in widths)
    if target_type == "all_wide":
        return all(w == 2 for w in widths)
    return any(w == 1 for w in widths) and any(w == 2 for w in widths)


def _validate_generated(grid: np.ndarray, target_type: str, room_count: int) -> bool:
    try:
        structure = _extract_structure(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if len(structure["room_cells"]) != room_count:
        return False
    if not np.any(out != grid):
        return False
    if int((out == int(structure["source_color"])).sum()) <= 1:
        return False

    source_cid = int(structure["source_corridor"])
    source_rooms = list(structure["corridor_rooms"][source_cid])
    if len(source_rooms) != 1:
        return False
    source_room = int(source_rooms[0])
    exits = [
        cid
        for cid in structure["room_to_corridors"][source_room]
        if cid != source_cid
    ]
    widths = [int(structure["corridor_width"][cid]) for cid in exits]
    return _branch_type_matches(widths, target_type)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_size = rng.choice([3, 4])
    gap = rng.choice([2, 3])
    margin = gap

    max_rows = max(
        n
        for n in range(3, 5)
        if n * room_size + (n + 1) * gap <= 30
    )
    max_cols = max(
        n
        for n in range(3, 5)
        if n * room_size + (n + 1) * gap <= 30
    )
    rows = rng.randint(3, max_rows)
    cols = rng.randint(3, max_cols)
    room_count = rng.randint(2, min(rows * cols, 6))
    source_color = rng.choice(SIGNAL_COLORS)
    target_type = rng.choice(BRANCH_TYPES)

    tree = _random_tree(rows, cols, room_count, rng)
    if tree is None:
        return None
    rooms, edges = tree

    room_links: dict[tuple[int, int], dict[str, tuple[str, int]]] = {
        room: {} for room in rooms
    }
    edge_recs: list[dict] = []
    for eid, (a, b) in enumerate(edges):
        side = _direction_between(a, b)
        edge_recs.append(
            {
                "rooms": (a, b),
                "side_a": side,
                "width": rng.choice([1, 2]),
                "offset": None,
            }
        )
        room_links[a][side] = ("edge", eid)
        room_links[b][OPPOSITE[side]] = ("edge", eid)

    source_candidates: list[tuple[int, int]] = []
    for room in rooms:
        unused = [
            side
            for side, _nb in _slot_neighbors(room, rows, cols)
            if side not in room_links[room]
        ]
        if unused and len(room_links[room]) + len(unused) - 1 >= 2:
            source_candidates.append(room)
    if not source_candidates:
        return None

    source_room = rng.choice(source_candidates)
    unused_source = [
        side
        for side, _nb in _slot_neighbors(source_room, rows, cols)
        if side not in room_links[source_room]
    ]
    source_side = rng.choice(unused_source)

    stub_recs: list[dict] = []
    source_stub_id = 0
    stub_recs.append(
        {
            "room": source_room,
            "side": source_side,
            "width": rng.choice([1, 2]),
            "offset": None,
            "length": rng.randint(2, gap),
            "color": source_color,
        }
    )
    room_links[source_room][source_side] = ("stub", source_stub_id)

    remaining_unused = [
        side
        for side, _nb in _slot_neighbors(source_room, rows, cols)
        if side not in room_links[source_room]
    ]
    current_exits = list(room_links[source_room].values())
    current_exits = [link for link in current_exits if link != ("stub", source_stub_id)]
    max_exit_total = len(current_exits) + len(remaining_unused)
    if max_exit_total < 2:
        return None

    desired_exit_total = rng.randint(2, min(3, max_exit_total))
    need_stubs = max(0, desired_exit_total - len(current_exits))
    if need_stubs > len(remaining_unused):
        return None

    rng.shuffle(remaining_unused)
    chosen_source_stubs = remaining_unused[:need_stubs]
    for side in chosen_source_stubs:
        sid = len(stub_recs)
        stub_recs.append(
            {
                "room": source_room,
                "side": side,
                "width": rng.choice([1, 2]),
                "offset": None,
                "length": rng.randint(2, gap),
                "color": None,
            }
        )
        room_links[source_room][side] = ("stub", sid)

    for room in rooms:
        if room == source_room:
            continue
        free = [
            side
            for side, _nb in _slot_neighbors(room, rows, cols)
            if side not in room_links[room]
        ]
        rng.shuffle(free)
        extra = rng.randint(0, min(2, len(free)))
        if len(room_links[room]) <= 1 and free and rng.random() < 0.65:
            extra = max(extra, 1)
        for side in free[:extra]:
            sid = len(stub_recs)
            stub_recs.append(
                {
                    "room": room,
                    "side": side,
                    "width": rng.choice([1, 2]),
                    "offset": None,
                    "length": rng.randint(2, gap),
                    "color": None,
                }
            )
            room_links[room][side] = ("stub", sid)

    exit_links = [
        link
        for link in room_links[source_room].values()
        if link != ("stub", source_stub_id)
    ]
    if len(exit_links) < 2:
        return None

    enforced = _choose_branch_widths(exit_links, target_type, rng)
    for link, width in enforced.items():
        kind, idx = link
        if kind == "edge":
            edge_recs[idx]["width"] = width
        else:
            stub_recs[idx]["width"] = width

    for rec in edge_recs:
        rec["offset"] = rng.randint(0, room_size - int(rec["width"]))
    for rec in stub_recs:
        rec["offset"] = rng.randint(0, room_size - int(rec["width"]))

    h = rows * room_size + (rows + 1) * gap
    w = cols * room_size + (cols + 1) * gap
    grid = np.full((h, w), GREY, dtype=int)

    for room in rooms:
        _paint_room(grid, _slot_top_left(room, room_size, gap, margin), room_size)
    for rec in edge_recs:
        _paint_edge(
            grid,
            rec["rooms"][0],
            rec["rooms"][1],
            room_size,
            gap,
            margin,
            int(rec["width"]),
            int(rec["offset"]),
        )
    for rec in stub_recs:
        _paint_stub(
            grid,
            rec["room"],
            rec["side"],
            room_size,
            gap,
            margin,
            int(rec["width"]),
            int(rec["offset"]),
            int(rec["length"]),
            rec["color"],
            rng,
        )

    if not _validate_generated(grid, target_type, room_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid maze after retries")
