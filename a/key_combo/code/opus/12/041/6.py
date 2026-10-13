from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)
RIGHT = (0, 1)
DOWN = (1, 0)
MAX_GEN_ATTEMPTS = 500


Room = Dict[str, object]
Marker = Dict[str, object]


def _sorted_coords(coords: np.ndarray) -> List[Tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _collapsed_full_lines(mask: np.ndarray) -> List[int]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    lines = [int(idx[0])]
    for val in idx[1:]:
        if int(val) != lines[-1] + 1:
            lines.append(int(val))
    return lines


def _room_spans(lines: List[int], limit: int) -> List[Tuple[int, int]]:
    assert len(lines) >= 2
    assert lines[0] == 0 and lines[-1] == limit - 1
    spans = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2
        spans.append((int(a + 1), int(b - 1)))
    return spans


def _parse_rooms(grid: np.ndarray) -> Tuple[List[List[Room]], Dict[Tuple[int, int], Tuple[int, int]]]:
    h, w = grid.shape
    row_lines = _collapsed_full_lines(np.all(grid == GREY, axis=1))
    col_lines = _collapsed_full_lines(np.all(grid == GREY, axis=0))
    row_spans = _room_spans(row_lines, h)
    col_spans = _room_spans(col_lines, w)
    rooms: List[List[Room]] = []
    pixel_to_room: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for rr, (r0, r1) in enumerate(row_spans):
        row_rooms = []
        for cc, (c0, c1) in enumerate(col_spans):
            room = {
                "row": rr,
                "col": cc,
                "bbox": (r0, c0, r1, c1),
                "center2": (r0 + r1, c0 + c1),
            }
            row_rooms.append(room)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    pixel_to_room[(r, c)] = (rr, cc)
        rooms.append(row_rooms)
    assert rooms and rooms[0]
    return rooms, pixel_to_room


def _extract_markers(grid: np.ndarray, pixel_to_room: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Marker]:
    markers = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 2
        pts = _sorted_coords(coords)
        anchor = pts[0]
        other = pts[1]
        dr = other[0] - anchor[0]
        dc = other[1] - anchor[1]
        assert abs(dr) + abs(dc) == 1
        room0 = pixel_to_room[anchor]
        room1 = pixel_to_room[other]
        assert room0 == room1
        markers.append(
            {
                "color": int(color),
                "anchor": anchor,
                "other": other,
                "direction": (int(dr), int(dc)),
                "room": room0,
            }
        )
    markers.sort(key=lambda m: tuple(m["anchor"]))
    assert 2 <= len(markers) <= 4
    return markers


def _trajectory(marker: Marker, room_rows: int, room_cols: int) -> List[Tuple[int, int]]:
    rr, cc = marker["room"]
    dr, dc = marker["direction"]
    out = []
    while 0 <= rr < room_rows and 0 <= cc < room_cols:
        out.append((int(rr), int(cc)))
        rr += int(dr)
        cc += int(dc)
    return out


def _dist2(anchor: Tuple[int, int], room: Room) -> int:
    center2_r, center2_c = room["center2"]
    return abs(2 * int(anchor[0]) - int(center2_r)) + abs(2 * int(anchor[1]) - int(center2_c))


def _analyze(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rooms, pixel_to_room = _parse_rooms(g)
    markers = _extract_markers(g, pixel_to_room)
    room_rows = len(rooms)
    room_cols = len(rooms[0])
    claims: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    winners: Dict[Tuple[int, int], int] = {}
    winner_meta: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for order, marker in enumerate(markers):
        path = _trajectory(marker, room_rows, room_cols)
        marker["trajectory"] = path
        for rr, cc in path:
            room = rooms[rr][cc]
            dist = _dist2(marker["anchor"], room)
            claims.setdefault((rr, cc), []).append((order, int(marker["color"]), dist))
            key = (dist, -order)
            if (rr, cc) not in winner_meta or key < winner_meta[(rr, cc)]:
                winner_meta[(rr, cc)] = key
                winners[(rr, cc)] = int(marker["color"])
    return {
        "rooms": rooms,
        "markers": markers,
        "claims": claims,
        "winners": winners,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every room crossed by each directional domino marker along the room grid.

    Input structure:
    - The grid has black room interiors separated by full grey divider rows and columns that form a rectangular room array.
    - Some rooms contain a two-pixel same-color domino marker; the lexicographically earlier pixel is the anchor, and the second pixel gives the slide direction.

    Rule:
    - Parse the grey divider lattice into rooms, then for each marker start from its room and advance room-by-room in the indicated cardinal direction until leaving the grid.
    - Every room on that trajectory is filled with the marker color across the whole room interior; if multiple trajectories hit the same room, choose the marker whose anchor is closer to that room center by Manhattan distance, breaking ties by later anchor order from top-left to bottom-right.

    Key insight:
    - The domino does not move to another object; it encodes a scan direction, and the output depends on the interaction between that scan line and the room partition.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    out = g.copy()
    for (rr, cc), color in info["winners"].items():
        r0, c0, r1, c1 = info["rooms"][rr][cc]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _build_room_grid(row_heights: List[int], col_widths: List[int]) -> Tuple[np.ndarray, List[List[Room]]]:
    h = 1 + sum(row_heights) + len(row_heights)
    w = 1 + sum(col_widths) + len(col_widths)
    grid = np.full((h, w), BG, dtype=int)
    r = 0
    grid[r, :] = GREY
    for height in row_heights:
        r += 1 + height
        grid[r, :] = GREY
    c = 0
    grid[:, c] = GREY
    for width in col_widths:
        c += 1 + width
        grid[:, c] = GREY
    rooms, _ = _parse_rooms(grid)
    return grid, rooms


def _domino_positions(room: Room, direction: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    r0, c0, r1, c1 = room["bbox"]
    out = []
    if direction == RIGHT:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1):
                out.append(((r, c), (r, c + 1)))
    elif direction == DOWN:
        for r in range(r0, r1):
            for c in range(c0, c1 + 1):
                out.append(((r, c), (r + 1, c)))
    else:
        raise ValueError("unsupported direction")
    return out


def _pick_conflict_pair(
    rooms: List[List[Room]], want_tie: bool
) -> List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    room_rows = len(rooms)
    room_cols = len(rooms[0])
    candidates = []
    for inter_r in range(1, room_rows):
        for inter_c in range(1, room_cols):
            for h_col in range(inter_c):
                for v_row in range(inter_r):
                    candidates.append((inter_r, inter_c, h_col, v_row))
    random.shuffle(candidates)
    for inter_r, inter_c, h_col, v_row in candidates:
        inter_room = rooms[inter_r][inter_c]
        horiz_room = rooms[inter_r][h_col]
        vert_room = rooms[v_row][inter_c]
        horiz_positions = _domino_positions(horiz_room, RIGHT)
        vert_positions = _domino_positions(vert_room, DOWN)
        random.shuffle(horiz_positions)
        random.shuffle(vert_positions)
        for h_anchor, h_other in horiz_positions:
            h_dist = _dist2(h_anchor, inter_room)
            for v_anchor, v_other in vert_positions:
                v_dist = _dist2(v_anchor, inter_room)
                if want_tie and h_dist == v_dist:
                    return [
                        (0, h_anchor, h_other, (inter_r, h_col)),
                        (1, v_anchor, v_other, (v_row, inter_c)),
                    ]
                if not want_tie and h_dist != v_dist:
                    return [
                        (0, h_anchor, h_other, (inter_r, h_col)),
                        (1, v_anchor, v_other, (v_row, inter_c)),
                    ]
    return []


def _random_sizes(count: int, allow_variable: bool) -> List[int]:
    if allow_variable:
        return [random.randint(3, 5) for _ in range(count)]
    size = random.randint(3, 5)
    return [size] * count


def _random_marker_room(
    rooms: List[List[Room]], used_rooms: set[Tuple[int, int]]
) -> Tuple[Tuple[int, int], Tuple[int, int]] | None:
    room_rows = len(rooms)
    room_cols = len(rooms[0])
    choices = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if (rr, cc) in used_rooms:
                continue
            if cc < room_cols - 1:
                choices.append(((rr, cc), RIGHT))
            if rr < room_rows - 1:
                choices.append(((rr, cc), DOWN))
    return random.choice(choices) if choices else None


def _paint_markers(
    grid: np.ndarray, placements: List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]], colors: List[int]
) -> np.ndarray:
    out = grid.copy()
    for marker_id, anchor, other, _room in placements:
        color = colors[marker_id]
        out[anchor] = color
        out[other] = color
    return out


def _has_tie_room(info: Dict[str, object]) -> bool:
    for claims in info["claims"].values():
        if len(claims) < 2:
            continue
        min_dist = min(dist for _order, _color, dist in claims)
        if sum(1 for _order, _color, dist in claims if dist == min_dist) >= 2:
            return True
    return False


def _has_strict_conflict_room(info: Dict[str, object]) -> bool:
    for claims in info["claims"].values():
        if len(claims) < 2:
            continue
        dists = sorted(dist for _order, _color, dist in claims)
        if dists[0] < dists[1]:
            return True
    return False


def _valid_example(grid: np.ndarray, want_tie: bool) -> bool:
    try:
        info = _analyze(grid)
    except Exception:
        return False
    room_count = len(info["rooms"]) * len(info["rooms"][0])
    touched = len(info["winners"])
    if touched >= room_count:
        return False
    if touched <= len(info["markers"]):
        return False
    conflict_rooms = [room for room, claims in info["claims"].items() if len(claims) >= 2]
    if not conflict_rooms:
        return False
    if want_tie and not _has_tie_room(info):
        return False
    if not want_tie and not _has_strict_conflict_room(info):
        return False
    if any(len(marker["trajectory"]) < 2 for marker in info["markers"]):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(3, 4)
    room_cols = random.randint(3, 4)
    want_tie = random.random() < 0.4
    allow_variable = (not want_tie) and (random.random() < 0.6)
    row_heights = _random_sizes(room_rows, allow_variable)
    col_widths = _random_sizes(room_cols, allow_variable or random.random() < 0.5)
    h = 1 + sum(row_heights) + room_rows
    w = 1 + sum(col_widths) + room_cols
    if h > 30 or w > 30:
        return None

    grid, rooms = _build_room_grid(row_heights, col_widths)
    placements = _pick_conflict_pair(rooms, want_tie)
    if not placements:
        return None

    used_rooms = {room for _marker_id, _anchor, _other, room in placements}
    marker_count = random.randint(2, 4)
    while len(placements) < marker_count:
        choice = _random_marker_room(rooms, used_rooms)
        if choice is None:
            return None
        room_idx, direction = choice
        room = rooms[room_idx[0]][room_idx[1]]
        pos = _domino_positions(room, direction)
        if not pos:
            return None
        anchor, other = random.choice(pos)
        placements.append((len(placements), anchor, other, room_idx))
        used_rooms.add(room_idx)

    colors = random.sample(list(MARKER_COLORS), len(placements))
    grid = _paint_markers(grid, placements, colors)
    if not _valid_example(grid, want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid directional-room-sliding input")
