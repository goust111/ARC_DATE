from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Room = tuple[int, int, int, int]

NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 700
ROOM_LAYOUTS: tuple[tuple[int, int], ...] = ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
ROOM_SIDE_MIN = 7
ROOM_SIDE_MAX = 9

_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue

            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _full_lines(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == int(color)))]
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == int(color)))]
    return rows, cols


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        rows, cols = _full_lines(grid, color)
        if not rows and not cols:
            continue

        line_mask = np.zeros(grid.shape, dtype=bool)
        if rows:
            line_mask[rows, :] = True
        if cols:
            line_mask[:, cols] = True
        if np.array_equal(grid == color, line_mask):
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected one visible full-span divider color"
    return candidates[0]


def _rooms_from_lines(shape: tuple[int, int], divider_rows: list[int], divider_cols: list[int]) -> list[Room]:
    h, w = shape
    row_breaks = [-1] + list(divider_rows) + [h]
    col_breaks = [-1] + list(divider_cols) + [w]
    rooms: list[Room] = []

    for a, b in zip(row_breaks, row_breaks[1:]):
        assert b - a > 1, "divider rows must leave positive-height rooms"
        r0 = a + 1
        r1 = b - 1
        for c, d in zip(col_breaks, col_breaks[1:]):
            assert d - c > 1, "divider columns must leave positive-width rooms"
            rooms.append((r0, c + 1, r1, d - 1))

    assert len(rooms) >= 2, "expected at least two rooms"
    return rooms


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    bg = _get_bg(grid)
    divider, divider_rows, divider_cols = _find_dividers(grid, bg)
    rooms = _rooms_from_lines(grid.shape, divider_rows, divider_cols)

    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), int(divider)}]
    role_candidates: list[tuple[int, list[dict]]] = []

    for marker in colors:
        marker_mask = grid == marker
        marker_components = _components(marker_mask)
        if not marker_components or any(comp.shape[0] != 1 for comp in marker_components):
            continue

        infos: list[dict] = []
        object_colors: list[int] = []
        marker_total = 0
        ok = True

        for room in rooms:
            r0, c0, r1, c1 = room
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            present = set(map(int, np.unique(sub))) - {int(bg), int(marker)}
            if len(present) != 1:
                ok = False
                break

            obj_color = int(next(iter(present)))
            obj_mask = sub == obj_color
            obj_components = _components(obj_mask)
            if len(obj_components) != 1 or obj_components[0].shape[0] < 4:
                ok = False
                break

            votes = int(np.count_nonzero(sub == marker))
            marker_total += votes
            object_colors.append(obj_color)
            infos.append(
                {
                    "room": room,
                    "color": obj_color,
                    "votes": votes,
                    "area": int(np.count_nonzero(obj_mask)),
                }
            )

        if not ok:
            continue
        if marker_total != int(np.count_nonzero(marker_mask)):
            continue
        if len(set(object_colors)) != len(object_colors):
            continue
        role_candidates.append((marker, infos))

    assert len(role_candidates) == 1, "expected a unique marker color and one object per room"
    marker, infos = role_candidates[0]
    return bg, divider, marker, infos


def _dilation_targets(sub: np.ndarray, obj_color: int, bg: int) -> np.ndarray:
    add = np.zeros(sub.shape, dtype=bool)
    for r, c in np.argwhere(sub == int(obj_color)):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and int(sub[nr, nc]) == int(bg):
                add[nr, nc] = True
    return add


def _erosion_targets(sub: np.ndarray, obj_color: int, bg: int) -> np.ndarray:
    remove = np.zeros(sub.shape, dtype=bool)
    for r, c in np.argwhere(sub == int(obj_color)):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and int(sub[nr, nc]) == int(bg):
                remove[rr, cc] = True
                break
    return remove


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the object in the room with the most marker votes and shrink all other room objects.

    Input structure:
    - A dominant background is partitioned into rectangular rooms by one-color full-span divider lines.
    - Every room contains exactly one 4-connected monochrome solid object; object colors are unique across rooms.
    - One global marker color appears as isolated vote pixels on room background cells, and one room has strictly the most markers.

    Rule:
    - Count the marker pixels separately inside each rectangular room.
    - The object in the unique highest-vote room dilates by one 4-neighbor layer into same-room background cells, clipped by room edges and dividers.
    - Every other room object erodes by one 4-neighbor layer: object cells touching same-room background become background.
    - Finally all marker pixels are erased to the background and divider lines are left unchanged.

    Key insight:
    - The winning object is selected by marker count, not by its size; the votes choose which room gains one morphological ring while every other room loses one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _divider, marker, infos = _parse_scene(g)
    votes = [int(info["votes"]) for info in infos]
    max_votes = max(votes)
    assert votes.count(max_votes) == 1, "expected a unique room with the most marker votes"
    winner_idx = votes.index(max_votes)

    out = g.copy()
    for idx, info in enumerate(infos):
        r0, c0, r1, c1 = info["room"]
        color = int(info["color"])
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        dst = out[r0 : r1 + 1, c0 : c1 + 1]

        if idx == winner_idx:
            dst[_dilation_targets(sub, color, bg)] = color
        else:
            dst[_erosion_targets(sub, color, bg)] = bg

    out[g == int(marker)] = bg
    return out


def _ranges_from_lengths(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = 0
    for i, size in enumerate(lengths):
        ranges.append((cursor, cursor + int(size) - 1))
        cursor += int(size)
        if i != len(lengths) - 1:
            dividers.append(cursor)
            cursor += 1
    return ranges, dividers, cursor


def _make_rooms(row_count: int, col_count: int, rng: random.Random) -> tuple[tuple[int, int], list[Room], list[int], list[int]]:
    row_lengths = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(row_count)]
    col_lengths = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(col_count)]
    row_ranges, divider_rows, height = _ranges_from_lengths(row_lengths)
    col_ranges, divider_cols, width = _ranges_from_lengths(col_lengths)
    if height > 30 or width > 30:
        raise AssertionError("generated grid would exceed ARC size")

    rooms: list[Room] = []
    for rr0, rr1 in row_ranges:
        for cc0, cc1 in col_ranges:
            rooms.append((rr0, cc0, rr1, cc1))
    return (height, width), rooms, divider_rows, divider_cols


def _sample_object_size(room: Room, kind: str, rng: random.Random) -> tuple[int, int] | None:
    r0, c0, r1, c1 = room
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    max_h = room_h - 2
    max_w = room_w - 2
    if kind == "small":
        lo, hi = 3, 4
    elif kind == "large":
        lo, hi = 5, 6
    else:
        lo, hi = 3, 5

    hi_h = min(hi, max_h)
    hi_w = min(hi, max_w)
    if hi_h < lo or hi_w < lo:
        return None
    return rng.randint(lo, hi_h), rng.randint(lo, hi_w)


def _place_rect(room: Room, height: int, width: int, rng: random.Random, touch_side: str | None) -> list[Coord] | None:
    r0, c0, r1, c1 = room

    if touch_side == "top":
        top = r0
    elif touch_side == "bottom":
        top = r1 - height + 1
    else:
        lo = r0 + 1
        hi = r1 - height
        if lo > hi:
            return None
        top = rng.randint(lo, hi)

    if touch_side == "left":
        left = c0
    elif touch_side == "right":
        left = c1 - width + 1
    else:
        lo = c0 + 1
        hi = c1 - width
        if lo > hi:
            return None
        left = rng.randint(lo, hi)

    if top < r0 or left < c0 or top + height - 1 > r1 or left + width - 1 > c1:
        return None
    return [(r, c) for r in range(top, top + height) for c in range(left, left + width)]


def _touches_object(grid: np.ndarray, r: int, c: int, object_color: int) -> bool:
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == int(object_color):
            return True
    return False


def _marker_candidates(grid: np.ndarray, room: Room, bg: int, object_color: int, placed: list[Coord]) -> list[Coord]:
    r0, c0, r1, c1 = room
    candidates: list[Coord] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != int(bg):
                continue
            if _touches_object(grid, r, c, object_color):
                continue
            if any(abs(r - pr) <= 1 and abs(c - pc) <= 1 for pr, pc in placed):
                continue
            candidates.append((r, c))
    return candidates


def _place_markers(
    grid: np.ndarray,
    room: Room,
    bg: int,
    marker_color: int,
    object_color: int,
    count: int,
    rng: random.Random,
) -> list[Coord] | None:
    placed: list[Coord] = []
    for _ in range(count):
        candidates = _marker_candidates(grid, room, bg, object_color, placed)
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        grid[r, c] = marker_color
        placed.append((r, c))
    return placed


def _object_touches_room_edge(coords: list[Coord], room: Room) -> bool:
    r0, c0, r1, c1 = room
    return any(r in (r0, r1) or c in (c0, c1) for r, c in coords)


def _room_object_count(grid: np.ndarray, room: Room, color: int) -> int:
    r0, c0, r1, c1 = room
    return int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _expected_from_metadata(grid: np.ndarray, bg: int, marker: int, objects: list[dict], winner_idx: int) -> np.ndarray:
    out = grid.copy()
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["room"]
        color = int(obj["color"])
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        dst = out[r0 : r1 + 1, c0 : c1 + 1]
        if idx == winner_idx:
            dst[_dilation_targets(sub, color, bg)] = color
        else:
            dst[_erosion_targets(sub, color, bg)] = bg
    out[grid == int(marker)] = bg
    return out


def _valid_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    marker: int,
    objects: list[dict],
    winner_idx: int,
    branch: int,
    clipped_required: bool,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if _get_bg(grid) != int(bg):
        return False

    votes = [int(obj["votes"]) for obj in objects]
    if votes.count(max(votes)) != 1 or votes.index(max(votes)) != winner_idx:
        return False

    areas = [int(obj["area"]) for obj in objects]
    if branch == 0 and not all(areas[winner_idx] > area for i, area in enumerate(areas) if i != winner_idx):
        return False
    if branch != 0 and not any(i != winner_idx and area > areas[winner_idx] for i, area in enumerate(areas)):
        return False
    if clipped_required and not _object_touches_room_edge(objects[winner_idx]["coords"], objects[winner_idx]["room"]):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.any(solved == int(marker)):
        return False

    for idx, obj in enumerate(objects):
        before = _room_object_count(grid, obj["room"], obj["color"])
        after = _room_object_count(expected, obj["room"], obj["color"])
        if idx == winner_idx:
            if after <= before:
                return False
        elif after >= before:
            return False

    return True


def _gen_attempt(rng: random.Random, branch: int) -> np.ndarray | None:
    if branch == 0:
        row_count, col_count = rng.choice(((2, 2), (2, 3), (3, 2)))
        winner_largest = True
        clipped_winner = True
    elif branch == 1:
        row_count, col_count = rng.choice(((1, 3), (3, 1)))
        winner_largest = False
        clipped_winner = False
    else:
        row_count, col_count = rng.choice(ROOM_LAYOUTS)
        winner_largest = False
        clipped_winner = True

    shape, rooms, divider_rows, divider_cols = _make_rooms(row_count, col_count, rng)
    room_count = len(rooms)
    if room_count + 3 > len(PALETTE):
        return None

    colors = list(PALETTE)
    rng.shuffle(colors)
    bg = int(colors[0])
    divider = int(colors[1])
    marker = int(colors[2])
    object_colors = [int(c) for c in colors[3 : 3 + room_count]]

    grid = np.full(shape, bg, dtype=int)
    for r in divider_rows:
        grid[r, :] = divider
    for c in divider_cols:
        grid[:, c] = divider

    winner_idx = rng.randrange(room_count)
    big_loser_idx = None
    if not winner_largest:
        loser_indices = [i for i in range(room_count) if i != winner_idx]
        big_loser_idx = rng.choice(loser_indices)

    objects: list[dict] = []
    for idx, room in enumerate(rooms):
        if idx == winner_idx:
            kind = "large" if winner_largest else "small"
        elif idx == big_loser_idx:
            kind = "large"
        else:
            kind = rng.choice(("small", "medium"))

        size = _sample_object_size(room, kind, rng)
        if size is None:
            return None
        height, width = size
        touch_side = rng.choice(("top", "bottom", "left", "right")) if idx == winner_idx and clipped_winner else None
        coords = _place_rect(room, height, width, rng, touch_side)
        if coords is None:
            return None

        color = object_colors[idx]
        for r, c in coords:
            grid[r, c] = color
        objects.append(
            {
                "room": room,
                "color": color,
                "coords": coords,
                "area": len(coords),
                "votes": 0,
            }
        )

    areas = [int(obj["area"]) for obj in objects]
    if winner_largest:
        if not all(areas[winner_idx] > area for i, area in enumerate(areas) if i != winner_idx):
            return None
    else:
        if not any(i != winner_idx and area > areas[winner_idx] for i, area in enumerate(areas)):
            return None

    winner_votes = rng.randint(3, 6)
    vote_counts: list[int] = []
    for idx, obj in enumerate(objects):
        if idx == winner_idx:
            vote_counts.append(winner_votes)
        else:
            vote_counts.append(rng.randint(1, min(3, winner_votes - 1)))

    for idx, obj in enumerate(objects):
        placed = _place_markers(
            grid,
            obj["room"],
            bg,
            marker,
            obj["color"],
            vote_counts[idx],
            rng,
        )
        if placed is None:
            return None
        obj["votes"] = vote_counts[idx]
        obj["markers"] = placed

    expected = _expected_from_metadata(grid, bg, marker, objects, winner_idx)
    if not _valid_semantics(grid, expected, bg, marker, objects, winner_idx, branch, clipped_winner):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng, branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid vote-room morphology input")
