from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
PATTERN = 5
WALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MIN_ROOM = 4
MAX_EXTRA_SPLITS = 3


@dataclass(frozen=True)
class Room:
    room_id: int
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass
class Wall:
    orientation: str
    coord: int
    start: int
    end: int
    color: int = -1
    door: int = -1


SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _room_for_cell(rooms: list[Room], r: int, c: int) -> Room | None:
    for room in rooms:
        if room.r0 <= r <= room.r1 and room.c0 <= c <= room.c1:
            return room
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = mask.shape
    view = grid[top : top + rows, left : left + cols]
    view[mask != 0] = color


def _close_wall_mask(shape: tuple[int, int], walls: list[Wall]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for wall in walls:
        if wall.orientation == "v":
            mask[wall.start : wall.end + 1, wall.coord] = True
        else:
            mask[wall.coord, wall.start : wall.end + 1] = True
    return mask


def _detect_walls(grid: np.ndarray) -> list[Wall]:
    walls: list[Wall] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, PATTERN)):
        coords = np.argwhere(grid == color)
        best: tuple[int, int, Wall] | None = None

        row_vals = sorted({int(r) for r in coords[:, 0]})
        for row in row_vals:
            cols = sorted(int(c) for c in coords[coords[:, 0] == row][:, 1])
            if len(cols) < 2:
                continue
            span_start, span_end = cols[0], cols[-1]
            missing = sorted(set(range(span_start, span_end + 1)) - set(cols))
            if len(missing) != 1 or missing[0] in (span_start, span_end):
                continue
            wall = Wall("h", row, span_start, span_end, color=color, door=missing[0])
            score = (len(cols), span_end - span_start)
            if best is None or score > best[:2]:
                best = (score[0], score[1], wall)

        col_vals = sorted({int(c) for c in coords[:, 1]})
        for col in col_vals:
            rows = sorted(int(r) for r in coords[coords[:, 1] == col][:, 0])
            if len(rows) < 2:
                continue
            span_start, span_end = rows[0], rows[-1]
            missing = sorted(set(range(span_start, span_end + 1)) - set(rows))
            if len(missing) != 1 or missing[0] in (span_start, span_end):
                continue
            wall = Wall("v", col, span_start, span_end, color=color, door=missing[0])
            score = (len(rows), span_end - span_start)
            if best is None or score > best[:2]:
                best = (score[0], score[1], wall)

        assert best is not None, f"could not recover wall for color {color}"
        walls.append(best[2])
    return walls


def _label_rooms(shape: tuple[int, int], wall_mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = shape
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []
    room_id = 0
    for r in range(h):
        for c in range(w):
            if wall_mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = room_id
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = room_id
                        q.append((nr, nc))
            rooms.append(np.asarray(pts, dtype=int))
            room_id += 1
    return labels, rooms


def _extract_pattern_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = PATTERN
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each room's key color to choose a wall-door pair, then rotate and push the
    room's grey pattern through that door into the neighboring room.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls = _detect_walls(g)
    wall_by_color = {wall.color: wall for wall in walls}

    wall_mask = _close_wall_mask(g.shape, walls)
    labels, room_cells = _label_rooms(g.shape, wall_mask)

    actions: list[tuple[np.ndarray, tuple[int, int], np.ndarray, tuple[int, int], int]] = []

    for room_id, coords in enumerate(room_cells):
        room_mask = labels == room_id
        grey_coords = np.argwhere(room_mask & (g == PATTERN))
        interior_color_coords = np.argwhere(room_mask & (g != BG) & (g != PATTERN))

        if grey_coords.size == 0:
            assert interior_color_coords.size == 0, "empty rooms should not contain stray key pixels"
            continue

        assert len(_components(room_mask & (g == PATTERN))) == 1, "each source room must hold one grey component"
        assert interior_color_coords.shape[0] == 1, "each source room must hold exactly one key"

        key_r, key_c = map(int, interior_color_coords[0])
        key_color = int(g[key_r, key_c])
        assert key_color in wall_by_color, "key color must identify a wall"
        wall = wall_by_color[key_color]

        patch = _extract_pattern_patch(g, grey_coords)
        if wall.orientation == "v":
            door_r = wall.door
            door_c = wall.coord
            left_room = int(labels[door_r, door_c - 1])
            right_room = int(labels[door_r, door_c + 1])
            rotated = np.rot90(patch, -1)
            if room_id == left_room:
                target_room = right_room
                top, left = door_r, door_c + 1
            else:
                assert room_id == right_room, "key wall must touch the source room at its door"
                target_room = left_room
                top, left = door_r, door_c - rotated.shape[1]
        else:
            door_r = wall.coord
            door_c = wall.door
            upper_room = int(labels[door_r - 1, door_c])
            lower_room = int(labels[door_r + 1, door_c])
            rotated = np.rot90(patch, 1)
            if room_id == upper_room:
                target_room = lower_room
                top, left = door_r + 1, door_c
            else:
                assert room_id == lower_room, "key wall must touch the source room at its door"
                target_room = upper_room
                top, left = door_r - rotated.shape[0], door_c

        rh, rw = rotated.shape
        assert 0 <= top and 0 <= left and top + rh <= h and left + rw <= w
        target_labels = labels[top : top + rh, left : left + rw]
        assert np.all(target_labels == target_room), "moved pattern bbox must stay inside the neighboring room"

        actions.append((grey_coords, (key_r, key_c), rotated, (top, left), target_room))

    assert actions, "expected at least one source room"
    out = g.copy()
    for grey_coords, key_pos, _rotated, _target_tl, _target_room in actions:
        out[grey_coords[:, 0], grey_coords[:, 1]] = BG
        out[key_pos] = BG

    for _grey_coords, _key_pos, rotated, (top, left), target_room in actions:
        view = out[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        target_labels = labels[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        assert np.all(target_labels == target_room)
        assert np.all(view[rotated == PATTERN] == BG), "target room should be empty before placement"
        view[rotated == PATTERN] = PATTERN

    return out


def _can_split(room: Room, orientation: str) -> bool:
    if orientation == "v":
        return room.width >= 2 * MIN_ROOM + 1 and room.height >= 3
    return room.height >= 2 * MIN_ROOM + 1 and room.width >= 3


def _split_room(rng: random.Random, room: Room, orientation: str, next_room_id: int) -> tuple[list[Room], Wall]:
    if orientation == "v":
        split_col = rng.randint(room.c0 + MIN_ROOM, room.c1 - MIN_ROOM)
        children = [
            Room(next_room_id, room.r0, room.r1, room.c0, split_col - 1),
            Room(next_room_id + 1, room.r0, room.r1, split_col + 1, room.c1),
        ]
        wall = Wall("v", split_col, room.r0, room.r1)
    else:
        split_row = rng.randint(room.r0 + MIN_ROOM, room.r1 - MIN_ROOM)
        children = [
            Room(next_room_id, room.r0, split_row - 1, room.c0, room.c1),
            Room(next_room_id + 1, split_row + 1, room.r1, room.c0, room.c1),
        ]
        wall = Wall("h", split_row, room.c0, room.c1)
    return children, wall


def _door_options(wall: Wall, closed_mask: np.ndarray) -> list[int]:
    options: list[int] = []
    if wall.orientation == "v":
        if not (0 < wall.coord < closed_mask.shape[1] - 1):
            return options
        for row in range(wall.start + 1, wall.end):
            if not closed_mask[row, wall.coord - 1] and not closed_mask[row, wall.coord + 1]:
                options.append(row)
    else:
        if not (0 < wall.coord < closed_mask.shape[0] - 1):
            return options
        for col in range(wall.start + 1, wall.end):
            if not closed_mask[wall.coord - 1, col] and not closed_mask[wall.coord + 1, col]:
                options.append(col)
    return options


def _bbox_fits_room(room: Room, top: int, left: int, height: int, width: int) -> bool:
    return room.r0 <= top and room.c0 <= left and top + height - 1 <= room.r1 and left + width - 1 <= room.c1


def _key_candidates(room: Room, pattern_mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = [(top + r, left + c) for r, c in np.argwhere(pattern_mask != 0)]
    pr0, pc0 = top, left
    pr1 = top + pattern_mask.shape[0] - 1
    pc1 = left + pattern_mask.shape[1] - 1
    cand: list[tuple[int, int]] = []
    rr0 = max(room.r0, pr0 - 2)
    rr1 = min(room.r1, pr1 + 2)
    cc0 = max(room.c0, pc0 - 2)
    cc1 = min(room.c1, pc1 + 2)
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            if pattern_mask[r - top, c - left] != 0 if pr0 <= r <= pr1 and pc0 <= c <= pc1 else False:
                continue
            if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in pts):
                continue
            cand.append((r, c))
    return cand


def _target_top_left(wall: Wall, direction: str, moved_shape: np.ndarray) -> tuple[int, int]:
    if direction == "right":
        return wall.door, wall.coord + 1
    if direction == "left":
        return wall.door, wall.coord - moved_shape.shape[1]
    if direction == "down":
        return wall.coord + 1, wall.door
    return wall.coord - moved_shape.shape[0], wall.door


def _plan_move(
    rng: random.Random,
    source_room: Room,
    target_room: Room,
    wall: Wall,
    direction: str,
) -> dict[str, object] | None:
    shape_order = SHAPES[:]
    rng.shuffle(shape_order)
    for shape in shape_order:
        moved = np.rot90(shape, -1) if wall.orientation == "v" else np.rot90(shape, 1)
        target_top, target_left = _target_top_left(wall, direction, moved)
        if not _bbox_fits_room(target_room, target_top, target_left, moved.shape[0], moved.shape[1]):
            continue

        placements: list[tuple[int, int, tuple[int, int]]] = []
        for top in range(source_room.r0, source_room.r1 - shape.shape[0] + 2):
            for left in range(source_room.c0, source_room.c1 - shape.shape[1] + 2):
                keys = _key_candidates(source_room, shape, top, left)
                if not keys:
                    continue
                placements.append((top, left, rng.choice(keys)))
        if not placements:
            continue

        top, left, key_pos = rng.choice(placements)
        return {
            "shape": shape,
            "source_top_left": (top, left),
            "key_pos": key_pos,
            "target_top_left": (target_top, target_left),
            "moved_shape": moved,
            "wall": wall,
        }
    return None


def _build_partition(rng: random.Random, h: int, w: int) -> tuple[list[Room], list[Wall]] | None:
    rooms = [Room(0, 0, h - 1, 0, w - 1)]
    walls: list[Wall] = []
    next_room_id = 1

    first = rng.choice(["v", "h"])
    second = "h" if first == "v" else "v"

    for orientation in (first, second):
        candidates = [room for room in rooms if _can_split(room, orientation)]
        if not candidates:
            return None
        room = rng.choice(candidates)
        rooms.remove(room)
        children, wall = _split_room(rng, room, orientation, next_room_id)
        next_room_id += 2
        rooms.extend(children)
        walls.append(wall)

    for _ in range(rng.randint(1, MAX_EXTRA_SPLITS)):
        options = [(room, orientation) for room in rooms for orientation in ("v", "h") if _can_split(room, orientation)]
        if not options or len(walls) >= len(WALL_COLORS):
            break
        room, orientation = rng.choice(options)
        rooms.remove(room)
        children, wall = _split_room(rng, room, orientation, next_room_id)
        next_room_id += 2
        rooms.extend(children)
        walls.append(wall)

    return rooms, walls


def _candidate_moves(rooms: list[Room], walls: list[Wall]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for wall in walls:
        if wall.orientation == "v":
            left_room = _room_for_cell(rooms, wall.door, wall.coord - 1)
            right_room = _room_for_cell(rooms, wall.door, wall.coord + 1)
            if left_room is None or right_room is None:
                continue
            candidates.append(
                {
                    "wall": wall,
                    "orientation": "v",
                    "room_pair": {left_room.room_id, right_room.room_id},
                    "left": left_room,
                    "right": right_room,
                }
            )
        else:
            upper_room = _room_for_cell(rooms, wall.coord - 1, wall.door)
            lower_room = _room_for_cell(rooms, wall.coord + 1, wall.door)
            if upper_room is None or lower_room is None:
                continue
            candidates.append(
                {
                    "wall": wall,
                    "orientation": "h",
                    "room_pair": {upper_room.room_id, lower_room.room_id},
                    "up": upper_room,
                    "down": lower_room,
                }
            )
    return candidates


def _select_moves(rng: random.Random, candidates: list[dict[str, object]]) -> list[dict[str, object]] | None:
    vertical = [cand for cand in candidates if cand["orientation"] == "v"]
    horizontal = [cand for cand in candidates if cand["orientation"] == "h"]
    pair_options = [
        (v_cand, h_cand)
        for v_cand in vertical
        for h_cand in horizontal
        if cand_room_ids(v_cand).isdisjoint(cand_room_ids(h_cand))
    ]
    if not pair_options:
        return None

    chosen_v, chosen_h = rng.choice(pair_options)
    selected = [chosen_v, chosen_h]
    used = cand_room_ids(chosen_v) | cand_room_ids(chosen_h)
    remainder = [cand for cand in candidates if cand not in selected and cand_room_ids(cand).isdisjoint(used)]
    if remainder and rng.random() < 0.5:
        selected.append(rng.choice(remainder))
    return selected


def cand_room_ids(candidate: dict[str, object]) -> set[int]:
    return set(int(v) for v in candidate["room_pair"])


def _instantiate_moves(
    rng: random.Random,
    selected: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    plans: list[dict[str, object]] = []
    for candidate in selected:
        wall = candidate["wall"]
        if candidate["orientation"] == "v":
            if rng.random() < 0.5:
                source_room = candidate["left"]
                target_room = candidate["right"]
                direction = "right"
            else:
                source_room = candidate["right"]
                target_room = candidate["left"]
                direction = "left"
        else:
            if rng.random() < 0.5:
                source_room = candidate["up"]
                target_room = candidate["down"]
                direction = "down"
            else:
                source_room = candidate["down"]
                target_room = candidate["up"]
                direction = "up"

        plan = _plan_move(
            rng,
            source_room=source_room,
            target_room=target_room,
            wall=wall,
            direction=direction,
        )
        if plan is None:
            return None
        plans.append(plan)
    return plans


def _render_base_grid(shape: tuple[int, int], walls: list[Wall]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for wall in walls:
        if wall.orientation == "v":
            grid[wall.start : wall.end + 1, wall.coord] = wall.color
            grid[wall.door, wall.coord] = BG
        else:
            grid[wall.coord, wall.start : wall.end + 1] = wall.color
            grid[wall.coord, wall.door] = BG
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 20)
    w = rng.randint(13, 20)
    built = _build_partition(rng, h, w)
    if built is None:
        return None
    rooms, walls = built
    if len(walls) > len(WALL_COLORS):
        return None

    colors = rng.sample(WALL_COLORS, len(walls))
    for wall, color in zip(walls, colors):
        wall.color = color

    closed_mask = _close_wall_mask((h, w), walls)
    for wall in walls:
        options = _door_options(wall, closed_mask)
        if not options:
            return None
        wall.door = rng.choice(options)

    candidates = _candidate_moves(rooms, walls)
    selected = _select_moves(rng, candidates)
    if selected is None:
        return None
    plans = _instantiate_moves(rng, selected)
    if plans is None:
        return None

    inp = _render_base_grid((h, w), walls)
    out = inp.copy()

    for plan in plans:
        shape = np.asarray(plan["shape"], dtype=int)
        source_top, source_left = plan["source_top_left"]
        target_top, target_left = plan["target_top_left"]
        key_r, key_c = plan["key_pos"]
        wall = plan["wall"]
        moved = np.asarray(plan["moved_shape"], dtype=int)

        _paint_mask(inp, shape, int(source_top), int(source_left), PATTERN)
        inp[int(key_r), int(key_c)] = int(wall.color)
        _paint_mask(out, moved, int(target_top), int(target_left), PATTERN)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
