from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 1
MARKER = 6
SPRITE_COLORS = (2, 3, 4, 7, 8, 9)
ROOM_SIZE_CHOICES = (
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 3),
    (4, 4),
    (4, 5),
)
GEN_RETRIES = 400

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, rect: Rect) -> np.ndarray:
    r0, c0, r1, c1 = rect
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_from_marker(grid: np.ndarray) -> tuple[Coord, np.ndarray]:
    marker_coords = np.argwhere(grid == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one marker pixel"
    start = tuple(map(int, marker_coords[0]))
    reach = np.zeros_like(grid, dtype=bool)
    q = deque([start])
    reach[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or reach[nr, nc]:
                continue
            if int(grid[nr, nc]) not in (BG, MARKER):
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return start, reach


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _all_zero(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return _rect_sum(prefix, r0, c0, r1, c1) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _outside_zero_contacts(grid: np.ndarray, rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    h, w = grid.shape
    contacts: set[Coord] = set()
    for c in range(c0, c1 + 1):
        if r0 > 0 and grid[r0 - 1, c] == BG:
            contacts.add((r0 - 1, c))
        if r1 + 1 < h and grid[r1 + 1, c] == BG:
            contacts.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if c0 > 0 and grid[r, c0 - 1] == BG:
            contacts.add((r, c0 - 1))
        if c1 + 1 < w and grid[r, c1 + 1] == BG:
            contacts.add((r, c1 + 1))
    return len(contacts)


def _detect_cavities(template: np.ndarray, allowed_sizes: set[tuple[int, int]]) -> list[Rect]:
    zero = (template == BG).astype(int)
    prefix = zero.cumsum(axis=0).cumsum(axis=1)
    h, w = template.shape
    by_size: dict[tuple[int, int], list[Rect]] = {size: [] for size in allowed_sizes}

    for room_h, room_w in sorted(allowed_sizes):
        if room_h < 2 or room_w < 2 or room_h * room_w < 6:
            continue
        for r0 in range(h - room_h + 1):
            r1 = r0 + room_h - 1
            for c0 in range(w - room_w + 1):
                c1 = c0 + room_w - 1
                if not _all_zero(prefix, r0, c0, r1, c1):
                    continue
                if r0 > 0 and _all_zero(prefix, r0 - 1, c0, r1, c1):
                    continue
                if r1 + 1 < h and _all_zero(prefix, r0, c0, r1 + 1, c1):
                    continue
                if c0 > 0 and _all_zero(prefix, r0, c0 - 1, r1, c1):
                    continue
                if c1 + 1 < w and _all_zero(prefix, r0, c0, r1, c1 + 1):
                    continue
                rect = (r0, c0, r1, c1)
                if _outside_zero_contacts(template, rect) <= 2:
                    by_size[(room_h, room_w)].append(rect)

    cavities = []
    for size, candidates in by_size.items():
        assert candidates, f"no cavity found for size {size}"
        best = min(
            candidates,
            key=lambda rect: (
                _outside_zero_contacts(template, rect),
                rect[0],
                rect[1],
            ),
        )
        cavities.append(best)
    return cavities


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, WALL, MARKER)):
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            sprite = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == color, color, BG)
            sprites.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "size": (r1 - r0 + 1, c1 - c0 + 1),
                    "sprite": sprite,
                }
            )
    return sprites


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed only the sprites whose matching rectangular cavities are reachable from the corner marker.

    Input structure:
    - Black background fills the canvas; a unique magenta marker sits on one outer corner.
    - Blue pixels form one large wall template whose interior contains several rectangular zero
      cavities, plus optional one-cell corridors from the template edge into some cavities.
    - Outside the template are several separated monochrome sprites; each sprite's bbox size
      uniquely matches exactly one cavity size.

    Rule:
    - Flood-fill from the magenta marker through black cells using 4-connectivity; blue walls block.
    - Inside the blue template bbox, find the rectangular zero cavities whose sizes match sprite bboxes.
    - A cavity stays active iff the flood reaches at least one cell inside it; otherwise fill that
      entire cavity with blue.
    - For each active cavity, place the uniquely size-matched sprite into that cavity with top-left
      alignment, erase the marker, and return only the template bbox.

    Key insight:
    - Reachability changes the cavity pool before size matching, so topology decides which sprites
      still have a valid slot.
    """
    g = np.asarray(grid, dtype=int)
    blue_coords = np.argwhere(g == WALL)
    assert len(blue_coords) > 0, "expected a blue template"
    template_bbox = _bbox_coords(blue_coords)
    template = _crop(g, template_bbox)

    _marker, reach = _flood_from_marker(g)
    sprites = _find_sprites(g)
    sprite_by_size: dict[tuple[int, int], np.ndarray] = {}
    for sprite in sprites:
        size = sprite["size"]
        assert size not in sprite_by_size, f"duplicate sprite size {size}"
        sprite_by_size[size] = sprite["sprite"]

    cavities = _detect_cavities(template, set(sprite_by_size))
    tr0, tc0, _tr1, _tc1 = template_bbox
    out = template.copy()
    out[(out != BG) & (out != WALL)] = BG

    for lr0, lc0, lr1, lc1 in cavities:
        size = (lr1 - lr0 + 1, lc1 - lc0 + 1)
        abs_rect = (tr0 + lr0, tc0 + lc0, tr0 + lr1, tc0 + lc1)
        active = bool(reach[abs_rect[0] : abs_rect[2] + 1, abs_rect[1] : abs_rect[3] + 1].any())
        if not active:
            out[lr0 : lr1 + 1, lc0 : lc1 + 1] = WALL
            continue
        _paste(out, sprite_by_size[size], lr0, lc0)

    return out


def _expanded_rect(rect: Rect, shape: tuple[int, int], pad: int = 1) -> Rect:
    r0, c0, r1, c1 = rect
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _mark_blocked(mask: np.ndarray, rect: Rect) -> None:
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_sprite_mask(h: int, w: int) -> np.ndarray:
    max_cells = min(8, h * w)
    for _ in range(80):
        target = random.randint(3, max_cells)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        coords = np.array(sorted(cells), dtype=int)
        if len(coords) < 3:
            continue
        if _bbox_coords(coords) == (0, 0, h - 1, w - 1):
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return mask

    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        mask[0, :] = True
        mask[:, -1] = True
    else:
        mask[:, 0] = True
        mask[-1, :] = True
    if random.random() < 0.5:
        mask = mask[::-1, :]
    if random.random() < 0.5:
        mask = mask[:, ::-1]
    return mask


def _place_rooms(frame_h: int, frame_w: int, sizes: list[tuple[int, int]]) -> list[dict] | None:
    reserved = np.zeros((frame_h, frame_w), dtype=bool)
    rooms = []
    for room_h, room_w in sorted(sizes, key=lambda size: (size[0] + 2) * (size[1] + 2), reverse=True):
        candidates = []
        for r0 in range(1, frame_h - room_h):
            r1 = r0 + room_h - 1
            if r1 >= frame_h - 1:
                continue
            for c0 in range(1, frame_w - room_w):
                c1 = c0 + room_w - 1
                if c1 >= frame_w - 1:
                    continue
                rr0, cc0, rr1, cc1 = _expanded_rect((r0, c0, r1, c1), (frame_h, frame_w), pad=1)
                if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((r0, c0, r1, c1))
        if not candidates:
            return None
        rect = random.choice(candidates)
        rr0, cc0, rr1, cc1 = _expanded_rect(rect, (frame_h, frame_w), pad=1)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        rooms.append({"rect_local": rect, "size": (room_h, room_w)})
    return rooms


def _door_candidates(rect: Rect) -> list[Coord]:
    r0, c0, r1, c1 = rect
    doors = []

    def ordered(points: list[Coord]) -> list[Coord]:
        if len(points) <= 2:
            return points
        return points[1:-1] + points[:1] + points[-1:]

    sides = [
        ordered([(r0 - 1, c) for c in range(c0, c1 + 1)]),
        ordered([(r1 + 1, c) for c in range(c0, c1 + 1)]),
        ordered([(r, c0 - 1) for r in range(r0, r1 + 1)]),
        ordered([(r, c1 + 1) for r in range(r0, r1 + 1)]),
    ]
    random.shuffle(sides)
    for side in sides:
        doors.extend(side)
    return doors


def _corridor_path(frame: np.ndarray, rooms: list[dict], room_index: int) -> list[Coord] | None:
    h, w = frame.shape
    target_room = rooms[room_index]
    expanded = np.zeros((h, w), dtype=bool)
    for idx, room in enumerate(rooms):
        rect = room["rect_local"]
        blocked_rect = _expanded_rect(rect, (h, w), pad=1)
        if idx == room_index:
            _mark_blocked(expanded, blocked_rect)
            r0, c0, r1, c1 = rect
            expanded[r0 : r1 + 1, c0 : c1 + 1] = True
        else:
            _mark_blocked(expanded, blocked_rect)

    for start in _door_candidates(target_room["rect_local"]):
        if not _in_bounds(frame.shape, *start):
            continue
        if frame[start] != WALL:
            continue
        blocked = expanded.copy()
        blocked[start] = False
        q = deque([start])
        parents: dict[Coord, Coord | None] = {start: None}
        goal = None
        while q:
            r, c = q.popleft()
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                goal = (r, c)
                break
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(frame.shape, nr, nc):
                    continue
                if (nr, nc) in parents:
                    continue
                if blocked[nr, nc] or frame[nr, nc] != WALL:
                    continue
                parents[(nr, nc)] = (r, c)
                q.append((nr, nc))
        if goal is None:
            continue
        path = []
        cur: Coord | None = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()
        return path
    return None


def _place_sprite_bbox(
    blocked: np.ndarray,
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
) -> tuple[int, int] | None:
    gh, gw = grid_shape
    ph, pw = patch_shape
    candidates = []
    for r0 in range(gh - ph + 1):
        r1 = r0 + ph - 1
        for c0 in range(gw - pw + 1):
            c1 = c0 + pw - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(case: dict) -> np.ndarray:
    out = case["template"].copy()
    for room in case["rooms"]:
        r0, c0, r1, c1 = room["rect_local"]
        if not room["active"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = WALL
            continue
        _paste(out, room["sprite"], r0, c0)
    return out


def _generate_case() -> dict | None:
    room_count = random.randint(3, 5)
    sizes = random.sample(ROOM_SIZE_CHOICES, room_count)
    reserved_area = sum((h + 2) * (w + 2) for h, w in sizes)
    min_side = max(14, int(np.ceil(reserved_area ** 0.5)))
    frame_h = random.randint(min_side, min(20, min_side + 4))
    frame_w = random.randint(min_side, min(20, min_side + 4))
    if frame_h * frame_w < reserved_area + 20:
        return None

    rooms = _place_rooms(frame_h, frame_w, sizes)
    if rooms is None:
        return None

    frame = np.full((frame_h, frame_w), WALL, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["rect_local"]
        frame[r0 : r1 + 1, c0 : c1 + 1] = BG

    max_active = room_count - 1
    min_active = 2 if room_count >= 4 else 1
    active_count = random.randint(min_active, max_active)
    active_indices = set(random.sample(range(room_count), active_count))
    for idx in active_indices:
        path = _corridor_path(frame, rooms, idx)
        if path is None:
            return None
        for r, c in path:
            frame[r, c] = BG
        rooms[idx]["active"] = True
    for idx in range(room_count):
        rooms[idx].setdefault("active", False)

    colors = random.sample(SPRITE_COLORS, room_count)
    for room, color in zip(random.sample(rooms, len(rooms)), colors):
        h, w = room["size"]
        mask = _random_sprite_mask(h, w)
        room["color"] = color
        room["sprite"] = np.where(mask, color, BG)

    grid_h = min(30, frame_h + random.randint(8, 10))
    grid_w = min(30, frame_w + random.randint(8, 10))
    if grid_h < frame_h + 6 or grid_w < frame_w + 6:
        return None
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_r0 = random.randint(3, grid_h - frame_h - 3)
    frame_c0 = random.randint(3, grid_w - frame_w - 3)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame
    template_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    marker = random.choice([(0, 0), (0, grid_w - 1), (grid_h - 1, 0), (grid_h - 1, grid_w - 1)])
    grid[marker] = MARKER

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_blocked(blocked, _expanded_rect(template_bbox, (grid_h, grid_w), pad=1))
    _mark_blocked(blocked, _expanded_rect((marker[0], marker[1], marker[0], marker[1]), (grid_h, grid_w), pad=1))

    for room in rooms:
        r0, c0, r1, c1 = room["rect_local"]
        room["rect_abs"] = (frame_r0 + r0, frame_c0 + c0, frame_r0 + r1, frame_c0 + c1)

    placement_order = random.sample(rooms, len(rooms))
    for room in placement_order:
        sprite = room["sprite"]
        loc = _place_sprite_bbox(blocked, grid.shape, sprite.shape)
        if loc is None:
            return None
        sr, sc = loc
        _paste(grid, sprite, sr, sc)
        room["sprite_bbox"] = (sr, sc, sr + sprite.shape[0] - 1, sc + sprite.shape[1] - 1)
        _mark_blocked(blocked, _expanded_rect(room["sprite_bbox"], (grid_h, grid_w), pad=1))

    case = {
        "grid": grid,
        "template": frame,
        "template_bbox": template_bbox,
        "marker": marker,
        "rooms": rooms,
    }
    return case


def _verify_case(case: dict) -> bool:
    grid = case["grid"]
    rooms = case["rooms"]
    sizes = [room["size"] for room in rooms]
    assert len(sizes) == len(set(sizes)), "room sizes must be unique"
    assert any(room["active"] for room in rooms), "need at least one active room"
    assert any(not room["active"] for room in rooms), "need at least one inactive room"

    start, reach = _flood_from_marker(grid)
    assert start == case["marker"]
    for room in rooms:
        r0, c0, r1, c1 = room["rect_abs"]
        reached = bool(reach[r0 : r1 + 1, c0 : c1 + 1].any())
        assert reached == room["active"], "reachability must match room activity"

    sprites = _find_sprites(grid)
    sprite_sizes = sorted(sprite["size"] for sprite in sprites)
    assert len(sprites) == len(rooms), "sprite count must equal cavity count"
    assert sprite_sizes == sorted(sizes), "sprite bbox sizes must exactly match room sizes"

    detected = sorted(_detect_cavities(case["template"], set(sizes)))
    expected_rects = sorted(room["rect_local"] for room in rooms)
    assert detected == expected_rects, "generator must produce solver-detectable cavities"

    expected = _expected_output(case)
    assert expected.shape == case["template"].shape

    all_used = case["template"].copy()
    for room in rooms:
        r0, c0, _r1, _c1 = room["rect_local"]
        _paste(all_used, room["sprite"], r0, c0)
    assert not np.array_equal(expected, all_used), "flood gating must change the matching pool"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the semantic construction"
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        try:
            if _verify_case(case):
                return case["grid"]
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid reachability-gated cavity puzzle")
