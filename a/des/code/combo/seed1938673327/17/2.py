import random
from collections import deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
GRID_MIN = 22
GRID_MAX = 30
ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_COUNT_WEIGHTS = (2, 2, 3)
ROOM_PLACEMENT_ATTEMPTS = 200
OBJECT_PLACEMENT_ATTEMPTS = 200
MASK_ATTEMPTS = 200


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _get_wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert values.size > 0, "expected at least one non-background color"
    return int(values[np.argmax(counts)])


def _rect_ring(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[dict] = []
    for comp in _components_mask(g == bg):
        if _touches_edge(comp, h, w):
            continue
        ir0, ic0, ir1, ic1 = _bbox(comp)
        if (ir1 - ir0 + 1) * (ic1 - ic0 + 1) != len(comp):
            continue
        if ir0 == 0 or ic0 == 0 or ir1 == h - 1 or ic1 == w - 1:
            continue

        or0, oc0, or1, oc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        ring = _rect_ring(or0, oc0, or1, oc1)
        entrances = []
        valid = True
        for rr, cc in ring:
            val = int(g[rr, cc])
            if val == wall:
                continue
            if val == bg:
                valid = False
                break
            entrances.append((rr, cc, val))
        if not valid or len(entrances) != 1:
            continue

        er, ec, color = entrances[0]
        sides = []
        if er == or0:
            sides.append("top")
        if er == or1:
            sides.append("bottom")
        if ec == oc0:
            sides.append("left")
        if ec == oc1:
            sides.append("right")
        if len(sides) != 1:
            continue

        rooms.append(
            {
                "color": color,
                "side": sides[0],
                "entrance": (er, ec),
                "outer_bbox": (or0, oc0, or1, oc1),
                "interior_bbox": (ir0, ic0, ir1, ic1),
            }
        )

    assert rooms, "expected at least one enclosed room"
    colors = [room["color"] for room in rooms]
    assert len(colors) == len(set(colors)), "room entrance colors must be unique"
    return rooms


def _extract_object(grid: np.ndarray, color: int, entrance: tuple[int, int]) -> dict:
    comps = _components_mask(grid == color)
    matches = []
    entrance_seen = False
    for comp in comps:
        if entrance in comp:
            entrance_seen = True
            assert len(comp) == 1, "entrance pixel must be isolated from the outside object"
        else:
            matches.append(comp)
    assert entrance_seen, "matching entrance pixel must exist"
    assert len(matches) == 1, "each entrance color must match exactly one outside object"
    coords = matches[0]
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {"coords": coords, "bbox": (r0, c0, r1, c1), "mask": mask}


def _flip_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.flipud(mask)
    return np.fliplr(mask)


def _place_inside_room(mask: np.ndarray, room: dict) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = room["interior_bbox"]
    er, ec = room["entrance"]
    side = room["side"]
    obj_h, obj_w = mask.shape
    room_h = ir1 - ir0 + 1
    room_w = ic1 - ic0 + 1
    assert obj_h <= room_h and obj_w <= room_w, "object must fit inside the room"

    if side == "top":
        row0 = ir1 - obj_h + 1
        anchor = ec - ic0
        col0 = ic0 + max(0, min(room_w - obj_w, anchor - obj_w // 2))
    elif side == "bottom":
        row0 = ir0
        anchor = ec - ic0
        col0 = ic0 + max(0, min(room_w - obj_w, anchor - obj_w // 2))
    elif side == "left":
        col0 = ic1 - obj_w + 1
        anchor = er - ir0
        row0 = ir0 + max(0, min(room_h - obj_h, anchor - obj_h // 2))
    else:
        col0 = ic0
        anchor = er - ir0
        row0 = ir0 + max(0, min(room_h - obj_h, anchor - obj_h // 2))

    coords: list[tuple[int, int]] = []
    rows, cols = np.where(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist()):
        rr = row0 + dr
        cc = col0 + dc
        assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "placed object must stay in the interior"
        coords.append((rr, cc))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each outside object into the room whose entrance has the same color.

    Input structure:
    - The dominant color is the black-like background.
    - One other color forms several 1-pixel-thick rectangular room walls.
    - Each room has exactly one colored entrance pixel on its wall.
    - Outside the rooms, there is exactly one 4-connected object per entrance
      color.

    Rule:
    - Match each room with the outside object of the same color.
    - Flip the object according to the entrance side.
    - Move it to the opposite interior wall, aligned with the entrance axis.
    - Erase the original object and restore the entrance to the wall color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall = _get_wall_color(g, bg)
    rooms = _find_rooms(g, bg, wall)

    out = g.copy()
    for room in rooms:
        color = int(room["color"])
        obj = _extract_object(g, color, room["entrance"])
        flipped = _flip_mask(obj["mask"], room["side"])
        target_coords = _place_inside_room(flipped, room)

        for r, c in obj["coords"]:
            out[r, c] = bg
        er, ec = room["entrance"]
        out[er, ec] = wall
        for r, c in target_coords:
            assert out[r, c] == bg, "room interior must be empty before insertion"
            out[r, c] = color
    return out


def _choose_anchor(rng: random.Random, span: int, obj_span: int) -> int:
    lo = obj_span // 2
    hi = span - (obj_span + 1) // 2
    assert lo <= hi, "room span must fit the object span"
    return rng.randint(lo, hi)


def _random_mask(rng: random.Random) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        max_cells = h * w - 1
        if max_cells < 3:
            continue
        target = rng.randint(3, max_cells)
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier = set()
            rows, cols = np.where(mask)
            for r, c in zip(rows.tolist(), cols.tolist()):
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(tuple(frontier))
            mask[nr, nc] = True

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if np.array_equal(cropped, np.flipud(cropped)):
            continue
        if np.array_equal(cropped, np.fliplr(cropped)):
            continue
        return cropped
    raise RuntimeError("failed to sample a visibly asymmetric mask")


def _mask_variant(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    variant = mask.copy()
    if rng.random() < 0.5:
        variant = np.flipud(variant)
    if rng.random() < 0.5:
        variant = np.fliplr(variant)
    return variant


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    return not np.any(reserved[rr0:rr1, cc0:cc1])


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _sample_bbox(
    rng: random.Random, reserved: np.ndarray, box_h: int, box_w: int, attempts: int
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if box_h > h or box_w > w:
        return None
    for _ in range(attempts):
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        if _can_place(reserved, r0, c0, box_h, box_w):
            return r0, c0
    return None


def _draw_room(grid: np.ndarray, room: dict) -> None:
    or0, oc0, or1, oc1 = room["outer_bbox"]
    color = int(room["color"])
    for r, c in _rect_ring(or0, oc0, or1, oc1):
        grid[r, c] = WALL_COLOR
    er, ec = room["entrance"]
    grid[er, ec] = color


def _room_spec(rng: random.Random, color: int, side: str) -> dict:
    base_mask = _mask_variant(_random_mask(rng), rng)
    obj_h, obj_w = base_mask.shape

    if side in ("top", "bottom"):
        room_h = obj_h + rng.randint(2, 5)
        room_w = obj_w + rng.randint(0, 4)
        anchor = _choose_anchor(rng, room_w, obj_w)
        entrance_local = (0 if side == "top" else room_h + 1, anchor + 1)
    else:
        room_h = obj_h + rng.randint(0, 4)
        room_w = obj_w + rng.randint(2, 5)
        anchor = _choose_anchor(rng, room_h, obj_h)
        entrance_local = (anchor + 1, 0 if side == "left" else room_w + 1)

    return {
        "color": color,
        "side": side,
        "object_mask": base_mask,
        "room_h": room_h,
        "room_w": room_w,
        "outer_h": room_h + 2,
        "outer_w": room_w + 2,
        "entrance_local": entrance_local,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    room_count = rng.choices(ROOM_COUNT_CHOICES, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    colors = rng.sample(OBJECT_COLORS, room_count)
    directions = list(DIRECTIONS)
    rng.shuffle(directions)
    room_specs = [_room_spec(rng, color, side) for color, side in zip(colors, directions[:room_count])]

    placed_rooms: list[dict] = []
    for spec in room_specs:
        pos = _sample_bbox(rng, reserved, spec["outer_h"], spec["outer_w"], ROOM_PLACEMENT_ATTEMPTS)
        if pos is None:
            return None
        or0, oc0 = pos
        or1 = or0 + spec["outer_h"] - 1
        oc1 = oc0 + spec["outer_w"] - 1
        er = or0 + spec["entrance_local"][0]
        ec = oc0 + spec["entrance_local"][1]
        room = {
            "color": spec["color"],
            "side": spec["side"],
            "entrance": (er, ec),
            "outer_bbox": (or0, oc0, or1, oc1),
            "interior_bbox": (or0 + 1, oc0 + 1, or1 - 1, oc1 - 1),
            "object_mask": spec["object_mask"],
        }
        _draw_room(grid, room)
        _mark_reserved(reserved, or0, oc0, spec["outer_h"], spec["outer_w"])
        placed_rooms.append(room)

    for room in placed_rooms:
        mask = room["object_mask"]
        obj_h, obj_w = mask.shape
        pos = _sample_bbox(rng, reserved, obj_h, obj_w, OBJECT_PLACEMENT_ATTEMPTS)
        if pos is None:
            return None
        r0, c0 = pos
        rows, cols = np.where(mask)
        for dr, dc in zip(rows.tolist(), cols.tolist()):
            grid[r0 + dr, c0 + dc] = int(room["color"])
        _mark_reserved(reserved, r0, c0, obj_h, obj_w)

    if _get_bg(grid) != BG_COLOR:
        return None
    if _get_wall_color(grid, BG_COLOR) != WALL_COLOR:
        return None

    rooms = _find_rooms(grid, BG_COLOR, WALL_COLOR)
    if len(rooms) != room_count:
        return None
    for room in rooms:
        _extract_object(grid, int(room["color"]), room["entrance"])

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
