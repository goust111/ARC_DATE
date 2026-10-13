import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
WALL = 5
PATTERN = 4
PORTAL_COLORS = (1, 2, 3, 6, 7, 8, 9)
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_IDX = {side: idx for idx, side in enumerate(SIDE_ORDER)}

GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
ROOM_GAP = 1
GEN_MAX_ATTEMPTS = 400


def _rot_cw(coord, turns):
    r, c = coord
    turns %= 4
    if turns == 0:
        return int(r), int(c)
    if turns == 1:
        return int(c), int(-r)
    if turns == 2:
        return int(-r), int(-c)
    return int(-c), int(r)


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _normalize_shape(coords):
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate_many(coords, turns):
    return [_rot_cw(coord, turns) for coord in coords]


def _component_to_room(grid, pts, bg):
    if len(pts) < 8:
        return None

    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    comp = set(pts)
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    if comp != border:
        return None

    colors = Counter(int(grid[r, c]) for r, c in pts)
    wall_color, _ = colors.most_common(1)[0]
    portal_cells = [(r, c) for r, c in pts if int(grid[r, c]) != wall_color]
    if len(portal_cells) != 1:
        return None

    pr, pc = portal_cells[0]
    if (pr in (r0, r1)) and (pc in (c0, c1)):
        return None
    if pr == r0:
        side = "top"
    elif pc == c1:
        side = "right"
    elif pr == r1:
        side = "bottom"
    elif pc == c0:
        side = "left"
    else:
        return None

    inner_fg = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                inner_fg.append((r, c))

    pattern = []
    if inner_fg:
        sub = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
        for r, c in inner_fg:
            sub[r - r0 - 1, c - c0 - 1] = True
        inner_comps = _components(sub)
        assert len(inner_comps) == 1, "room pattern must be one 4-connected component"
        pattern = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in inner_comps[0]]

    return {
        "bbox": (r0, c0, r1, c1),
        "portal": (pr, pc),
        "portal_color": int(grid[pr, pc]),
        "side": side,
        "side_idx": SIDE_TO_IDX[side],
        "pattern": pattern,
    }


def _extract_rooms(grid, bg):
    rooms = []
    for pts in _components(grid != bg):
        room = _component_to_room(grid, pts, bg)
        if room is not None:
            rooms.append(room)
    assert rooms, "no valid rooms found"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rooms by portal color, rotate the source pattern by the portal-side
    difference, and paint it into the paired empty room.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(Counter(g.flat).most_common(1)[0][0])
    rooms = _extract_rooms(g, bg)

    by_color = defaultdict(list)
    for room in rooms:
        by_color[room["portal_color"]].append(room)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, f"portal color {color} must appear in exactly two rooms"
        sources = [room for room in pair if room["pattern"]]
        targets = [room for room in pair if not room["pattern"]]
        assert len(sources) == 1 and len(targets) == 1, "each portal pair needs one source room and one empty room"

        src = sources[0]
        dst = targets[0]
        turns = (dst["side_idx"] - src["side_idx"]) % 4
        spr, spc = src["portal"]
        dpr, dpc = dst["portal"]
        dr0, dc0, dr1, dc1 = dst["bbox"]

        for r, c in src["pattern"]:
            off_r, off_c = _rot_cw((r - spr, c - spc), turns)
            rr = dpr + off_r
            cc = dpc + off_c
            assert dr0 < rr < dr1 and dc0 < cc < dc1, "teleported pattern must stay inside the target room"
            assert out[rr, cc] == bg, "teleported pattern cannot overwrite walls or existing content"
            out[rr, cc] = g[r, c]

    return out


def _build_top_room(inner_h, inner_w, portal_col):
    portal = (0, portal_col)
    wall_rel = []
    interior_rel = []
    for r in range(inner_h + 2):
        for c in range(inner_w + 2):
            if r in (0, inner_h + 1) or c in (0, inner_w + 1):
                if (r, c) != portal:
                    wall_rel.append((r - portal[0], c - portal[1]))
            else:
                interior_rel.append((r - portal[0], c - portal[1]))
    return {"wall_rel": wall_rel, "interior_rel": interior_rel}


def _normalize_room(wall_rel, interior_rel, side_idx):
    all_coords = list(wall_rel) + list(interior_rel) + [(0, 0)]
    min_r, min_c, max_r, max_c = _bbox(all_coords)
    shift_r = -min_r
    shift_c = -min_c
    wall = tuple(sorted((r + shift_r, c + shift_c) for r, c in wall_rel))
    interior = tuple(sorted((r + shift_r, c + shift_c) for r, c in interior_rel))
    portal = (shift_r, shift_c)
    return {
        "wall": wall,
        "interior": interior,
        "portal": portal,
        "shape": (max_r - min_r + 1, max_c - min_c + 1),
        "side_idx": side_idx,
        "side": SIDE_ORDER[side_idx],
        "shift": (shift_r, shift_c),
    }


def _orient_room(base_room, side_idx):
    wall_rel = _rotate_many(base_room["wall_rel"], side_idx)
    interior_rel = _rotate_many(base_room["interior_rel"], side_idx)
    return _normalize_room(wall_rel, interior_rel, side_idx)


def _pattern_to_local(base_pattern, side_idx, shift):
    sr, sc = shift
    return tuple(sorted((r + sr, c + sc) for r, c in _rotate_many(base_pattern, side_idx)))


def _random_connected_pattern(interior_rel, turns):
    cells = set(interior_rel)
    max_size = min(4, len(cells))
    min_size = 3 if turns == 2 and max_size >= 3 else 2
    sizes = list(range(min_size, max_size + 1))

    for strict in (True, False):
        for _ in range(200):
            size = random.choice(sizes)
            start = random.choice(tuple(cells))
            pattern = {start}
            while len(pattern) < size:
                frontier = []
                for r, c in pattern:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nxt = (r + dr, c + dc)
                        if nxt in cells and nxt not in pattern:
                            frontier.append(nxt)
                if not frontier:
                    break
                pattern.add(random.choice(frontier))
            if len(pattern) != size:
                continue
            if strict and turns:
                if _normalize_shape(pattern) == _normalize_shape(_rotate_many(pattern, turns)):
                    continue
            return tuple(sorted(pattern))
    return None


def _boxes_separated(box_a, box_b, gap):
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _place_rooms(instances):
    total_area = sum(inst["spec"]["shape"][0] * inst["spec"]["shape"][1] for inst in instances)
    max_h = max(inst["spec"]["shape"][0] for inst in instances)
    max_w = max(inst["spec"]["shape"][1] for inst in instances)
    min_h = max(GRID_MIN_SIDE, max_h + 2)
    min_w = max(GRID_MIN_SIDE, max_w + 2)

    for _ in range(120):
        h = random.randint(min_h, GRID_MAX_SIDE)
        w = random.randint(min_w, GRID_MAX_SIDE)
        if h * w < total_area * 3:
            continue

        order = list(range(len(instances)))
        random.shuffle(order)
        placements = {}
        boxes = []
        ok = True

        for idx in order:
            room_h, room_w = instances[idx]["spec"]["shape"]
            if h - room_h - 1 < 1 or w - room_w - 1 < 1:
                ok = False
                break

            placed = False
            for _ in range(300):
                top = random.randint(1, h - room_h - 1)
                left = random.randint(1, w - room_w - 1)
                box = (top, left, top + room_h - 1, left + room_w - 1)
                if all(_boxes_separated(box, other, ROOM_GAP) for other in boxes):
                    placements[idx] = (top, left)
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok:
            return h, w, placements
    return None


def _draw_instance(grid, instance, top, left):
    spec = instance["spec"]
    for r, c in spec["wall"]:
        grid[top + r, left + c] = WALL
    pr, pc = spec["portal"]
    grid[top + pr, left + pc] = instance["portal_color"]
    for r, c in instance["pattern_local"]:
        grid[top + r, left + c] = PATTERN


def _gen_attempt():
    pair_count = random.choice((1, 2, 3))
    portal_colors = random.sample(PORTAL_COLORS, pair_count)
    instances = []
    expected_pattern_pixels = 0

    for portal_color in portal_colors:
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        portal_col = random.randint(1, inner_w)
        src_side = random.randrange(4)
        turns = random.randrange(4)

        base_room = _build_top_room(inner_h, inner_w, portal_col)
        base_pattern = _random_connected_pattern(base_room["interior_rel"], turns)
        if base_pattern is None:
            return None

        src_spec = _orient_room(base_room, src_side)
        dst_spec = _orient_room(base_room, (src_side + turns) % 4)
        src_pattern_local = _pattern_to_local(base_pattern, src_side, src_spec["shift"])

        expected_pattern_pixels += len(src_pattern_local)
        instances.append(
            {
                "spec": src_spec,
                "portal_color": portal_color,
                "pattern_local": src_pattern_local,
            }
        )
        instances.append(
            {
                "spec": dst_spec,
                "portal_color": portal_color,
                "pattern_local": tuple(),
            }
        )

    placement = _place_rooms(instances)
    if placement is None:
        return None

    h, w, placements = placement
    grid = np.full((h, w), BG, dtype=int)
    for idx, instance in enumerate(instances):
        top, left = placements[idx]
        _draw_instance(grid, instance, top, left)

    rooms = _extract_rooms(grid, BG)
    if len(rooms) != 2 * pair_count:
        return None

    out = main(grid)
    if np.count_nonzero(grid == PATTERN) != expected_pattern_pixels:
        return None
    if np.count_nonzero(out == PATTERN) != 2 * expected_pattern_pixels:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-room puzzle")
