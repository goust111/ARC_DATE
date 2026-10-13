from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

BG = 0
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
ROOM_COUNT_BY_AREA = [
    (360, (1, 2)),
    (520, (2, 3)),
    (10**9, (2, 4)),
]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(values: np.ndarray) -> int:
    return Counter(map(int, values.reshape(-1))).most_common(1)[0][0]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _extract_rooms(g: np.ndarray) -> tuple[int, int, list[dict], np.ndarray]:
    bg = _most_common_color(g)
    non_bg = [int(v) for v in g.reshape(-1) if int(v) != bg]
    assert non_bg, "grid must contain non-background structure"
    wall = Counter(non_bg).most_common(1)[0][0]

    room_mask = np.zeros_like(g, dtype=bool)
    rooms: list[dict] = []
    for comp in _components(g != wall, connectivity=4):
        if _touches_border(comp, *g.shape):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        if len(comp) != room_h * room_w:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted({int(v) for v in patch.reshape(-1) if int(v) != bg})
        assert len(colors) == 1, "each room interior must contain exactly one stamp color"
        color = colors[0]

        stamp_cells = np.argwhere(patch == color)
        assert len(stamp_cells) >= 2, "stamp must contain at least two cells"
        sr0, sc0 = map(int, stamp_cells.min(axis=0))
        sr1, sc1 = map(int, stamp_cells.max(axis=0))
        stamp_mask = patch[sr0 : sr1 + 1, sc0 : sc1 + 1] == color

        room_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "stamp_mask": stamp_mask,
            }
        )

    assert rooms, "at least one enclosed room must be present"
    return bg, wall, rooms, room_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find enclosed rooms, count same-colored corridor bars, then tile each room's stamp pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall, rooms, room_mask = _extract_rooms(g)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        color = int(room["color"])
        stamp_mask = np.asarray(room["stamp_mask"], dtype=bool)
        ph, pw = map(int, stamp_mask.shape)
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1

        corridor_bars = (g == color) & (~room_mask)
        count = len(_components(corridor_bars, connectivity=4))
        assert count >= 1, "each room needs at least one same-colored corridor bar"

        out[corridor_bars] = bg
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

        tiles_per_row = room_w // pw
        tiles_per_col = room_h // ph
        assert tiles_per_row > 0 and tiles_per_col > 0
        assert tiles_per_row * tiles_per_col >= count, "room is too small for the requested tiling count"

        for idx in range(count):
            tr = idx // tiles_per_row
            tc = idx % tiles_per_row
            ar = r0 + tr * ph
            ac = c0 + tc * pw
            patch = out[ar : ar + ph, ac : ac + pw]
            patch[stamp_mask] = color

    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _random_connected_stamp(rng: random.Random) -> np.ndarray:
    dims = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    box_h, box_w = rng.choice(dims)
    max_cells = min(6, box_h * box_w)
    target = rng.randint(2, max_cells)

    cells = {(rng.randrange(box_h), rng.randrange(box_w))}
    frontier = set(cells)
    while len(cells) < target:
        cr, cc = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in DIR4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if nbrs:
            pick = rng.choice(nbrs)
            cells.add(pick)
            frontier.add(pick)
        else:
            frontier.remove((cr, cc))
            if not frontier:
                frontier = set(cells)

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _room_count_for_area(area: int, rng: random.Random) -> int:
    for limit, bounds in ROOM_COUNT_BY_AREA:
        if area <= limit:
            lo, hi = bounds
            return rng.randint(lo, hi)
    return 1


def _build_room_specs(rng: random.Random, count: int) -> list[dict]:
    specs: list[dict] = []
    for _ in range(count):
        stamp_mask = _random_connected_stamp(rng)
        ph, pw = map(int, stamp_mask.shape)

        row_tiles = rng.randint(1, 3)
        col_tiles = rng.randint(1, 4)
        room_h = max(4, min(9, row_tiles * ph + rng.randint(0, max(0, ph - 1) + 1)))
        room_w = max(5, min(12, col_tiles * pw + rng.randint(0, max(0, pw - 1) + 1)))
        capacity = (room_h // ph) * (room_w // pw)
        if capacity < 1:
            return []

        specs.append(
            {
                "stamp_mask": stamp_mask,
                "room_h": room_h,
                "room_w": room_w,
                "outer_h": room_h + 2,
                "outer_w": room_w + 2,
                "capacity": capacity,
            }
        )
    return specs


def _place_rooms(rng: random.Random, h: int, w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda x: x["outer_h"] * x["outer_w"], reverse=True):
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        if h - oh - 2 < 2 or w - ow - 2 < 2:
            return None

        positions = [
            (r0, c0)
            for r0 in range(2, h - oh - 1)
            for c0 in range(2, w - ow - 1)
        ]
        rng.shuffle(positions)

        placed_here = False
        for r0, c0 in positions:
            reserve_box = (r0 - 1, c0 - 1, r0 + oh, c0 + ow)
            if any(_rectangles_overlap(reserve_box, other) for other in reserved):
                continue

            room = dict(spec)
            room["outer_bbox"] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            room["inner_bbox"] = (r0 + 1, c0 + 1, r0 + oh - 2, c0 + ow - 2)
            placed.append(room)
            reserved.append(reserve_box)
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _bar_candidates(room: dict) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = room["outer_bbox"]
    room_h = int(room["room_h"])
    room_w = int(room["room_w"])
    candidates: list[list[tuple[int, int]]] = []

    for length in range(2, min(4, room_w) + 1):
        for off in range(room_w - length + 1):
            start_c = c0 + 1 + off
            top = [(r0 - 1, start_c + d) for d in range(length)]
            bottom = [(r1 + 1, start_c + d) for d in range(length)]
            candidates.append(top)
            candidates.append(bottom)

    for length in range(2, min(4, room_h) + 1):
        for off in range(room_h - length + 1):
            start_r = r0 + 1 + off
            left = [(start_r + d, c0 - 1) for d in range(length)]
            right = [(start_r + d, c1 + 1) for d in range(length)]
            candidates.append(left)
            candidates.append(right)

    return candidates


def _segments_touch(seg_a: list[tuple[int, int]], seg_b: list[tuple[int, int]]) -> bool:
    pts_b = set(seg_b)
    for r, c in seg_a:
        for dr, dc in DIR8 + [(0, 0)]:
            if (r + dr, c + dc) in pts_b:
                return True
    return False


def _choose_bars(rng: random.Random, room: dict) -> tuple[int, list[list[tuple[int, int]]]] | None:
    candidates = _bar_candidates(room)
    if not candidates:
        return None

    best: list[list[tuple[int, int]]] = []
    for _ in range(40):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        chosen: list[list[tuple[int, int]]] = []
        for cand in shuffled:
            if any(_segments_touch(cand, prev) for prev in chosen):
                continue
            chosen.append(cand)
        if len(chosen) > len(best):
            best = chosen

    max_count = min(4, int(room["capacity"]), len(best))
    if max_count < 1:
        return None

    if max_count == 1:
        target = 1
    elif rng.random() < 0.5:
        target = 1
    else:
        target = rng.randint(2, max_count)

    for _ in range(80):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        chosen = []
        for cand in shuffled:
            if any(_segments_touch(cand, prev) for prev in chosen):
                continue
            chosen.append(cand)
            if len(chosen) == target:
                return target, chosen

    return None


def _paint_room_walls(grid: np.ndarray, room: dict, wall: int) -> None:
    r0, c0, r1, c1 = room["outer_bbox"]
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall


def _place_stamp(grid: np.ndarray, room: dict, color: int, rng: random.Random) -> None:
    stamp_mask = np.asarray(room["stamp_mask"], dtype=bool)
    ph, pw = map(int, stamp_mask.shape)
    r0, c0, r1, c1 = room["inner_bbox"]
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1

    max_or = min(max(0, room_h - ph), 1)
    max_oc = min(max(0, room_w - pw), 2)
    off_r = rng.randint(0, max_or) if max_or > 0 else 0
    off_c = rng.randint(0, max_oc) if max_oc > 0 else 0

    patch = grid[r0 + off_r : r0 + off_r + ph, c0 + off_c : c0 + off_c + pw]
    patch[stamp_mask] = color


def _validate_candidate(grid: np.ndarray, wall: int, room_count: int) -> bool:
    try:
        bg, parsed_wall, rooms, _room_mask = _extract_rooms(grid)
        if bg != BG or parsed_wall != wall or len(rooms) != room_count:
            return False
        counts = Counter(map(int, grid.reshape(-1)))
        wall_count = counts[wall]
        for color, count in counts.items():
            if color not in (bg, wall) and count >= wall_count:
                return False
        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not np.any(out != grid):
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    room_count = _room_count_for_area(h * w, rng)
    specs = _build_room_specs(rng, room_count)
    if len(specs) != room_count:
        return None

    rooms = _place_rooms(rng, h, w, specs)
    if rooms is None or len(rooms) != room_count:
        return None

    palette = [c for c in range(1, 10)]
    wall = rng.choice(palette)
    palette.remove(wall)
    if len(palette) < room_count:
        return None
    stamp_colors = rng.sample(palette, room_count)

    grid = np.full((h, w), BG, dtype=int)
    for room in rooms:
        _paint_room_walls(grid, room, wall)

    for room, color in zip(rooms, stamp_colors):
        room["color"] = color
        _place_stamp(grid, room, color, rng)

    for room in rooms:
        chosen = _choose_bars(rng, room)
        if chosen is None:
            return None
        _count, bars = chosen
        color = int(room["color"])
        for seg in bars:
            for r, c in seg:
                if grid[r, c] != BG:
                    return None
                grid[r, c] = color

    if not _validate_candidate(grid, wall, room_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid sample")
