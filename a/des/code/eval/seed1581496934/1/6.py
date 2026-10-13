from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
COUNT = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_NAMES = tuple(DIRS)

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
ROOM_H_RANGE = (5, 8)
ROOM_W_RANGE = (5, 9)
MAX_COUNT_DOTS = 6
PLACE_TRIES = 80
GEN_ATTEMPTS = 400

_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _gap_candidates(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _gap_side(gap: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    r, c = gap
    sides = []
    if r == r0:
        sides.append("up")
    if r == r1:
        sides.append("down")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    assert len(sides) == 1, "gap must lie on exactly one side"
    return sides[0]


def _ray_cells(canvas: np.ndarray, gap: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    h, w = canvas.shape
    dr, dc = DIRS[side]
    rr, cc = gap[0] + dr, gap[1] + dc
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(canvas[rr, cc]) != BG:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _build_frame_occupancy(rooms: list[dict], h: int, w: int) -> np.ndarray:
    occ = np.zeros((h, w), dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        gap = room["gap"]
        for cell in _border_coords(r0, c0, r1, c1):
            if cell != gap:
                occ[cell] = True
    return occ


def _free_len(room: dict, rooms: list[dict], h: int, w: int) -> int:
    canvas = _build_frame_occupancy(rooms, h, w).astype(int)
    return len(_ray_cells(canvas, room["gap"], room["side"], max(h, w)))


def _inside_any_box(cell: tuple[int, int], rooms: list[dict]) -> bool:
    r, c = cell
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit one outward ray per holed grey room.

    - The gap side gives the ray direction.
    - The interior maroon pixel count gives the ray length.
    - The interior singleton seed gives the ray color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rooms = []
    for comp in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        box = (r0, c0, r1, c1)
        border = set(_border_coords(r0, c0, r1, c1))
        comp_set = set(comp)
        assert comp_set <= border, "grey component must stay on a rectangle border"
        missing = [cell for cell in border if cell not in comp_set]
        assert len(missing) == 1, "each room must have exactly one border gap"
        gap = missing[0]
        side = _gap_side(gap, box)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (inner != BG) & (inner != FRAME) & (inner != COUNT)
        seed_coords = np.argwhere(seed_mask)
        assert seed_coords.shape[0] == 1, "expected exactly one seed pixel per room"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(inner[sr, sc])
        count = int(np.count_nonzero(inner == COUNT))
        rooms.append((r0, c0, gap, side, seed_color, count))

    rooms.sort(key=lambda item: (item[0], item[1]))
    for _r0, _c0, gap, side, seed_color, count in rooms:
        for rr, cc in _ray_cells(out, gap, side, count):
            out[rr, cc] = seed_color
    return out


def _place_rooms(rng: random.Random, h: int, w: int) -> list[dict] | None:
    area = h * w
    if area < 270:
        room_count = rng.choice((2, 3))
    else:
        room_count = rng.choice((2, 3, 4))

    rooms: list[dict] = []
    seed_colors = rng.sample(SEED_COLORS, room_count)
    for idx in range(room_count):
        for _ in range(PLACE_TRIES):
            rh = rng.randint(*ROOM_H_RANGE)
            rw = rng.randint(*ROOM_W_RANGE)
            max_dots = min(MAX_COUNT_DOTS, (rh - 2) * (rw - 2) - 1)
            if max_dots < 2:
                continue
            if h - rh - 1 < 1 or w - rw - 1 < 1:
                continue
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_overlap(_expand(box, 1), _expand(other["bbox"], 1)) for other in rooms):
                continue
            rooms.append({
                "bbox": box,
                "seed_color": seed_colors[idx],
                "max_dots": max_dots,
            })
            break
        else:
            return None
    return rooms


def _assign_sides_and_gaps(rng: random.Random, rooms: list[dict]) -> None:
    pool = list(DIR_NAMES)
    rng.shuffle(pool)
    for idx, room in enumerate(rooms):
        if idx < len(pool) and rng.random() < 0.8:
            side = pool[idx]
        else:
            side = rng.choice(DIR_NAMES)
        room["side"] = side
        room["gap"] = rng.choice(_gap_candidates(room["bbox"], side))


def _force_blockable_choice(rng: random.Random, rooms: list[dict], idx: int, h: int, w: int) -> bool:
    options: list[tuple[int, str, tuple[int, int]]] = []
    room = rooms[idx]
    original_side = room["side"]
    original_gap = room["gap"]
    for side in DIR_NAMES:
        for gap in _gap_candidates(room["bbox"], side):
            room["side"] = side
            room["gap"] = gap
            free = _free_len(room, rooms, h, w)
            if free < room["max_dots"]:
                options.append((free, side, gap))
    room["side"] = original_side
    room["gap"] = original_gap
    if not options:
        return False
    best_free = min(free for free, _side, _gap in options)
    best = [(side, gap) for free, side, gap in options if free == best_free]
    room["side"], room["gap"] = rng.choice(best)
    return True


def _populate_room(rng: random.Random, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    rng.shuffle(cells)
    room["seed"] = cells[0]
    room["dots"] = cells[1 : 1 + room["count"]]


def _render_input(h: int, w: int, rooms: list[dict]) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        for cell in _border_coords(r0, c0, r1, c1):
            g[cell] = FRAME
        g[room["gap"]] = BG
        sr, sc = room["seed"]
        g[sr, sc] = room["seed_color"]
        for dr, dc in room["dots"]:
            g[dr, dc] = COUNT
    return g


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    rooms = _place_rooms(rng, h, w)
    if rooms is None:
        return None

    _assign_sides_and_gaps(rng, rooms)
    want_blocked = rng.random() < 0.5
    blocked_idx = rng.randrange(len(rooms)) if want_blocked else None
    if blocked_idx is not None and not _force_blockable_choice(rng, rooms, blocked_idx, h, w):
        return None

    if len({room["side"] for room in rooms}) < 2:
        return None

    for idx, room in enumerate(rooms):
        free = _free_len(room, rooms, h, w)
        if free < 1:
            return None
        room["blocked"] = idx == blocked_idx
        if room["blocked"]:
            if free >= room["max_dots"]:
                return None
            room["count"] = rng.randint(free + 1, room["max_dots"])
        else:
            upper = min(free, room["max_dots"])
            if upper < 1:
                return None
            room["count"] = rng.randint(1, upper)
        _populate_room(rng, room)

    if len({room["count"] for room in rooms}) < 2:
        return None

    g = _render_input(h, w, rooms)
    expected = g.copy()
    total_painted = 0
    ray_lengths: list[int] = []
    for room in sorted(rooms, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        cells = _ray_cells(expected, room["gap"], room["side"], room["count"])
        if room["blocked"]:
            if len(cells) >= room["count"]:
                return None
        elif len(cells) != room["count"]:
            return None
        if any(_inside_any_box(cell, rooms) for cell in cells):
            return None
        for rr, cc in cells:
            expected[rr, cc] = room["seed_color"]
        total_painted += len(cells)
        ray_lengths.append(len(cells))

    if total_painted < len(rooms) or max(ray_lengths) < 2:
        return None
    if np.array_equal(g, expected):
        return None
    if not np.array_equal(main(g), expected):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
