from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 5
COLOR_POOL = tuple(c for c in range(1, 10) if c != WALL)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in deltas:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _palette_component(grid: np.ndarray, boundary_markers: set[tuple[int, int]]) -> list[tuple[int, int]]:
    colored = (grid != BLACK) & (grid != WALL)
    palette_mask = colored.copy()
    for r, c in boundary_markers:
        palette_mask[r, c] = False

    candidates: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for comp in _components(palette_mask, connectivity=4):
        rows = {r for r, _ in comp}
        cols = sorted(c for _, c in comp)
        if len(rows) != 1:
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        row = next(iter(rows))
        if row not in (0, h - 1):
            continue
        if cols[0] != 0 and cols[-1] != w - 1:
            continue
        candidates.append(comp)

    assert candidates, "expected a corner palette row"
    return max(candidates, key=len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black room with the palette color indexed by how many wall markers surround it.

    Input structure:
    - Background is black.
    - Grey one-cell-thick walls enclose one or more black connected "rooms".
    - Some wall cells are replaced by non-black, non-grey marker colors.
    - A disconnected horizontal palette row of distinct colors sits in one grid corner.

    Rule:
    - Find every black 4-connected component that does not touch the grid border; each is a room.
    - For each room, inspect the non-black cells in its 8-neighborhood and count how many are not grey.
    - Read the corner palette from left to right, using 1-based positions.
    - Fill the room's black cells with the palette color at position N, where N is that room's marker count.

    Key insight:
    - Marker colors themselves are irrelevant; only the number of marked wall cells matters, and that count indexes the palette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rooms = [comp for comp in _components(g == BLACK, connectivity=4) if not _touches_border(comp, h, w)]
    room_info: list[tuple[list[tuple[int, int]], int]] = []
    boundary_markers: set[tuple[int, int]] = set()

    for room in rooms:
        room_set = set(room)
        boundary: set[tuple[int, int]] = set()
        for r, c in room:
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in room_set and g[nr, nc] != BLACK:
                        boundary.add((nr, nc))
        marker_count = sum(int(g[r, c] != WALL) for r, c in boundary)
        room_info.append((room, marker_count))
        for r, c in boundary:
            if g[r, c] not in (BLACK, WALL):
                boundary_markers.add((r, c))

    palette_cells = _palette_component(g, boundary_markers)
    palette_cells.sort(key=lambda rc: rc[1])
    palette = [int(g[r, c]) for r, c in palette_cells]

    out = g.copy()
    for room, marker_count in room_info:
        assert 1 <= marker_count <= len(palette), "room marker count must index the palette"
        fill = palette[marker_count - 1]
        for r, c in room:
            out[r, c] = fill
    return out


def _random_polyomino(rng: random.Random, h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_room_interior(rng: random.Random) -> np.ndarray | None:
    if rng.random() < 0.4:
        inner_h = rng.randint(3, 7)
        inner_w = rng.randint(3, 7)
        return np.ones((inner_h, inner_w), dtype=bool)

    coarse_h = rng.randint(2, 4)
    coarse_w = rng.randint(2, 4)
    max_area = coarse_h * coarse_w
    area = rng.randint(max(3, (max_area + 1) // 2), max_area)
    coarse = _random_polyomino(rng, coarse_h, coarse_w, area)
    if coarse is None:
        return None
    scale_h = rng.choice([2, 3])
    scale_w = rng.choice([2, 3])
    return np.kron(coarse, np.ones((scale_h, scale_w), dtype=bool))


def _room_patch(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    room = np.pad(interior.astype(bool), 1)
    wall = _dilate8(room) & ~room
    return room, wall


def _placeable(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    return not occupied[top - 1 : top + height + 1, left - 1 : left + width + 1].any()


def _palette_cells(h: int, w: int, length: int, corner: str) -> list[tuple[int, int]]:
    row = 0 if corner in ("tl", "tr") else h - 1
    if corner in ("tl", "bl"):
        cols = range(0, length)
    else:
        cols = range(w - length, w)
    return [(row, c) for c in cols]


def _reserve_palette_zone(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = max(0, min(rows) - 2)
    r1 = min(occupied.shape[0], max(rows) + 3)
    c0 = max(0, min(cols) - 1)
    c1 = min(occupied.shape[1], max(cols) + 2)
    occupied[r0:r1, c0:c1] = True


def _marker_slots(rng: random.Random, wall: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(wall)]
    rng.shuffle(coords)
    chosen: list[tuple[int, int]] = []
    for r, c in coords:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((r, c))
    return chosen if chosen else coords


def _choose_counts(rng: random.Random, capacities: list[int], palette_len: int) -> list[int] | None:
    limits = [min(cap, palette_len) for cap in capacities]
    if any(limit < 1 for limit in limits):
        return None
    counts = [rng.randint(1, limit) for limit in limits]

    if max(limits) >= 2 and max(counts) == 1:
        choices = [i for i, limit in enumerate(limits) if limit >= 2]
        idx = rng.choice(choices)
        counts[idx] = rng.randint(2, limits[idx])

    if len(counts) > 1 and len(set(counts)) == 1 and max(limits) >= 2:
        choices = [i for i, limit in enumerate(limits) if limit >= 2]
        idx = rng.choice(choices)
        options = [v for v in range(1, limits[idx] + 1) if v != counts[idx]]
        counts[idx] = rng.choice(options)

    if len(counts) > 1 and len(set(counts)) == 1:
        return None
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    palette_len = rng.randint(3, min(6, w))
    corner = rng.choice(["tl", "tr", "bl", "br"])

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    palette_cells = _palette_cells(h, w, palette_len, corner)
    _reserve_palette_zone(occupied, palette_cells)

    if h * w >= 650:
        max_rooms = 5
    elif h * w >= 450:
        max_rooms = 4
    else:
        max_rooms = 3
    target_rooms = rng.randint(2, max_rooms)

    rooms: list[dict[str, object]] = []
    for _ in range(target_rooms):
        placed = False
        for _ in range(120):
            interior = _make_room_interior(rng)
            if interior is None:
                continue
            room_mask, wall_mask = _room_patch(interior)
            patch_h, patch_w = room_mask.shape
            if patch_h + 2 >= h or patch_w + 2 >= w:
                continue
            top_min = 1
            top_max = h - patch_h - 1
            left_min = 1
            left_max = w - patch_w - 1
            if top_min > top_max or left_min > left_max:
                continue
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)
            if not _placeable(occupied, top, left, patch_h, patch_w):
                continue
            occupied[top - 1 : top + patch_h + 1, left - 1 : left + patch_w + 1] = True
            rooms.append(
                {
                    "top": top,
                    "left": left,
                    "room": room_mask,
                    "wall": wall_mask,
                    "slots": _marker_slots(rng, wall_mask),
                }
            )
            placed = True
            break
        if not placed:
            break

    if len(rooms) < 2:
        return None

    capacities = [len(room["slots"]) for room in rooms]
    counts = _choose_counts(rng, capacities, palette_len)
    if counts is None:
        return None

    palette_colors = rng.sample(COLOR_POOL, palette_len)
    for (r, c), color in zip(palette_cells, palette_colors):
        grid[r, c] = color

    for room in rooms:
        top = int(room["top"])
        left = int(room["left"])
        wall_mask = room["wall"]
        for r, c in np.argwhere(wall_mask):
            grid[top + int(r), left + int(c)] = WALL

    for room, count in zip(rooms, counts):
        slots = list(room["slots"])
        rng.shuffle(slots)
        if len(slots) < count:
            return None
        fill = palette_colors[count - 1]
        for r, c in slots[:count]:
            marker_choices = [color for color in COLOR_POOL if color != WALL]
            grid[int(room["top"]) + r, int(room["left"]) + c] = rng.choice(marker_choices)
        room["fill"] = fill

    expected = grid.copy()
    for room in rooms:
        top = int(room["top"])
        left = int(room["left"])
        room_mask = room["room"]
        fill = int(room["fill"])
        expected[top : top + room_mask.shape[0], left : left + room_mask.shape[1]][room_mask] = fill

    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
