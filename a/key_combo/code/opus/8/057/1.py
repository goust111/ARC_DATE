from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PARTICLE_COLORS = (BLUE, GREEN, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _marker_side(grid: np.ndarray) -> str:
    sides = []
    if np.all(grid[0, :] == GRAY):
        sides.append("top")
    if np.all(grid[-1, :] == GRAY):
        sides.append("bottom")
    if np.all(grid[:, 0] == GRAY):
        sides.append("left")
    if np.all(grid[:, -1] == GRAY):
        sides.append("right")
    assert len(sides) == 1, "expected exactly one full gray edge marker"
    return sides[0]


def _gravity_delta(side: str) -> tuple[int, int]:
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    if side == "right":
        return 0, 1
    raise AssertionError(side)


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid != RED, dtype=bool)


def _exterior_reachable(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and passable[r, c] and not seen[r, c]:
            seen[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            add(rr, cc)

    return seen


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _particle_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, PARTICLE_COLORS)


def _particle_sort_key(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    dr, dc = delta
    if dr == 1:
        return (-r, c)
    if dr == -1:
        return (r, c)
    if dc == 1:
        return (-c, r)
    return (c, r)


def _settle_region(
    out: np.ndarray,
    grid: np.ndarray,
    region: list[tuple[int, int]],
    delta: tuple[int, int],
) -> None:
    region_cells = set(region)
    particles = [
        (r, c, int(grid[r, c]))
        for r, c in region
        if int(grid[r, c]) in PARTICLE_COLORS
    ]
    particles.sort(key=lambda item: _particle_sort_key((item[0], item[1]), delta))

    occupied: set[tuple[int, int]] = set()
    dr, dc = delta
    for r, c, color in particles:
        rr, cc = r, c
        while (rr + dr, cc + dc) in region_cells and (rr + dr, cc + dc) not in occupied:
            rr += dr
            cc += dc
        occupied.add((rr, cc))
        out[rr, cc] = color


def _enclosed_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    passable = _passable_mask(grid)
    exterior = _exterior_reachable(passable)
    enclosed = passable & ~exterior
    return _mask_components(enclosed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only particles trapped inside red enclosures, then slide the survivors in the gray-marked gravity direction.

    Input structure:
    - The grid has black background, red one-cell walls, isolated colored particles, and one full gray marker line on exactly one outer edge.
    - Some red structures are fully closed and create enclosed passable regions; others have a gap and stay connected to the outside.
    - Colored particles may appear inside either closed or open regions.

    Rule:
    - Treat only red cells as walls. Flood-fill all non-red cells reachable from the grid boundary; particles in those open regions are erased.
    - Particles inside non-boundary-reachable regions survive.
    - The full gray edge marker gives gravity directly: top means up, bottom means down, left means left, right means right.
    - Within each enclosed region, slide surviving particles one by one along that direction until blocked by a red wall, the region boundary, or an already settled particle.
    - Remove the gray marker line in the output and keep red walls unchanged.

    Key insight:
    - Survival is decided by topology of red-enclosed space, while final position is decided by a second independent rule: gravity from the gray boundary marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side = _marker_side(g)
    delta = _gravity_delta(side)
    enclosed_regions = _enclosed_components(g)

    out = g.copy()
    out[out == GRAY] = BG
    out[_particle_mask(out)] = BG
    for region in enclosed_regions:
        _settle_region(out, g, region, delta)
    return out


def _room_box(room: dict) -> tuple[int, int, int, int]:
    return room["r0"], room["c0"], room["r0"] + room["h"] - 1, room["c0"] + room["w"] - 1


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _make_room(rng: random.Random, kind: str, active: bool) -> dict:
    room = {"kind": kind, "active": active, "h": 4, "w": 4}
    _assign_open_gap(rng, room)
    return room


def _assign_open_gap(rng: random.Random, room: dict) -> None:
    if room["kind"] != "open":
        return
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        gap_len = 2 if room["w"] >= 6 and rng.random() < 0.5 else 1
        start = rng.randint(1, room["w"] - 1 - gap_len)
    else:
        gap_len = 2 if room["h"] >= 6 and rng.random() < 0.5 else 1
        start = rng.randint(1, room["h"] - 1 - gap_len)
    room["gap"] = (side, start, start + gap_len - 1)


def _make_sized_room(rng: random.Random, kind: str, active: bool, h: int, w: int) -> dict:
    room = {"kind": kind, "active": active, "h": h, "w": w}
    _assign_open_gap(rng, room)
    return room


def _place_rooms(rng: random.Random, h: int, w: int, rooms: list[dict]) -> bool:
    placed: list[dict] = []
    order = sorted(rooms, key=lambda room: room["h"] * room["w"], reverse=True)
    for room in order:
        if room["h"] > h - 2 or room["w"] > w - 2:
            return False
        options = []
        for r0 in range(1, h - room["h"]):
            for c0 in range(1, w - room["w"]):
                candidate = (r0, c0, r0 + room["h"] - 1, c0 + room["w"] - 1)
                if all(not _boxes_too_close(candidate, _room_box(other)) for other in placed):
                    options.append((r0, c0))
        if not options:
            return False
        r0, c0 = rng.choice(options)
        room["r0"] = r0
        room["c0"] = c0
        placed.append(room)
    return True


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = _room_box(room)
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    if room["kind"] == "open":
        side, start, end = room["gap"]
        if side == "top":
            grid[r0, c0 + start : c0 + end + 1] = BG
        elif side == "bottom":
            grid[r1, c0 + start : c0 + end + 1] = BG
        elif side == "left":
            grid[r0 + start : r0 + end + 1, c0] = BG
        else:
            grid[r0 + start : r0 + end + 1, c1] = BG


def _interior_cells(room: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _room_box(room)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _pick_non_adjacent_cells(
    rng: random.Random,
    cells: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    cells = cells[:]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _active_positions(room: dict, side: str) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = _room_box(room)
    inner_h = room["h"] - 2
    inner_w = room["w"] - 2
    if side in ("top", "bottom"):
        assert inner_h >= 4
        col = c0 + 1 + inner_w // 2
        if side == "bottom":
            rows = [r0 + 1, r0 + 2, r0 + inner_h]
        else:
            rows = [r0 + inner_h, r0 + inner_h - 1, r0 + 1]
        return [(row, col) for row in rows]
    assert inner_w >= 4
    row = r0 + 1 + inner_h // 2
    if side == "right":
        cols = [c0 + 1, c0 + 2, c0 + inner_w]
    else:
        cols = [c0 + inner_w, c0 + inner_w - 1, c0 + 1]
    return [(row, col) for col in cols]


def _paint_particles(
    rng: random.Random,
    grid: np.ndarray,
    rooms: list[dict],
    side: str,
) -> bool:
    closed_rooms = [room for room in rooms if room["kind"] == "closed"]
    open_rooms = [room for room in rooms if room["kind"] == "open"]
    active_room = next(room for room in closed_rooms if room["active"])

    for cell in _active_positions(active_room, side):
        grid[cell] = rng.choice(PARTICLE_COLORS)

    for room in rooms:
        cells = _interior_cells(room)
        target = 1
        if room is active_room:
            continue
        if room["kind"] == "closed" and len(cells) >= 6 and rng.random() < 0.6:
            target = 2
        if room["kind"] == "open" and len(cells) >= 6 and rng.random() < 0.35:
            target = 2
        picked = _pick_non_adjacent_cells(rng, [cell for cell in cells if grid[cell] == BG], target)
        if picked is None:
            return False
        for cell in picked:
            grid[cell] = rng.choice(PARTICLE_COLORS)

    return len(closed_rooms) >= 2 and len(open_rooms) >= 2


def _place_marker(grid: np.ndarray, side: str) -> None:
    if side == "top":
        grid[0, :] = GRAY
    elif side == "bottom":
        grid[-1, :] = GRAY
    elif side == "left":
        grid[:, 0] = GRAY
    else:
        grid[:, -1] = GRAY


def _surviving_input_positions(grid: np.ndarray) -> set[tuple[int, int]]:
    enclosed = np.zeros_like(grid, dtype=bool)
    for region in _enclosed_components(grid):
        for r, c in region:
            enclosed[r, c] = True
    return {
        (r, c)
        for r, c in np.argwhere(enclosed & _particle_mask(grid))
    }


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    enclosed_regions = _enclosed_components(grid)
    enclosed_with_particles = 0
    enclosed_particle_total = 0
    for region in enclosed_regions:
        has_particle = any(int(grid[r, c]) in PARTICLE_COLORS for r, c in region)
        enclosed_with_particles += int(has_particle)
        enclosed_particle_total += sum(int(grid[r, c]) in PARTICLE_COLORS for r, c in region)

    total_particles = int(np.count_nonzero(_particle_mask(grid)))
    out_particles = int(np.count_nonzero(_particle_mask(out)))
    if enclosed_with_particles < 2:
        return False
    if total_particles - enclosed_particle_total < 2:
        return False
    if out_particles != enclosed_particle_total:
        return False
    if np.any(out == GRAY):
        return False

    before = _surviving_input_positions(grid)
    after = {tuple(map(int, rc)) for rc in np.argwhere(_particle_mask(out))}
    if before == after:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(14, 16)
    w = rng.randint(14, 16)
    side = rng.choice(["top", "bottom", "left", "right"])

    mid_r = h // 2
    mid_c = w // 2
    slots = {
        "tl": (1, mid_r - 1, 1, mid_c - 1),
        "tr": (1, mid_r - 1, mid_c + 1, w - 2),
        "bl": (mid_r + 1, h - 2, 1, mid_c - 1),
        "br": (mid_r + 1, h - 2, mid_c + 1, w - 2),
    }

    def slot_shape(name: str) -> tuple[int, int]:
        r0, r1, c0, c1 = slots[name]
        return r1 - r0 + 1, c1 - c0 + 1

    if side in ("top", "bottom"):
        active_candidates = [name for name in slots if slot_shape(name)[0] >= 6]
    else:
        active_candidates = [name for name in slots if slot_shape(name)[1] >= 6]
    if not active_candidates:
        return None

    active_slot = rng.choice(active_candidates)
    remaining_slots = [name for name in slots if name != active_slot]
    closed_slot = rng.choice(remaining_slots)

    room_specs = {
        active_slot: ("closed", True),
        closed_slot: ("closed", False),
    }
    for name in remaining_slots:
        if name not in room_specs:
            room_specs[name] = ("open", False)

    rooms = []
    for name in ["tl", "tr", "bl", "br"]:
        kind, active = room_specs[name]
        slot_h, slot_w = slot_shape(name)
        if active and side in ("top", "bottom"):
            room_h = rng.randint(6, slot_h)
            room_w = rng.randint(4, slot_w)
        elif active:
            room_h = rng.randint(4, slot_h)
            room_w = rng.randint(6, slot_w)
        else:
            room_h = rng.randint(4, slot_h)
            room_w = rng.randint(4, slot_w)
        room = _make_sized_room(rng, kind, active, room_h, room_w)
        slot_r0, slot_r1, slot_c0, slot_c1 = slots[name]
        room["r0"] = rng.randint(slot_r0, slot_r1 - room_h + 1)
        room["c0"] = rng.randint(slot_c0, slot_c1 - room_w + 1)
        rooms.append(room)

    grid = np.zeros((h, w), dtype=int)
    for room in rooms:
        _draw_room(grid, room)
    if not _paint_particles(rng, grid, rooms, side):
        return None
    _place_marker(grid, side)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
