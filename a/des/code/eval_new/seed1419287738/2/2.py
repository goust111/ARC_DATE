from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 8
WALL = 5
HIDDEN = 9
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7]
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _key_orientation(comp: dict) -> tuple[int, int] | None:
    if int(comp["area"]) != 3:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 2:
        return None

    cells = {(int(r), int(c)) for r, c in comp["coords"]}
    corner = None
    degree_count = Counter()
    for r, c in cells:
        deg = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            deg += int((r + dr, c + dc) in cells)
        degree_count[deg] += 1
        if deg == 2:
            corner = (int(r), int(c))
    if degree_count != Counter({1: 2, 2: 1}) or corner is None:
        return None

    cr, cc = corner
    dr = -1 if cr == r0 else 1
    dc = -1 if cc == c0 else 1
    return int(dr), int(dc)


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((int(r0), int(c)))
        cells.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.append((int(r), int(c0)))
        cells.append((int(r), int(c1)))
    return cells


def _validate_room(
    grid: np.ndarray,
    bg: int,
    door_pos: tuple[int, int],
    side: str,
    bbox: tuple[int, int, int, int],
) -> dict | None:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, door_pos)

    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        return None
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    if side == "top":
        if dr != r0 or dc != (c0 + c1) // 2 or ((c1 - c0 + 1) % 2) != 1:
            return None
        if dr == 0 or int(grid[dr - 1, dc]) != int(bg) or int(grid[dr + 1, dc]) == int(WALL):
            return None
    elif side == "bottom":
        if dr != r1 or dc != (c0 + c1) // 2 or ((c1 - c0 + 1) % 2) != 1:
            return None
        if dr == h - 1 or int(grid[dr + 1, dc]) != int(bg) or int(grid[dr - 1, dc]) == int(WALL):
            return None
    elif side == "left":
        if dc != c0 or dr != (r0 + r1) // 2 or ((r1 - r0 + 1) % 2) != 1:
            return None
        if dc == 0 or int(grid[dr, dc - 1]) != int(bg) or int(grid[dr, dc + 1]) == int(WALL):
            return None
    elif side == "right":
        if dc != c1 or dr != (r0 + r1) // 2 or ((r1 - r0 + 1) % 2) != 1:
            return None
        if dc == w - 1 or int(grid[dr, dc + 1]) != int(bg) or int(grid[dr, dc - 1]) == int(WALL):
            return None
    else:
        return None

    boundary = _boundary_cells(r0, c0, r1, c1)
    non_wall = [(r, c) for r, c in boundary if int(grid[r, c]) != int(WALL)]
    if non_wall != [door_pos]:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == int(WALL)) or not np.any(interior == int(HIDDEN)):
        return None

    return {
        "color": int(grid[dr, dc]),
        "door": (int(dr), int(dc)),
        "side": side,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _room_from_door(grid: np.ndarray, bg: int, door_pos: tuple[int, int]) -> dict | None:
    r, c = map(int, door_pos)
    h, w = grid.shape
    if not (0 < r < h - 1 and 0 < c < w - 1):
        return None

    matches: list[dict] = []

    if int(grid[r, c - 1]) == int(WALL) and int(grid[r, c + 1]) == int(WALL):
        c0 = c - 1
        while c0 - 1 >= 0 and int(grid[r, c0 - 1]) == int(WALL):
            c0 -= 1
        c1 = c + 1
        while c1 + 1 < w and int(grid[r, c1 + 1]) == int(WALL):
            c1 += 1

        rr = r + 1
        while rr < h and int(grid[rr, c0]) == int(WALL) and int(grid[rr, c1]) == int(WALL):
            rr += 1
        top_candidate = _validate_room(grid, bg, (r, c), "top", (r, c0, rr - 1, c1))
        if top_candidate is not None:
            matches.append(top_candidate)

        rr = r - 1
        while rr >= 0 and int(grid[rr, c0]) == int(WALL) and int(grid[rr, c1]) == int(WALL):
            rr -= 1
        bottom_candidate = _validate_room(grid, bg, (r, c), "bottom", (rr + 1, c0, r, c1))
        if bottom_candidate is not None:
            matches.append(bottom_candidate)

    if int(grid[r - 1, c]) == int(WALL) and int(grid[r + 1, c]) == int(WALL):
        r0 = r - 1
        while r0 - 1 >= 0 and int(grid[r0 - 1, c]) == int(WALL):
            r0 -= 1
        r1 = r + 1
        while r1 + 1 < h and int(grid[r1 + 1, c]) == int(WALL):
            r1 += 1

        cc = c + 1
        while cc < w and int(grid[r0, cc]) == int(WALL) and int(grid[r1, cc]) == int(WALL):
            cc += 1
        left_candidate = _validate_room(grid, bg, (r, c), "left", (r0, c, r1, cc - 1))
        if left_candidate is not None:
            matches.append(left_candidate)

        cc = c - 1
        while cc >= 0 and int(grid[r0, cc]) == int(WALL) and int(grid[r1, cc]) == int(WALL):
            cc -= 1
        right_candidate = _validate_room(grid, bg, (r, c), "right", (r0, cc + 1, r1, c))
        if right_candidate is not None:
            matches.append(right_candidate)

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each L-shaped key's orientation to decide which same-colored room door opens.

    Input structure:
    - The background is azure.
    - Several monochrome 3-pixel L triominoes act as keys, each encoded by its corner orientation.
    - Several grey rectangular rooms contain maroon interior patterns, and each room has exactly one single-pixel colored door centered on one side.

    Rule:
    - Recover every key's orientation from the L triomino corner: the corner pixel points diagonally away from the open 2x2 corner.
    - Match each room to the key that shares its door color, then compare the door side to the corresponding orientation axis component.
    - If the orientation matches the door side, erase the door pixel to open the room and keep the maroon pattern; otherwise keep the door closed and erase all maroon pixels inside that room.
    - Erase every key from the output.

    Key insight:
    The L triomino is not moved anywhere; its geometry encodes a directional signal, and that signal controls whether the same-colored room reveals or hides its interior pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    comps = _find_components(g, bg)

    key_dirs: dict[int, tuple[int, int]] = {}
    key_coords: list[tuple[int, int]] = []
    for comp in comps:
        if int(comp["color"]) in {int(WALL), int(HIDDEN)}:
            continue
        orient = _key_orientation(comp)
        if orient is None:
            continue
        key_dirs[int(comp["color"])] = orient
        key_coords.extend((int(r), int(c)) for r, c in comp["coords"])

    for r, c in key_coords:
        out[r, c] = int(bg)

    rooms: list[dict] = []
    for comp in comps:
        if int(comp["color"]) in {int(bg), int(WALL), int(HIDDEN)} or int(comp["area"]) != 1:
            continue
        room = _room_from_door(g, bg, comp["coords"][0])
        if room is not None:
            rooms.append(room)

    seen_doors: set[tuple[int, int]] = set()
    unique_rooms: list[dict] = []
    for room in rooms:
        if room["door"] not in seen_doors:
            seen_doors.add(room["door"])
            unique_rooms.append(room)

    for room in unique_rooms:
        color = int(room["color"])
        r0, c0, r1, c1 = map(int, room["bbox"])
        dr, dc = map(int, room["door"])
        orient = key_dirs.get(color)
        opened = False
        if orient is not None:
            vr, vc = orient
            if room["side"] == "top":
                opened = int(vr) < 0
            elif room["side"] == "bottom":
                opened = int(vr) > 0
            elif room["side"] == "left":
                opened = int(vc) < 0
            elif room["side"] == "right":
                opened = int(vc) > 0

        if opened:
            out[dr, dc] = int(bg)
        else:
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == int(HIDDEN)] = int(bg)

    return out


def _grow_connected_mask(height: int, width: int, area: int, rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    start = rng.choice(cells)
    chosen = {start}
    while len(chosen) < area:
        frontier = []
        for r, c in chosen:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return None
        chosen.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _make_pattern(interior_h: int, interior_w: int, rng: random.Random) -> np.ndarray | None:
    max_area = min(interior_h * interior_w - 1, max(2, (interior_h * interior_w) // 2))
    if max_area < 2:
        return None
    for _ in range(40):
        area = rng.randint(2, max_area)
        mask = _grow_connected_mask(interior_h, interior_w, area, rng)
        if mask is not None and mask.any():
            return mask
    return None


def _room_patch(room_h: int, room_w: int, side: str, door_color: int, pattern: np.ndarray) -> np.ndarray:
    patch = np.full((room_h, room_w), BG, dtype=int)
    patch[0, :] = WALL
    patch[-1, :] = WALL
    patch[:, 0] = WALL
    patch[:, -1] = WALL

    if side == "top":
        patch[0, room_w // 2] = door_color
    elif side == "bottom":
        patch[-1, room_w // 2] = door_color
    elif side == "left":
        patch[room_h // 2, 0] = door_color
    else:
        patch[room_h // 2, -1] = door_color

    patch[1:-1, 1:-1][pattern] = HIDDEN
    return patch


def _key_patch(color: int, side: str, opened: bool, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    if side == "top":
        corner_row = 0 if opened else 1
        corner_col = rng.choice([0, 1])
    elif side == "bottom":
        corner_row = 1 if opened else 0
        corner_col = rng.choice([0, 1])
    elif side == "left":
        corner_col = 0 if opened else 1
        corner_row = rng.choice([0, 1])
    else:
        corner_col = 1 if opened else 0
        corner_row = rng.choice([0, 1])

    patch = np.full((2, 2), BG, dtype=int)
    patch[:, :] = color
    patch[1 - corner_row, 1 - corner_col] = BG
    orient = (-1 if corner_row == 0 else 1, -1 if corner_col == 0 else 1)
    return patch, orient


def _place_rectangles(
    grid_h: int,
    grid_w: int,
    sizes: list[tuple[int, int]],
    rng: random.Random,
    padding: int = 1,
    edge_margin: int = 1,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(sizes)

    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(edge_margin, grid_h - h - edge_margin + 1):
            for c0 in range(edge_margin, grid_w - w - edge_margin + 1):
                rr0 = max(0, r0 - padding)
                cc0 = max(0, c0 - padding)
                rr1 = min(grid_h, r0 + h + padding)
                cc1 = min(grid_w, c0 + w + padding)
                if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements[idx] = (int(r0), int(c0))
        occupied[r0 : r0 + h, c0 : c0 + w] = True
    return placements


def _build_expected(inp: np.ndarray, room_specs: list[dict], key_specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in key_specs:
        kr, kc = map(int, spec["pos"])
        out[kr : kr + 2, kc : kc + 2][spec["patch"] != int(BG)] = int(BG)

    for spec in room_specs:
        rr, cc = map(int, spec["pos"])
        room_h, room_w = map(int, spec["patch"].shape)
        door_r, door_c = rr + spec["door_local"][0], cc + spec["door_local"][1]
        if spec["opened"]:
            out[door_r, door_c] = int(BG)
        else:
            inner = out[rr + 1 : rr + room_h - 1, cc + 1 : cc + room_w - 1]
            inner[inner == int(HIDDEN)] = int(BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_rooms = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, n_rooms)

    opened_flags = [True] * (n_rooms // 2) + [False] * (n_rooms // 2)
    if n_rooms % 2 == 1:
        opened_flags.append(rng.choice([True, False]))
    rng.shuffle(opened_flags)
    if len(set(opened_flags)) == 1:
        opened_flags[0] = not opened_flags[0]
        rng.shuffle(opened_flags)

    room_specs: list[dict] = []
    key_specs: list[dict] = []
    object_sizes: list[tuple[int, int]] = []

    for color, opened in zip(colors, opened_flags):
        room_h = rng.choice([5, 7, 9])
        room_w = rng.choice([5, 7, 9])
        side = rng.choice(["top", "bottom", "left", "right"])
        pattern = _make_pattern(room_h - 2, room_w - 2, rng)
        if pattern is None:
            return None
        room_patch = _room_patch(room_h, room_w, side, color, pattern)
        if side == "top":
            door_local = (0, room_w // 2)
        elif side == "bottom":
            door_local = (room_h - 1, room_w // 2)
        elif side == "left":
            door_local = (room_h // 2, 0)
        else:
            door_local = (room_h // 2, room_w - 1)

        key_patch, orient = _key_patch(color, side, opened, rng)
        room_specs.append(
            {
                "color": color,
                "opened": opened,
                "side": side,
                "patch": room_patch,
                "door_local": door_local,
            }
        )
        key_specs.append({"color": color, "patch": key_patch, "orient": orient})
        object_sizes.append((room_h, room_w))
        object_sizes.append((2, 2))

    min_side = max(15, max(max(h for h, _ in object_sizes), max(w for _, w in object_sizes)) + 4)
    total_bbox_area = sum(h * w for h, w in object_sizes)
    dim_choices = [
        (h, w)
        for h in range(min_side, 31)
        for w in range(min_side, 31)
        if h * w >= total_bbox_area * 3
    ]
    if not dim_choices:
        return None
    grid_h, grid_w = rng.choice(dim_choices)
    placements = _place_rectangles(grid_h, grid_w, object_sizes, rng, padding=1)
    if placements is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    place_index = 0
    for i, spec in enumerate(room_specs):
        rr, cc = placements[place_index]
        place_index += 1
        spec["pos"] = (int(rr), int(cc))
        inp[rr : rr + spec["patch"].shape[0], cc : cc + spec["patch"].shape[1]] = spec["patch"]

        kr, kc = placements[place_index]
        place_index += 1
        key_spec = key_specs[i]
        key_spec["pos"] = (int(kr), int(kc))
        mask = key_spec["patch"] != int(BG)
        inp[kr : kr + 2, kc : kc + 2][mask] = int(key_spec["color"])

    expected = _build_expected(inp, room_specs, key_specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
