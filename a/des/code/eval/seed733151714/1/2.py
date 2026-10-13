from __future__ import annotations

import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touches_border(coords: np.ndarray, height: int, width: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == height - 1)
        or np.any(cols == 0)
        or np.any(cols == width - 1)
    )


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _infer_background(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _infer_wall_color(g: np.ndarray, bg: int) -> int:
    wall_pixels = g[g != int(bg)]
    colors, counts = np.unique(wall_pixels, return_counts=True)
    assert len(colors) > 0
    return int(colors[int(np.argmax(counts))])


def _monochrome_components(g: np.ndarray, excluded: set[int]) -> list[dict]:
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color in excluded or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": _bbox_from_coords(arr),
                }
            )
    return components


def _extract_key_rel_coords(component: dict) -> list[tuple[int, int]]:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    expected = {(0, cc) for cc in range(width)}
    expected |= {(rr, 0) for rr in range(height)}
    assert rel == expected
    assert width >= height >= 2
    return sorted(rel)


def _placement_coords(rel_coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c + 1) for r, c in rel_coords]


def _rotate_key(rel_coords: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        return [(r, c) for r, c in rel_coords]
    if direction == "down":
        return [(c, -r) for r, c in rel_coords]
    if direction == "left":
        return [(-r, -c) for r, c in rel_coords]
    if direction == "up":
        return [(-c, r) for r, c in rel_coords]
    raise AssertionError(f"unknown direction {direction}")


def _find_doors(g: np.ndarray, bg: int, wall: int) -> list[tuple[int, int]]:
    height, width = g.shape
    doors: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if int(g[r, c]) != int(bg):
                continue
            up = r > 0 and int(g[r - 1, c]) == int(wall)
            down = r + 1 < height and int(g[r + 1, c]) == int(wall)
            left = c > 0 and int(g[r, c - 1]) == int(wall)
            right = c + 1 < width and int(g[r, c + 1]) == int(wall)
            if left and right and not up and not down:
                doors.append((r, c))
            elif up and down and not left and not right:
                doors.append((r, c))
    doors.sort()
    return doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect walled rooms, match each room's door-count to an L-shaped key of the same area,
    rotate the matched key toward the room interior, erase the source keys, and stamp a copy
    of that key at every door of the room.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _infer_background(g)
    wall = _infer_wall_color(g, bg)

    key_specs: dict[int, dict] = {}
    for component in _monochrome_components(g, excluded={bg, wall}):
        area = int(component["area"])
        assert area not in key_specs
        rel_coords = _extract_key_rel_coords(component)
        key_specs[area] = {
            "color": int(component["color"]),
            "coords": np.asarray(component["coords"], dtype=int),
            "rel_coords": rel_coords,
            "place_coords": _placement_coords(rel_coords),
        }

    doors = _find_doors(g, bg, wall)
    closed = g.copy()
    for r, c in doors:
        closed[r, c] = wall

    room_components = [
        coords
        for coords in _connected_components(closed == int(bg))
        if not _touches_border(coords, height, width)
    ]
    room_components.sort(key=lambda coords: _bbox_from_coords(coords)[:2])

    out = g.copy()
    for spec in key_specs.values():
        coords = spec["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for room in room_components:
        room_set = {tuple(cell) for cell in room.tolist()}
        room_doors = [
            door
            for door in doors
            if any((door[0] + dr, door[1] + dc) in room_set for dr, dc in DIRS.values())
        ]
        room_doors.sort()
        door_count = len(room_doors)
        assert door_count in key_specs
        key = key_specs[door_count]

        for door_r, door_c in room_doors:
            inward = [
                name
                for name, (dr, dc) in DIRS.items()
                if (door_r + dr, door_c + dc) in room_set
            ]
            assert len(inward) == 1
            for dr, dc in _rotate_key(key["place_coords"], inward[0]):
                rr = int(door_r + dr)
                cc = int(door_c + dc)
                assert 0 <= rr < height and 0 <= cc < width
                assert int(out[rr, cc]) == int(bg)
                out[rr, cc] = int(key["color"])

    return out


# --------------- generator ---------------

BG_COLOR = 0
WALL_COLOR = 9
KEY_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
DOOR_COUNT_OPTIONS = [3, 4, 5, 6]
GRID_SIDE_MIN = 24
GRID_SIDE_MAX = 32
ROOM_COUNT_OPTIONS = [2, 3, 4]
KEY_ZONE_SHAPE = (8, 12)
ROOM_GAP = 2
ROOM_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _mask_value(mask: np.ndarray, r: int, c: int) -> bool:
    height, width = mask.shape
    return 0 <= r < height and 0 <= c < width and bool(mask[r, c])


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _key_rel_coords(area: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for short in (2, 3):
        long = int(area - short + 1)
        if short >= 2 and long >= short and long + short - 1 == area:
            if area == 3 or long > short:
                options.append((long, short))
    assert options
    long, short = random.choice(options)
    coords = {(0, c) for c in range(long)}
    coords |= {(r, 0) for r in range(short)}
    return sorted(coords)


def _patch_from_rel_coords(rel_coords: list[tuple[int, int]], color: int) -> np.ndarray:
    max_r = max(r for r, _ in rel_coords)
    max_c = max(c for _, c in rel_coords)
    patch = np.full((max_r + 1, max_c + 1), BG_COLOR, dtype=int)
    for r, c in rel_coords:
        patch[r, c] = color
    return patch


def _sample_room_interior(min_side: int) -> np.ndarray:
    height = random.randint(min_side, min_side + 3)
    width = random.randint(min_side, min_side + 3)
    interior = np.ones((height, width), dtype=bool)
    if random.random() < 0.5 and height >= 7 and width >= 7:
        cut_h = random.randint(2, height - 3)
        cut_w = random.randint(2, width - 3)
        corner = random.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            interior[:cut_h, :cut_w] = False
        elif corner == "tr":
            interior[:cut_h, width - cut_w :] = False
        elif corner == "bl":
            interior[height - cut_h :, :cut_w] = False
        else:
            interior[height - cut_h :, width - cut_w :] = False
    return np.pad(interior, 1, constant_values=False)


def _door_candidates(
    interior: np.ndarray,
    wall: np.ndarray,
    place_coords: list[tuple[int, int]],
) -> list[dict]:
    height, width = interior.shape
    candidates: list[dict] = []
    for r, c in np.argwhere(wall):
        inward_dirs = [
            name
            for name, (dr, dc) in DIRS.items()
            if _mask_value(interior, int(r + dr), int(c + dc))
        ]
        if len(inward_dirs) != 1:
            continue
        direction = inward_dirs[0]
        opp_dr, opp_dc = DIRS[OPPOSITE[direction]]
        if _mask_value(interior, int(r + opp_dr), int(c + opp_dc)):
            continue

        if direction in ("up", "down"):
            if not (_mask_value(wall, int(r), int(c - 1)) and _mask_value(wall, int(r), int(c + 1))):
                continue
            if _mask_value(wall, int(r - 1), int(c)) or _mask_value(wall, int(r + 1), int(c)):
                continue
        else:
            if not (_mask_value(wall, int(r - 1), int(c)) and _mask_value(wall, int(r + 1), int(c))):
                continue
            if _mask_value(wall, int(r), int(c - 1)) or _mask_value(wall, int(r), int(c + 1)):
                continue

        footprint: set[tuple[int, int]] = set()
        ok = True
        for dr, dc in _rotate_key(place_coords, direction):
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < height and 0 <= cc < width):
                ok = False
                break
            if (dr, dc) != (0, 0) and not bool(interior[rr, cc]):
                ok = False
                break
            footprint.add((rr, cc))
        if ok:
            candidates.append(
                {
                    "cell": (int(r), int(c)),
                    "direction": direction,
                    "footprint": footprint,
                }
            )
    return candidates


def _compatible_door(candidate: dict, chosen: list[dict]) -> bool:
    cell_r, cell_c = candidate["cell"]
    footprint = candidate["footprint"]
    for prev in chosen:
        prev_r, prev_c = prev["cell"]
        if abs(cell_r - prev_r) + abs(cell_c - prev_c) == 1:
            return False
        if footprint & prev["footprint"]:
            return False
    return True


def _choose_doors(candidates: list[dict], door_count: int) -> list[dict] | None:
    by_dir: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_dir.setdefault(candidate["direction"], []).append(candidate)
    directions = [direction for direction, items in by_dir.items() if items]

    for _ in range(LAYOUT_ATTEMPTS):
        chosen: list[dict] = []
        must_dirs = directions[:]
        random.shuffle(must_dirs)
        if len(must_dirs) > door_count:
            must_dirs = must_dirs[:door_count]

        ok = True
        for direction in must_dirs:
            options = by_dir[direction][:]
            random.shuffle(options)
            picked = None
            for candidate in options:
                if _compatible_door(candidate, chosen):
                    picked = candidate
                    break
            if picked is None:
                ok = False
                break
            chosen.append(picked)
        if not ok:
            continue

        pool = candidates[:]
        random.shuffle(pool)
        for candidate in pool:
            if len(chosen) == door_count:
                break
            if candidate in chosen:
                continue
            if _compatible_door(candidate, chosen):
                chosen.append(candidate)
        if len(chosen) != door_count:
            continue
        return sorted(chosen, key=lambda item: item["cell"])
    return None


def _generate_room_patch(key_rel_coords: list[tuple[int, int]], door_count: int) -> dict | None:
    place_coords = _placement_coords(key_rel_coords)
    key_height = max(r for r, _ in place_coords) + 1
    key_width = max(c for _, c in place_coords) + 1
    min_side = max(7, key_width + 1, door_count)
    if key_height >= 3:
        min_side = max(min_side, key_height + 4)

    for _ in range(ROOM_ATTEMPTS):
        interior = _sample_room_interior(min_side)
        wall = _dilate4(interior) & ~interior
        candidates = _door_candidates(interior, wall, place_coords)
        if len(candidates) < door_count:
            continue
        chosen = _choose_doors(candidates, door_count)
        if chosen is None:
            continue

        patch = np.full(interior.shape, BG_COLOR, dtype=int)
        patch[wall] = WALL_COLOR
        for door in chosen:
            rr, cc = door["cell"]
            patch[rr, cc] = BG_COLOR

        if set(_find_doors(patch, BG_COLOR, WALL_COLOR)) != {door["cell"] for door in chosen}:
            continue
        closed = patch.copy()
        for door in chosen:
            rr, cc = door["cell"]
            closed[rr, cc] = WALL_COLOR
        rooms = [
            coords
            for coords in _connected_components(closed == BG_COLOR)
            if not _touches_border(coords, patch.shape[0], patch.shape[1])
        ]
        if len(rooms) != 1:
            continue

        return {
            "patch": patch,
            "doors": chosen,
            "door_count": door_count,
            "key_rel_coords": key_rel_coords,
            "place_coords": place_coords,
        }
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def _key_zone_rect(height: int, width: int, corner: str) -> tuple[int, int, int, int]:
    zone_h, zone_w = KEY_ZONE_SHAPE
    if corner == "tl":
        return 0, 0, zone_h, zone_w
    if corner == "tr":
        return 0, width - zone_w, zone_h, zone_w
    if corner == "bl":
        return height - zone_h, 0, zone_h, zone_w
    return height - zone_h, width - zone_w, zone_h, zone_w


def _place_rooms(grid: np.ndarray, rooms: list[dict], corner: str) -> bool:
    height, width = grid.shape
    reserved = [_key_zone_rect(height, width, corner)]
    order = sorted(range(len(rooms)), key=lambda idx: rooms[idx]["patch"].size, reverse=True)

    for idx in order:
        patch = rooms[idx]["patch"]
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(1, height - ph)
            for c in range(1, width - pw)
            if all(
                not _rectangles_overlap((r, c, ph, pw), rect, ROOM_GAP)
                for rect in reserved
            )
        ]
        if not positions:
            return False
        top, left = random.choice(positions)
        rooms[idx]["top"] = int(top)
        rooms[idx]["left"] = int(left)
        reserved.append((int(top), int(left), int(ph), int(pw)))
        patch_mask = patch == WALL_COLOR
        grid[top : top + ph, left : left + pw][patch_mask] = WALL_COLOR
    return True


def _slot_positions(zone_h: int, zone_w: int) -> list[tuple[int, int]]:
    del zone_h, zone_w
    return [(1, 1), (1, 6), (4, 1), (4, 6)]


def _place_keys(grid: np.ndarray, key_specs: list[dict], corner: str) -> bool:
    height, width = grid.shape
    zone_r, zone_c, zone_h, zone_w = _key_zone_rect(height, width, corner)
    slots = _slot_positions(zone_h, zone_w)
    assert len(key_specs) <= len(slots)
    chosen_slots = random.sample(slots, len(key_specs))
    random.shuffle(chosen_slots)

    for key_spec, (local_r, local_c) in zip(key_specs, chosen_slots):
        patch = key_spec["patch"]
        ph, pw = patch.shape
        top = zone_r + local_r
        left = zone_c + local_c
        if top < 0 or left < 0 or top + ph > height or left + pw > width:
            return False
        target = grid[top : top + ph, left : left + pw]
        colored = patch != BG_COLOR
        if np.any(target[colored] != BG_COLOR):
            return False
        target[colored] = patch[colored]
        key_spec["coords"] = np.argwhere(patch != BG_COLOR) + np.array([top, left], dtype=int)
    return True


def _build_expected(inp: np.ndarray, rooms: list[dict], key_specs_by_area: dict[int, dict]) -> np.ndarray:
    out = inp.copy()
    for key_spec in key_specs_by_area.values():
        coords = key_spec["coords"]
        out[coords[:, 0], coords[:, 1]] = BG_COLOR

    for room in rooms:
        key_spec = key_specs_by_area[int(room["door_count"])]
        for door in room["doors"]:
            door_r = int(room["top"] + door["cell"][0])
            door_c = int(room["left"] + door["cell"][1])
            for dr, dc in _rotate_key(key_spec["place_coords"], door["direction"]):
                rr = int(door_r + dr)
                cc = int(door_c + dc)
                assert int(out[rr, cc]) == BG_COLOR
                out[rr, cc] = int(key_spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    corner = random.choice(["tl", "tr", "bl", "br"])
    room_count = random.choice(ROOM_COUNT_OPTIONS)
    door_counts = random.sample(DOOR_COUNT_OPTIONS, room_count)
    key_colors = random.sample(KEY_COLOR_POOL, room_count)

    rooms: list[dict] = []
    key_specs_by_area: dict[int, dict] = {}
    for door_count, color in zip(door_counts, key_colors):
        rel_coords = _key_rel_coords(door_count)
        room = _generate_room_patch(rel_coords, door_count)
        if room is None:
            return None
        rooms.append(room)
        key_specs_by_area[int(door_count)] = {
            "color": int(color),
            "rel_coords": rel_coords,
            "place_coords": _placement_coords(rel_coords),
            "patch": _patch_from_rel_coords(rel_coords, int(color)),
        }

    inp = np.full((height, width), BG_COLOR, dtype=int)
    if not _place_rooms(inp, rooms, corner):
        return None
    if not _place_keys(inp, list(key_specs_by_area.values()), corner):
        return None

    expected = _build_expected(inp, rooms, key_specs_by_area)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door-key puzzle")
