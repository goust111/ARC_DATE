import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_monochrome_components(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            pts: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(pts)),
                }
            )

    return comps


def _is_rectangular_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False
    return True


def _extract_room_objects(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    mask = np.zeros_like(grid, dtype=bool)
    mask[inner_r0:r1, inner_c0:c1] = True
    mask[inner_r0, inner_c0] = False
    mask &= grid != bg

    comps = _find_monochrome_components(grid, mask)
    objects: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        if color == frame_color:
            continue
        coords = comp["coords"]
        rr0, cc0, rr1, cc1 = comp["bbox"]
        objects.append(
            {
                "color": color,
                "coords": coords,
                "area": int(comp["area"]),
                "sort_key": (-int(comp["area"]), int(rr0), int(cc0), int(rr1), int(cc1)),
            }
        )

    objects.sort(key=lambda obj: obj["sort_key"])
    return objects


def _find_rooms(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _find_monochrome_components(g, g != bg)
    rooms: list[dict] = []

    for comp in comps:
        if not _is_rectangular_frame(g, comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        seed_pos = (r0 + 1, c0 + 1)
        seed_color = int(g[seed_pos])
        frame_color = int(comp["color"])
        if seed_color in (bg, frame_color):
            continue
        objects = _extract_room_objects(g, bg, frame_color, comp["bbox"])
        rooms.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "frame_color": frame_color,
                "seed_color": seed_color,
                "seed_pos": seed_pos,
                "objects": objects,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return bg, rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rooms by the seed pixel in their top-left interior corner, then
    swap the colors of same-rank objects between the two rooms in each group.
    Objects are ranked by 4-connected area from largest to smallest.
    """
    g = np.asarray(grid, dtype=int)
    _bg, rooms = _find_rooms(g)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for room in rooms:
        grouped[int(room["seed_color"])].append(room)

    out = g.copy()
    for seed_color, pair in grouped.items():
        assert len(pair) == 2, f"seed color {seed_color} must define exactly one pair"
        objs_a = pair[0]["objects"]
        objs_b = pair[1]["objects"]
        assert len(objs_a) == len(objs_b), "paired rooms must contain the same number of objects"

        for obj_a, obj_b in zip(objs_a, objs_b):
            out[obj_a["coords"][:, 0], obj_a["coords"][:, 1]] = int(obj_b["color"])
            out[obj_b["coords"][:, 0], obj_b["coords"][:, 1]] = int(obj_a["color"])

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _fallback_shape(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    choices: list[tuple[int, int]] = []
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            if h * w >= area:
                choices.append((h, w))
    choices.sort(key=lambda hw: (hw[0] * hw[1], hw[0] + hw[1], hw[0]))
    h, w = choices[0]
    cells: list[tuple[int, int]] = []
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            cells.append((r, c))
            if len(cells) == area:
                return tuple(sorted(cells))
    raise AssertionError("fallback shape construction failed")


def _random_shape(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    cap_h = min(max_h, 4)
    cap_w = min(max_w, 4)
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.append(cand)
            cells.add(random.choice(frontier))

        transformed = set(cells)
        if random.random() < 0.5:
            transformed = {(c, r) for r, c in transformed}
        if random.random() < 0.5:
            transformed = {(-r, c) for r, c in transformed}
        if random.random() < 0.5:
            transformed = {(r, -c) for r, c in transformed}

        shape = _normalize_shape(transformed)
        sh, sw = _shape_dims(shape)
        if sh <= cap_h and sw <= cap_w:
            return shape

    return _fallback_shape(area, cap_h, cap_w)


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _build_room(
    bg: int,
    border_color: int,
    seed_color: int,
    rank_colors: list[int],
    rank_areas: list[int],
) -> np.ndarray | None:
    n_objects = len(rank_colors)
    for _ in range(160):
        inner_h = random.randint(5 + (n_objects > 2), 8 + (n_objects > 3))
        inner_w = random.randint(6 + (n_objects > 2), 10 + (n_objects > 3))
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        blocked[: min(2, inner_h), : min(2, inner_w)] = True
        placements: list[tuple[int, list[tuple[int, int]]]] = []
        ok = True

        for area, color in zip(rank_areas, rank_colors):
            placed = False
            for _ in range(180):
                shape = _random_shape(area, inner_h, inner_w)
                sh, sw = _shape_dims(shape)
                if sh > inner_h or sw > inner_w:
                    continue
                base_r = random.randint(0, inner_h - sh)
                base_c = random.randint(0, inner_w - sw)
                coords = [(base_r + dr, base_c + dc) for dr, dc in shape]
                if any(blocked[r, c] for r, c in coords):
                    continue
                placements.append((color, coords))
                _mark_blocked(blocked, coords)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        room = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        room[0, :] = border_color
        room[-1, :] = border_color
        room[:, 0] = border_color
        room[:, -1] = border_color
        room[1, 1] = seed_color
        for color, coords in placements:
            for r, c in coords:
                room[r + 1, c + 1] = color
        return room

    return None


def _derangement(values: list[int]) -> list[int]:
    assert len(values) >= 2
    base = list(values)
    for _ in range(60):
        perm = list(base)
        random.shuffle(perm)
        if all(a != b for a, b in zip(base, perm)):
            return perm
    return base[1:] + base[:1]


def _compose_canvas(rooms: list[np.ndarray], bg: int) -> np.ndarray | None:
    heights = [room.shape[0] for room in rooms]
    widths = [room.shape[1] for room in rooms]
    max_h = max(heights)
    max_w = max(widths)
    padded_area = sum((h + 2) * (w + 2) for h, w in zip(heights, widths))
    base = int(np.ceil(np.sqrt(padded_area * 1.55)))

    for _ in range(180):
        h = random.randint(max(max_h + 2, base - 2), max(max_h + 3, base + 6))
        w = random.randint(max(max_w + 2, base - 2), max(max_w + 3, base + 8))
        grid = np.full((h, w), bg, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        order = list(range(len(rooms)))
        random.shuffle(order)
        order.sort(key=lambda idx: rooms[idx].shape[0] * rooms[idx].shape[1], reverse=True)

        ok = True
        for idx in order:
            room = rooms[idx]
            rh, rw = room.shape
            placed = False
            for _ in range(300):
                top = random.randint(1, h - rh - 1)
                left = random.randint(1, w - rw - 1)
                r0 = max(0, top - 1)
                r1 = min(h, top + rh + 1)
                c0 = max(0, left - 1)
                c1 = min(w, left + rw + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                grid[top : top + rh, left : left + rw] = room
                occupied[r0:r1, c0:c1] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and int(np.count_nonzero(grid == bg)) > max(
            int(np.count_nonzero(grid == color)) for color in np.unique(grid) if int(color) != bg
        ):
            return grid

    return None


def _build_pair(
    bg: int, border_color: int, seed_color: int, object_palette: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.randint(2, min(4, len(object_palette)))
    rank_colors_a = random.sample(object_palette, n_objects)
    rank_colors_b = _derangement(rank_colors_a)
    rank_areas_a = sorted(random.sample(range(1, 8), n_objects), reverse=True)
    rank_areas_b = sorted(random.sample(range(1, 8), n_objects), reverse=True)

    room_a = _build_room(bg, border_color, seed_color, rank_colors_a, rank_areas_a)
    room_b = _build_room(bg, border_color, seed_color, rank_colors_b, rank_areas_b)
    if room_a is None or room_b is None:
        return None
    return room_a, room_b


def _valid_generated_grid(grid: np.ndarray) -> bool:
    bg, rooms = _find_rooms(grid)
    if len(rooms) < 2:
        return False
    grouped: dict[int, list[dict]] = defaultdict(list)
    for room in rooms:
        grouped[int(room["seed_color"])].append(room)
        areas = [obj["area"] for obj in room["objects"]]
        if len(areas) < 2 or len(areas) != len(set(areas)):
            return False

    if any(len(pair) != 2 for pair in grouped.values()):
        return False
    for pair in grouped.values():
        if len(pair[0]["objects"]) != len(pair[1]["objects"]):
            return False

    if int(np.count_nonzero(grid == bg)) <= max(
        int(np.count_nonzero(grid == color)) for color in np.unique(grid) if int(color) != bg
    ):
        return False

    return not np.array_equal(main(grid), grid)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices([1, 2, 3], weights=[2, 5, 3])[0]
    bg, border_color = random.sample(list(ALL_COLORS), 2)
    remaining = [color for color in ALL_COLORS if color not in (bg, border_color)]
    seed_colors = random.sample(remaining, pair_count)
    object_palette = [color for color in remaining if color not in seed_colors]

    rooms: list[np.ndarray] = []
    for seed_color in seed_colors:
        pair = _build_pair(bg, border_color, seed_color, object_palette)
        if pair is None:
            return None
        rooms.extend(pair)

    random.shuffle(rooms)
    grid = _compose_canvas(rooms, bg)
    if grid is None or not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
