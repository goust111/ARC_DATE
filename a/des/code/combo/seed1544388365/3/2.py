import random
from typing import Optional

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == int(bg) or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = r0
            min_c = max_c = c0
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min_r, min_c, max_r, max_c),
                }
            )
    return components


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    return cells


def _direction_shift(
    direction: str, r0: int, c0: int, r1: int, c1: int, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "top":
        shift = int(rows.max() - r0 + 1)
        return rows - shift, cols
    if direction == "bottom":
        shift = int(r1 - rows.min() + 1)
        return rows + shift, cols
    if direction == "left":
        shift = int(cols.max() - c0 + 1)
        return rows, cols - shift
    shift = int(c1 - cols.min() + 1)
    return rows, cols + shift


def _detect_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []

    for comp in _find_monochrome_components(g, bg=BG):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        coords = set(comp["coords"])
        perimeter = set(_perimeter_cells(r0, c0, r1, c1))
        if len(coords) != len(perimeter) - 1 or not coords.issubset(perimeter):
            continue

        missing = list(perimeter - coords)
        if len(missing) != 1:
            continue
        gap_r, gap_c = missing[0]
        side_hits = [
            gap_r == r0,
            gap_r == r1,
            gap_c == c0,
            gap_c == c1,
        ]
        if sum(side_hits) != 1:
            continue

        if gap_r == r0:
            direction = "top"
        elif gap_r == r1:
            direction = "bottom"
        elif gap_c == c0:
            direction = "left"
        else:
            direction = "right"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        frag_rel = np.argwhere(interior != BG)
        if len(frag_rel) == 0:
            continue

        fragments = [
            (r0 + 1 + int(dr), c0 + 1 + int(dc), int(interior[dr, dc]))
            for dr, dc in frag_rel
        ]
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "fragments": fragments,
            }
        )

    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's interior fragment pattern rigidly outward along the side
    indicated by the border gap, then clear the room interior.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    rooms = _detect_rooms(g)

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        direction = room["direction"]
        fragments = room["fragments"]
        rows = np.array([r for r, _c, _color in fragments], dtype=int)
        cols = np.array([c for _r, c, _color in fragments], dtype=int)
        dst_rows, dst_cols = _direction_shift(direction, r0, c0, r1, c1, rows, cols)

        for (dst_r, dst_c), (_src_r, _src_c, color) in zip(zip(dst_rows, dst_cols), fragments):
            nr = int(dst_r)
            nc = int(dst_c)
            if not (0 <= nr < h and 0 <= nc < w):
                raise AssertionError("translated fragment left the grid")
            if int(out[nr, nc]) != BG:
                raise AssertionError("translated fragments overlap existing structure")
            out[nr, nc] = int(color)

    return out


GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (16, 30)
ROOM_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
ROOM_OUTER_RANGE = (5, 9)
FRAGMENT_COLORS_MAX = 3


def _is_open_frame_shape(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    perimeter = set(_perimeter_cells(r0, c0, r1, c1))
    return len(cells) == len(perimeter) - 1 and cells.issubset(perimeter) and len(perimeter - cells) == 1


def _gen_fragment_shape(inner_h: int, inner_w: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(80):
        box_h = rng.randint(1, inner_h)
        box_w = rng.randint(1, inner_w)
        base_r = rng.randint(0, inner_h - box_h)
        base_c = rng.randint(0, inner_w - box_w)
        max_area = box_h * box_w
        target = rng.randint(1, min(max_area, max(3, (2 * max_area) // 3)))

        seed = (rng.randrange(box_h), rng.randrange(box_w))
        occupied = {seed}
        while len(occupied) < target:
            candidates = []
            for r, c in occupied:
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in occupied:
                        candidates.append((rr, cc))
            if candidates:
                occupied.add(rng.choice(candidates))
            else:
                remaining = [(r, c) for r in range(box_h) for c in range(box_w) if (r, c) not in occupied]
                occupied.add(rng.choice(remaining))

        if max_area - len(occupied) > 0 and rng.random() < 0.6:
            extra_cap = min(2, max_area - len(occupied))
            for cell in rng.sample(
                [(r, c) for r in range(box_h) for c in range(box_w) if (r, c) not in occupied],
                rng.randint(0, extra_cap),
            ):
                occupied.add(cell)

        placed = {(base_r + r, base_c + c) for r, c in occupied}
        if _is_open_frame_shape(placed):
            continue
        return sorted(placed)

    return [(rng.randrange(inner_h), rng.randrange(inner_w))]


def _assign_fragment_colors(
    shape: list[tuple[int, int]], border_color: int, rng: random.Random
) -> list[tuple[int, int, int]]:
    palette = [color for color in range(1, 10) if color != border_color]
    num_colors = min(len(shape), rng.randint(1, FRAGMENT_COLORS_MAX))
    chosen = rng.sample(palette, num_colors)
    colors = [rng.choice(chosen) for _ in shape]
    for i, color in enumerate(chosen):
        colors[i] = color
    rng.shuffle(colors)
    return [(r, c, color) for (r, c), color in zip(shape, colors)]


def _moved_relative_fragments(room: dict) -> list[tuple[int, int, int]]:
    h = int(room["height"])
    w = int(room["width"])
    direction = room["direction"]
    frags = room["fragments"]
    rows = np.array([1 + r for r, _c, _color in frags], dtype=int)
    cols = np.array([1 + c for _r, c, _color in frags], dtype=int)
    if direction == "top":
        shift = int(rows.max() + 1)
        dst_rows = rows - shift
        dst_cols = cols
    elif direction == "bottom":
        shift = int(h - rows.min())
        dst_rows = rows + shift
        dst_cols = cols
    elif direction == "left":
        shift = int(cols.max() + 1)
        dst_rows = rows
        dst_cols = cols - shift
    else:
        shift = int(w - cols.min())
        dst_rows = rows
        dst_cols = cols + shift
    return [(int(r), int(c), int(color)) for r, c, (_sr, _sc, color) in zip(dst_rows, dst_cols, frags)]


def _room_footprint_bbox(room: dict) -> tuple[int, int, int, int]:
    moved = room["moved_fragments"]
    rows = [0, int(room["height"]) - 1]
    cols = [0, int(room["width"]) - 1]
    rows.extend(int(r) for r, _c, _color in moved)
    cols.extend(int(c) for _r, c, _color in moved)
    return min(rows), min(cols), max(rows), max(cols)


def _gen_room_spec(border_color: int, rng: random.Random) -> dict:
    height = rng.randint(*ROOM_OUTER_RANGE)
    width = rng.randint(*ROOM_OUTER_RANGE)
    direction = rng.choice(["top", "bottom", "left", "right"])

    if direction in {"top", "bottom"}:
        gap_rel = (0 if direction == "top" else height - 1, rng.randint(1, width - 2))
    else:
        gap_rel = (rng.randint(1, height - 2), 0 if direction == "left" else width - 1)

    inner_h = height - 2
    inner_w = width - 2
    shape = _gen_fragment_shape(inner_h, inner_w, rng)
    fragments = _assign_fragment_colors(shape, border_color, rng)
    room = {
        "border_color": border_color,
        "height": height,
        "width": width,
        "direction": direction,
        "gap_rel": gap_rel,
        "fragments": fragments,
    }
    room["moved_fragments"] = _moved_relative_fragments(room)
    room["foot_bbox_rel"] = _room_footprint_bbox(room)
    return room


def _candidate_positions(room: dict, shape: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape
    fr0, fc0, fr1, fc1 = room["foot_bbox_rel"]
    r_min = -fr0
    r_max = h - 1 - fr1
    c_min = -fc0
    c_max = w - 1 - fc1
    if r_min > r_max or c_min > c_max:
        return []

    positions = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            ar0 = r0 + fr0
            ac0 = c0 + fc0
            ar1 = r0 + fr1
            ac1 = c0 + fc1
            pr0 = max(0, ar0 - 1)
            pc0 = max(0, ac0 - 1)
            pr1 = min(h - 1, ar1 + 1)
            pc1 = min(w - 1, ac1 + 1)
            if not blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                positions.append((r0, c0))
    return positions


def _mark_blocked(room: dict, blocked: np.ndarray) -> None:
    h, w = blocked.shape
    fr0, fc0, fr1, fc1 = room["foot_bbox_rel"]
    ar0 = int(room["r0"]) + fr0
    ac0 = int(room["c0"]) + fc0
    ar1 = int(room["r0"]) + fr1
    ac1 = int(room["c0"]) + fc1
    pr0 = max(0, ar0 - 1)
    pc0 = max(0, ac0 - 1)
    pr1 = min(h - 1, ar1 + 1)
    pc1 = min(w - 1, ac1 + 1)
    blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _place_rooms(shape: tuple[int, int], rooms: list[dict], rng: random.Random) -> Optional[list[dict]]:
    blocked = np.zeros(shape, dtype=bool)
    placed: list[dict] = []

    for room in sorted(rooms, key=lambda spec: (spec["foot_bbox_rel"][2] - spec["foot_bbox_rel"][0] + 1) * (spec["foot_bbox_rel"][3] - spec["foot_bbox_rel"][1] + 1), reverse=True):
        positions = _candidate_positions(room, shape, blocked)
        if not positions:
            return None
        rng.shuffle(positions)
        chosen_r0, chosen_c0 = positions[0]
        placed_room = dict(room)
        placed_room["r0"] = chosen_r0
        placed_room["c0"] = chosen_c0
        placed.append(placed_room)
        _mark_blocked(placed_room, blocked)

    return placed


def _draw_room_border(grid: np.ndarray, room: dict) -> None:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["height"])
    w = int(room["width"])
    color = int(room["border_color"])
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    gap_r, gap_c = room["gap_rel"]
    grid[r0 + int(gap_r), c0 + int(gap_c)] = BG


def _render_input(shape: tuple[int, int], rooms: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for room in rooms:
        _draw_room_border(grid, room)
        for r, c, color in room["fragments"]:
            grid[int(room["r0"]) + 1 + int(r), int(room["c0"]) + 1 + int(c)] = int(color)
    return grid


def _render_output(shape: tuple[int, int], rooms: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for room in rooms:
        _draw_room_border(grid, room)
        for r, c, color in room["moved_fragments"]:
            grid[int(room["r0"]) + int(r), int(room["c0"]) + int(c)] = int(color)
    return grid


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    shape = (rng.randint(*GRID_SIDE_RANGE), rng.randint(*GRID_SIDE_RANGE))
    room_count = rng.choice(ROOM_COUNT_CHOICES)
    border_colors = rng.sample(range(1, 10), room_count)
    rooms = [_gen_room_spec(color, rng) for color in border_colors]
    placed = _place_rooms(shape, rooms, rng)
    if placed is None:
        return None

    inp = _render_input(shape, placed)
    expected = _render_output(shape, placed)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-push scene")
