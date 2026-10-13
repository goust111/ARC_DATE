import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
ARROW = 2
EXIT = 5
ARROW_EXIT = 6
OBJECT_COLORS = (3, 4, 7, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# The missing 2x2 corner is read in clockwise order as top, right, bottom, left.
MISSING_TO_DIR = {
    (1, 1): "up",
    (0, 0): "down",
    (0, 1): "left",
    (1, 0): "right",
}

ARROW_SHAPES = {
    "up": [(0, 0), (0, 1), (1, 0)],
    "down": [(0, 1), (1, 0), (1, 1)],
    "left": [(0, 0), (1, 0), (1, 1)],
    "right": [(0, 0), (0, 1), (1, 1)],
}

BASE_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
]

ROOM_RETRIES = 200
GRID_RETRIES = 300
PLACEMENT_RETRIES = 100


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _normalize_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _rotate_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h = max(r for r, _ in cells) + 1
    return _normalize_shape([(c, h - 1 - r) for r, c in cells])


def _shape_variants(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []
    cur = _normalize_shape(cells)
    for _ in range(4):
        if cur not in variants:
            variants.append(cur)
        cur = _rotate_shape(cur)
    return variants


SHAPE_VARIANTS = [variant for base in BASE_SHAPES for variant in _shape_variants(base)]


def _sort_key_for_push(
    cells: list[tuple[int, int]],
    direction: str,
    region_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = region_bbox
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if direction == "up":
        return (min(rows) - r0, min(rows), min(cols))
    if direction == "down":
        return (r1 - max(rows), -max(rows), min(cols))
    if direction == "left":
        return (min(cols) - c0, min(cols), min(rows))
    return (c1 - max(cols), -max(cols), min(rows))


def _arrow_direction(arrow_cells: list[tuple[int, int]]) -> str:
    assert len(arrow_cells) == 3
    r0, c0, r1, c1 = _bbox(arrow_cells)
    assert r1 - r0 == 1 and c1 - c0 == 1
    local = {(r - r0, c - c0) for r, c in arrow_cells}
    missing = next((rr, cc) for rr in range(2) for cc in range(2) if (rr, cc) not in local)
    return MISSING_TO_DIR[missing]


def _find_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    passable = (grid != WALL) & (grid != EXIT)
    regions: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for comp in _component_cells(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _transform_region(
    grid: np.ndarray,
    out: np.ndarray,
    region_cells: list[tuple[int, int]],
) -> dict[str, object]:
    region_set = set(region_cells)
    region_bbox = _bbox(region_cells)
    region_mask = np.zeros_like(grid, dtype=bool)
    for r, c in region_cells:
        region_mask[r, c] = True

    arrow_mask = region_mask & (grid == ARROW)
    arrow_components = _component_cells(arrow_mask)
    assert len(arrow_components) == 1
    arrow_cells = sorted(arrow_components[0])
    direction = _arrow_direction(arrow_cells)
    dr, dc = DIRS[direction]

    exit_present = False
    for r, c in region_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not region_mask[nr, nc]:
                if int(grid[nr, nc]) == EXIT:
                    exit_present = True
                    break
        if exit_present:
            break

    object_mask = region_mask & (grid != BG) & (grid != ARROW)
    objects = []
    for comp in _component_cells(object_mask):
        cells = sorted(comp)
        values = [int(grid[r, c]) for r, c in cells]
        objects.append({"cells": cells, "values": values})

    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = BG

    occupied = set(arrow_cells)
    ordered = sorted(objects, key=lambda obj: _sort_key_for_push(obj["cells"], direction, region_bbox))
    transformed_objects = []
    for obj in ordered:
        current = list(obj["cells"])
        while True:
            trial = _shift_cells(current, dr, dc)
            if all((rr, cc) in region_set and (rr, cc) not in occupied for rr, cc in trial):
                current = trial
            else:
                break
        for (rr, cc), value in zip(current, obj["values"]):
            out[rr, cc] = value
        occupied.update(current)
        transformed_objects.append(
            {
                "input_cells": obj["cells"],
                "output_cells": current,
                "values": obj["values"],
            }
        )

    arrow_color = ARROW_EXIT if exit_present else ARROW
    for r, c in arrow_cells:
        out[r, c] = arrow_color

    return {
        "direction": direction,
        "exit_present": exit_present,
        "region_set": region_set,
        "arrow_cells": arrow_cells,
        "objects": transformed_objects,
    }


def _room_stacking_used(meta: dict[str, object]) -> bool:
    direction = str(meta["direction"])
    dr, dc = DIRS[direction]
    region_set = set(meta["region_set"])
    arrow_cells = set(meta["arrow_cells"])
    object_cells = [set(obj["output_cells"]) for obj in meta["objects"]]
    for i, cells in enumerate(object_cells):
        trial = {(r + dr, c + dc) for r, c in cells}
        if any(cell not in region_set for cell in trial):
            continue
        if trial & arrow_cells:
            continue
        if any(trial & other for j, other in enumerate(object_cells) if j != i):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push every non-arrow object inside each walled room toward the wall indicated by the room's red L marker.

    Input structure:
    - The background is black.
    - Blue pixels form one-pixel-thick rectangular room walls; some wall cells may be replaced by a single grey exit marker.
    - Each enclosed room contains one red 3-pixel L shape in a 2x2 box and 1-3 other small colored connected objects.

    Rule:
    - Treat blue and grey wall cells as fixed room boundaries; each non-border passable component is one room interior.
    - In each room, read the red L marker by its missing 2x2 corner, using the clockwise side order top/right/bottom/left, and push in the opposite cardinal direction.
    - Move all non-arrow colored objects in that direction, processing them from nearest to the target wall to farthest so they stack against the wall or against already moved objects.
    - Keep the red marker fixed in place, recoloring it to magenta iff that room has any grey exit cell on its surrounding wall.

    Key insight:
    - The L marker does not move; its missing corner encodes a push direction, the objects compact in directional order, and the separate grey exit feature controls only the marker recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region_cells in _find_regions(g):
        _transform_region(g, out, region_cells)
    return out


def _shape_dims(shape: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _preferred_arrow_positions(ih: int, iw: int, direction: str) -> list[tuple[int, int]]:
    center = (max(0, ih // 2 - 1), max(0, iw // 2 - 1))
    corners = {
        "up": [(ih - 2, 0), (ih - 2, iw - 2), center, (0, 0), (0, iw - 2)],
        "down": [(0, 0), (0, iw - 2), center, (ih - 2, 0), (ih - 2, iw - 2)],
        "left": [(0, iw - 2), (ih - 2, iw - 2), center, (0, 0), (ih - 2, 0)],
        "right": [(0, 0), (ih - 2, 0), center, (0, iw - 2), (ih - 2, iw - 2)],
    }
    seen = set()
    valid = []
    for r, c in corners[direction]:
        r = min(max(r, 0), ih - 2)
        c = min(max(c, 0), iw - 2)
        if (r, c) not in seen:
            valid.append((r, c))
            seen.add((r, c))
    return valid


def _touches_forbidden(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _choose_shape(rng: random.Random, ih: int, iw: int) -> list[tuple[int, int]]:
    variants = SHAPE_VARIANTS[:]
    rng.shuffle(variants)
    for shape in variants:
        sh, sw = _shape_dims(shape)
        if sh <= ih and sw <= iw:
            return shape
    raise RuntimeError("no shape fits room interior")


def _room_meta(room: np.ndarray) -> dict[str, object]:
    out = room.copy()
    regions = _find_regions(room)
    assert len(regions) == 1
    meta = _transform_region(room, out, regions[0])
    meta["output"] = out
    meta["stacking_used"] = _room_stacking_used(meta)
    return meta


def _generate_room(
    rng: random.Random,
    direction: str,
    exit_present: bool,
    object_count: int,
    require_stacking: bool,
) -> np.ndarray | None:
    for _ in range(ROOM_RETRIES):
        outer_h = rng.randint(8, 11)
        outer_w = rng.randint(8, 11)
        ih, iw = outer_h - 2, outer_w - 2
        room = np.full((outer_h, outer_w), BG, dtype=int)
        room[0, :] = WALL
        room[-1, :] = WALL
        room[:, 0] = WALL
        room[:, -1] = WALL

        if exit_present:
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"}:
                c = rng.randint(1, outer_w - 2)
                room[0 if side == "top" else outer_h - 1, c] = EXIT
            else:
                r = rng.randint(1, outer_h - 2)
                room[r, 0 if side == "left" else outer_w - 1] = EXIT

        arrow_positions = _preferred_arrow_positions(ih, iw, direction)
        rng.shuffle(arrow_positions)
        arrow_r, arrow_c = arrow_positions[0]
        arrow_global = []
        for dr, dc in ARROW_SHAPES[direction]:
            rr = 1 + arrow_r + dr
            cc = 1 + arrow_c + dc
            room[rr, cc] = ARROW
            arrow_global.append((rr, cc))

        occupied = set(arrow_global)
        placed_objects: list[list[tuple[int, int]]] = []
        colors = list(OBJECT_COLORS)
        rng.shuffle(colors)
        stack_bias = object_count > 1 and (require_stacking or rng.random() < 0.75)
        anchor_row = None
        anchor_col = None

        ok = True
        for idx in range(object_count):
            color = colors[idx]
            shape = _choose_shape(rng, ih, iw)
            sh, sw = _shape_dims(shape)

            candidates = []
            for _ in range(PLACEMENT_RETRIES):
                if direction in {"left", "right"} and stack_bias and anchor_row is not None:
                    lo = max(0, anchor_row - 1)
                    hi = min(ih - sh, anchor_row + 1)
                    top = rng.randint(lo, hi) if lo <= hi else rng.randint(0, ih - sh)
                    left = rng.randint(0, iw - sw)
                elif direction in {"up", "down"} and stack_bias and anchor_col is not None:
                    top = rng.randint(0, ih - sh)
                    lo = max(0, anchor_col - 1)
                    hi = min(iw - sw, anchor_col + 1)
                    left = rng.randint(lo, hi) if lo <= hi else rng.randint(0, iw - sw)
                else:
                    top = rng.randint(0, ih - sh)
                    left = rng.randint(0, iw - sw)

                cells = [(1 + top + r, 1 + left + c) for r, c in shape]
                candidates.append(cells)

            placed = False
            rng.shuffle(candidates)
            for cells in candidates:
                if _touches_forbidden(cells, occupied, room.shape):
                    continue
                for r, c in cells:
                    room[r, c] = color
                occupied.update(cells)
                placed_objects.append(cells)
                top_rows = [r for r, _ in cells]
                left_cols = [c for _, c in cells]
                if anchor_row is None:
                    anchor_row = min(top_rows) - 1
                    anchor_col = min(left_cols) - 1
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        meta = _room_meta(room)
        if len(meta["objects"]) != object_count:
            continue
        if not all(set(obj["input_cells"]) != set(obj["output_cells"]) for obj in meta["objects"]):
            continue
        if bool(meta["exit_present"]) != exit_present:
            continue
        if require_stacking and not bool(meta["stacking_used"]):
            continue
        if np.array_equal(room, meta["output"]):
            continue
        return room
    return None


def _can_place_room(
    room: np.ndarray,
    top: int,
    left: int,
    occupied: np.ndarray,
) -> bool:
    h, w = room.shape
    H, W = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_room(grid: np.ndarray, room: np.ndarray, top: int, left: int, occupied: np.ndarray) -> None:
    h, w = room.shape
    grid[top : top + h, left : left + w] = room
    occupied[top : top + h, left : left + w] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(22, 30)
    W = rng.randint(22, 30)
    room_count = rng.randint(2, 4)

    directions = rng.sample(list(DIRS), k=room_count)
    exit_count = room_count // 2 if rng.random() < 0.5 else (room_count + 1) // 2
    exit_flags = [True] * exit_count + [False] * (room_count - exit_count)
    rng.shuffle(exit_flags)

    object_counts = [rng.randint(1, 3) for _ in range(room_count)]
    if max(object_counts) == 1:
        object_counts[rng.randrange(room_count)] = rng.randint(2, 3)

    require_stack_index = None
    multi_rooms = [i for i, count in enumerate(object_counts) if count > 1]
    if multi_rooms:
        require_stack_index = rng.choice(multi_rooms)

    rooms = []
    for i in range(room_count):
        room = _generate_room(
            rng,
            direction=directions[i],
            exit_present=exit_flags[i],
            object_count=object_counts[i],
            require_stacking=i == require_stack_index,
        )
        if room is None:
            return None
        rooms.append(room)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    order = sorted(range(room_count), key=lambda i: rooms[i].shape[0] * rooms[i].shape[1], reverse=True)
    for idx in order:
        room = rooms[idx]
        h, w = room.shape
        placed = False
        for _ in range(200):
            top = rng.randint(0, H - h)
            left = rng.randint(0, W - w)
            if _can_place_room(room, top, left, occupied):
                _place_room(grid, room, top, left, occupied)
                placed = True
                break
        if not placed:
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    regions = _find_regions(grid)
    if len(regions) != room_count:
        return None

    seen_exit = any(flag for flag in exit_flags)
    seen_no_exit = any(not flag for flag in exit_flags)
    if not (seen_exit and seen_no_exit):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GRID_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-push ARC input")
