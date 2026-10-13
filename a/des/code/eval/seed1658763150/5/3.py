from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
WALL_COLORS = tuple(color for color in range(10) if color not in {BLACK, GREY})
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _is_solid_rect(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _find_rooms(g: np.ndarray) -> list[dict]:
    wall_mask = (g != BLACK) & (g != GREY)
    rooms: list[dict] = []
    for component in _find_components(~wall_mask):
        if not _is_solid_rect(component):
            continue
        coords = component["coords"]
        values = g[coords[:, 0], coords[:, 1]]
        room = dict(component)
        room["grey_coords"] = coords[values == GREY]
        room["is_empty"] = bool(np.all(values == BLACK))
        rooms.append(room)
    assert rooms, "expected at least one rectangular room"
    return rooms


def _uniform_wall_color(segment: np.ndarray) -> int | None:
    values = np.unique(segment)
    if values.size != 1:
        return None
    color = int(values[0])
    if color in {BLACK, GREY}:
        return None
    return color


def _shared_wall(g: np.ndarray, a: dict, b: dict) -> dict | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if int(ac1) + 2 == int(bc0) or int(bc1) + 2 == int(ac0):
        overlap_r0 = max(int(ar0), int(br0))
        overlap_r1 = min(int(ar1), int(br1))
        if overlap_r0 <= overlap_r1:
            wall_col = int(ac1) + 1 if int(ac1) < int(bc0) else int(bc1) + 1
            color = _uniform_wall_color(g[overlap_r0 : overlap_r1 + 1, wall_col])
            if color is not None:
                return {"orientation": "vertical", "axis": wall_col, "color": int(color)}

    if int(ar1) + 2 == int(br0) or int(br1) + 2 == int(ar0):
        overlap_c0 = max(int(ac0), int(bc0))
        overlap_c1 = min(int(ac1), int(bc1))
        if overlap_c0 <= overlap_c1:
            wall_row = int(ar1) + 1 if int(ar1) < int(br0) else int(br1) + 1
            color = _uniform_wall_color(g[wall_row, overlap_c0 : overlap_c1 + 1])
            if color is not None:
                return {"orientation": "horizontal", "axis": wall_row, "color": int(color)}

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each grey room pattern into its unique adjacent empty room.

    Rooms are rectangular black interiors enclosed by non-black, non-grey wall
    pixels. For each room containing grey pixels, locate the unique neighboring
    room whose interior is fully black. Mirror the grey pattern across the
    shared wall: left/right neighbors use a left-right reflection and
    top/bottom neighbors use an up-down reflection. The copied pixels take the
    shared wall's color, while the source grey pattern and all walls stay
    unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _find_rooms(g)
    room_index = np.full(g.shape, -1, dtype=int)
    for idx, room in enumerate(rooms):
        coords = room["coords"]
        room_index[coords[:, 0], coords[:, 1]] = int(idx)

    adjacency: dict[int, list[dict]] = {idx: [] for idx in range(len(rooms))}
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            wall = _shared_wall(g, rooms[i], rooms[j])
            if wall is None:
                continue
            adjacency[i].append({"other": int(j), **wall})
            adjacency[j].append({"other": int(i), **wall})

    out = g.copy()
    for idx, room in enumerate(rooms):
        grey_coords = room["grey_coords"]
        if grey_coords.size == 0:
            continue

        candidates = [info for info in adjacency[idx] if rooms[info["other"]]["is_empty"]]
        assert len(candidates) == 1, "each patterned room must have a unique empty neighbor"
        info = candidates[0]
        target_idx = int(info["other"])
        axis = int(info["axis"])
        color = int(info["color"])

        for r, c in grey_coords:
            rr = int(r)
            cc = int(c)
            if info["orientation"] == "vertical":
                nr, nc = rr, 2 * axis - cc
            else:
                nr, nc = 2 * axis - rr, cc
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "reflection must stay inside the grid"
            assert room_index[nr, nc] == target_idx, "reflected pixel must land in the paired room"
            assert int(g[nr, nc]) == BLACK, "target room must start empty"
            assert int(out[nr, nc]) == BLACK, "different propagations must not overlap"
            out[nr, nc] = color

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
MODULE_COUNT_RANGE = (1, 4)
ROOM_SPAN_BY_COUNT = {
    1: (4, 8),
    2: (4, 7),
    3: (3, 6),
    4: (3, 5),
}
PATTERN_BOX_MAX = 5
PATTERN_CELLS_MIN = 3
PATTERN_CELLS_MAX = 7
PATTERN_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
MODULE_MARGIN = 1
MODULE_GAP = 1


def _grow_pattern(height: int, width: int, flip_kind: str) -> np.ndarray | None:
    if height < 2 or width < 2:
        return None

    max_cells = min(PATTERN_CELLS_MAX, height * width - 1)
    if max_cells < PATTERN_CELLS_MIN:
        return None

    for _ in range(PATTERN_ATTEMPTS):
        target = random.randint(PATTERN_CELLS_MIN, max_cells)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}

        while len(cells) < target:
            frontier = sorted(
                {
                    (r + dr, c + dc)
                    for r, c in cells
                    for dr, dc in DIRS4
                    if 0 <= r + dr < height and 0 <= c + dc < width and (r + dr, c + dc) not in cells
                }
            )
            if not frontier:
                break
            weights = [1 + sum((fr + dr, fc + dc) in cells for dr, dc in DIRS4) for fr, fc in frontier]
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != target:
            continue

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        flipped = np.fliplr(mask) if flip_kind == "lr" else np.flipud(mask)
        if np.array_equal(mask, flipped):
            continue
        return mask

    return None


def _sample_module(module_count: int) -> dict | None:
    lo, hi = ROOM_SPAN_BY_COUNT[module_count]
    orientation = random.choice(("vertical", "horizontal"))
    divider_color = random.choice(WALL_COLORS)
    frame_pool = [color for color in WALL_COLORS if color != divider_color]
    frame_colors = random.sample(frame_pool, 4)

    if orientation == "vertical":
        room_h = random.randint(lo + 1, hi + 1)
        left_w = random.randint(lo, hi)
        right_w = random.randint(lo, hi)
        source_side = random.choice(("left", "right"))

        source_w = left_w if source_side == "left" else right_w
        target_w = right_w if source_side == "left" else left_w
        usable_w = min(source_w, target_w)
        ph_max = min(room_h, PATTERN_BOX_MAX)
        pw_max = min(usable_w, PATTERN_BOX_MAX)
        if ph_max < 2 or pw_max < 2:
            return None

        pattern_h = random.randint(2, ph_max)
        pattern_w = random.randint(2, pw_max)
        pattern = _grow_pattern(pattern_h, pattern_w, "lr")
        if pattern is None:
            return None

        row0 = random.randint(0, room_h - pattern_h)
        if source_side == "left":
            col_lo = source_w - usable_w
            col_hi = source_w - pattern_w
        else:
            col_lo = 0
            col_hi = usable_w - pattern_w
        col0 = random.randint(col_lo, col_hi)

        return {
            "orientation": orientation,
            "room_h": room_h,
            "left_w": left_w,
            "right_w": right_w,
            "source_side": source_side,
            "divider_color": divider_color,
            "frame_colors": frame_colors,
            "pattern": pattern,
            "pattern_r0": row0,
            "pattern_c0": col0,
            "shape": (room_h + 2, left_w + right_w + 3),
        }

    top_h = random.randint(lo, hi)
    bottom_h = random.randint(lo, hi)
    room_w = random.randint(lo + 1, hi + 2)
    source_side = random.choice(("top", "bottom"))

    source_h = top_h if source_side == "top" else bottom_h
    target_h = bottom_h if source_side == "top" else top_h
    usable_h = min(source_h, target_h)
    ph_max = min(usable_h, PATTERN_BOX_MAX)
    pw_max = min(room_w, PATTERN_BOX_MAX)
    if ph_max < 2 or pw_max < 2:
        return None

    pattern_h = random.randint(2, ph_max)
    pattern_w = random.randint(2, pw_max)
    pattern = _grow_pattern(pattern_h, pattern_w, "ud")
    if pattern is None:
        return None

    if source_side == "top":
        row_lo = source_h - usable_h
        row_hi = source_h - pattern_h
    else:
        row_lo = 0
        row_hi = usable_h - pattern_h
    row0 = random.randint(row_lo, row_hi)
    col0 = random.randint(0, room_w - pattern_w)

    return {
        "orientation": orientation,
        "top_h": top_h,
        "bottom_h": bottom_h,
        "room_w": room_w,
        "source_side": source_side,
        "divider_color": divider_color,
        "frame_colors": frame_colors,
        "pattern": pattern,
        "pattern_r0": row0,
        "pattern_c0": col0,
        "shape": (top_h + bottom_h + 3, room_w + 2),
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_modules(shape: tuple[int, int], modules: list[dict]) -> list[tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(modules)), key=lambda idx: modules[idx]["shape"][0] * modules[idx]["shape"][1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(modules)
    padded_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        mh, mw = modules[idx]["shape"]
        if h < mh + 2 * MODULE_MARGIN or w < mw + 2 * MODULE_MARGIN:
            return None

        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(MODULE_MARGIN, h - mh - MODULE_MARGIN)
            c0 = random.randint(MODULE_MARGIN, w - mw - MODULE_MARGIN)
            r1 = r0 + mh - 1
            c1 = c0 + mw - 1
            padded = (
                max(0, r0 - MODULE_GAP),
                max(0, c0 - MODULE_GAP),
                min(h - 1, r1 + MODULE_GAP),
                min(w - 1, c1 + MODULE_GAP),
            )
            if any(_boxes_overlap(padded, other) for other in padded_boxes):
                continue
            found = (r0, c0)
            padded_boxes.append(padded)
            break

        if found is None:
            return None
        placed[idx] = found

    if any(pos is None for pos in placed):
        return None
    return [tuple(map(int, pos)) for pos in placed]


def _draw_module(inp: np.ndarray, out: np.ndarray, module: dict, top: int, left: int) -> None:
    mh, mw = module["shape"]
    r0, c0 = int(top), int(left)
    r1, c1 = r0 + mh - 1, c0 + mw - 1
    top_color, bottom_color, left_color, right_color = module["frame_colors"]

    for grid in (inp, out):
        grid[r0, c0 : c1 + 1] = int(top_color)
        grid[r1, c0 : c1 + 1] = int(bottom_color)
        grid[r0 : r1 + 1, c0] = int(left_color)
        grid[r0 : r1 + 1, c1] = int(right_color)

    pattern = module["pattern"]
    divider_color = int(module["divider_color"])
    pattern_cells = np.argwhere(pattern)

    if module["orientation"] == "vertical":
        room_h = int(module["room_h"])
        left_w = int(module["left_w"])
        right_w = int(module["right_w"])
        divider_col = c0 + left_w + 1
        for grid in (inp, out):
            grid[r0 + 1 : r0 + room_h + 1, divider_col] = divider_color

        if module["source_side"] == "left":
            source_left = c0 + 1
        else:
            source_left = divider_col + 1
        source_top = r0 + 1

        for dr, dc in pattern_cells:
            rr = source_top + int(module["pattern_r0"]) + int(dr)
            cc = source_left + int(module["pattern_c0"]) + int(dc)
            inp[rr, cc] = GREY
            out[rr, cc] = GREY
            out[rr, 2 * divider_col - cc] = divider_color
        return

    top_h = int(module["top_h"])
    divider_row = r0 + top_h + 1
    room_w = int(module["room_w"])
    for grid in (inp, out):
        grid[divider_row, c0 + 1 : c0 + room_w + 1] = divider_color

    if module["source_side"] == "top":
        source_top = r0 + 1
    else:
        source_top = divider_row + 1
    source_left = c0 + 1

    for dr, dc in pattern_cells:
        rr = source_top + int(module["pattern_r0"]) + int(dr)
        cc = source_left + int(module["pattern_c0"]) + int(dc)
        inp[rr, cc] = GREY
        out[rr, cc] = GREY
        out[2 * divider_row - rr, cc] = divider_color


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    module_count = random.randint(*MODULE_COUNT_RANGE)
    modules = []
    for _ in range(module_count):
        module = _sample_module(module_count)
        if module is None:
            return None
        modules.append(module)

    placements = _place_modules((grid_h, grid_w), modules)
    if placements is None:
        return None

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = inp.copy()
    for module, (r0, c0) in zip(modules, placements):
        _draw_module(inp, out, module, r0, c0)

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
