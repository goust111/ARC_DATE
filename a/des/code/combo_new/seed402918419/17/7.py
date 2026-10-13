from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_TRAIL_COLOR = 9

_GEN_MAX_ATTEMPTS = 500
_GEN_GRID_H_RANGE = (18, 29)
_GEN_GRID_W_RANGE = (18, 29)
_GEN_ROOM_COUNT_RANGE = (2, 5)
_GEN_INNER_H_RANGE = (4, 8)
_GEN_INNER_W_RANGE = (4, 8)
_GEN_OBJECT_H_RANGE = (1, 3)
_GEN_OBJECT_W_RANGE = (1, 3)
_GEN_ROOM_TRIES = 300
_GEN_PALETTE = tuple(color for color in range(10) if color != _TRAIL_COLOR)
_SIDES = ("up", "down", "left", "right")


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            for rr, cc in queue:
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _contiguous_run(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    if int(idx[-1]) - int(idx[0]) + 1 != int(idx.size):
        return None
    return int(idx[0]), int(idx[-1])


def _gap_on_line(line: np.ndarray, divider: int, bg: int) -> tuple[int, int] | None:
    arr = np.asarray(line, dtype=int)
    if arr.size < 3 or int(arr[0]) != divider or int(arr[-1]) != divider:
        return None
    inner = arr[1:-1]
    if np.any((inner != divider) & (inner != bg)):
        return None
    run = _contiguous_run(inner != divider)
    if run is None:
        return None
    return run[0] + 1, run[1] + 1


def _divider_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one foreground color"
    best_score: tuple[int, int] | None = None
    best_color = colors[0]
    for color in colors:
        mask = grid == color
        h, w = mask.shape
        line_cells = 0
        for r, c in np.argwhere(mask):
            rr, cc = int(r), int(c)
            has_aligned_neighbor = (
                (rr > 0 and bool(mask[rr - 1, cc]))
                or (rr + 1 < h and bool(mask[rr + 1, cc]))
                or (cc > 0 and bool(mask[rr, cc - 1]))
                or (cc + 1 < w and bool(mask[rr, cc + 1]))
            )
            if has_aligned_neighbor:
                line_cells += 1
        score = (line_cells, int(np.count_nonzero(mask)))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    return int(best_color)


def _room_from_component(grid: np.ndarray, bg: int, divider: int, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    boundary = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if any((r, c) not in boundary for r, c in coords):
        return None

    top_gap = _gap_on_line(grid[r0, c0 : c1 + 1], divider, bg)
    bottom_gap = _gap_on_line(grid[r1, c0 : c1 + 1], divider, bg)
    left_gap = _gap_on_line(grid[r0 : r1 + 1, c0], divider, bg)
    right_gap = _gap_on_line(grid[r0 : r1 + 1, c1], divider, bg)
    side_to_gap = {
        "up": top_gap,
        "down": bottom_gap,
        "left": left_gap,
        "right": right_gap,
    }
    gap_sides = [(side, gap) for side, gap in side_to_gap.items() if gap is not None]
    if len(gap_sides) != 1:
        return None

    for side, gap in side_to_gap.items():
        if gap is not None:
            continue
        if side == "up" and not np.all(grid[r0, c0 : c1 + 1] == divider):
            return None
        if side == "down" and not np.all(grid[r1, c0 : c1 + 1] == divider):
            return None
        if side == "left" and not np.all(grid[r0 : r1 + 1, c0] == divider):
            return None
        if side == "right" and not np.all(grid[r0 : r1 + 1, c1] == divider):
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == divider):
        return None

    object_mask = (interior != bg) & (interior != divider)
    object_comps = _components(object_mask)
    if len(object_comps) != 1:
        return None

    obj_local = object_comps[0]
    obj_rows = [r for r, _ in obj_local]
    obj_cols = [c for _, c in obj_local]
    olr0, olc0, olr1, olc1 = min(obj_rows), min(obj_cols), max(obj_rows), max(obj_cols)
    obj_h = olr1 - olr0 + 1
    obj_w = olc1 - olc0 + 1
    obj_patch = interior[olr0 : olr1 + 1, olc0 : olc1 + 1]
    obj_color = int(obj_patch[0, 0])
    if not np.all(obj_patch == obj_color):
        return None
    if int(np.count_nonzero(object_mask)) != obj_h * obj_w:
        return None

    obj_r0 = r0 + 1 + olr0
    obj_c0 = c0 + 1 + olc0
    obj_r1 = obj_r0 + obj_h - 1
    obj_c1 = obj_c0 + obj_w - 1

    side, (gap_start, gap_end) = gap_sides[0]
    if side in ("up", "down"):
        if (obj_c0, obj_c1) != (c0 + gap_start, c0 + gap_end):
            return None
    else:
        if (obj_r0, obj_r1) != (r0 + gap_start, r0 + gap_end):
            return None

    return {
        "room_bbox": (r0, c0, r1, c1),
        "object_bbox": (obj_r0, obj_c0, obj_r1, obj_c1),
        "color": obj_color,
        "direction": side,
    }


def _find_rooms(grid: np.ndarray, bg: int, divider: int) -> list[dict]:
    rooms = []
    for comp in _components(grid == divider):
        room = _room_from_component(grid, bg, divider, comp)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: room["room_bbox"])
    return rooms


def _translated_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _max_slide_steps(
    grid: np.ndarray,
    bg: int,
    object_bbox: tuple[int, int, int, int],
    direction: str,
) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = object_bbox
    base = grid.copy()
    base[r0 : r1 + 1, c0 : c1 + 1] = bg
    dr, dc = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]

    steps = 0
    while True:
        nr0, nc0, nr1, nc1 = _translated_bbox(object_bbox, dr * (steps + 1), dc * (steps + 1))
        if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
            break
        if np.any(base[nr0 : nr1 + 1, nc0 : nc1 + 1] != bg):
            break
        steps += 1
    return steps


def _paint_trail(
    out: np.ndarray,
    start_bbox: tuple[int, int, int, int],
    final_bbox: tuple[int, int, int, int],
    direction: str,
) -> None:
    sr0, sc0, sr1, sc1 = start_bbox
    fr0, fc0, fr1, fc1 = final_bbox
    if direction == "up" and fr1 + 1 < sr0:
        out[fr1 + 1 : sr0, sc0 : sc1 + 1] = _TRAIL_COLOR
    elif direction == "down" and sr1 + 1 < fr0:
        out[sr1 + 1 : fr0, sc0 : sc1 + 1] = _TRAIL_COLOR
    elif direction == "left" and fc1 + 1 < sc0:
        out[sr0 : sr1 + 1, fc1 + 1 : sc0] = _TRAIL_COLOR
    elif direction == "right" and sc1 + 1 < fc0:
        out[sr0 : sr1 + 1, sc1 + 1 : fc0] = _TRAIL_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored room object out through its unique wall gap and leave a maroon trail behind it.

    Input structure:
    - The grid has a dominant background color, one divider color used for one-cell-thick horizontal and vertical room walls, and several smaller colored rectangular room objects.
    - Each room is a rectangular divider outline containing exactly one solid colored rectangle.
    - Exactly one wall of each room has a contiguous gap whose length matches the object's span on that wall.

    Rule:
    - Keep all divider lines unchanged.
    - For each room, detect the unique gapped wall to determine the slide direction.
    - Move the room's colored rectangle straight through that gap until the next step would leave the grid or hit another divider line.
    - Clear the object's original position to background, and fill the path between the old and new positions with maroon while leaving the final resting object unchanged.

    Key insight:
    - The opening in the room wall is the only directional cue: it both selects the motion axis and defines the corridor the object uses to escape before stopping at the next structural barrier.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    divider = _divider_color(g, bg)
    rooms = _find_rooms(g, bg, divider)
    assert rooms, "expected at least one valid room"

    out = g.copy()
    for room in rooms:
        object_bbox = room["object_bbox"]
        color = int(room["color"])
        direction = room["direction"]
        steps = _max_slide_steps(g, bg, object_bbox, direction)
        assert steps > 0, "room object must be able to slide through its gap"

        dr, dc = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[direction]
        final_bbox = _translated_bbox(object_bbox, dr * steps, dc * steps)

        r0, c0, r1, c1 = object_bbox
        fr0, fc0, fr1, fc1 = final_bbox

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        _paint_trail(out, object_bbox, final_bbox, direction)
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = color

    return out


def _side_vector(side: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _line_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _sample_room_geometry(rng: random.Random, side: str) -> dict:
    for _ in range(120):
        obj_h = rng.randint(*_GEN_OBJECT_H_RANGE)
        obj_w = rng.randint(*_GEN_OBJECT_W_RANGE)

        inner_h_min = max(_GEN_INNER_H_RANGE[0], obj_h + 3)
        inner_h_max = _GEN_INNER_H_RANGE[1]
        inner_w_min = max(_GEN_INNER_W_RANGE[0], obj_w + 3)
        inner_w_max = _GEN_INNER_W_RANGE[1]
        if inner_h_min > inner_h_max or inner_w_min > inner_w_max:
            continue

        inner_h = rng.randint(inner_h_min, inner_h_max)
        inner_w = rng.randint(inner_w_min, inner_w_max)
        obj_y = rng.randint(1, inner_h - obj_h - 1)
        obj_x = rng.randint(1, inner_w - obj_w - 1)

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        obj_r0 = 1 + obj_y
        obj_c0 = 1 + obj_x
        obj_r1 = obj_r0 + obj_h - 1
        obj_c1 = obj_c0 + obj_w - 1

        gap = (obj_c0, obj_c1) if side in ("up", "down") else (obj_r0, obj_r1)
        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "obj_bbox_rel": (obj_r0, obj_c0, obj_r1, obj_c1),
            "gap": gap,
        }
    raise RuntimeError("failed to sample room geometry")


def _room_feature_cells(
    top: int,
    left: int,
    geometry: dict,
    side: str,
    divider: int,
    color: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    patch = np.full((geometry["outer_h"], geometry["outer_w"]), bg, dtype=int)
    patch[0, :] = divider
    patch[-1, :] = divider
    patch[:, 0] = divider
    patch[:, -1] = divider

    gap0, gap1 = geometry["gap"]
    if side == "up":
        patch[0, gap0 : gap1 + 1] = bg
    elif side == "down":
        patch[-1, gap0 : gap1 + 1] = bg
    elif side == "left":
        patch[gap0 : gap1 + 1, 0] = bg
    else:
        patch[gap0 : gap1 + 1, -1] = bg

    obj_r0, obj_c0, obj_r1, obj_c1 = geometry["obj_bbox_rel"]
    patch[obj_r0 : obj_r1 + 1, obj_c0 : obj_c1 + 1] = color
    return patch, (top + obj_r0, left + obj_c0, top + obj_r1, left + obj_c1)


def _blocker_cells(
    h: int,
    w: int,
    side: str,
    object_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    sr0, sc0, sr1, sc1 = object_bbox
    rr0, rc0, rr1, rc1 = room_bbox

    if side == "up":
        hi = min(rr0 - 2, sr0 - 4)
        if hi < 0:
            return None
        row = rng.randint(0, hi)
        extra_l = rng.randint(1, min(3, sc0))
        extra_r = rng.randint(1, min(3, w - 1 - sc1))
        return [(row, c) for c in range(sc0 - extra_l, sc1 + extra_r + 1)]

    if side == "down":
        lo = max(rr1 + 2, sr1 + 4)
        if lo >= h:
            return None
        row = rng.randint(lo, h - 1)
        extra_l = rng.randint(1, min(3, sc0))
        extra_r = rng.randint(1, min(3, w - 1 - sc1))
        return [(row, c) for c in range(sc0 - extra_l, sc1 + extra_r + 1)]

    if side == "left":
        hi = min(rc0 - 2, sc0 - 4)
        if hi < 0:
            return None
        col = rng.randint(0, hi)
        extra_u = rng.randint(1, min(3, sr0))
        extra_d = rng.randint(1, min(3, h - 1 - sr1))
        return [(r, col) for r in range(sr0 - extra_u, sr1 + extra_d + 1)]

    lo = max(rc1 + 2, sc1 + 4)
    if lo >= w:
        return None
    col = rng.randint(lo, w - 1)
    extra_u = rng.randint(1, min(3, sr0))
    extra_d = rng.randint(1, min(3, h - 1 - sr1))
    return [(r, col) for r in range(sr0 - extra_u, sr1 + extra_d + 1)]


def _sweep_bbox(
    start_bbox: tuple[int, int, int, int],
    final_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    sr0, sc0, sr1, sc1 = start_bbox
    fr0, fc0, fr1, fc1 = final_bbox
    return min(sr0, fr0), min(sc0, fc0), max(sr1, fr1), max(sc1, fc1)


def _place_room(
    grid: np.ndarray,
    reserved: np.ndarray,
    side: str,
    stop_kind: str,
    divider: int,
    color: int,
    bg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    geometry = _sample_room_geometry(rng, side)

    for _ in range(_GEN_ROOM_TRIES):
        top = rng.randint(0, h - geometry["outer_h"])
        left = rng.randint(0, w - geometry["outer_w"])
        room_bbox = (top, left, top + geometry["outer_h"] - 1, left + geometry["outer_w"] - 1)
        patch, object_bbox = _room_feature_cells(top, left, geometry, side, divider, color, bg)
        sr0, sc0, sr1, sc1 = object_bbox

        if stop_kind == "boundary":
            if side == "up":
                if sr0 < 4:
                    continue
                final_bbox = (0, sc0, sr1 - sr0, sc1)
            elif side == "down":
                if h - 1 - sr1 < 4:
                    continue
                final_bbox = (h - (sr1 - sr0 + 1), sc0, h - 1, sc1)
            elif side == "left":
                if sc0 < 4:
                    continue
                final_bbox = (sr0, 0, sr1, sc1 - sc0)
            else:
                if w - 1 - sc1 < 4:
                    continue
                final_bbox = (sr0, w - (sc1 - sc0 + 1), sr1, w - 1)
            blocker = []
        else:
            blocker = _blocker_cells(h, w, side, object_bbox, room_bbox, rng)
            if blocker is None:
                continue
            if side == "up":
                row = blocker[0][0]
                final_bbox = (row + 1, sc0, row + 1 + (sr1 - sr0), sc1)
            elif side == "down":
                row = blocker[0][0]
                final_bbox = (row - (sr1 - sr0) - 1, sc0, row - 1, sc1)
            elif side == "left":
                col = blocker[0][1]
                final_bbox = (sr0, col + 1, sr1, col + 1 + (sc1 - sc0))
            else:
                col = blocker[0][1]
                final_bbox = (sr0, col - (sc1 - sc0) - 1, sr1, col - 1)

        steps = max(abs(final_bbox[0] - sr0), abs(final_bbox[1] - sc0))
        if steps < 4:
            continue

        feature_cells = {(top + r, left + c) for r, c in np.argwhere(patch != bg)}
        blocker_cells = set(blocker)
        sweep_cells = _expand_bbox(_sweep_bbox(object_bbox, final_bbox), 1, h, w)
        footprint = set()
        footprint |= _expand_bbox(room_bbox, 1, h, w)
        footprint |= sweep_cells
        if blocker_cells:
            footprint |= _expand_bbox(_line_bbox(blocker), 1, h, w)

        if any(bool(reserved[r, c]) for r, c in footprint):
            continue

        temp = grid.copy()
        for r, c in feature_cells:
            temp[r, c] = int(patch[r - top, c - left])
        for r, c in blocker_cells:
            temp[r, c] = divider

        bg_guess = _most_common_color(temp)
        divider_guess = _divider_color(temp, bg_guess)
        rooms = _find_rooms(temp, bg_guess, divider_guess)
        if sum(int(room["color"] == color) for room in rooms) == 0:
            continue

        for r, c in feature_cells:
            grid[r, c] = int(patch[r - top, c - left])
        for r, c in blocker_cells:
            grid[r, c] = divider
        for r, c in footprint:
            reserved[r, c] = True
        return True

    return False


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*_GEN_GRID_H_RANGE)
    w = rng.randint(*_GEN_GRID_W_RANGE)
    bg, divider = rng.sample(_GEN_PALETTE, 2)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    room_count = rng.randint(*_GEN_ROOM_COUNT_RANGE)
    colors = [color for color in _GEN_PALETTE if color not in {bg, divider}]
    if len(colors) < room_count:
        return None
    rng.shuffle(colors)

    stop_kinds = ["boundary", "line"] * ((room_count + 1) // 2)
    rng.shuffle(stop_kinds)
    stop_kinds = stop_kinds[:room_count]
    if room_count >= 2 and len(set(stop_kinds)) == 1:
        stop_kinds[0] = "line" if stop_kinds[0] == "boundary" else "boundary"

    directions = [rng.choice(_SIDES) for _ in range(room_count)]

    placed = 0
    for idx in range(room_count):
        if _place_room(grid, reserved, directions[idx], stop_kinds[idx], divider, colors[idx], bg, rng):
            placed += 1
    if placed != room_count:
        return None

    if int(np.count_nonzero(grid == bg)) <= (grid.size // 2):
        return None

    divider_guess = _divider_color(grid, bg)
    rooms = _find_rooms(grid, bg, divider_guess)
    if len(rooms) != room_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid room-gap puzzle after {_GEN_MAX_ATTEMPTS} attempts")
