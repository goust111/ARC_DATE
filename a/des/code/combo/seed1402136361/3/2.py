from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
ROTATE_TO_UP = {"top": 0, "right": 1, "bottom": 2, "left": 3}
ROTATE_FROM_UP = {"top": 0, "right": 3, "bottom": 2, "left": 1}

GEN_ROOM_COUNT = (1, 4)
GEN_ROOM_H = (5, 10)
GEN_ROOM_W = (5, 10)
GEN_RETRIES = 400
GEN_ROOM_RETRIES = 160
GEN_MASK_RETRIES = 120


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None
    return r0, c0, r1, c1


def _parse_room(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> dict | None:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None

    comps = _components(interior != bg)
    if len(comps) != 1:
        return None
    obj_coords = comps[0]
    obj_colors = {int(interior[r, c]) for r, c in obj_coords}
    if len(obj_colors) != 1:
        return None
    obj_color = next(iter(obj_colors))
    if obj_color == wall_color:
        return None

    touches: list[str] = []
    if any(r == 0 for r, _ in obj_coords):
        touches.append("top")
    if any(c == interior.shape[1] - 1 for _, c in obj_coords):
        touches.append("right")
    if any(r == interior.shape[0] - 1 for r, _ in obj_coords):
        touches.append("bottom")
    if any(c == 0 for _, c in obj_coords):
        touches.append("left")
    if len(touches) != 1:
        return None

    return {
        "frame_bbox": frame_bbox,
        "obj_bbox": _bbox(obj_coords),
        "obj_coords": obj_coords,
        "obj_color": int(obj_color),
        "side": touches[0],
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: dict[int, list[dict]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rooms: list[dict] = []
        ok = True
        for coords in _components(grid == color):
            frame_bbox = _rect_frame_bbox(coords)
            if frame_bbox is None:
                ok = False
                break
            room = _parse_room(grid, bg, color, frame_bbox)
            if room is None:
                ok = False
                break
            rooms.append(room)
        if ok and rooms:
            candidates[color] = rooms

    assert len(candidates) == 1, "expected a unique wall color whose components are all rooms"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's interior object from the touched wall to the room center and rotate it to the upright pose.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    rooms = _find_rooms(g, bg)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["frame_bbox"]
        ir0, ic0 = r0 + 1, c0 + 1
        interior = g[ir0:r1, ic0:c1]
        br0, bc0, br1, bc1 = room["obj_bbox"]
        obj_color = int(room["obj_color"])
        obj_mask = interior[br0 : br1 + 1, bc0 : bc1 + 1] == obj_color
        rotated = np.rot90(obj_mask, ROTATE_TO_UP[room["side"]])

        for rr, cc in room["obj_coords"]:
            out[ir0 + rr, ic0 + cc] = bg

        ih, iw = interior.shape
        ph, pw = rotated.shape
        assert ph <= ih and pw <= iw, "rotated object must fit in the room interior"
        dst_r = (ih - ph) // 2
        dst_c = (iw - pw) // 2
        out[ir0 + dst_r : ir0 + dst_r + ph, ic0 + dst_c : ic0 + dst_c + pw][rotated] = obj_color

    return out


def _draw_frame(patch: np.ndarray, color: int) -> None:
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    min_cells = max(1, max(h, w))
    for _ in range(GEN_MASK_RETRIES):
        target = random.randint(min_cells, h * w)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(rr) + dr, int(cc) + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break

            preferred = [cell for cell in frontier if not bool(mask[cell[0], :].any()) or not bool(mask[:, cell[1]].any())]
            pick = random.choice(preferred or list(frontier))
            mask[pick] = True

        if bool(mask.any(axis=0).all()) and bool(mask.any(axis=1).all()):
            return mask
    return None


def _make_room_patches(
    bg: int,
    wall_color: int,
    obj_color: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(GEN_ROOM_RETRIES):
        room_h = random.randint(*GEN_ROOM_H)
        room_w = random.randint(*GEN_ROOM_W)
        ih, iw = room_h - 2, room_w - 2

        if side in ("top", "bottom"):
            max_h = ih - 1
            max_w = iw - 2
        else:
            max_h = min(ih, iw - 1)
            max_w = min(iw, ih - 2)
        if max_h < 1 or max_w < 1:
            continue

        obj_h = random.randint(1, max_h)
        obj_w = random.randint(1, max_w)
        canonical = _random_connected_mask(obj_h, obj_w)
        if canonical is None:
            continue

        out_patch = np.full((room_h, room_w), bg, dtype=int)
        _draw_frame(out_patch, wall_color)
        out_r = 1 + (ih - obj_h) // 2
        out_c = 1 + (iw - obj_w) // 2
        out_patch[out_r : out_r + obj_h, out_c : out_c + obj_w][canonical] = obj_color

        input_mask = np.rot90(canonical, ROTATE_FROM_UP[side])
        ph, pw = input_mask.shape
        inp_patch = np.full((room_h, room_w), bg, dtype=int)
        _draw_frame(inp_patch, wall_color)

        if side in ("top", "bottom"):
            if iw - pw - 1 < 1:
                continue
            src_r = 1 if side == "top" else room_h - ph - 1
            src_c = 1 + random.randint(1, iw - pw - 1)
        else:
            if ih - ph - 1 < 1:
                continue
            src_r = 1 + random.randint(1, ih - ph - 1)
            src_c = 1 if side == "left" else room_w - pw - 1

        inp_patch[src_r : src_r + ph, src_c : src_c + pw][input_mask] = obj_color
        if np.array_equal(inp_patch, out_patch):
            continue
        return inp_patch, out_patch
    return None


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    room_count = random.randint(*GEN_ROOM_COUNT)
    min_side = 8 + 4 * (room_count - 1)
    h = random.randint(min_side, min_side + 8)
    w = random.randint(min_side, min_side + 8)

    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    wall_color = random.choice(colors)
    colors.remove(wall_color)
    obj_colors = random.sample(colors, room_count)
    sides = random.sample(list(SIDES), room_count)

    room_patches: list[tuple[np.ndarray, np.ndarray]] = []
    for obj_color, side in zip(obj_colors, sides):
        pair = _make_room_patches(bg, wall_color, obj_color, side)
        if pair is None:
            return None
        room_patches.append(pair)

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    order = list(range(room_count))
    random.shuffle(order)
    for idx in order:
        inp_patch, out_patch = room_patches[idx]
        ph, pw = inp_patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)

        placed_here = False
        for r0, c0 in positions:
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_too_close(rect, other) for other in placed):
                continue
            inp[r0 : r0 + ph, c0 : c0 + pw] = inp_patch
            out[r0 : r0 + ph, c0 : c0 + pw] = out_patch
            placed.append(rect)
            placed_here = True
            break
        if not placed_here:
            return None

    if _dominant_color(inp) != bg:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
