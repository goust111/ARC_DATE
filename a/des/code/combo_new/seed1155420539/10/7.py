from __future__ import annotations

import random
from collections import deque

import numpy as np


_CCW_STEPS = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}

_DIR_BAG: list[str] = []


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _room_from_boundary(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    cell_set = set(cells)
    if cell_set != perimeter:
        return None

    counts: dict[int, list[tuple[int, int]]] = {}
    for r, c in cells:
        counts.setdefault(int(grid[r, c]), []).append((r, c))
    if len(counts) != 2:
        return None

    ordered = sorted(counts.items(), key=lambda item: len(item[1]))
    door_color, door_cells = ordered[0]
    border_color, border_cells = ordered[1]
    if len(door_cells) != 1 or len(border_cells) != len(perimeter) - 1:
        return None

    door_r, door_c = door_cells[0]
    side = None
    if door_r == r0 and c0 < door_c < c1:
        side = "top"
    elif door_r == r1 and c0 < door_c < c1:
        side = "bottom"
    elif door_c == c0 and r0 < door_r < r1:
        side = "left"
    elif door_c == c1 and r0 < door_r < r1:
        side = "right"
    if side is None:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": int(border_color),
        "door_color": int(door_color),
        "door": (int(door_r), int(door_c)),
        "side": side,
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    rooms = []
    for cells in _components(grid != bg):
        room = _room_from_boundary(grid, cells)
        if room is not None:
            rooms.append(room)
    assert rooms, "expected at least one room boundary"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room's interior pattern according to the side containing the single door pixel.

    Input structure:
    - The background is the most common color.
    - Several one-pixel-thick rectangular room frames appear as closed borders of a single color, with exactly one border pixel replaced by a different door color.
    - Each room contains a non-background pattern strictly inside the frame, separated from the border by background cells.

    Rule:
    - Detect each rectangular frame and locate its unique off-color door pixel.
    - A door on the top edge means keep the interior pattern unchanged; right means rotate it clockwise 90 degrees; bottom means rotate it 180 degrees; left means rotate it clockwise 270 degrees.
    - Rotate the tight bounding box of the room's full interior pattern around its own center and place it back at the same center.
    - Leave the frame and door unchanged.

    Key insight:
    The lone odd-colored pixel on the frame is a control marker whose side encodes the rotation to apply to the pattern inside that room.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for room in _find_rooms(g, bg):
        r0, c0, r1, c1 = room["bbox"]
        side = room["side"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = inner != bg
        if not bool(mask.any()):
            continue

        pr0, pc0, pr1, pc1 = _mask_bbox(mask)
        crop = inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        k = _CCW_STEPS[side]
        rotated = np.rot90(crop, k=k)

        room_out = out[r0 + 1 : r1, c0 + 1 : c1]
        room_out[mask] = bg

        nr0 = pr0 + (crop.shape[0] - rotated.shape[0]) // 2
        nc0 = pc0 + (crop.shape[1] - rotated.shape[1]) // 2
        nr1 = nr0 + rotated.shape[0]
        nc1 = nc0 + rotated.shape[1]
        assert 0 <= nr0 <= nr1 <= room_out.shape[0]
        assert 0 <= nc0 <= nc1 <= room_out.shape[1]

        target = room_out[nr0:nr1, nc0:nc1]
        target[rotated != bg] = rotated[rotated != bg]

    return out


def _next_direction() -> str:
    global _DIR_BAG
    if not _DIR_BAG:
        _DIR_BAG = ["top", "right", "bottom", "left"]
        random.shuffle(_DIR_BAG)
    return _DIR_BAG.pop()


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            mask[r, c] = True
        while c != tc:
            c += 1 if tc > c else -1
            mask[r, c] = True
    else:
        while c != tc:
            c += 1 if tc > c else -1
            mask[r, c] = True
        while r != tr:
            r += 1 if tr > r else -1
            mask[r, c] = True


def _grow_connected(mask: np.ndarray, target: int) -> None:
    h, w = mask.shape
    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if not frontier:
            return
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True


def _pattern_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        order = anchors[:]
        random.shuffle(order)
        for a, b in zip(order, order[1:]):
            _draw_path(mask, a, b)

        min_area = max(int(mask.sum()), max(4, int(np.ceil(0.35 * h * w))))
        max_area = max(min_area, int(np.floor(0.7 * h * w)))
        target = random.randint(min_area, max_area)
        _grow_connected(mask, target)
        if bool(mask[0].any()) and bool(mask[-1].any()) and bool(mask[:, 0].any()) and bool(mask[:, -1].any()):
            return mask
    return None


def _colorized_crop(h: int, w: int, bg: int, palette: list[int], side: str) -> np.ndarray | None:
    for _ in range(80):
        mask = _pattern_mask(h, w)
        if mask is None:
            continue

        crop = np.full((h, w), bg, dtype=int)
        color_count = random.randint(1, min(3, len(palette)))
        colors = random.sample(palette, color_count)
        cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        random.shuffle(cells)
        for idx, (r, c) in enumerate(cells):
            crop[r, c] = colors[idx] if idx < color_count else random.choice(colors)

        if side != "top":
            inv = np.rot90(crop, k=(4 - _CCW_STEPS[side]) % 4)
            if np.array_equal(inv, crop):
                continue
        return crop
    return None


def _room_spec(total_rooms: int, side: str, bg: int, pattern_palette: list[int]) -> dict | None:
    size_ranges = {
        1: (5, 12),
        2: (5, 10),
        3: (5, 8),
        4: (5, 7),
    }
    inner_lo, inner_hi = size_ranges[total_rooms]
    for _ in range(120):
        inner_h = random.randint(inner_lo, inner_hi)
        inner_w = random.randint(inner_lo, inner_hi)
        support_max = min(7, inner_h - 2, inner_w - 2)
        if support_max < 3:
            continue

        support = random.randint(3, support_max)
        dims = [
            (h, w)
            for h in range(2, support + 1)
            for w in range(2, support + 1)
            if h * w >= 4 and (support - h) % 2 == 0 and (support - w) % 2 == 0
        ]
        random.shuffle(dims)
        for box_h, box_w in dims:
            out_crop = _colorized_crop(box_h, box_w, bg, pattern_palette, side)
            if out_crop is None:
                continue

            in_crop = np.rot90(out_crop, k=(4 - _CCW_STEPS[side]) % 4)
            in_h, in_w = map(int, in_crop.shape)

            support_r0 = random.randint(1, inner_h - support - 1)
            support_c0 = random.randint(1, inner_w - support - 1)

            out_r0 = support_r0 + (support - box_h) // 2
            out_c0 = support_c0 + (support - box_w) // 2
            in_r0 = support_r0 + (support - in_h) // 2
            in_c0 = support_c0 + (support - in_w) // 2

            return {
                "side": side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "out_crop": out_crop,
                "in_crop": in_crop,
                "out_box": (out_r0, out_c0),
                "in_box": (in_r0, in_c0),
            }
    return None


def _place_rooms(grid_h: int, grid_w: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)

    for idx in order:
        oh = int(specs[idx]["outer_h"])
        ow = int(specs[idx]["outer_w"])
        candidates = []
        for r0 in range(grid_h - oh + 1):
            for c0 in range(grid_w - ow + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r0 + oh + 1)
                cc1 = min(grid_w, c0 + ow + 1)
                if bool(occupied[rr0:rr1, cc0:cc1].any()):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + oh, c0 : c0 + ow] = True

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _draw_room_frame(grid: np.ndarray, r0: int, c0: int, oh: int, ow: int, border: int, door: int, side: str) -> None:
    r1 = r0 + oh - 1
    c1 = c0 + ow - 1
    grid[r0 : r1 + 1, c0] = border
    grid[r0 : r1 + 1, c1] = border
    grid[r0, c0 : c1 + 1] = border
    grid[r1, c0 : c1 + 1] = border

    if side == "top":
        door_r, door_c = r0, random.randint(c0 + 1, c1 - 1)
    elif side == "bottom":
        door_r, door_c = r1, random.randint(c0 + 1, c1 - 1)
    elif side == "left":
        door_r, door_c = random.randint(r0 + 1, r1 - 1), c0
    else:
        door_r, door_c = random.randint(r0 + 1, r1 - 1), c1
    grid[door_r, door_c] = door


def _paste_crop(grid: np.ndarray, top: int, left: int, crop: np.ndarray, bg: int) -> None:
    h, w = crop.shape
    region = grid[top : top + h, left : left + w]
    region[crop != bg] = crop[crop != bg]


def _unique_bg_mode(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    max_count = counts.max()
    return int(np.sum(counts == max_count)) == 1 and int(vals[int(np.argmax(counts))]) == bg


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    room_count = random.randint(1, 4)
    grid_min = {1: 14, 2: 18, 3: 21, 4: 24}[room_count]
    grid_h = random.randint(grid_min, 30)
    grid_w = random.randint(grid_min, 30)

    colors = list(range(10))
    bg, border, door = random.sample(colors, 3)
    pattern_palette = [c for c in colors if c not in {bg, border, door}]

    sides = [_next_direction() for _ in range(room_count)]
    specs = []
    for side in sides:
        spec = _room_spec(room_count, side, bg, pattern_palette)
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_rooms(grid_h, grid_w, specs)
    if placements is None or len(placements) != room_count:
        return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)

    for spec, (r0, c0) in zip(specs, placements):
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        side = str(spec["side"])
        _draw_room_frame(inp, r0, c0, oh, ow, border, door, side)
        _draw_room_frame(out, r0, c0, oh, ow, border, door, side)

        in_r0, in_c0 = spec["in_box"]
        out_r0, out_c0 = spec["out_box"]
        _paste_crop(inp, r0 + 1 + in_r0, c0 + 1 + in_c0, spec["in_crop"], bg)
        _paste_crop(out, r0 + 1 + out_r0, c0 + 1 + out_c0, spec["out_crop"], bg)

    if not _unique_bg_mode(inp, bg):
        return None
    if not _unique_bg_mode(out, bg):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is None:
            continue
        inp, out = case
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
