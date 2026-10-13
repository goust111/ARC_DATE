import random
from collections import Counter, deque

import numpy as np


_BG = 0
_FRAME = 5
_PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _room_from_component(
    grid: np.ndarray, bg: int, frame_color: int, coords: np.ndarray
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = _perimeter_cells(r0, c0, r1, c1)
    border_set = set(border)
    coord_set = {(int(r), int(c)) for r, c in coords}
    if len(coord_set) != len(border_set) - 1:
        return None
    if not coord_set.issubset(border_set):
        return None

    missing = list(border_set - coord_set)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if gap_r in (r0, r1) and gap_c in (c0, c1):
        return None

    if gap_r == r0:
        direction = (-1, 0)
    elif gap_r == r1:
        direction = (1, 0)
    elif gap_c == c0:
        direction = (0, -1)
    else:
        direction = (0, 1)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior_colors = set(map(int, np.unique(interior))) - {bg, frame_color}
    if len(interior_colors) != 1:
        return None
    pattern_color = next(iter(interior_colors))
    pattern_comps = _components(interior == pattern_color)
    if len(pattern_comps) != 1:
        return None
    pattern_coords = [
        (int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in pattern_comps[0]
    ]
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap_r, gap_c),
        "direction": direction,
        "frame_color": int(frame_color),
        "pattern_color": int(pattern_color),
        "pattern_coords": pattern_coords,
        "border_coords": border,
    }


def _find_rooms(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    rooms: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for comp in _components(grid == color):
            room = _room_from_component(grid, bg, color, comp)
            if room is not None:
                rooms.append(room)
    assert rooms, "expected at least one valid room"
    frame_colors = {room["frame_color"] for room in rooms}
    assert len(frame_colors) == 1, "all rooms should share one frame color"
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for room in rooms:
        assert room["bbox"] not in seen_bboxes, "duplicate room detection"
        seen_bboxes.add(room["bbox"])
    return bg, sorted(rooms, key=lambda room: room["bbox"])


def _fully_outside(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = bbox
    return all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in coords)


def _shift_coords(
    coords: list[tuple[int, int]], dr: int, dc: int, steps: int
) -> list[tuple[int, int]]:
    return [(r + dr * steps, c + dc * steps) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's monochrome inner sprite out through the side indicated by the room's unique border gap.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick rectangular room borders appear; each border has exactly one one-cell gap on one side.
    - Each room interior contains one connected monochrome sprite in a unique non-grey, non-black color.

    Rule:
    - Detect every rectangular room and read the gap position as an outward direction: up, down, left, or right.
    - Clear each room interior to black while preserving the grey border and its gap.
    - Translate the whole interior sprite rigidly one cell at a time in the gap direction until the first position where all of its pixels lie outside that room's bounding box; do not let it pass through another room border.

    Key insight:
    - The border gap is not a place where the shape must physically fit; it only encodes the direction of a rigid projection away from the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg, rooms = _find_rooms(g)
    frame_mask = np.zeros_like(g, dtype=bool)
    for room in rooms:
        for r, c in room["border_coords"]:
            frame_mask[r, c] = True

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for room in rooms:
        dr, dc = room["direction"]
        bbox = room["bbox"]
        pattern = room["pattern_coords"]
        color = room["pattern_color"]
        other_frames = frame_mask.copy()
        for r, c in room["border_coords"]:
            other_frames[r, c] = False

        steps = 0
        while True:
            candidate = _shift_coords(pattern, dr, dc, steps + 1)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in candidate):
                break
            if any(other_frames[r, c] for r, c in candidate):
                break
            steps += 1
            if _fully_outside(candidate, bbox):
                break

        shifted = _shift_coords(pattern, dr, dc, steps)
        assert _fully_outside(shifted, bbox), "projection must fully exit its own room"
        for r, c in shifted:
            assert out[r, c] == bg, "projected sprite must land on clear background"
            out[r, c] = color

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _sample_pattern(inner_h: int, inner_w: int) -> np.ndarray:
    max_h = min(inner_h, 4)
    max_w = min(inner_w, 4)
    for _ in range(80):
        ph = random.randint(1, max_h)
        pw = random.randint(1, max_w)
        max_area = min(ph * pw, max(1, min(inner_h * inner_w - 1, 6)))
        area = random.randint(1, max_area)
        seed = (random.randrange(ph), random.randrange(pw))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((ph, pw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop(mask)
        if mask.shape[0] <= inner_h and mask.shape[1] <= inner_w:
            return mask
    return np.ones((1, 1), dtype=bool)


def _sample_room_spec(color: int, direction: tuple[int, int]) -> dict:
    room_h = random.randint(4, 8)
    room_w = random.randint(4, 8)
    inner_h, inner_w = room_h - 2, room_w - 2
    patch = _sample_pattern(inner_h, inner_w)
    ph, pw = patch.shape
    off_r = random.randint(0, inner_h - ph)
    off_c = random.randint(0, inner_w - pw)
    pr0 = 1 + off_r
    pc0 = 1 + off_c
    pattern_rel = [(pr0 + r, pc0 + c) for r, c in np.argwhere(patch)]

    rs = [r for r, _ in pattern_rel]
    cs = [c for _, c in pattern_rel]
    dr, dc = direction
    if dr == -1:
        steps = max(rs) + 1
        gap = (0, random.randint(1, room_w - 2))
    elif dr == 1:
        steps = room_h - min(rs)
        gap = (room_h - 1, random.randint(1, room_w - 2))
    elif dc == -1:
        steps = max(cs) + 1
        gap = (random.randint(1, room_h - 2), 0)
    else:
        steps = room_w - min(cs)
        gap = (random.randint(1, room_h - 2), room_w - 1)

    projected_rel = [(r + dr * steps, c + dc * steps) for r, c in pattern_rel]
    all_rows = [0, room_h - 1] + [r for r, _ in projected_rel]
    all_cols = [0, room_w - 1] + [c for _, c in projected_rel]
    union_rel = (min(all_rows), min(all_cols), max(all_rows), max(all_cols))
    return {
        "color": int(color),
        "direction": direction,
        "room_h": room_h,
        "room_w": room_w,
        "gap_rel": gap,
        "pattern_patch": patch,
        "pattern_rel": pattern_rel,
        "projected_rel": projected_rel,
        "union_rel": union_rel,
    }


def _expand_box(
    box: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _place_specs(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    occ = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    order = sorted(
        specs,
        key=lambda spec: (spec["union_rel"][2] - spec["union_rel"][0] + 1)
        * (spec["union_rel"][3] - spec["union_rel"][1] + 1),
        reverse=True,
    )
    for spec in order:
        ur0, uc0, ur1, uc1 = spec["union_rel"]
        r_min = -ur0
        r_max = h - 1 - ur1
        c_min = -uc0
        c_max = w - 1 - uc1
        if r_min > r_max or c_min > c_max:
            return None
        anchors = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
        random.shuffle(anchors)
        picked = None
        for base_r, base_c in anchors:
            union_abs = (ur0 + base_r, uc0 + base_c, ur1 + base_r, uc1 + base_c)
            reserve = _expand_box(union_abs, 1, h, w)
            rr0, cc0, rr1, cc1 = reserve
            if occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            picked = (base_r, base_c, union_abs, reserve)
            break
        if picked is None:
            return None
        base_r, base_c, _union_abs, reserve = picked
        rr0, cc0, rr1, cc1 = reserve
        occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        item = dict(spec)
        item["base_r"] = base_r
        item["base_c"] = base_c
        placed.append(item)
    return placed


def _draw_room(grid: np.ndarray, spec: dict) -> None:
    r0, c0 = spec["base_r"], spec["base_c"]
    h, w = spec["room_h"], spec["room_w"]
    color = spec["color"]
    grid[r0, c0 : c0 + w] = _FRAME
    grid[r0 + h - 1, c0 : c0 + w] = _FRAME
    grid[r0 : r0 + h, c0] = _FRAME
    grid[r0 : r0 + h, c0 + w - 1] = _FRAME
    gr, gc = spec["gap_rel"]
    grid[r0 + gr, c0 + gc] = _BG
    for rr, cc in spec["pattern_rel"]:
        grid[r0 + rr, c0 + cc] = color


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0 = spec["base_r"], spec["base_c"]
        h, w = spec["room_h"], spec["room_w"]
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = _BG
    for spec in specs:
        color = spec["color"]
        r0, c0 = spec["base_r"], spec["base_c"]
        for rr, cc in spec["projected_rel"]:
            out[r0 + rr, c0 + cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(2, 4)
    min_side = {2: 16, 3: 20, 4: 24}[room_count]
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    colors = random.sample(_PATTERN_COLORS, room_count)
    directions = random.sample(_DIRS, room_count)
    specs = [_sample_room_spec(color, direction) for color, direction in zip(colors, directions)]
    placed = _place_specs(h, w, specs)
    if placed is None:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    for spec in placed:
        _draw_room(grid, spec)

    if _get_bg(grid) != _BG:
        return None

    expected = _expected_output(grid, placed)
    try:
        bg, rooms = _find_rooms(grid)
        if bg != _BG or len(rooms) != room_count:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-projection input")
