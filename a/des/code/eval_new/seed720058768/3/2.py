from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER = 9
PORTAL_COLORS = (1, 2, 3, 4, 6, 7, 8)
CONTENT_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIDE_TO_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _side_positions(room: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
    }


def _extract_rooms(grid: np.ndarray) -> list[dict[str, int]]:
    mask = np.asarray(grid != BG, dtype=bool)
    rooms: list[dict[str, int]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        comp_set = set(comp)
        border = _border_cells(r0, c0, r1, c1)
        if comp_set != border:
            continue
        wall_count = sum(1 for r, c in comp if int(grid[r, c]) == WALL)
        if wall_count < len(border) - 2:
            continue
        rooms.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 - 1),
                "w": int(c1 - c0 - 1),
            }
        )
    rooms.sort(key=lambda room: (room["r0"], room["c0"], room["r1"], room["c1"]))
    if not rooms:
        raise AssertionError("expected at least one framed room")
    return rooms


def _marker_side(grid: np.ndarray, room: dict[str, int]) -> str:
    hits = [side for side, cells in _side_positions(room).items() if any(int(grid[r, c]) == MARKER for r, c in cells)]
    if len(hits) != 1:
        raise AssertionError("each room must have exactly one direction marker")
    return hits[0]


def _portal_info(grid: np.ndarray, room: dict[str, int]) -> tuple[int, tuple[int, int]] | None:
    boundary = []
    for cells in _side_positions(room).values():
        boundary.extend(cells)
    portal_cells = [(r, c) for r, c in boundary if int(grid[r, c]) not in (WALL, MARKER)]
    if not portal_cells:
        return None
    if len(portal_cells) != 1:
        raise AssertionError("each room may have at most one portal")
    r, c = portal_cells[0]
    return int(grid[r, c]), (int(r), int(c))


def _fit_same_origin(arr: np.ndarray, shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    hh = min(h, int(arr.shape[0]))
    ww = min(w, int(arr.shape[1]))
    out[:hh, :ww] = np.asarray(arr[:hh, :ww], dtype=int)
    return out


def _rotate_centered(arr: np.ndarray, k: int, bg: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=int)
    rotated = np.rot90(arr, k=k % 4)
    if rotated.shape == arr.shape:
        return rotated.copy()

    h, w = arr.shape
    rh, rw = rotated.shape
    out = np.full((h, w), bg, dtype=int)
    src_r0 = max((rh - h) // 2, 0)
    src_c0 = max((rw - w) // 2, 0)
    dst_r0 = max((h - rh) // 2, 0)
    dst_c0 = max((w - rw) // 2, 0)
    copy_h = min(h, rh)
    copy_w = min(w, rw)
    out[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = rotated[
        src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w
    ]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap room contents across same-color wall portals, then rotate each room by its maroon wall marker.

    Input structure:
    - bg = black. Several grey 1-pixel rectangular wall frames appear on the grid, each enclosing one rectangular room interior.
    - Every room frame contains exactly one maroon wall pixel on one side; its side encodes the room's rotation.
    - Some room frames also contain one extra colored wall pixel; equal-colored pairs are portals linking two rooms.
    - The room interiors contain colored content pixels on black, while walls, portals, and maroon markers stay on the boundary.

    Rule:
    - Detect each framed room, keeping only its interior as movable content and treating the wall frame itself as fixed structure.
    - Pair rooms whose portal colors match, and swap their interior contents while preserving the source content's relative coordinates inside the destination room; rooms without portals keep their own contents.
    - After swapping, rotate each destination room's interior by the angle encoded by its maroon marker position: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 90 degrees counterclockwise.
    - Write the transformed interior back into each room, leaving the grey walls, portal pixels, and maroon marker pixels unchanged.

    Key insight:
    - The matching colored wall pixels are transport links between rooms, while the maroon wall marker is not a target pointer at all: it encodes the destination room's local rotation, so the correct sequence is swap first and rotate second.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    rooms = _extract_rooms(g)

    contents: list[np.ndarray] = []
    marker_sides: list[str] = []
    portals_by_color: dict[int, list[int]] = {}
    for idx, room in enumerate(rooms):
        contents.append(g[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]].copy())
        marker_sides.append(_marker_side(g, room))
        portal = _portal_info(g, room)
        if portal is not None:
            color, _pos = portal
            portals_by_color.setdefault(int(color), []).append(int(idx))

    sources = list(range(len(rooms)))
    for color, pair in portals_by_color.items():
        if len(pair) != 2:
            raise AssertionError(f"portal color {color} must appear in exactly two rooms")
        a, b = pair
        sources[a] = b
        sources[b] = a

    out = g.copy()
    for idx, room in enumerate(rooms):
        payload = _fit_same_origin(contents[sources[idx]], (room["h"], room["w"]), bg)
        rotated = _rotate_centered(payload, SIDE_TO_K[marker_sides[idx]], bg)
        out[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]] = rotated
    return out


def _placeable(occ: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, gap: int) -> bool:
    h, w = occ.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + outer_h + gap)
    c1 = min(w, left + outer_w + gap)
    return not bool(np.any(occ[r0:r1, c0:c1]))


def _stamp_room(grid: np.ndarray, occ: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> dict[str, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left] = WALL
    grid[top : top + outer_h, left + outer_w - 1] = WALL
    grid[top, left : left + outer_w] = WALL
    grid[top + outer_h - 1, left : left + outer_w] = WALL
    occ[top : top + outer_h, left : left + outer_w] = True
    return {
        "r0": int(top),
        "c0": int(left),
        "r1": int(top + outer_h - 1),
        "c1": int(left + outer_w - 1),
        "h": int(inner_h),
        "w": int(inner_w),
    }


def _choose_layout() -> tuple[int, int]:
    return random.randint(20, 30), random.randint(20, 30)


def _room_count_bounds(height: int, width: int) -> tuple[int, int]:
    area = height * width
    if area < 500:
        return 4, 5
    if area < 700:
        return 4, 6
    return 5, 7


def _balanced_sides(n: int) -> list[str]:
    sides = ["top", "right", "bottom", "left"]
    pool = (sides * ((n + 3) // 4))[:n]
    random.shuffle(pool)
    return pool


def _middle_cell(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return cells[len(cells) // 2]


def _boundary_choices(room: dict[str, int], blocked: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    for side_cells in _side_positions(room).values():
        for cell in side_cells:
            if cell != blocked:
                cells.append(cell)
    return cells


def _random_pattern(inner_h: int, inner_w: int, used: set[bytes]) -> np.ndarray:
    usable = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    if len(usable) < 3:
        raise AssertionError("room interior must be large enough to keep content off the wall")

    max_pixels = min(len(usable), 7)
    for _ in range(300):
        arr = np.full((inner_h, inner_w), BG, dtype=int)
        color_count = random.randint(2, min(3, len(CONTENT_COLORS)))
        colors = random.sample(CONTENT_COLORS, color_count)
        pixel_count = random.randint(3, max_pixels)
        chosen = random.sample(usable, pixel_count)

        for idx, (r, c) in enumerate(chosen):
            arr[r, c] = colors[idx] if idx < color_count else random.choice(colors)

        filled = np.argwhere(arr != BG)
        if len(filled) < 3:
            continue
        r0, c0 = filled.min(axis=0)
        r1, c1 = filled.max(axis=0)
        if int(r1 - r0) < 1 or int(c1 - c0) < 1:
            continue
        if any(np.array_equal(arr, np.rot90(arr, k=k)) for k in (1, 2, 3)):
            continue
        sig = arr.tobytes() + str(arr.shape).encode()
        if sig in used:
            continue
        used.add(sig)
        return arr
    raise RuntimeError("failed to generate a non-degenerate content pattern")


def _choose_portal_pairs(room_count: int) -> list[tuple[int, int]]:
    max_pairs = min(len(PORTAL_COLORS), room_count // 2)
    if room_count < 4 or max_pairs < 1:
        return []
    pair_cap = min(max_pairs, (room_count - 1) // 2)
    pair_count = random.randint(1, pair_cap)
    chosen = random.sample(range(room_count), 2 * pair_count)
    random.shuffle(chosen)
    return [(chosen[i], chosen[i + 1]) for i in range(0, len(chosen), 2)]


def _crop_to_objects(grid: np.ndarray) -> np.ndarray:
    cells = np.argwhere(grid != BG)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    top_pad = random.randint(0, min(2, int(r0)))
    left_pad = random.randint(0, min(2, int(c0)))
    bottom_pad = random.randint(0, min(2, int(grid.shape[0] - 1 - r1)))
    right_pad = random.randint(0, min(2, int(grid.shape[1] - 1 - c1)))
    return grid[
        int(r0) - top_pad : int(r1) + bottom_pad + 1,
        int(c0) - left_pad : int(c1) + right_pad + 1,
    ].copy()


def _room_changes(inp: np.ndarray, out: np.ndarray) -> int:
    changed = 0
    for room in _extract_rooms(inp):
        before = inp[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]]
        after = out[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]]
        if not np.array_equal(before, after):
            changed += 1
    return changed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height, width = _choose_layout()
        room_min, room_max = _room_count_bounds(height, width)
        target_rooms = random.randint(room_min, room_max)
        grid = np.full((height, width), BG, dtype=int)
        occ = np.zeros((height, width), dtype=bool)
        rooms: list[dict[str, int]] = []

        room_specs = []
        for _room_idx in range(target_rooms):
            inner_h = random.randint(4, 6)
            inner_w = random.randint(4, 6)
            room_specs.append((inner_h, inner_w))
        room_specs.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

        failed = False
        for inner_h, inner_w in room_specs:
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            options = [
                (top, left)
                for top in range(1, height - outer_h)
                for left in range(1, width - outer_w)
                if _placeable(occ, top, left, outer_h, outer_w, gap=2)
            ]
            if not options:
                failed = True
                break
            top, left = random.choice(options)
            rooms.append(_stamp_room(grid, occ, top, left, inner_h, inner_w))
        if failed or len(rooms) < 4:
            continue

        sides = _balanced_sides(len(rooms))
        marker_positions: list[tuple[int, int]] = []
        for room, side in zip(rooms, sides):
            pos = _middle_cell(_side_positions(room)[side])
            grid[pos] = MARKER
            marker_positions.append(pos)

        portal_pairs = _choose_portal_pairs(len(rooms))
        if not portal_pairs:
            continue
        for color, (a, b) in zip(random.sample(PORTAL_COLORS, len(portal_pairs)), portal_pairs):
            for room_idx in (a, b):
                candidates = _boundary_choices(rooms[room_idx], marker_positions[room_idx])
                if not candidates:
                    failed = True
                    break
                grid[random.choice(candidates)] = int(color)
            if failed:
                break
        if failed:
            continue

        used_patterns: set[bytes] = set()
        for room in rooms:
            pattern = _random_pattern(room["h"], room["w"], used_patterns)
            grid[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]] = pattern

        grid = _crop_to_objects(grid)
        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(grid, out):
            continue
        if _room_changes(grid, out) < 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid portal-room puzzle")
