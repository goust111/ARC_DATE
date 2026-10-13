import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZES_SMALL = (5, 7)
ROOM_SIZES_LARGE = (5, 7, 9)
DIST_RANGE = (3, 8)
PLACEMENT_PAD = 1
ROOM_PLACEMENT_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts.update((r1, c) for c in range(c0, c1 + 1))
    pts.update((r, c0) for r in range(r0, r1 + 1))
    pts.update((r, c1) for r in range(r0, r1 + 1))
    return pts


def _anchor_offset(height: int, width: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, width // 2
    if direction == "down":
        return height - 1, width // 2
    if direction == "left":
        return height // 2, 0
    if direction == "right":
        return height // 2, width - 1
    raise ValueError(f"unknown direction: {direction}")


def _make_room_patch(height: int, width: int, color: int, direction: str, rng: random.Random) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = WALL
    patch[-1, :] = WALL
    patch[:, 0] = WALL
    patch[:, -1] = WALL

    seed_r = rng.randint(1, height - 2)
    seed_c = rng.randint(1, width - 2)
    patch[seed_r, seed_c] = color

    anchor_r, anchor_c = _anchor_offset(height, width, direction)
    patch[anchor_r, anchor_c] = color
    return patch


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    for pts in _components(g != BG):
        if len(pts) < 8:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue

        coords = set(pts)
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        if coords != perimeter:
            continue

        values = [int(g[r, c]) for r, c in pts]
        special = [(r, c) for r, c in pts if int(g[r, c]) != WALL]
        if len(special) != 1:
            continue
        anchor = special[0]
        color = int(g[anchor])
        if color in (BG, WALL):
            continue
        if any(v not in (WALL, color) for v in values):
            continue

        direction_by_anchor = {
            (r0, c0 + width // 2): "up",
            (r1, c0 + width // 2): "down",
            (r0 + height // 2, c0): "left",
            (r0 + height // 2, c1): "right",
        }
        if anchor not in direction_by_anchor:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(inner == WALL):
            continue
        seeds = np.argwhere((inner != BG) & (inner != WALL))
        if seeds.shape[0] != 1:
            continue
        sr, sc = map(int, seeds[0])
        seed = (r0 + 1 + sr, c0 + 1 + sc)
        if int(g[seed]) != color:
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "anchor": anchor,
                "direction": direction_by_anchor[anchor],
                "color": color,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return rooms


def _find_target(grid: np.ndarray, room: dict) -> tuple[int, int]:
    h, w = grid.shape
    r, c = room["anchor"]
    dr, dc = DIRS[room["direction"]]
    color = room["color"]
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        if int(grid[nr, nc]) == color:
            return nr, nc
        nr += dr
        nc += dc
    raise AssertionError("each room must find a same-color target marker along its anchor direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each grey-walled room to the same-color marker indicated by its colored border anchor.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular rooms enclose black interiors, and each room contains exactly one colored seed pixel.
    - On exactly one side midpoint of each room border, the grey wall is replaced by a pixel matching the room's seed color.
    - Elsewhere in the grid, each room color appears once more as a distant singleton target marker.

    Rule:
    - Detect every closed rectangular room by its grey perimeter and locate its unique colored border anchor.
    - The anchor side gives the movement direction: top means up, bottom means down, left means left, and right means right.
    - Search outward from the anchor along that direction until the same-color target marker is found.
    - Translate the whole room, including its grey border and interior seed, so that the anchor aligns with that target marker.
    - Erase the original room to black; if translated cells fall outside the grid, clip them.

    Key insight:
    - The off-grey wall pixel is not damage to the frame: it both identifies the room color and points to the ray that leads to the matching destination marker.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    rooms = _find_rooms(g)
    assert rooms, "expected at least one valid room"

    for room in rooms:
        target_r, target_c = _find_target(g, room)
        anchor_r, anchor_c = room["anchor"]
        delta_r = target_r - anchor_r
        delta_c = target_c - anchor_c
        r0, c0, _, _ = room["bbox"]
        patch = room["patch"]

        for pr in range(patch.shape[0]):
            for pc in range(patch.shape[1]):
                value = int(patch[pr, pc])
                if value == BG:
                    continue
                nr = r0 + pr + delta_r
                nc = c0 + pc + delta_c
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    out[nr, nc] = value

    return out


def _inflate_bbox(
    bbox: tuple[int, int, int, int], height: int, width: int, pad: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(height - 1, r1 + pad), min(width - 1, c1 + pad)


def _bbox_is_free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    r0, c0, r1, c1 = _inflate_bbox(bbox, mask.shape[0], mask.shape[1], pad)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = _inflate_bbox(bbox, mask.shape[0], mask.shape[1], pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _cell_is_free(mask: np.ndarray, cell: tuple[int, int], pad: int = 0) -> bool:
    r, c = cell
    return _bbox_is_free(mask, (r, c, r, c), pad=pad)


def _mark_cell(mask: np.ndarray, cell: tuple[int, int], pad: int = 0) -> None:
    r, c = cell
    _mark_bbox(mask, (r, c, r, c), pad=pad)


def _corridor_clear(mask: np.ndarray, start: tuple[int, int], direction: str, steps: int) -> bool:
    dr, dc = DIRS[direction]
    r, c = start
    for step in range(1, steps):
        nr = r + dr * step
        nc = c + dc * step
        if mask[nr, nc]:
            return False
    return True


def _sample_output_top_left(
    grid_h: int, grid_w: int, room_h: int, room_w: int, direction: str, dist: int, rng: random.Random
) -> tuple[int, int] | None:
    dr, dc = DIRS[direction]
    row_lo = 0
    row_hi = grid_h - room_h
    col_lo = 0
    col_hi = grid_w - room_w

    if dr == -1:
        row_hi -= dist
    elif dr == 1:
        row_lo += dist
    if dc == -1:
        col_hi -= dist
    elif dc == 1:
        col_lo += dist

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    size_pool = ROOM_SIZES_LARGE if min(grid_h, grid_w) >= 24 and room_count <= 3 else ROOM_SIZES_SMALL

    colors = rng.sample(PALETTE, room_count)
    directions = list(DIRS)
    rng.shuffle(directions)
    directions = directions[:room_count]

    specs: list[dict] = []
    for color, direction in zip(colors, directions):
        room_h = rng.choice(size_pool)
        room_w = rng.choice(size_pool)
        dist = rng.randint(*DIST_RANGE)
        specs.append(
            {
                "color": color,
                "direction": direction,
                "room_h": room_h,
                "room_w": room_w,
                "dist": dist,
            }
        )

    input_exact = np.zeros((grid_h, grid_w), dtype=bool)
    input_blocked = np.zeros((grid_h, grid_w), dtype=bool)
    output_blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        placed_room = None
        for _ in range(ROOM_PLACEMENT_ATTEMPTS):
            sampled = _sample_output_top_left(
                grid_h, grid_w, spec["room_h"], spec["room_w"], spec["direction"], spec["dist"], rng
            )
            if sampled is None:
                break
            out_r0, out_c0 = sampled
            dr, dc = DIRS[spec["direction"]]
            in_r0 = out_r0 - spec["dist"] * dr
            in_c0 = out_c0 - spec["dist"] * dc
            in_r1 = in_r0 + spec["room_h"] - 1
            in_c1 = in_c0 + spec["room_w"] - 1
            out_r1 = out_r0 + spec["room_h"] - 1
            out_c1 = out_c0 + spec["room_w"] - 1

            input_bbox = (in_r0, in_c0, in_r1, in_c1)
            output_bbox = (out_r0, out_c0, out_r1, out_c1)
            if not _bbox_is_free(input_blocked, input_bbox):
                continue
            if not _bbox_is_free(output_blocked, output_bbox):
                continue

            anchor_off_r, anchor_off_c = _anchor_offset(spec["room_h"], spec["room_w"], spec["direction"])
            input_anchor = (in_r0 + anchor_off_r, in_c0 + anchor_off_c)
            target = (out_r0 + anchor_off_r, out_c0 + anchor_off_c)

            if not _cell_is_free(input_blocked, target, pad=PLACEMENT_PAD):
                continue
            if not _corridor_clear(input_exact, input_anchor, spec["direction"], spec["dist"]):
                continue

            patch = _make_room_patch(spec["room_h"], spec["room_w"], spec["color"], spec["direction"], rng)
            placed_room = {
                "patch": patch,
                "input_top_left": (in_r0, in_c0),
                "input_bbox": input_bbox,
                "output_bbox": output_bbox,
                "target": target,
                "color": spec["color"],
            }
            break

        if placed_room is None:
            return None

        _mark_bbox(input_exact, placed_room["input_bbox"])
        _mark_bbox(input_blocked, placed_room["input_bbox"], pad=PLACEMENT_PAD)
        _mark_cell(input_exact, placed_room["target"])
        _mark_cell(input_blocked, placed_room["target"], pad=PLACEMENT_PAD)
        _mark_bbox(output_blocked, placed_room["output_bbox"], pad=PLACEMENT_PAD)
        placed.append(placed_room)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for room in placed:
        r0, c0 = room["input_top_left"]
        patch = room["patch"]
        for pr in range(patch.shape[0]):
            for pc in range(patch.shape[1]):
                value = int(patch[pr, pc])
                if value != BG:
                    grid[r0 + pr, c0 + pc] = value
        tr, tc = room["target"]
        grid[tr, tc] = room["color"]

    rooms = _find_rooms(grid)
    if len(rooms) != room_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to create a valid puzzle instance")
