from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("top", "right", "bottom", "left")
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as N x 2 coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            stack = [(sr, sc)]
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _door_side(door: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = door
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if c == c1:
        return "right"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    raise AssertionError("door must lie on the room perimeter")


def _find_rooms(g: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components(g == GREY):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 6 and width >= 6

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        assert int(coords.shape[0]) == len(perimeter) - 1
        for r, c in coords:
            assert int(r) in (r0, r1) or int(c) in (c0, c1)

        missing = [(r, c) for r, c in perimeter if int(g[r, c]) != GREY]
        assert len(missing) == 1
        door = missing[0]
        rooms.append({"bbox": (r0, c0, r1, c1), "door": door, "side": _door_side(door, (r0, c0, r1, c1))})

    rooms.sort(key=lambda room: room["bbox"])
    assert rooms
    return rooms


def _centered_origin(lo: int, hi: int, size: int) -> int:
    return int((int(lo) + int(hi) - int(size) + 1) // 2)


def _sprite_patch(g: np.ndarray, room_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = room_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    sprite_mask = (interior != BLACK) & (interior != GREY)
    comps = _components(sprite_mask)
    assert len(comps) == 1
    local = comps[0]
    vals = interior[local[:, 0], local[:, 1]]
    colors = set(map(int, vals))
    assert len(colors) == 1

    sr0, sc0 = local.min(axis=0)
    sr1, sc1 = local.max(axis=0)
    sr0, sc0, sr1, sc1 = map(int, (sr0, sc0, sr1, sc1))
    patch = np.full((sr1 - sr0 + 1, sc1 - sc0 + 1), BLACK, dtype=int)
    patch[local[:, 0] - sr0, local[:, 1] - sc0] = int(vals[0])

    global_coords = local.copy()
    global_coords[:, 0] += r0 + 1
    global_coords[:, 1] += c0 + 1
    return patch, global_coords


def _rotate_clockwise(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -int(turns))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room's inner sprite according to the side that contains its door.

    Input structure:
    - The background is black and every room is a grey rectangular perimeter with exactly one single-pixel door gap.
    - Door gaps may be on the top, right, bottom, or left side of a room.
    - Each room interior contains one small 4-connected monochrome sprite in a non-black, non-grey color.
    - Sprites are strictly inside their rooms and do not touch the grey border.

    Rule:
    - Identify the door side of each room: top means 0 degrees, right means 90 degrees clockwise, bottom means 180 degrees, and left means 270 degrees.
    - Remove the original inner sprite, rotate its tight bounding-box patch by the door's angle, and place the rotated patch back inside the same room.
    - The rotated patch is centered in the room interior; the grey border, the door gap, and the black background remain unchanged.

    Key insight: The missing border pixel is not damage to repair; its side is the instruction that tells how far to rotate the free sprite inside that room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    turns_for_side = {"top": 0, "right": 1, "bottom": 2, "left": 3}

    for room in _find_rooms(g):
        r0, c0, r1, c1 = room["bbox"]
        patch, coords = _sprite_patch(g, room["bbox"])
        out[coords[:, 0], coords[:, 1]] = BLACK

        rotated = _rotate_clockwise(patch, turns_for_side[room["side"]])
        ph, pw = rotated.shape
        rr0 = _centered_origin(r0 + 1, r1 - 1, ph)
        cc0 = _centered_origin(c0 + 1, c1 - 1, pw)
        assert r0 < rr0 and rr0 + ph - 1 < r1
        assert c0 < cc0 and cc0 + pw - 1 < c1

        mask = rotated != BLACK
        target = out[rr0 : rr0 + ph, cc0 : cc0 + pw]
        assert np.all((target[mask] == BLACK) | (target[mask] == rotated[mask]))
        target[mask] = rotated[mask]

    return out


# --------------- generator ---------------

BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _tight_patch(cells: list[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _random_sprite_patch(color: int) -> np.ndarray:
    cells = list(random.choice(BASE_SHAPES))
    patch = _tight_patch(cells, color)
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    rows, cols = np.where(patch != BLACK)
    return patch[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1].copy()


def _room_inner_sizes(patch: np.ndarray, side: str) -> list[tuple[int, int]]:
    turns = DIRECTIONS.index(side)
    rotated = _rotate_clockwise(patch, turns)
    ph, pw = patch.shape
    rh, rw = rotated.shape
    sizes: list[tuple[int, int]] = []
    for ih in range(4, 9):
        for iw in range(4, 9):
            if ih < max(ph, rh) + 2 or iw < max(pw, rw) + 2:
                continue
            if (ih - rh) % 2 != 0 or (iw - rw) % 2 != 0:
                continue
            sizes.append((ih, iw))
    random.shuffle(sizes)
    return sizes


def _paint_room(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, side: str) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    if side == "top":
        door = (r0, random.randint(c0 + 1, c1 - 1))
    elif side == "right":
        door = (random.randint(r0 + 1, r1 - 1), c1)
    elif side == "bottom":
        door = (r1, random.randint(c0 + 1, c1 - 1))
    else:
        door = (random.randint(r0 + 1, r1 - 1), c0)
    grid[door] = BLACK


def _place_sprite(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    row_min = r0 + 2
    row_max = r0 + inner_h - ph
    col_min = c0 + 2
    col_max = c0 + inner_w - pw
    assert row_min <= row_max and col_min <= col_max
    sr = random.randint(row_min, row_max)
    sc = random.randint(col_min, col_max)
    mask = patch != BLACK
    grid[sr : sr + ph, sc : sc + pw][mask] = patch[mask]


def _sample_directions(room_count: int) -> list[str]:
    if room_count >= 4 and random.random() < 0.75:
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        for _ in range(room_count - 4):
            dirs.append(random.choice(DIRECTIONS))
        random.shuffle(dirs)
        return dirs
    return [random.choice(DIRECTIONS) for _ in range(room_count)]


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    room_count = random.randint(2, 4)
    colors = random.sample(COLORS, room_count)
    sides = _sample_directions(room_count)

    specs = []
    for color, side in zip(colors, sides):
        for _ in range(50):
            patch = _random_sprite_patch(color)
            sizes = _room_inner_sizes(patch, side)
            if sizes:
                inner_h, inner_w = sizes[0]
                specs.append({"color": color, "side": side, "patch": patch, "inner_h": inner_h, "inner_w": inner_w})
                break
        else:
            return None

    random.shuffle(specs)
    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for spec in specs:
        outer_h = int(spec["inner_h"]) + 2
        outer_w = int(spec["inner_w"]) + 2
        if outer_h + 2 > height or outer_w + 2 > width:
            return None
        candidates = [
            (r, c)
            for r in range(1, height - outer_h)
            for c in range(1, width - outer_w)
            if _can_place(blocked, r, c, outer_h, outer_w)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _paint_room(grid, r0, c0, int(spec["inner_h"]), int(spec["inner_w"]), str(spec["side"]))
        _place_sprite(grid, r0, c0, int(spec["inner_h"]), int(spec["inner_w"]), spec["patch"])
        _mark_blocked(blocked, r0, c0, outer_h, outer_w)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
