from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROOM_RETRY_LIMIT = 300
FURNITURE_RETRY_LIMIT = 200
GENERATE_RETRY_LIMIT = 400


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _is_single_component(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False

    start = tuple(map(int, cells[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    return np.rot90(arr, -turns)


def _door_side(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if (
        int(grid[r0, c0]) != WALL
        or int(grid[r0, c1]) != WALL
        or int(grid[r1, c0]) != WALL
        or int(grid[r1, c1]) != WALL
    ):
        return None

    sides = {
        "top": grid[r0, c0 + 1 : c1],
        "right": grid[r0 + 1 : r1, c1],
        "bottom": grid[r1, c0 + 1 : c1],
        "left": grid[r0 + 1 : r1, c0],
    }

    door_side = None
    for side, segment in sides.items():
        non_wall = np.flatnonzero(segment != WALL)
        if np.any((segment != WALL) & (segment != BG)):
            return None
        if len(non_wall) > 1:
            return None
        if len(non_wall) == 1:
            if door_side is not None:
                return None
            door_side = side

    return door_side


def _extract_furniture(inner: np.ndarray) -> tuple[np.ndarray, int]:
    mask = (inner != BG) & (inner != WALL)
    assert mask.any(), "room must contain furniture"
    colors = np.unique(inner[mask])
    assert len(colors) == 1, "furniture must be monochrome"
    assert _is_single_component(mask), "furniture must be a single component"

    coords = np.argwhere(mask)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    patch = inner[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch, int(colors[0])


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_ps = _prefix_sum(g == WALL)
    furniture_ps = _prefix_sum((g != BG) & (g != WALL))

    rooms: list[dict] = []
    for r0 in range(h - 4):
        for r1 in range(r0 + 4, h):
            for c0 in range(w - 4):
                for c1 in range(c0 + 4, w):
                    if _rect_sum(wall_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _rect_sum(furniture_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) == 0:
                        continue

                    room_h = r1 - r0 + 1
                    room_w = c1 - c0 + 1
                    perimeter = 2 * (room_h + room_w) - 4
                    if _rect_sum(wall_ps, r0, c0, r1, c1) != perimeter - 1:
                        continue

                    side = _door_side(g, r0, c0, r1, c1)
                    if side is None:
                        continue

                    inner = g[r0 + 1 : r1, c0 + 1 : c1]
                    try:
                        patch, color = _extract_furniture(inner)
                    except AssertionError:
                        continue

                    rooms.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "door_side": side,
                            "color": color,
                            "patch": patch,
                        }
                    )

    return sorted(rooms, key=lambda room: room["bbox"])


def _apply_room_rotation(grid: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    turns_by_side = {"top": 0, "right": 1, "bottom": 2, "left": 3}

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[(inner != BG) & (inner != WALL)] = BG

        rotated = _rotate_cw(room["patch"], turns_by_side[room["door_side"]])
        ph, pw = rotated.shape
        ih, iw = inner.shape
        top = (ih - ph) // 2
        left = (iw - pw) // 2
        mask = rotated != BG
        inner[top : top + ph, left : left + pw][mask] = rotated[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each room's furniture according to the side containing its single wall gap.

    Input structure:
    - Black background surrounds several axis-aligned rectangular rooms.
    - Grey pixels form one-cell-thick room walls, and each room boundary has
      exactly one one-cell door gap on one side.
    - Every room interior contains one monochrome asymmetric furniture sprite;
      all rooms share the same sprite shape but use different colors.
    - A corner reference room has its door on the top side, showing the
      unrotated furniture orientation.

    Rule:
    - Keep the full grid size, all grey wall pixels, and every door gap
      unchanged.
    - Detect each rectangular room from its almost-complete grey border and
      locate the side containing the unique missing wall cell.
    - Leave top-door furniture unchanged, rotate right-door furniture 90 degrees
      clockwise, bottom-door furniture 180 degrees, and left-door furniture 270
      degrees clockwise.
    - Recenter the rotated furniture inside the same room interior.

    Key insight:
    - The door side is the orientation cue: once a room is recognized from its
      wall rectangle and single gap, the furniture rotation is determined only
      by that gap direction, not by color or room position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rooms = _find_rooms(g)
    assert rooms, "expected at least one room"
    return _apply_room_rotation(g, rooms)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _extend_to_side(mask: np.ndarray, side: str) -> None:
    h, w = mask.shape
    cells = np.argwhere(mask)
    assert cells.size > 0

    if side == "top":
        target = (0, random.randrange(w))
    elif side == "bottom":
        target = (h - 1, random.randrange(w))
    elif side == "left":
        target = (random.randrange(h), 0)
    else:
        target = (random.randrange(h), w - 1)

    sr, sc = min((tuple(map(int, cell)) for cell in cells), key=lambda rc: abs(rc[0] - target[0]) + abs(rc[1] - target[1]))
    tr, tc = target

    if random.random() < 0.5:
        step = 1 if tr >= sr else -1
        for r in range(sr, tr + step, step):
            mask[r, sc] = True
        step = 1 if tc >= sc else -1
        for c in range(sc, tc + step, step):
            mask[tr, c] = True
    else:
        step = 1 if tc >= sc else -1
        for c in range(sc, tc + step, step):
            mask[sr, c] = True
        step = 1 if tr >= sr else -1
        for r in range(sr, tr + step, step):
            mask[r, tc] = True


def _random_furniture_mask() -> np.ndarray:
    dims = random.choice([(3, 5), (5, 3)])
    h, w = dims

    for _ in range(FURNITURE_RETRY_LIMIT):
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        target_area = random.randint(max(h, w) + 1, h * w - 3)

        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True

        for side in ("top", "bottom", "left", "right"):
            if side == "top" and mask[0, :].any():
                continue
            if side == "bottom" and mask[-1, :].any():
                continue
            if side == "left" and mask[:, 0].any():
                continue
            if side == "right" and mask[:, -1].any():
                continue
            _extend_to_side(mask, side)

        if not _touches_all_sides(mask):
            continue
        if not _is_single_component(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask

    raise RuntimeError("failed to generate asymmetric furniture mask")


def _place_patch(inner: np.ndarray, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    ih, iw = inner.shape
    top = (ih - ph) // 2
    left = (iw - pw) // 2
    mask = patch != BG
    inner[top : top + ph, left : left + pw][mask] = patch[mask]


def _paint_room(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    door_side: str,
    door_offset: int,
    patch: np.ndarray,
) -> dict:
    room_h = inner_h + 2
    room_w = inner_w + 2
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL

    if door_side == "top":
        grid[r0, c0 + door_offset] = BG
    elif door_side == "right":
        grid[r0 + door_offset, c1] = BG
    elif door_side == "bottom":
        grid[r1, c0 + door_offset] = BG
    else:
        grid[r0 + door_offset, c0] = BG

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    _place_patch(inner, patch)
    return {"bbox": (r0, c0, r1, c1), "door_side": door_side, "patch": patch.copy()}


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _reference_corner_position(corner: str, H: int, W: int, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, W - w
    if corner == "bl":
        return H - h, 0
    return H - h, W - w


def _expected_output(inp: np.ndarray, rooms: list[dict]) -> np.ndarray:
    return _apply_room_rotation(inp, rooms)


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choice([4, 5])
    mask = _random_furniture_mask()
    base_orientations = ["top", "right", "bottom", "left"]
    if room_count == 5:
        base_orientations.append(random.choice(["top", "right", "bottom", "left"]))
    tail = base_orientations[1:]
    random.shuffle(tail)
    base_orientations = [base_orientations[0], *tail]

    patch_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], room_count)
    patches = [np.where(mask, color, BG).astype(int) for color in patch_colors]

    max_patch_dim = max(mask.shape)
    inner_choices = [size for size in (max_patch_dim + 2, max_patch_dim + 4) if size <= 9]
    if not inner_choices:
        return None

    room_specs = []
    for door_side, patch in zip(base_orientations, patches):
        inner_h = random.choice(inner_choices)
        inner_w = random.choice(inner_choices)
        room_specs.append(
            {
                "door_side": door_side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "room_h": inner_h + 2,
                "room_w": inner_w + 2,
                "patch": patch,
            }
        )

    H = random.randint(20, 30)
    W = random.randint(20, 30)
    occupied = np.zeros((H, W), dtype=bool)
    grid = np.full((H, W), BG, dtype=int)
    rooms: list[dict] = []

    reference = room_specs[0]
    corner = random.choice(["tl", "tr", "bl", "br"])
    ref_r0, ref_c0 = _reference_corner_position(corner, H, W, reference["room_h"], reference["room_w"])
    if not _can_place(occupied, ref_r0, ref_c0, reference["room_h"], reference["room_w"], padding=1):
        return None
    _mark_occupied(occupied, ref_r0, ref_c0, reference["room_h"], reference["room_w"])
    ref_door = random.randint(1, reference["room_w"] - 2)
    rooms.append(
        _paint_room(
            grid,
            r0=ref_r0,
            c0=ref_c0,
            inner_h=reference["inner_h"],
            inner_w=reference["inner_w"],
            door_side="top",
            door_offset=ref_door,
            patch=reference["patch"],
        )
    )

    for spec in room_specs[1:]:
        placed = False
        for _ in range(ROOM_RETRY_LIMIT):
            r0 = random.randint(0, H - spec["room_h"])
            c0 = random.randint(0, W - spec["room_w"])
            if not _can_place(occupied, r0, c0, spec["room_h"], spec["room_w"], padding=1):
                continue
            _mark_occupied(occupied, r0, c0, spec["room_h"], spec["room_w"])
            if spec["door_side"] in ("top", "bottom"):
                door_offset = random.randint(1, spec["room_w"] - 2)
            else:
                door_offset = random.randint(1, spec["room_h"] - 2)
            rooms.append(
                _paint_room(
                    grid,
                    r0=r0,
                    c0=c0,
                    inner_h=spec["inner_h"],
                    inner_w=spec["inner_w"],
                    door_side=spec["door_side"],
                    door_offset=door_offset,
                    patch=spec["patch"],
                )
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, rooms)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_find_rooms(grid)) != room_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-rotation puzzle")
