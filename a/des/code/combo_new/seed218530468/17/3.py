from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

ROOM_COLORS = (3, 4, 6, 7, 8, 9)
ROOM_SIZE_CHOICES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _door_rotation(patch: np.ndarray) -> int:
    n = int(patch.shape[0])
    mid = n // 2
    options = {
        (0, mid): 0,
        (mid, n - 1): 1,
        (n - 1, mid): 2,
        (mid, 0): 3,
    }
    found = [rot for (r, c), rot in options.items() if int(patch[r, c]) == RED]
    assert len(found) == 1
    return int(found[0])


def _extract_room(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h >= 5 and h % 2 == 1

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    n = h
    border = np.zeros((n, n), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    door_rot = _door_rotation(patch)
    door_pos = {
        0: (0, n // 2),
        1: (n // 2, n - 1),
        2: (n - 1, n // 2),
        3: (n // 2, 0),
    }[door_rot]

    red_mask = patch == RED
    assert int(red_mask.sum()) == 1
    assert bool(red_mask[door_pos])
    assert not bool(red_mask[1:-1, 1:-1].any())
    assert int((patch == BLUE).sum()) == 4 * n - 5
    assert np.all(patch[border & (~red_mask)] == BLUE)
    assert np.all(patch[~border] != BLUE)

    interior = patch[1:-1, 1:-1]
    inner_colors = {int(v) for v in np.unique(interior)} - {BG, BLUE, RED}
    assert len(inner_colors) == 1
    color = int(next(iter(inner_colors)))
    mask = interior == color
    assert bool(mask.any())

    return {
        "bbox": (r0, c0, r1, c1),
        "size": n,
        "door_rot": door_rot,
        "color": color,
        "mask": mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each room interior back to the template orientation and concatenate the normalized interiors.

    Input structure:
    - The background is black.
    - Several disjoint blue odd-sized square room frames appear in the grid, and each frame has exactly one red door pixel at the midpoint of one side.
    - Every room contains the same interior shape in a single non-black, non-blue, non-red color; rooms differ only by color and by 0/90/180/270-degree rotation.
    - A grey reference copy of the base shape sits in one corner on an unframed black (N-2)x(N-2) patch.

    Rule:
    - Detect every blue square room and read the door position as an orientation code: top=0, right=90, bottom=180, left=270.
    - Remove the blue frame, then rotate the room interior in the opposite direction so the room matches the standard orientation defined by the reference with the door on top.
    - Sort rooms by their positions in reading order across the input and place the normalized interiors in a single row with one black separator column between neighboring interiors.

    Key insight:
    - The red door is not part of the interior pattern; it is an external orientation marker telling how much the shared shape must be reverse-rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = [_extract_room(g, comp) for comp in _components(g == BLUE)]
    assert rooms
    sizes = {room["size"] for room in rooms}
    assert len(sizes) == 1

    inner = int(next(iter(sizes)) - 2)
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))

    out = np.zeros((inner, len(rooms) * inner + len(rooms) - 1), dtype=int)
    for idx, room in enumerate(rooms):
        block = np.zeros((inner, inner), dtype=int)
        block[np.rot90(room["mask"], int(room["door_rot"]))] = int(room["color"])
        c0 = idx * (inner + 1)
        out[:, c0 : c0 + inner] = block
    return out


def _add_frontier(frontier: set[tuple[int, int]], mask: np.ndarray, r: int, c: int) -> None:
    side = int(mask.shape[0])
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < side and 0 <= nc < side and not bool(mask[nr, nc]):
            frontier.add((nr, nc))


def _make_base_mask(side: int) -> np.ndarray | None:
    area = side * side
    lo = max(3, int(np.ceil(area * 0.25)))
    hi = max(lo, int(np.floor(area * 0.55)))
    for _ in range(200):
        target = random.randint(lo, hi)
        mask = np.zeros((side, side), dtype=bool)
        start = (random.randrange(side), random.randrange(side))
        mask[start] = True
        frontier: set[tuple[int, int]] = set()
        _add_frontier(frontier, mask, start[0], start[1])

        while int(mask.sum()) < target and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            mask[r, c] = True
            _add_frontier(frontier, mask, r, c)

        if int(mask.sum()) != target:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    return None


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _balanced_orientations(count: int) -> list[int]:
    out: list[int] = []
    while len(out) < count:
        cycle = [0, 1, 2, 3]
        random.shuffle(cycle)
        out.extend(cycle)
    return out[:count]


def _corner_reserve(corner: str, inner: int) -> tuple[int, int, int, int]:
    gap = inner + 1
    if corner == "tl":
        return gap, 0, gap, 0
    if corner == "tr":
        return gap, 0, 0, gap
    if corner == "bl":
        return 0, gap, gap, 0
    return 0, gap, 0, gap


def _place_template(grid: np.ndarray, mask: np.ndarray, corner: str) -> None:
    side = int(mask.shape[0])
    h, w = grid.shape
    r0 = 0 if corner[0] == "t" else h - side
    c0 = 0 if corner[1] == "l" else w - side
    patch = grid[r0 : r0 + side, c0 : c0 + side]
    patch[mask] = GREY


def _room_patch(size: int, color: int, base_mask: np.ndarray, door_rot: int) -> np.ndarray:
    patch = np.full((size, size), BLUE, dtype=int)
    patch[1:-1, 1:-1] = BG
    rotated = np.rot90(base_mask, (-door_rot) % 4)
    interior = patch[1:-1, 1:-1]
    interior[rotated] = color

    mid = size // 2
    door_pos = {
        0: (0, mid),
        1: (mid, size - 1),
        2: (size - 1, mid),
        3: (mid, 0),
    }[door_rot]
    patch[door_pos] = RED
    return patch


def _expected_output(inner: int, base_mask: np.ndarray, rooms: list[dict]) -> np.ndarray:
    ordered = sorted(rooms, key=lambda room: (room["bbox"][0], room["bbox"][1]))
    out = np.zeros((inner, len(ordered) * inner + len(ordered) - 1), dtype=int)
    for idx, room in enumerate(ordered):
        block = np.zeros((inner, inner), dtype=int)
        block[base_mask] = int(room["color"])
        c0 = idx * (inner + 1)
        out[:, c0 : c0 + inner] = block
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(ROOM_SIZE_CHOICES)
    inner = size - 2
    max_rooms = min(len(ROOM_COLORS), (30 + 1) // (inner + 1))
    room_count = random.randint(2, max_rooms)

    base_mask = _make_base_mask(inner)
    if base_mask is None:
        return None

    corner = random.choice(("tl", "tr", "bl", "br"))
    row_count = random.randint(1, min(3, room_count))
    row_sizes = _random_partition(room_count, row_count)
    row_gaps = [random.randint(1, 3) for _ in range(row_count - 1)]
    col_gaps = [[random.randint(1, 3) for _ in range(n - 1)] for n in row_sizes]

    row_widths = [
        n * size + sum(gaps)
        for n, gaps in zip(row_sizes, col_gaps, strict=False)
    ]
    arrangement_h = row_count * size + sum(row_gaps)
    arrangement_w = max(row_widths)

    top_res, bottom_res, left_res, right_res = _corner_reserve(corner, inner)
    base_h = top_res + arrangement_h + bottom_res
    base_w = left_res + arrangement_w + right_res
    if base_h > 30 or base_w > 30:
        return None

    top_pad = random.randint(0, min(4, 30 - base_h))
    bottom_pad = random.randint(0, min(4, 30 - base_h - top_pad))
    left_pad = random.randint(0, min(4, 30 - base_w))
    right_pad = random.randint(0, min(4, 30 - base_w - left_pad))

    h = base_h + top_pad + bottom_pad
    w = base_w + left_pad + right_pad
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    _place_template(grid, base_mask, corner)

    colors = random.sample(list(ROOM_COLORS), room_count)
    orientations = _balanced_orientations(room_count)
    random.shuffle(colors)
    random.shuffle(orientations)

    rooms: list[dict] = []
    y = top_res + top_pad
    room_idx = 0
    for row_idx, count in enumerate(row_sizes):
        row_width = row_widths[row_idx]
        x = left_res + left_pad + random.randint(0, arrangement_w - row_width)
        for col_idx in range(count):
            color = colors[room_idx]
            door_rot = orientations[room_idx]
            patch = _room_patch(size, color, base_mask, door_rot)
            grid[y : y + size, x : x + size] = patch
            rooms.append(
                {
                    "bbox": (y, x, y + size - 1, x + size - 1),
                    "color": color,
                }
            )
            room_idx += 1
            if col_idx < count - 1:
                x += size + col_gaps[row_idx][col_idx]
        if row_idx < row_count - 1:
            y += size + row_gaps[row_idx]

    expected = _expected_output(inner, base_mask, rooms)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
