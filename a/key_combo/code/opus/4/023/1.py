from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
WALL = 5
SPRITE_COLORS = (1, 2, 3, 4)
DIR_BY_HOLES = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def hole_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for comp in components_mask(~mask):
        touches = (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        )
        if not touches:
            count += 1
    return count


def sprite_hole_count(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return hole_count(sub)


def slide_coords(coords: np.ndarray, base: np.ndarray, dr: int, dc: int) -> np.ndarray:
    cur = coords.copy()
    h, w = base.shape
    while True:
        nxt = cur.copy()
        nxt[:, 0] += dr
        nxt[:, 1] += dc
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            return cur
        if np.any(base[nxt[:, 0], nxt[:, 1]] != BG):
            return cur
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite through its walled region according to the number of holes inside the sprite.

    Input structure:
    - Black (0) is the background and grey (5) forms one-cell-thick partition walls.
    - Three or four separate wall-bounded black regions each contain one monochrome 4-connected sprite.
    - Sprite colors are distinct and come from {blue, red, green, yellow}; the direction clue is not the color.

    Rule:
    - For every sprite, count the number of enclosed black 4-connected holes inside its own bounding box.
    - Decode hole count to direction: 0->up, 1->right, 2->down, 3->left.
    - Remove the sprite from its old cells, then slide it cell by cell through black space until one more step would hit a grey wall.
    - Keep the wall layout unchanged and paint only the final sprite positions.

    Key insight:
    - The motion code is topological: the sprite's internal hole count determines which way it moves inside the wall maze.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    object_mask = (g != BG) & (g != WALL)
    objects = components_mask(object_mask)
    assert objects, "expected at least one sprite"

    base = g.copy()
    for coords in objects:
        base[coords[:, 0], coords[:, 1]] = BG

    out = base.copy()
    occupied = np.zeros_like(object_mask, dtype=bool)

    for coords in objects:
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each sprite must be monochrome"
        color = int(colors[0])
        holes = sprite_hole_count(coords)
        assert holes in DIR_BY_HOLES, "supported hole counts are 0..3"
        dr, dc = DIR_BY_HOLES[holes]
        moved = slide_coords(coords, base, dr, dc)
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "final sprite positions must not overlap"
        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = color

    return out


GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 30
MIN_ROOM_SPAN = 8


def random_polyomino(size: int, rng: np.random.Generator) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        frontier_list = sorted(frontier)
        chosen = frontier_list[int(rng.integers(len(frontier_list)))]
        cells.add(chosen)
    pts = np.array(sorted(cells), dtype=int)
    pts[:, 0] -= pts[:, 0].min()
    pts[:, 1] -= pts[:, 1].min()
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def core_sprite_mask(hole_total: int, rng: np.random.Generator) -> np.ndarray:
    if hole_total == 0:
        return random_polyomino(int(rng.integers(4, 9)), rng)
    width = 2 * hole_total + 1
    mask = np.ones((3, width), dtype=bool)
    mask[1, 1:width - 1:2] = False
    if rng.random() < 0.5:
        mask = mask.T
    return mask


def add_protrusion(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    work = np.pad(mask.astype(bool), 1, constant_values=False)
    h, w = work.shape

    outside = np.zeros_like(work, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not work[r, c] and not outside[r, c]:
                    outside[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not work[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if work[r, c] or not outside[r, c]:
                continue
            neighbors = sum(bool(work[nr, nc]) for nr, nc in neighbors4(r, c))
            if neighbors == 1:
                candidates.append((r, c))
    if not candidates:
        return mask
    rr, cc = candidates[int(rng.integers(len(candidates)))]
    work[rr, cc] = True
    return trim_mask(work)


def random_sprite_mask(hole_total: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(40):
        mask = core_sprite_mask(hole_total, rng)
        for _ in range(int(rng.integers(0, 3))):
            mask = add_protrusion(mask, rng)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, k=int(rng.integers(4)))
        mask = trim_mask(mask)
        if hole_count(mask) == hole_total:
            return mask
    raise RuntimeError(f"failed to build sprite with {hole_total} holes")


def choose_split(total: int, rng: np.random.Generator) -> int:
    low = MIN_ROOM_SPAN + 1
    high_exclusive = total - MIN_ROOM_SPAN - 1
    return int(rng.integers(low, high_exclusive))


def build_partition(h: int, w: int, room_count: int, rng: np.random.Generator):
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    hr = choose_split(h, rng)
    vc = choose_split(w, rng)

    rooms: list[tuple[int, int, int, int]]
    if room_count == 4:
        grid[hr, :] = WALL
        grid[:, vc] = WALL
        rooms = [
            (1, 1, hr - 1, vc - 1),
            (1, vc + 1, hr - 1, w - 2),
            (hr + 1, 1, h - 2, vc - 1),
            (hr + 1, vc + 1, h - 2, w - 2),
        ]
        return grid, rooms

    layout = str(rng.choice(["left", "right", "top", "bottom"]))
    if layout == "left":
        grid[:, vc] = WALL
        grid[hr, : vc + 1] = WALL
        rooms = [
            (1, 1, hr - 1, vc - 1),
            (hr + 1, 1, h - 2, vc - 1),
            (1, vc + 1, h - 2, w - 2),
        ]
    elif layout == "right":
        grid[:, vc] = WALL
        grid[hr, vc:] = WALL
        rooms = [
            (1, 1, h - 2, vc - 1),
            (1, vc + 1, hr - 1, w - 2),
            (hr + 1, vc + 1, h - 2, w - 2),
        ]
    elif layout == "top":
        grid[hr, :] = WALL
        grid[: hr + 1, vc] = WALL
        rooms = [
            (1, 1, hr - 1, vc - 1),
            (1, vc + 1, hr - 1, w - 2),
            (hr + 1, 1, h - 2, w - 2),
        ]
    else:
        grid[hr, :] = WALL
        grid[hr:, vc] = WALL
        rooms = [
            (1, 1, hr - 1, w - 2),
            (hr + 1, 1, h - 2, vc - 1),
            (hr + 1, vc + 1, h - 2, w - 2),
        ]
    return grid, rooms


def room_shape(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    return r1 - r0 + 1, c1 - c0 + 1


def place_sprite(room: tuple[int, int, int, int], mask: np.ndarray, dr: int, dc: int, rng: np.random.Generator):
    r0, c0, r1, c1 = room
    h, w = mask.shape
    room_h, room_w = room_shape(room)
    if room_h < h + (1 if dr != 0 else 0) or room_w < w + (1 if dc != 0 else 0):
        return None

    if dr < 0:
        top_low, top_high = r0 + 1, r1 - h + 1
    elif dr > 0:
        top_low, top_high = r0, r1 - h
    else:
        top_low, top_high = r0, r1 - h + 1

    if dc < 0:
        left_low, left_high = c0 + 1, c1 - w + 1
    elif dc > 0:
        left_low, left_high = c0, c1 - w
    else:
        left_low, left_high = c0, c1 - w + 1

    if top_low > top_high or left_low > left_high:
        return None

    top = int(rng.integers(top_low, top_high + 1))
    left = int(rng.integers(left_low, left_high + 1))
    local = np.argwhere(mask)
    coords = local.copy()
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def try_add_stub(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    coords: np.ndarray,
    dr: int,
    dc: int,
    rng: np.random.Generator,
) -> bool:
    r0, c0, r1, c1 = room
    used_rows = set(map(int, coords[:, 0]))
    used_cols = set(map(int, coords[:, 1]))
    room_h, room_w = room_shape(room)

    if dr != 0:
        candidates = [c for c in range(c0, c1 + 1) if c not in used_cols]
        if not candidates or room_h < 4:
            return False
        col = int(rng.choice(candidates))
        length = int(rng.integers(2, room_h - 1))
        if rng.random() < 0.5:
            rows = range(r0, r0 + length)
        else:
            rows = range(r1 - length + 1, r1 + 1)
        cells = [(r, col) for r in rows]
    else:
        candidates = [r for r in range(r0, r1 + 1) if r not in used_rows]
        if not candidates or room_w < 4:
            return False
        row = int(rng.choice(candidates))
        length = int(rng.integers(2, room_w - 1))
        if rng.random() < 0.5:
            cols = range(c0, c0 + length)
        else:
            cols = range(c1 - length + 1, c1 + 1)
        cells = [(row, c) for c in cols]

    for r, c in cells:
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = WALL
    return True


def generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    room_count = 4 if rng.random() < 0.7 else 3
    grid, rooms = build_partition(h, w, room_count, rng)

    hole_counts = [0, 1, 2, 3]
    if room_count == 3:
        keep = sorted(rng.choice(hole_counts, size=3, replace=False).tolist())
    else:
        keep = hole_counts
    rng.shuffle(keep)

    colors = rng.choice(np.array(SPRITE_COLORS, dtype=int), size=len(keep), replace=False).tolist()
    placements: list[tuple[tuple[int, int, int, int], np.ndarray, int, int, int]] = []

    for room, holes, color in zip(rooms, keep, colors):
        dr, dc = DIR_BY_HOLES[holes]
        room_h, room_w = room_shape(room)
        placed = None
        for _ in range(60):
            mask = random_sprite_mask(holes, rng)
            mh, mw = mask.shape
            if room_h < mh + (1 if dr != 0 else 0) or room_w < mw + (1 if dc != 0 else 0):
                continue
            maybe = place_sprite(room, mask, dr, dc, rng)
            if maybe is None:
                continue
            placed = maybe
            break
        if placed is None:
            return None
        paint_coords(grid, placed, color)
        placements.append((room, placed, color, dr, dc))

    stub_added = False
    order = rng.permutation(len(placements))
    for idx in order:
        room, coords, _color, dr, dc = placements[int(idx)]
        if rng.random() < 0.75 and try_add_stub(grid, room, coords, dr, dc, rng):
            stub_added = True
    if not stub_added:
        for room, coords, _color, dr, dc in placements:
            if try_add_stub(grid, room, coords, dr, dc, rng):
                stub_added = True
                break
    if not stub_added:
        return None

    return grid if validate_instance(grid) else None


def validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(np.isin(g, [0, 1, 2, 3, 4, 5])):
        return False
    if np.any(g[0, :] != WALL) or np.any(g[-1, :] != WALL) or np.any(g[:, 0] != WALL) or np.any(g[:, -1] != WALL):
        return False

    objects = components_mask((g != BG) & (g != WALL))
    if len(objects) not in (3, 4):
        return False

    seen_colors = set()
    hole_values = []
    base = g.copy()
    for coords in objects:
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        if colors.size != 1:
            return False
        color = int(colors[0])
        if color in seen_colors:
            return False
        seen_colors.add(color)
        holes = sprite_hole_count(coords)
        if holes not in DIR_BY_HOLES:
            return False
        hole_values.append(holes)
        base[coords[:, 0], coords[:, 1]] = BG
    if len(set(hole_values)) != len(hole_values):
        return False

    regions = components_mask(base == BG)
    if len(regions) != len(objects):
        return False
    labels = -np.ones(g.shape, dtype=int)
    irregular = False
    for idx, region in enumerate(regions):
        labels[region[:, 0], region[:, 1]] = idx
        r0, c0, r1, c1 = bbox_coords(region)
        if region.shape[0] < (r1 - r0 + 1) * (c1 - c0 + 1):
            irregular = True
    if not irregular:
        return False

    occupied = np.zeros(g.shape, dtype=bool)
    moved_any = False
    for coords, holes in zip(objects, hole_values):
        region_ids = {int(labels[r, c]) for r, c in coords}
        if len(region_ids) != 1:
            return False
        dr, dc = DIR_BY_HOLES[holes]
        moved = slide_coords(coords, base, dr, dc)
        if np.array_equal(moved, coords):
            return False
        moved_any = True
        nxt = moved.copy()
        nxt[:, 0] += dr
        nxt[:, 1] += dc
        if (
            np.any(nxt[:, 0] < 0)
            or np.any(nxt[:, 0] >= g.shape[0])
            or np.any(nxt[:, 1] < 0)
            or np.any(nxt[:, 1] >= g.shape[1])
        ):
            return False
        if not np.any(base[nxt[:, 0], nxt[:, 1]] == WALL):
            return False
        if np.any(occupied[moved[:, 0], moved[:, 1]]):
            return False
        occupied[moved[:, 0], moved[:, 1]] = True
    if not moved_any:
        return False

    try:
        out = main(g)
    except Exception:
        return False
    if out.shape != g.shape:
        return False
    if np.array_equal(out, g):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-navigation input")
