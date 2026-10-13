import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == shape[0] - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == shape[1] - 1)
    )


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _room_distances(room_mask: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    dist = np.zeros(room_mask.shape, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(room_mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(room_mask.shape, nr, nc) and wall_mask[nr, nc]:
                dist[rr, cc] = 1
                q.append((rr, cc))
                break

    assert q, "room must have an inner wall surface"

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(room_mask.shape, nr, nc) and room_mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    assert np.all(dist[room_mask] > 0), "room interior must be fully reachable from the wall"
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_mask = g == WALL
    out = g.copy()

    for coords in _components(~wall_mask):
        if _touches_border(coords, g.shape):
            continue

        room_mask = np.zeros(g.shape, dtype=bool)
        room_mask[coords[:, 0], coords[:, 1]] = True

        seed_colors = sorted({int(g[r, c]) for r, c in coords if int(g[r, c]) != BG})
        assert len(seed_colors) <= 1, "each room must contain at most one seed color"
        if not seed_colors:
            continue

        seed_color = int(seed_colors[0])
        seed_mask = room_mask & (g == seed_color)
        seed_count = int(seed_mask.sum())
        assert seed_count > 0, "room must contain at least one seed"

        dist = _room_distances(room_mask, wall_mask)
        out[seed_mask] = BG
        out[room_mask & (dist <= seed_count)] = seed_color

    return out


GEN_ATTEMPTS = 400
ROOM_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
HEIGHT_RANGE = (20, 32)
WIDTH_RANGE = (20, 32)
ROOM_COUNT_CHOICES = [2, 3, 4, 5]
ROOM_COUNT_WEIGHTS = [3, 4, 3, 1]
INNER_H_RANGE = (5, 9)
INNER_W_RANGE = (5, 9)
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
ROOM_GAP = 1


def _random_segment(indices: np.ndarray) -> tuple[int, int]:
    lo = int(indices.min())
    hi = int(indices.max())
    span = hi - lo + 1
    if span <= 2:
        return lo, hi
    length = random.randint(2, span)
    start = random.randint(lo, hi - length + 1)
    return start, start + length - 1


def _generate_inner_shape(h: int, w: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)

    base_h = random.randint(max(4, h // 2 + 1), h)
    base_w = random.randint(max(4, w // 2 + 1), w)
    r0 = random.randint(0, h - base_h)
    c0 = random.randint(0, w - base_w)
    mask[r0 : r0 + base_h, c0 : c0 + base_w] = True

    steps = random.randint(2, h + w)
    for _ in range(steps):
        rows, cols = np.where(mask)
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(cols.min())
        right = int(cols.max())

        options: list[str] = []
        if top > 0 and np.count_nonzero(mask[top, :]) >= 2:
            options.append("top")
        if bottom < h - 1 and np.count_nonzero(mask[bottom, :]) >= 2:
            options.append("bottom")
        if left > 0 and np.count_nonzero(mask[:, left]) >= 2:
            options.append("left")
        if right < w - 1 and np.count_nonzero(mask[:, right]) >= 2:
            options.append("right")
        if not options:
            break

        side = random.choice(options)
        if side == "top":
            start, end = _random_segment(np.flatnonzero(mask[top, :]))
            mask[top - 1, start : end + 1] = True
        elif side == "bottom":
            start, end = _random_segment(np.flatnonzero(mask[bottom, :]))
            mask[bottom + 1, start : end + 1] = True
        elif side == "left":
            start, end = _random_segment(np.flatnonzero(mask[:, left]))
            mask[start : end + 1, left - 1] = True
        else:
            start, end = _random_segment(np.flatnonzero(mask[:, right]))
            mask[start : end + 1, right + 1] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    return mask


def _choose_seed_count(max_dist: int, eligible_count: int) -> int:
    upper = min(eligible_count, max_dist + 3)
    weighted: list[int] = []
    for count in range(1, upper + 1):
        weight = 1
        if count <= max_dist:
            weight += 2
        if count == max_dist:
            weight += 1
        if count > max_dist:
            weight += 1
        weighted.extend([count] * weight)
    return random.choice(weighted)


def _make_room_patch(seed_color: int) -> np.ndarray | None:
    for _ in range(ROOM_ATTEMPTS):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        inner = _generate_inner_shape(inner_h, inner_w)

        room_mask = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        room_mask[1:-1, 1:-1] = inner
        wall_mask = _dilate8(room_mask) & ~room_mask

        if len(_components(wall_mask)) != 1:
            continue

        dist = _room_distances(room_mask, wall_mask)
        eligible = np.argwhere(room_mask & (dist >= 2))
        if len(eligible) == 0:
            continue

        max_dist = int(dist[room_mask].max())
        if max_dist < 2:
            continue

        seed_count = _choose_seed_count(max_dist, len(eligible))
        if seed_count <= 0 or seed_count > len(eligible):
            continue

        seed_cells = random.sample([tuple(map(int, rc)) for rc in eligible], seed_count)

        patch = np.zeros(room_mask.shape, dtype=int)
        patch[wall_mask] = WALL
        for r, c in seed_cells:
            patch[r, c] = seed_color
        return patch
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, patch_shape: tuple[int, int]) -> bool:
    h, w = occupied.shape
    ph, pw = patch_shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - ROOM_GAP)
    c0 = max(0, left - ROOM_GAP)
    r1 = min(h, top + ph + ROOM_GAP)
    c1 = min(w, left + pw + ROOM_GAP)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _stamp_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    non_bg = patch != BG
    grid_slice = grid[top : top + ph, left : left + pw]
    grid_slice[non_bg] = patch[non_bg]

    r0 = max(0, top - ROOM_GAP)
    c0 = max(0, left - ROOM_GAP)
    r1 = min(grid.shape[0], top + ph + ROOM_GAP)
    c1 = min(grid.shape[1], left + pw + ROOM_GAP)
    occupied[r0:r1, c0:c1] = True


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False

    wall_mask = grid == WALL
    rooms = []
    for coords in _components(~wall_mask):
        if _touches_border(coords, grid.shape):
            continue
        rooms.append(coords)

    if len(rooms) < 2:
        return False

    for coords in rooms:
        room_mask = np.zeros(grid.shape, dtype=bool)
        room_mask[coords[:, 0], coords[:, 1]] = True

        colors = sorted({int(grid[r, c]) for r, c in coords if int(grid[r, c]) != BG})
        if len(colors) != 1:
            return False

        color = int(colors[0])
        seed_mask = room_mask & (grid == color)
        if int(seed_mask.sum()) == 0:
            return False

        dist = _room_distances(room_mask, wall_mask)
        if not np.any(seed_mask & (dist >= 2)):
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choices(ROOM_COUNT_CHOICES, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    min_dim = max(HEIGHT_RANGE[0], room_count * 4 + 8)
    if min_dim > HEIGHT_RANGE[1] or min_dim > WIDTH_RANGE[1]:
        return None

    h = random.randint(min_dim, HEIGHT_RANGE[1])
    w = random.randint(min_dim, WIDTH_RANGE[1])
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(COLOR_POOL, room_count)

    for seed_color in colors:
        patch = _make_room_patch(seed_color)
        if patch is None:
            return None

        placed = False
        ph, pw = patch.shape
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            if not _can_place(occupied, top, left, patch.shape):
                continue
            _stamp_patch(grid, occupied, patch, top, left)
            placed = True
            break

        if not placed:
            return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth puzzle")
