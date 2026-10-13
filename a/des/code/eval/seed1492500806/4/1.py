import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = tuple(color for color in range(1, 10) if color != GRAY)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
PAIR_COUNT_RANGE = (2, 4)
KEY_INTERIOR_H_RANGE = (1, 4)
KEY_INTERIOR_W_RANGE = (1, 4)
LOCK_INTERIOR_MAX = 8
PLACEMENT_RETRIES = 200
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_locks_and_keys(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    locks: list[dict] = []
    keys: list[dict] = []

    for coords in _components(g != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 3 or w < 3:
            continue

        border = _border_mask(h, w)
        border_vals = sub[border]
        interior = sub[1:-1, 1:-1]

        if np.all(border_vals != BLACK) and np.all(interior == BLACK):
            colored = border_vals[(border_vals != GRAY)]
            if colored.size == 1:
                locks.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "color": int(colored[0]),
                    }
                )
                continue

        colors = np.unique(sub[sub != BLACK])
        if colors.size == 1 and int(colors[0]) != GRAY and np.all(sub[border] == colors[0]):
            pattern = sub[1:-1, 1:-1].copy()
            if pattern.size > 0 and np.any(pattern == int(colors[0])):
                keys.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "color": int(colors[0]),
                        "pattern": pattern,
                        "coords": coords,
                    }
                )

    return locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the lock with the same keyhole color, then copy
    or tile the key's inner binary pattern into the lock interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    locks, keys = _find_locks_and_keys(g)
    assert locks, "expected at least one lock"
    assert keys, "expected at least one key"
    assert len(locks) == len(keys), "lock/key counts must match"

    lock_by_color: dict[int, dict] = {}
    for lock in locks:
        color = int(lock["color"])
        assert color not in lock_by_color, "lock colors must be unique"
        lock_by_color[color] = lock

    key_by_color: dict[int, dict] = {}
    for key in keys:
        color = int(key["color"])
        assert color not in key_by_color, "key colors must be unique"
        key_by_color[color] = key

    assert set(lock_by_color) == set(key_by_color), "lock colors must match key colors"

    out = g.copy()
    for key in keys:
        coords = key["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for color, lock in lock_by_color.items():
        key = key_by_color[color]
        r0, c0, r1, c1 = lock["bbox"]
        pattern = np.asarray(key["pattern"], dtype=int)
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        ph, pw = pattern.shape
        assert ih % ph == 0 and iw % pw == 0, "lock interior must be an integer multiple of key pattern"
        tiled = np.tile(pattern, (ih // ph, iw // pw))
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled

    return out


def _grow_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    area = h * w
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    boundary_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r == 0 or r == h - 1 or c == 0 or c == w - 1
    ]
    target = rng.randint(1, area - 1)
    mask = np.zeros((h, w), dtype=bool)
    seed = boundary_cells[rng.randrange(len(boundary_cells))]
    mask[seed] = True
    frontier = {seed}

    while int(mask.sum()) < target and frontier:
        cell = rng.choice(tuple(frontier))
        neighbors = []
        for nr, nc in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if neighbors:
            nxt = neighbors[rng.randrange(len(neighbors))]
            mask[nxt] = True
            frontier.add(nxt)
        else:
            frontier.remove(cell)

    return mask


def _make_key_patch(color: int, pattern_mask: np.ndarray) -> np.ndarray:
    ph, pw = pattern_mask.shape
    patch = np.zeros((ph + 2, pw + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][pattern_mask] = color
    return patch


def _choose_tile_factors(rng: random.Random, ph: int, pw: int) -> tuple[int, int]:
    factors: list[tuple[int, int]] = []
    for fh in (1, 2, 3):
        for fw in (1, 2, 3):
            if fh == 1 and fw == 1:
                continue
            if ph * fh <= LOCK_INTERIOR_MAX and pw * fw <= LOCK_INTERIOR_MAX:
                factors.append((fh, fw))
    assert factors, "at least one tiling factor must fit"
    return factors[rng.randrange(len(factors))]


def _make_lock_patch(color: int, interior_h: int, interior_w: int, rng: random.Random) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    patch[0, :] = GRAY
    patch[-1, :] = GRAY
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY

    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        col = rng.randint(1, interior_w)
        row = 0 if side == "top" else interior_h + 1
    else:
        row = rng.randint(1, interior_h)
        col = 0 if side == "left" else interior_w + 1
    patch[row, col] = color
    return patch


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    rh, rw = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(rh, top + h + 1)
    c1 = min(rw, left + w + 1)
    return not np.any(reserved[r0:r1, c0:c1])


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    rh, rw = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(rh, top + h + 1)
    c1 = min(rw, left + w + 1)
    reserved[r0:r1, c0:c1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _build_expected(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for item in placements:
        if item["kind"] == "key":
            patch = item["patch"]
            top = item["top"]
            left = item["left"]
            h, w = patch.shape
            view = expected[top : top + h, left : left + w]
            view[patch != BLACK] = BLACK

    for item in placements:
        if item["kind"] != "lock":
            continue
        top = item["top"]
        left = item["left"]
        interior = item["interior"]
        ih, iw = interior.shape
        expected[top + 1 : top + 1 + ih, left + 1 : left + 1 + iw] = interior
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, pair_count)

    branch_seed = rng.randrange(2)
    branches = [((i + branch_seed) % 2) == 0 for i in range(pair_count)]
    rng.shuffle(branches)

    specs: list[dict] = []
    max_h = 0
    max_w = 0
    for color, direct_copy in zip(colors, branches):
        ph = rng.randint(*KEY_INTERIOR_H_RANGE)
        pw = rng.randint(*KEY_INTERIOR_W_RANGE)
        pattern_mask = _grow_connected_mask(rng, ph, pw)
        key_patch = _make_key_patch(color, pattern_mask)

        if direct_copy:
            ih, iw = ph, pw
        else:
            fh, fw = _choose_tile_factors(rng, ph, pw)
            ih, iw = ph * fh, pw * fw
        pattern = np.where(pattern_mask, color, BLACK)
        interior = np.tile(pattern, (ih // ph, iw // pw))
        lock_patch = _make_lock_patch(color, ih, iw, rng)

        specs.append(
            {
                "color": color,
                "key_patch": key_patch,
                "lock_patch": lock_patch,
                "interior": interior,
            }
        )
        max_h = max(max_h, key_patch.shape[0], lock_patch.shape[0])
        max_w = max(max_w, key_patch.shape[1], lock_patch.shape[1])

    min_h = max(GRID_H_RANGE[0], max_h + 2)
    min_w = max(GRID_W_RANGE[0], max_w + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    objects: list[dict] = []
    for spec in specs:
        objects.append(
            {
                "kind": "lock",
                "color": spec["color"],
                "patch": spec["lock_patch"],
                "interior": spec["interior"],
            }
        )
        objects.append(
            {
                "kind": "key",
                "color": spec["color"],
                "patch": spec["key_patch"],
            }
        )
    objects.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    for obj in objects:
        patch = obj["patch"]
        oh, ow = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, h - oh)
            left = rng.randint(0, w - ow)
            if not _can_place(reserved, top, left, oh, ow):
                continue
            _paste(grid, patch, top, left)
            _reserve(reserved, top, left, oh, ow)
            placed = True
            entry = dict(obj)
            entry["top"] = top
            entry["left"] = left
            placements.append(entry)
            break
        if not placed:
            return None

    expected = _build_expected(grid, placements)
    try:
        parsed_locks, parsed_keys = _find_locks_and_keys(grid)
        if len(parsed_locks) != pair_count or len(parsed_keys) != pair_count:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key sample")
