import random
from collections import deque

import numpy as np


BACKGROUND = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

# Generator tuning.
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PACK_RETRIES = 160
PLACE_RETRIES = 250
OBJECT_COUNT_CHOICES = (2, 3, 4)
LOCK_GAP = 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.ascontiguousarray(mask.astype(np.uint8))
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _rotation_keys(mask: np.ndarray) -> list[tuple[int, int, bytes]]:
    return [_shape_key(np.rot90(mask, k)) for k in range(4)]


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_rotation_keys(mask))


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    bg = ~mask
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r, c in np.argwhere(bg):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        touches_boundary = False
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                touches_boundary = True
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        if touches_boundary:
            continue
        hole = np.zeros_like(mask, dtype=bool)
        for hr, hc in pts:
            hole[hr, hc] = True
        holes.append(hole)
    return holes


def _scene_objects(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys: list[dict[str, object]] = []
    locks: list[dict[str, object]] = []
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    for color in colors:
        for coords in _components(grid == color, connectivity=8):
            mask, bbox = _crop_mask(coords)
            holes = _enclosed_holes(mask)
            info = {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
            }
            if len(holes) == 0:
                keys.append(info)
            elif len(holes) == 1:
                hole_points = [(int(r), int(c)) for r, c in np.argwhere(holes[0])]
                hole_mask, hole_bbox = _crop_mask(hole_points)
                info["hole_mask"] = hole_mask
                info["hole_bbox"] = hole_bbox
                locks.append(info)
            else:
                raise AssertionError("each object must be either solid or a single-hole lock")
    return keys, locks


def _match_locks_to_keys(
    locks: list[dict[str, object]], keys: list[dict[str, object]]
) -> dict[int, int]:
    key_options = [set(_rotation_keys(key["mask"])) for key in keys]
    lock_candidates: dict[int, list[int]] = {}
    for li, lock in enumerate(locks):
        hole_key = _shape_key(lock["hole_mask"])
        candidates = [ki for ki, options in enumerate(key_options) if hole_key in options]
        assert candidates, "every lock must match at least one key"
        lock_candidates[li] = candidates

    order = sorted(lock_candidates, key=lambda li: len(lock_candidates[li]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        li = order[idx]
        for ki in lock_candidates[li]:
            if ki in used:
                continue
            used.add(ki)
            assignment[li] = ki
            if dfs(idx + 1):
                return True
            used.remove(ki)
            del assignment[li]
        return False

    assert dfs(0), "must find a full lock-key matching"
    return assignment


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "ARC grids must stay within 30x30"

    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    keys, locks = _scene_objects(g, bg)

    assert keys, "expected at least one key"
    assert locks, "expected at least one lock"
    assert len(keys) == len(locks), "keys and locks must be paired one-to-one"

    assignment = _match_locks_to_keys(locks, keys)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg

    for li, ki in assignment.items():
        lock = locks[li]
        color = int(keys[ki]["color"])
        r0, c0, _, _ = lock["bbox"]
        hr0, hc0, _, _ = lock["hole_bbox"]
        for rr, cc in np.argwhere(lock["hole_mask"]):
            out[r0 + hr0 + int(rr), c0 + hc0 + int(cc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match solid key shapes to lock cavities and fill each lock with the matched key color.

    Input structure:
    - The background is black.
    - The grid contains several solid connected objects in distinct colors that act as keys.
    - The grid also contains several one-pixel-thick monochrome ring objects that each enclose one cavity acting as a lock.

    Rule:
    - Extract the cavity mask from every lock and the solid mask from every key.
    - Compare each key to each lock cavity under the four rotations 0, 90, 180, and 270 degrees.
    - For each lock, find the uniquely matching key shape, remove that key from the grid, and fill the lock cavity with the key's color.
    - Keep the lock outlines unchanged and leave the rest of the background untouched.

    Key insight:
    - Shape under rotation determines the pairing, while color is only the payload transferred from the matched key into the lock cavity.
    """
    return _solve(grid)


def _adjacent_count(shape: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in shape for dr, dc in DIR4)


def _frontier(shape: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in shape:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape:
                frontier.add((nr, nc))
    return list(frontier)


def _grow_shape(h: int, w: int, target: int, mode: str) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    shape = {start}
    while len(shape) < target:
        frontier = _frontier(shape, h, w)
        if not frontier:
            return None
        if mode == "compact":
            frontier.sort(key=lambda cell: (_adjacent_count(shape, cell), random.random()), reverse=True)
            cell = frontier[0]
        elif mode == "stringy":
            frontier.sort(key=lambda cell: (_adjacent_count(shape, cell), random.random()))
            cell = frontier[0]
        else:
            cell = random.choice(frontier)
        shape.add(cell)

    pts = list(shape)
    mask, _ = _crop_mask(pts)
    if len(_components(mask, connectivity=4)) != 1:
        return None
    if _enclosed_holes(mask):
        return None
    return mask


def _make_lock_patch(cavity_mask: np.ndarray, lock_color: int) -> np.ndarray | None:
    h, w = cavity_mask.shape
    cavity = np.zeros((h + 2, w + 2), dtype=bool)
    cavity[1 : h + 1, 1 : w + 1] = cavity_mask

    border = np.zeros_like(cavity, dtype=bool)
    for r, c in np.argwhere(cavity):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not cavity[nr, nc]:
                border[nr, nc] = True

    if len(_components(border, connectivity=8)) != 1:
        return None
    holes = _enclosed_holes(border)
    if len(holes) != 1 or not np.array_equal(holes[0], cavity):
        return None

    patch = np.zeros_like(cavity, dtype=int)
    patch[border] = int(lock_color)
    return patch


def _generate_base_shape() -> np.ndarray | None:
    mode = random.choice(("compact", "stringy", "mixed"))
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    target = random.randint(4, min(h * w, 9))
    return _grow_shape(h, w, target, mode)


def _rotational_symmetry_count(mask: np.ndarray) -> int:
    return len({_shape_key(rot) for rot in _rotation_variants(mask)})


def _build_shape_set(count: int, require_shared_area: bool) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for _ in range(SHAPE_RETRIES):
        if len(shapes) == count:
            areas = [int(shape.sum()) for shape in shapes]
            if require_shared_area and len(set(areas)) == len(areas):
                shapes = []
                seen.clear()
                continue
            return shapes
        shape = _generate_base_shape()
        if shape is None:
            continue
        key = _canonical_shape_key(shape)
        if key in seen:
            continue
        if _make_lock_patch(shape, 1) is None:
            continue
        seen.add(key)
        shapes.append(shape)
    return None


def _reserve_bbox(r: int, c: int, h: int, w: int, H: int, W: int, gap: int) -> tuple[int, int, int, int]:
    return (
        max(0, r - gap),
        max(0, c - gap),
        min(H - 1, r + h - 1 + gap),
        min(W - 1, c + w - 1 + gap),
    )


def _place_patches(patches: list[np.ndarray], H: int, W: int, gap: int) -> list[tuple[int, int]] | None:
    reserve = np.zeros((H, W), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        choices: list[tuple[int, int]] = []
        for _ in range(PLACE_RETRIES):
            r = random.randint(0, H - h)
            c = random.randint(0, W - w)
            rr0, cc0, rr1, cc1 = _reserve_bbox(r, c, h, w, H, W, gap)
            if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            choices.append((r, c))
            if len(choices) >= 8:
                break
        if not choices:
            return None
        r, c = random.choice(choices)
        positions[idx] = (r, c)
        rr0, cc0, rr1, cc1 = _reserve_bbox(r, c, h, w, H, W, gap)
        reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paint_scene(patches: list[np.ndarray], positions: list[tuple[int, int]], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        h, w = patch.shape
        target = grid[r0 : r0 + h, c0 : c0 + w]
        assert not np.any((target != BACKGROUND) & (patch != BACKGROUND))
        target[patch != BACKGROUND] = patch[patch != BACKGROUND]
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(OBJECT_COUNT_CHOICES)
    require_shared_area = pair_count >= 2 and random.random() < 0.6
    base_shapes = _build_shape_set(pair_count, require_shared_area)
    if base_shapes is None:
        return None

    lock_color = random.choice([color for color in range(1, 10)])
    key_colors = random.sample([color for color in range(1, 10) if color != lock_color], pair_count)

    key_rots = [random.randrange(4) for _ in range(pair_count)]
    lock_rots = [random.randrange(4) for _ in range(pair_count)]
    non_symmetric = [i for i, shape in enumerate(base_shapes) if _rotational_symmetry_count(shape) > 1]
    if non_symmetric:
        idx = random.choice(non_symmetric)
        lock_rots[idx] = (key_rots[idx] + random.choice((1, 2, 3))) % 4

    key_patches: list[np.ndarray] = []
    lock_patches: list[np.ndarray] = []
    for shape, key_color, key_rot, lock_rot in zip(base_shapes, key_colors, key_rots, lock_rots):
        key_mask = np.rot90(shape, key_rot)
        key_patch = np.zeros(key_mask.shape, dtype=int)
        key_patch[key_mask] = int(key_color)
        key_patches.append(key_patch)

        lock_mask = np.rot90(shape, lock_rot)
        lock_patch = _make_lock_patch(lock_mask, lock_color)
        if lock_patch is None:
            return None
        lock_patches.append(lock_patch)

    patches = key_patches + lock_patches
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    reserve_area = sum((patch.shape[0] + 2 * LOCK_GAP) * (patch.shape[1] + 2 * LOCK_GAP) for patch in patches)

    for _ in range(PACK_RETRIES):
        base = int(np.ceil(np.sqrt(reserve_area)))
        H = random.randint(max(max_h + 2, base - 2), min(30, max(max_h + 2, base + 6)))
        W = random.randint(max(max_w + 2, base - 2), min(30, max(max_w + 2, base + 6)))
        if H > 30 or W > 30:
            continue
        positions = _place_patches(patches, H, W, LOCK_GAP)
        if positions is None:
            continue
        grid = _paint_scene(patches, positions, H, W)
        try:
            out = _solve(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        keys, locks = _scene_objects(grid, BACKGROUND)
        if len(keys) != pair_count or len(locks) != pair_count:
            continue
        candidates = [
            sum(_shape_key(lock["hole_mask"]) in set(_rotation_keys(key["mask"])) for key in keys)
            for lock in locks
        ]
        if any(count != 1 for count in candidates):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
