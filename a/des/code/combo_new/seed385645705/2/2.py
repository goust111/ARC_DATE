import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIR8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors(r, c, h, w):
            if bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), np.ascontiguousarray(mask).tobytes()


def _component_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == BG:
                continue
            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors(r, c, h, w):
                    if not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for r, c in pts:
                mask[int(r - r0), int(c - c0)] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"], obj["color"], obj["area"]))
    return objects


def _are_complements(a: dict, b: dict) -> bool:
    if int(a["color"]) == int(b["color"]):
        return False
    if tuple(a["mask"].shape) != tuple(b["mask"].shape):
        return False
    return bool(np.all(a["mask"] ^ b["mask"]))


def _pair_objects(objects: list[dict]) -> list[tuple[int, int]]:
    n = len(objects)
    if n == 0 or n % 2:
        raise AssertionError("expected a positive even number of objects")
    candidates = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _are_complements(objects[i], objects[j]):
                candidates[i].append(j)
                candidates[j].append(i)

    def search(unmatched: frozenset[int]) -> list[tuple[int, int]] | None:
        if not unmatched:
            return []
        i = min(unmatched, key=lambda idx: (len([j for j in candidates[idx] if j in unmatched]), idx))
        for j in sorted(c for c in candidates[i] if c in unmatched):
            rest = search(frozenset(k for k in unmatched if k not in (i, j)))
            if rest is not None:
                return [(i, j)] + rest
        return None

    pairs = search(frozenset(range(n)))
    if pairs is None:
        raise AssertionError("objects must be pairable by complementary shape")
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments, recolor the smaller one, and compact each pair into a solid rectangle.

    Input structure:
    - The background is black (0).
    - The foreground is a set of single-color connected sprites scattered on the grid.
    - Sprites come in pairs with different colors; within each pair, the two tight bounding boxes have the same size.
    - When the paired bounding boxes are aligned, one sprite's occupied cells are exactly the other sprite's empty cells.

    Rule:
    - Identify pairs by complementary shape in equal-size bounding boxes.
    - For each pair, compare foreground-pixel counts and use the larger fragment's color.
    - Clear the moving fragments and draw a solid rectangle of the larger fragment's color at the larger fragment's original bbox.

    Key insight:
    - Pairing is determined by complementary geometry, while the larger area decides both the inherited color and the final position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _component_objects(g)
    pairs = _pair_objects(objects)

    out = np.full_like(g, BG)
    for i, j in pairs:
        a, b = objects[i], objects[j]
        assert int(a["area"]) != int(b["area"]), "paired fragments must have unequal areas"
        large = a if int(a["area"]) > int(b["area"]) else b
        r0, c0, r1, c1 = large["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(large["color"])
    return out


def _random_mask(h: int, w: int, used_shapes: set[tuple[int, int, bytes]]) -> np.ndarray | None:
    min_side_area = max(4, (h * w) // 4)
    for _ in range(1200):
        p = random.uniform(0.36, 0.64)
        mask = np.random.random((h, w)) < p
        area = int(mask.sum())
        if area <= min_side_area or h * w - area <= min_side_area or area * 2 == h * w:
            continue
        if not (_full_bbox(mask) and _full_bbox(~mask)):
            continue
        if not (_is_connected(mask) and _is_connected(~mask)):
            continue
        key = _shape_key(mask)
        comp_key = _shape_key(~mask)
        if key in used_shapes or comp_key in used_shapes:
            continue
        used_shapes.add(key)
        used_shapes.add(comp_key)
        return mask.copy()
    return None


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def _can_reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(H, r0 + h + 1), min(W, c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _add_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    r0: int,
    c0: int,
) -> dict:
    h, w = mask.shape
    _reserve(occupied, r0, c0, h, w)
    _place_mask(grid, mask, color, r0, c0)
    return {"r": int(r0), "c": int(c0), "mask": mask, "color": int(color)}


def _expected_output(shape: tuple[int, int], pairs: list[tuple[dict, dict]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for a, b in pairs:
        large = a if int(a["mask"].sum()) > int(b["mask"].sum()) else b
        h, w = large["mask"].shape
        out[large["r"] : large["r"] + h, large["c"] : large["c"] + w] = int(large["color"])
    return out


def _random_pair_spec(used_shapes: set[tuple[int, int, bytes]], h_range=(4, 8), w_range=(4, 8)):
    h = random.randint(*h_range)
    w = random.randint(*w_range)
    mask = _random_mask(h, w, used_shapes)
    if mask is None:
        return None
    c1, c2 = random.sample(PALETTE, 2)
    if random.random() < 0.5:
        return mask, ~mask, c1, c2
    return ~mask, mask, c1, c2


def _place_pair_aligned(
    grid: np.ndarray,
    occupied: np.ndarray,
    spec: tuple[np.ndarray, np.ndarray, int, int],
) -> tuple[dict, dict] | None:
    mask_a, mask_b, color_a, color_b = spec
    h, w = mask_a.shape
    orientation = random.choice(("horizontal", "vertical"))
    gap = random.randint(2, 7)

    if orientation == "horizontal":
        group_h, group_w = h, 2 * w + gap
    else:
        group_h, group_w = 2 * h + gap, w

    H, W = grid.shape
    if group_h > H - 2 or group_w > W - 2:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = random.randint(1, H - group_h - 1)
        c0 = random.randint(1, W - group_w - 1)
        if orientation == "horizontal":
            pos = [(r0, c0), (r0, c0 + w + gap)]
        else:
            pos = [(r0, c0), (r0 + h + gap, c0)]
        if random.random() < 0.5:
            pos.reverse()
        if all(_can_reserve(occupied, r, c, h, w) for r, c in pos):
            obj_a = _add_object(grid, occupied, mask_a, color_a, pos[0][0], pos[0][1])
            obj_b = _add_object(grid, occupied, mask_b, color_b, pos[1][0], pos[1][1])
            return obj_a, obj_b
    return None


def _clear_attempt() -> np.ndarray | None:
    used_shapes: set[tuple[int, int, bytes]] = set()
    n_pairs = random.randint(2, 4)
    specs = []
    for _ in range(n_pairs):
        spec = _random_pair_spec(used_shapes)
        if spec is None:
            return None
        specs.append(spec)

    H = random.randint(18, 30)
    W = random.randint(18, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    pairs = []

    specs.sort(key=lambda s: int(s[0].size), reverse=True)
    for spec in specs:
        placed = _place_pair_aligned(grid, occupied, spec)
        if placed is None:
            return None
        pairs.append(placed)

    expected = _expected_output(grid.shape, pairs)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def _blocked_attempt() -> np.ndarray | None:
    used_shapes: set[tuple[int, int, bytes]] = set()
    h0, w0 = random.randint(7, 9), random.randint(5, 7)
    h1 = random.randint(4, h0 - 2)
    w1 = random.randint(4, 6)
    gap0 = w1 + random.randint(3, 7)
    gap1 = random.randint(2, 4)

    spec0 = _random_pair_spec(used_shapes, (h0, h0), (w0, w0))
    spec1 = _random_pair_spec(used_shapes, (h1, h1), (w1, w1))
    if spec0 is None or spec1 is None:
        return None

    min_h = h0 + h1 + gap1 + 4
    min_w = 2 * w0 + gap0 + 3
    if min_h > 30 or min_w > 30:
        return None
    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    pairs: list[tuple[dict, dict]] = []

    c_left = random.randint(1, W - (2 * w0 + gap0) - 1)
    c_right = c_left + w0 + gap0
    block_c = c_left + w0 + random.randint(1, gap0 - w1 - 1)
    block_off_r = random.randint(1, h0 - h1 - 1)

    side = random.choice(("above", "below"))
    if side == "above":
        r_pair = random.randint(h1 + gap1 + 1, H - h0 - 2)
        mate_r = r_pair - h1 - gap1
    else:
        r_pair = random.randint(1, H - h0 - h1 - gap1 - 2)
        mate_r = r_pair + h0 + gap1
    block_r = r_pair + block_off_r

    positions0 = [(r_pair, c_left), (r_pair, c_right)]
    positions1 = [(block_r, block_c), (mate_r, block_c)]
    if random.random() < 0.5:
        positions0.reverse()
    if random.random() < 0.5:
        positions1.reverse()

    for r, c, h, w in [
        (*positions0[0], h0, w0),
        (*positions0[1], h0, w0),
        (*positions1[0], h1, w1),
        (*positions1[1], h1, w1),
    ]:
        if not _can_reserve(occupied, r, c, h, w):
            return None

    obj0a = _add_object(grid, occupied, spec0[0], spec0[2], positions0[0][0], positions0[0][1])
    obj0b = _add_object(grid, occupied, spec0[1], spec0[3], positions0[1][0], positions0[1][1])
    obj1a = _add_object(grid, occupied, spec1[0], spec1[2], positions1[0][0], positions1[0][1])
    obj1b = _add_object(grid, occupied, spec1[1], spec1[3], positions1[1][0], positions1[1][1])
    pairs.extend([(obj0a, obj0b), (obj1a, obj1b)])

    for _ in range(random.randint(0, 2)):
        spec = _random_pair_spec(used_shapes)
        if spec is None:
            return None
        placed = _place_pair_aligned(grid, occupied, spec)
        if placed is None:
            return None
        pairs.append(placed)

    transform = random.choice(("identity", "rot90", "rot180", "rot270", "flipud", "fliplr"))
    expected = _expected_output(grid.shape, pairs)
    if transform == "rot90":
        grid = np.rot90(grid).copy()
        expected = np.rot90(expected).copy()
    elif transform == "rot180":
        grid = np.rot90(grid, 2).copy()
        expected = np.rot90(expected, 2).copy()
    elif transform == "rot270":
        grid = np.rot90(grid, 3).copy()
        expected = np.rot90(expected, 3).copy()
    elif transform == "flipud":
        grid = np.flipud(grid).copy()
        expected = np.flipud(expected).copy()
    elif transform == "fliplr":
        grid = np.fliplr(grid).copy()
        expected = np.fliplr(expected).copy()

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        if random.random() < 0.5:
            grid = _blocked_attempt()
            if grid is not None:
                return grid
            grid = _clear_attempt()
        else:
            grid = _clear_attempt()
            if grid is not None:
                return grid
            grid = _blocked_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate complementary-fragment input")
