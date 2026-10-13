from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

MAX_GENERATION_ATTEMPTS = 500
MASK_POOL_ATTEMPTS = 6000
PLACEMENT_ATTEMPTS = 250

CASE_CYCLE = (
    (4, 4, 1),
    (5, 5, 2),
    (6, 4, 3),
    (4, 5, 0),
    (5, 4, 1),
    (6, 5, 2),
)

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGS: set[str] = set()
_MASK_POOLS: dict[int, list[np.ndarray]] = {}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        visited[rr, cc] = True
        stack = [(rr, cc)]
        coords: list[tuple[int, int]] = []

        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIRS8 if int(connectivity) == 8 else ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(np.array(coords, dtype=int))

    return comps


def _grid_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in arr.tolist())


def _rotation_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_grid_key(np.rot90(arr, k)) for k in range(4))


def _array_sig(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _occupied_corner_count(mask: np.ndarray) -> int:
    side_r, side_c = mask.shape
    return int(mask[0, 0]) + int(mask[0, side_c - 1]) + int(mask[side_r - 1, 0]) + int(mask[side_r - 1, side_c - 1])


def _occupied_edge_count(mask: np.ndarray) -> int:
    return int(mask[0, :].sum() + mask[-1, :].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())


def _shape_signature(mask: np.ndarray) -> tuple[int, int, int]:
    return int(mask.sum()), _occupied_edge_count(mask), _occupied_corner_count(mask)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _king_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]

    while (r, c) != (er, ec):
        dr = _sign(er - r)
        dc = _sign(ec - c)
        options: list[tuple[int, int]] = []
        weights: list[int] = []

        if dr != 0 and dc != 0:
            options.extend(((r + dr, c + dc), (r + dr, c), (r, c + dc)))
            weights.extend((5, 1, 1))
        elif dr != 0:
            options.append((r + dr, c))
            weights.append(1)
        else:
            options.append((r, c + dc))
            weights.append(1)

        r, c = random.choices(options, weights=weights, k=1)[0]
        path.append((r, c))

    return path


def _is_valid_shape(mask: np.ndarray) -> bool:
    if mask.shape[0] != mask.shape[1]:
        return False
    if not (6 <= int(mask.sum()) <= 8):
        return False
    comps = _components_mask(mask, connectivity=8)
    if len(comps) != 1:
        return False
    bbox = _bbox_coords(np.argwhere(mask))
    side = mask.shape[0]
    if bbox != (0, 0, side - 1, side - 1):
        return False
    if len({_grid_key(np.rot90(mask, k).astype(int)) for k in range(4)}) != 4:
        return False
    if np.all(mask):
        return False
    return True


def _random_shape(side: int, area: int) -> np.ndarray | None:
    for _ in range(200):
        hub = (random.randint(1, side - 2), random.randint(1, side - 2))
        anchors = [
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        ]
        random.shuffle(anchors)

        cells = {hub}
        for anchor in anchors:
            cells.update(_king_path(hub, anchor))
        if len(cells) > area:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break

            frontier_list = sorted(frontier)
            weights = []
            for r, c in frontier_list:
                weight = 1
                if r in (0, side - 1):
                    weight += 2
                if c in (0, side - 1):
                    weight += 2
                if (r, c) in ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)):
                    weight += 1
                weights.append(weight)

            nr, nc = random.choices(frontier_list, weights=weights, k=1)[0]
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue
        if _is_valid_shape(mask):
            return mask

    return None


def _ensure_mask_pool(side: int, target_size: int = 64) -> list[np.ndarray]:
    pool = _MASK_POOLS.setdefault(side, [])
    seen = {_grid_key(mask.astype(int)) for mask in pool}
    attempts = 0

    while len(pool) < target_size and attempts < MASK_POOL_ATTEMPTS:
        attempts += 1
        mask = _random_shape(side, random.randint(6, 8))
        if mask is None:
            continue
        key = _grid_key(mask.astype(int))
        if key in seen:
            continue
        seen.add(key)
        pool.append(mask)

    return pool


def _choose_shape_pair(side: int) -> tuple[np.ndarray, np.ndarray]:
    for target_size in (48, 72, 96):
        pool = _ensure_mask_pool(side, target_size=target_size)
        by_signature: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
        for mask in pool:
            by_signature[_shape_signature(mask)].append(mask)

        candidate_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for masks in by_signature.values():
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    if _rotation_key(masks[i].astype(int)) != _rotation_key(masks[j].astype(int)):
                        candidate_pairs.append((masks[i], masks[j]))

        if candidate_pairs:
            base, odd = random.choice(candidate_pairs)
            return base.copy(), odd.copy()

    raise RuntimeError(f"failed to find a same-signature non-rotational pair for side={side}")


def _rotation_schedule(base_count: int) -> list[int]:
    rots = [0, 1, 2, 3]
    while len(rots) < base_count:
        rots.append(random.randrange(4))
    random.shuffle(rots)
    return rots


def _sprite_from_mask(mask: np.ndarray, color: int, rot_k: int) -> np.ndarray:
    rotated = np.rot90(mask, rot_k)
    return rotated.astype(int) * int(color)


def _boxes_have_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_sprites(shape: tuple[int, int], sprites: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    order = list(range(len(sprites)))
    random.shuffle(order)

    for idx in order:
        sprite = sprites[idx]
        side = sprite.shape[0]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, h - side - 1)
            c0 = random.randint(1, w - side - 1)
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if any(not _boxes_have_gap(box, prev) for prev in boxes):
                continue
            patch = grid[r0 : r0 + side, c0 : c0 + side]
            patch[sprite != BLACK] = sprite[sprite != BLACK]
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return grid


def _choose_case() -> tuple[int, int, int]:
    global _GEN_CALL_COUNT
    case = CASE_CYCLE[_GEN_CALL_COUNT % len(CASE_CYCLE)]
    _GEN_CALL_COUNT += 1
    return case


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the single sprite whose shape is not rotation-equivalent to the others, then crop it.

    Input structure:
    - The grid has a black background and 5 to 7 separated monochrome sprites of one non-background color.
    - Every sprite is 8-connected and is considered through its tight bounding-box crop.
    - Most sprites are rotated copies of one underlying irregular shape; exactly one sprite is not.

    Rule:
    - Find every non-background 8-connected sprite and crop each one to its bounding box.
    - Group the crops by rotation equivalence: two crops are equivalent if one matches the other after a 0/90/180/270 degree rotation.
    - Identify the unique equivalence class with exactly one member.
    - Output that member's crop unchanged, preserving its original orientation.

    Key insight:
    - Visual difference alone is misleading here: several sprites look different only because they are rotated views of the same shape, so the odd one out appears only after rotation-based grouping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "expected multiple sprites"

    crops: list[np.ndarray] = []
    classes: dict[tuple[tuple[int, ...], ...], list[int]] = defaultdict(list)
    for idx, coords in enumerate(comps):
        crop = _crop_bbox(g, _bbox_coords(coords))
        crops.append(crop)
        shape_mask = (crop != bg).astype(int)
        classes[_rotation_key(shape_mask)].append(idx)

    singleton_members = [members[0] for members in classes.values() if len(members) == 1]
    assert len(singleton_members) == 1, "expected exactly one singleton rotation class"
    return crops[singleton_members[0]].copy()


def _build_candidate() -> tuple[np.ndarray, np.ndarray, int] | None:
    base_count, side, odd_rot = _choose_case()
    total_count = base_count + 1

    base_mask, odd_mask = _choose_shape_pair(side)
    color = random.choice(PALETTE)

    sprites = [_sprite_from_mask(base_mask, color, rot_k) for rot_k in _rotation_schedule(base_count)]
    odd_sprite = _sprite_from_mask(odd_mask, color, odd_rot % 4)
    sprites.append(odd_sprite)

    size_lo = 18 + max(0, total_count - 5)
    size_hi = min(26, size_lo + 5 + (side - 4))
    h = random.randint(size_lo, size_hi)
    w = random.randint(size_lo, size_hi)

    grid = _place_sprites((h, w), sprites)
    if grid is None:
        return None
    return grid, odd_sprite, base_count


def _validate_instance(grid: np.ndarray, odd_sprite: np.ndarray, base_count: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    bg = _get_bg(grid)
    if bg != BLACK:
        return False

    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    if len(colors) != 1:
        return False

    comps = _components_mask(grid != bg, connectivity=8)
    if len(comps) != base_count + 1:
        return False

    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    raw_counts: Counter[tuple[tuple[int, ...], ...]] = Counter()
    rot_classes: dict[tuple[tuple[int, ...], ...], list[int]] = defaultdict(list)

    for coords in comps:
        crop = _crop_bbox(grid, _bbox_coords(coords))
        mask = (crop != bg).astype(int)
        crops.append(crop)
        masks.append(mask)
        raw_counts[_grid_key(mask)] += 1

    if len({mask.shape for mask in masks}) != 1:
        return False
    if len({int(mask.sum()) for mask in masks}) != 1:
        return False

    for idx, mask in enumerate(masks):
        rot_classes[_rotation_key(mask)].append(idx)

    if sorted(len(members) for members in rot_classes.values()) != [1, base_count]:
        return False

    singleton_idx = [members[0] for members in rot_classes.values() if len(members) == 1]
    if len(singleton_idx) != 1:
        return False
    odd_idx = singleton_idx[0]

    if not np.array_equal(main(grid), crops[odd_idx]):
        return False
    if not np.array_equal(crops[odd_idx], odd_sprite):
        return False

    large_class = next(members for members in rot_classes.values() if len(members) == base_count)
    if len({_grid_key(masks[idx]) for idx in large_class}) != 4:
        return False

    exact_singletons = sum(1 for count in raw_counts.values() if count == 1)
    if exact_singletons <= 1:
        return False

    signatures = Counter(_shape_signature(mask.astype(bool)) for mask in masks)
    if signatures[_shape_signature(masks[odd_idx].astype(bool))] <= 1:
        return False

    out_sig = _array_sig(odd_sprite)
    if out_sig in _SEEN_OUTPUT_SIGS:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, odd_sprite, base_count = built
        if _validate_instance(grid, odd_sprite, base_count):
            _SEEN_OUTPUT_SIGS.add(_array_sig(odd_sprite))
            return grid
    raise RuntimeError("failed to generate a valid rotation-equivalence odd-sprite instance")
