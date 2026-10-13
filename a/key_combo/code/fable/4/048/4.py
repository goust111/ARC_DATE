from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = _DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in _DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in _DIR8:
        yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    neigh = _neighbors4 if connectivity == 4 else _neighbors8
    while q:
        r, c = q.popleft()
        for nr, nc in neigh(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_sprites(grid: np.ndarray) -> list[np.ndarray]:
    mask = grid == GREY
    seen = np.zeros_like(mask, dtype=bool)
    sprites: list[np.ndarray] = []
    neigh = _neighbors8
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in neigh(cr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comp = np.zeros_like(mask, dtype=bool)
        for cr, cc in cells:
            comp[cr, cc] = True
        r0, c0, r1, c1 = _bbox(comp)
        sprites.append(comp[r0 : r1 + 1, c0 : c1 + 1])
    return sprites


def _partner_mask(mask: np.ndarray) -> np.ndarray:
    return np.rot90(~mask, 2)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the lone grey sprite whose 180-degree complementary partner is missing, and output that missing partner.

    Input structure:
    - Black background with several separated grey 4-connected sprites.
    - Apart from one outlier, the sprites can be partitioned into pairs with equal bounding-box size.
    - In each valid pair, rotating one sprite by 180 degrees and aligning bounding boxes makes the two sprites fill the whole rectangle with no overlap.

    Rule:
    - Extract each grey sprite as a boolean mask in its own bounding box.
    - For every sprite, its unique partner is the 180-degree rotation of the bounding-box complement.
    - Use multiset pairing under that partner relation to find the unique sprite class with one unmatched copy, then output its missing partner in the same bounding-box size using grey on black.

    Key insight:
    - The correct pairing relation is complementary completion after a half-turn, not shape equality; repeated translated copies are only distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _extract_sprites(g)
    assert sprites, "expected at least one grey sprite"

    counts = Counter(_mask_key(sprite) for sprite in sprites)
    reps = {_mask_key(sprite): sprite for sprite in sprites}

    surplus: list[tuple[tuple[int, int], bytes]] = []
    handled: set[tuple[tuple[int, int], bytes]] = set()
    for key in counts:
        if key in handled:
            continue
        sprite = reps[key]
        partner = _partner_mask(sprite)
        partner_key = _mask_key(partner)
        c1 = counts[key]
        c2 = counts.get(partner_key, 0)
        if partner_key == key:
            if c1 % 2 == 1:
                surplus.extend([key] * (c1 % 2))
            handled.add(key)
            continue
        if c1 > c2:
            surplus.extend([key] * (c1 - c2))
        elif c2 > c1:
            surplus.extend([partner_key] * (c2 - c1))
        handled.add(key)
        handled.add(partner_key)

    assert len(surplus) == 1, "expected exactly one unmatched sprite"
    outlier = reps[surplus[0]]
    partner = _partner_mask(outlier)

    out = np.zeros(outlier.shape, dtype=int)
    out[partner] = GREY
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _random_sprite_mask(h: int, w: int, forbidden: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(400):
        density = random.uniform(0.35, 0.65)
        mask = np.random.rand(h, w) < density
        comp = ~mask
        if mask.all() or (~mask).all():
            continue
        if not (_touches_all_sides(mask) and _touches_all_sides(comp)):
            continue
        if not (_is_connected(mask, connectivity=8) and _is_connected(comp, connectivity=8)):
            continue
        if np.array_equal(mask, _partner_mask(mask)) and random.random() < 0.7:
            continue
        key = _mask_key(mask)
        partner_key = _mask_key(_partner_mask(mask))
        if key in forbidden or partner_key in forbidden:
            continue
        return mask
    return None


def _sprite_array(mask: np.ndarray) -> np.ndarray:
    arr = np.zeros(mask.shape, dtype=int)
    arr[mask] = GREY
    return arr


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_objects(arrays: list[np.ndarray], H: int, W: int) -> np.ndarray | None:
    order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[0] * arrays[i].shape[1], reverse=True)
    occupied = np.zeros((H, W), dtype=bool)
    grid = np.zeros((H, W), dtype=int)

    for idx in order:
        arr = arrays[idx]
        h, w = map(int, arr.shape)
        spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(spots)
        placed = False
        for r, c in spots:
            if not _can_place(occupied, r, c, h, w):
                continue
            grid[r : r + h, c : c + w][arr == GREY] = GREY
            occupied[r : r + h, c : c + w] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _choose_dims() -> tuple[int, int]:
    dims = [(h, w) for h in range(4, 8) for w in range(4, 8) if h % 2 == 0 or w % 2 == 0]
    return random.choice(dims)


def _build_pair_type(
    dims: tuple[int, int],
    forbidden: set[tuple[tuple[int, int], bytes]],
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _random_sprite_mask(dims[0], dims[1], forbidden)
    if mask is None:
        return None
    return mask, _partner_mask(mask)


def _gen_attempt() -> np.ndarray | None:
    special_dims = _choose_dims()
    forbidden: set[tuple[tuple[int, int], bytes]] = set()

    pair_specs: list[tuple[np.ndarray, np.ndarray, int]] = []

    special_pair = _build_pair_type(special_dims, forbidden)
    if special_pair is None:
        return None
    special_mult = 2 if random.random() < 0.75 else 1
    pair_specs.append((special_pair[0], special_pair[1], special_mult))
    forbidden.add(_mask_key(special_pair[0]))
    forbidden.add(_mask_key(special_pair[1]))

    if random.random() < 0.45:
        extra_special = _build_pair_type(special_dims, forbidden)
        if extra_special is None:
            return None
        extra_mult = 1 if special_mult == 2 else 2
        pair_specs.append((extra_special[0], extra_special[1], extra_mult))
        forbidden.add(_mask_key(extra_special[0]))
        forbidden.add(_mask_key(extra_special[1]))

    other_type_count = random.randint(1, 3)
    for _ in range(other_type_count):
        dims = _choose_dims()
        pair = _build_pair_type(dims, forbidden)
        if pair is None:
            return None
        mult = 2 if random.random() < 0.35 else 1
        pair_specs.append((pair[0], pair[1], mult))
        forbidden.add(_mask_key(pair[0]))
        forbidden.add(_mask_key(pair[1]))

    if all(mult == 1 for _, _, mult in pair_specs):
        i = random.randrange(len(pair_specs))
        base, partner, _ = pair_specs[i]
        pair_specs[i] = (base, partner, 2)

    outlier = _random_sprite_mask(special_dims[0], special_dims[1], forbidden)
    if outlier is None:
        return None
    if _mask_key(_partner_mask(outlier)) in forbidden:
        return None

    arrays: list[np.ndarray] = []
    for base, partner, mult in pair_specs:
        for _ in range(mult):
            arrays.append(_sprite_array(base))
            arrays.append(_sprite_array(partner))

    expected = _sprite_array(_partner_mask(outlier))
    arrays.append(_sprite_array(outlier))
    random.shuffle(arrays)

    total_area = sum((arr.shape[0] + 2) * (arr.shape[1] + 2) for arr in arrays)
    min_side = max(18, int(np.ceil(np.sqrt(total_area))) + 2)
    if min_side > 30:
        return None

    for _ in range(80):
        H = random.randint(min_side, 30)
        W = random.randint(min_side, 30)
        grid = _place_objects(arrays, H, W)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
