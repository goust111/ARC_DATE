from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _has_internal_void(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~mask) & (~outside)))


def _has_rotation_symmetry(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rotated = np.rot90(mask, k)
        if rotated.shape == mask.shape and np.array_equal(rotated, mask):
            return True
    return False


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return min(variants)


def _find_monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            coords: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[comp["coords"][:, 0] - r0, comp["coords"][:, 1] - c0] = True
    return mask


def _parse_lock(grid: np.ndarray, comp: dict) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None

    cavity = patch != color
    if not cavity.any():
        return None
    if cavity[0, :].any() or cavity[-1, :].any() or cavity[:, 0].any() or cavity[:, -1].any():
        return None
    if _mask_component_count(cavity) != 1:
        return None

    secret_colors = set(map(int, np.unique(patch[cavity]))) - {BG}
    if len(secret_colors) != 1:
        return None

    hole_box = _bbox_from_mask(cavity)
    assert hole_box is not None
    hr0, hc0, hr1, hc1 = hole_box
    hole_patch = patch[hr0 : hr1 + 1, hc0 : hc1 + 1]
    hole_mask = hole_patch != color
    if _has_rotation_symmetry(hole_mask):
        return None

    secret_color = secret_colors.pop()
    secret_patch = np.zeros_like(hole_patch)
    secret_patch[hole_patch == secret_color] = secret_color
    if not secret_patch.any():
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "hole_mask": hole_mask,
        "secret_patch": secret_patch,
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = _find_monochrome_components(grid)
    locks: list[dict] = []
    lock_comp_indices: set[int] = set()

    for i, comp in enumerate(comps):
        lock = _parse_lock(grid, comp)
        if lock is not None:
            lock["comp_index"] = i
            locks.append(lock)
            lock_comp_indices.add(i)

    keys: list[dict] = []
    for i, comp in enumerate(comps):
        if i in lock_comp_indices:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        inside_lock = any(
            lr0 <= r0 <= r1 <= lr1 and lc0 <= c0 <= c1 <= lc1
            for lr0, lc0, lr1, lc1 in (lock["bbox"] for lock in locks)
        )
        if inside_lock:
            continue

        keys.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "coords": comp["coords"],
                "mask": _component_mask(comp),
            }
        )

    keys.sort(key=lambda key: key["bbox"])
    locks.sort(key=lambda lock: lock["bbox"])
    return locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key to the lock cavity with the same outline under rotation,
    then replace the key with the lock's hidden pattern rotated into the key's orientation.
    """
    g = np.asarray(grid, dtype=int)
    locks, keys = _extract_objects(g)
    assert locks and keys

    out = g.copy()
    used_locks: set[int] = set()

    for key in keys:
        candidates: list[tuple[int, int]] = []
        for i, lock in enumerate(locks):
            if i in used_locks or lock["color"] != key["color"]:
                continue
            for k in range(4):
                rotated_hole = np.rot90(lock["hole_mask"], k)
                if rotated_hole.shape == key["mask"].shape and np.array_equal(rotated_hole, key["mask"]):
                    candidates.append((i, k))

        if not candidates:
            for i, lock in enumerate(locks):
                if i in used_locks:
                    continue
                for k in range(4):
                    rotated_hole = np.rot90(lock["hole_mask"], k)
                    if rotated_hole.shape == key["mask"].shape and np.array_equal(rotated_hole, key["mask"]):
                        candidates.append((i, k))

        assert len(candidates) == 1, "each key must match exactly one lock"
        lock_index, rotation = candidates[0]
        used_locks.add(lock_index)

        kr0, kc0, kr1, kc1 = key["bbox"]
        out[key["coords"][:, 0], key["coords"][:, 1]] = BG

        rotated_secret = np.rot90(locks[lock_index]["secret_patch"], rotation)
        sub = out[kr0 : kr1 + 1, kc0 : kc1 + 1]
        paint = rotated_secret != BG
        sub[paint] = rotated_secret[paint]

    return out


def _grow_connected_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    frontier = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        nr, nc = random.choice(list(frontier))
        frontier.discard((nr, nc))
        if (nr, nc) in cells:
            continue
        cells.add((nr, nc))
        for dr, dc in NEIGHBORS4:
            rr, cc = nr + dr, nc + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    if len(cells) != target_area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_hole_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        total = h * w
        min_area = max(4, total // 2)
        max_area = total - 1
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(h, w, target_area)
        if mask is None:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.all():
            continue
        if _mask_component_count(mask) != 1:
            continue
        if _has_internal_void(mask):
            continue
        if _has_rotation_symmetry(mask):
            continue
        return mask
    return None


def _random_secret_mask(hole_mask: np.ndarray) -> np.ndarray | None:
    hole_coords = np.argwhere(hole_mask)
    hole_area = int(hole_coords.shape[0])
    for _ in range(200):
        density = random.uniform(0.25, 0.75)
        secret = np.zeros_like(hole_mask, dtype=bool)
        for r, c in hole_coords:
            if random.random() < density:
                secret[int(r), int(c)] = True
        area = int(secret.sum())
        if area == 0 or area == hole_area:
            continue
        if _has_rotation_symmetry(secret):
            continue
        return secret
    return None


def _build_pair(lock_color: int, secret_color: int, used_shapes: set) -> dict | None:
    hole_mask = _random_hole_mask()
    if hole_mask is None:
        return None

    shape_key = _canonical_rotation_key(hole_mask)
    if shape_key in used_shapes:
        return None

    secret_mask = _random_secret_mask(hole_mask)
    if secret_mask is None:
        return None

    hole_h, hole_w = hole_mask.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)

    lock_patch = np.full((top + hole_h + bottom, left + hole_w + right), lock_color, dtype=int)
    lock_window = lock_patch[top : top + hole_h, left : left + hole_w]
    lock_window[hole_mask] = BG
    lock_window[secret_mask] = secret_color

    rotation = random.randrange(4)
    key_mask = np.rot90(hole_mask, rotation)
    key_patch = np.zeros(key_mask.shape, dtype=int)
    key_patch[key_mask] = lock_color

    secret_patch = np.zeros(hole_mask.shape, dtype=int)
    secret_patch[secret_mask] = secret_color

    used_shapes.add(shape_key)
    return {
        "lock_color": lock_color,
        "secret_color": secret_color,
        "rotation": rotation,
        "hole_mask": hole_mask,
        "key_mask": key_mask,
        "secret_patch": secret_patch,
        "lock_patch": lock_patch,
        "key_patch": key_patch,
    }


def _place_patches(patches: list[np.ndarray], grid_shape: tuple[int, int], pad: int = 1) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)

    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            rr0 = max(0, r - pad)
            rr1 = min(h, r + ph + pad)
            for c in range(w - pw + 1):
                cc0 = max(0, c - pad)
                cc1 = min(w, c + pw + pad)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        positions[idx] = (r, c)
        rr0 = max(0, r - pad)
        rr1 = min(h, r + ph + pad)
        cc0 = max(0, c - pad)
        cc1 = min(w, c + pw + pad)
        occupied[rr0:rr1, cc0:cc1] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _render_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != BG
    sub[mask] = patch[mask]


def _expected_output(inp: np.ndarray, pairs: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for pair, (top, left) in zip(pairs, positions, strict=False):
        key_mask = pair["key_mask"]
        rotated_secret = np.rot90(pair["secret_patch"], pair["rotation"])
        sub = out[top : top + key_mask.shape[0], left : left + key_mask.shape[1]]
        sub[key_mask] = BG
        paint = rotated_secret != BG
        sub[paint] = rotated_secret[paint]
    return out


def _generate_candidate() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, 2 * pair_count)
    lock_colors = colors[:pair_count]
    secret_colors = colors[pair_count:]

    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    pairs: list[dict] = []
    for lock_color, secret_color in zip(lock_colors, secret_colors, strict=False):
        pair = None
        for _ in range(100):
            pair = _build_pair(lock_color, secret_color, used_shapes)
            if pair is not None:
                break
        if pair is None:
            return None
        pairs.append(pair)

    patches = [pair["lock_patch"] for pair in pairs] + [pair["key_patch"] for pair in pairs]
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)

    base = max(18, int(np.ceil(np.sqrt(total_area * 2.4))))
    grid_h = random.randint(max(max_h + 2, base), max(max_h + 6, base + 8))
    grid_w = random.randint(max(max_w + 2, base), max(max_w + 6, base + 8))

    positions = _place_patches(patches, (grid_h, grid_w), pad=1)
    if positions is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for patch, (top, left) in zip(patches, positions, strict=False):
        _render_patch(inp, patch, top, left)

    expected = _expected_output(inp, pairs, positions[pair_count:])
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
