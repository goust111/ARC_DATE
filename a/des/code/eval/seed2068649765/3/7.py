from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
KEY_COLOR = 8
BORDER_COLORS = (5, 6, 7, 9)
BG_COLOR = 0

RECOLOR_BY_RINGS = {
    1: RED,
    2: GREEN,
    3: YELLOW,
}

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]

GENERATE_RETRIES = 300
PLACEMENT_PAD = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(x) for x in grid.flat).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.tobytes()


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _unique_rotations(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen = set()
    out = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = _mask_key(rotated)
        if key in seen:
            continue
        seen.add(key)
        out.append((k, rotated))
    return out


def _count_rings(grid: np.ndarray, body_bbox: tuple[int, int, int, int], bg: int) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = body_bbox
    count = 0
    layer = 1
    while True:
        rr0, cc0 = r0 - layer, c0 - layer
        rr1, cc1 = r1 + layer, c1 + layer
        if rr0 < 0 or cc0 < 0 or rr1 >= h or cc1 >= w:
            break
        shell_values = set()
        for r in range(rr0, rr1 + 1):
            for c in range(cc0, cc1 + 1):
                if r0 - (layer - 1) <= r <= r1 + (layer - 1) and c0 - (layer - 1) <= c <= c1 + (layer - 1):
                    continue
                shell_values.add(int(grid[r, c]))
        if len(shell_values) != 1:
            break
        value = next(iter(shell_values))
        if value in (bg, BLUE, KEY_COLOR):
            break
        count += 1
        layer += 1
    return count


def _extract_locks(grid: np.ndarray, bg: int) -> list[dict]:
    locks = []
    for body_coords in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(body_coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sub_bg = sub == bg
        notch_components = []
        for comp in _components(sub_bg):
            if np.any(comp[:, 0] == 0) or np.any(comp[:, 1] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1) or np.any(comp[:, 1] == sub.shape[1] - 1):
                notch_components.append(comp)
        if len(notch_components) != 1:
            continue
        notch_coords = notch_components[0]
        nr0, nc0, nr1, nc1 = _bbox(notch_coords)
        notch_mask = np.zeros((nr1 - nr0 + 1, nc1 - nc0 + 1), dtype=bool)
        notch_mask[notch_coords[:, 0] - nr0, notch_coords[:, 1] - nc0] = True
        ring_count = _count_rings(grid, (r0, c0, r1, c1), bg)
        if ring_count not in RECOLOR_BY_RINGS:
            continue
        locks.append(
            {
                "body_bbox": (r0, c0, r1, c1),
                "notch_bbox": (r0 + nr0, c0 + nc0, r0 + nr1, c0 + nc1),
                "notch_mask": notch_mask,
                "ring_count": ring_count,
            }
        )
    return locks


def _extract_keys(grid: np.ndarray) -> list[dict]:
    keys = []
    for coords in _components(grid == KEY_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        keys.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )
    return keys


def _match_locks_to_keys(locks: list[dict], keys: list[dict]) -> list[tuple[int, int]]:
    candidates: list[list[tuple[int, int]]] = []
    for lock in locks:
        matches = []
        for key_index, key in enumerate(keys):
            for rotation, rotated in _unique_rotations(key["mask"]):
                if np.array_equal(rotated, lock["notch_mask"]):
                    matches.append((key_index, rotation))
        candidates.append(matches)

    order = sorted(range(len(locks)), key=lambda idx: len(candidates[idx]))
    assignment: list[tuple[int, int] | None] = [None] * len(locks)
    used_keys: set[int] = set()

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        lock_index = order[depth]
        for key_index, rotation in candidates[lock_index]:
            if key_index in used_keys:
                continue
            used_keys.add(key_index)
            assignment[lock_index] = (key_index, rotation)
            if dfs(depth + 1):
                return True
            used_keys.remove(key_index)
            assignment[lock_index] = None
        return False

    assert dfs(0), "every lock should match a unique key"
    return [item for item in assignment if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each azure key with the blue lock notch it complements under rotation,
    recolor it by the number of surrounding border layers, and insert it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    locks = _extract_locks(g, bg)
    keys = _extract_keys(g)
    assert locks and len(locks) == len(keys)

    assignment = _match_locks_to_keys(locks, keys)

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = bg

    for lock, (key_index, rotation) in zip(locks, assignment):
        color = RECOLOR_BY_RINGS[lock["ring_count"]]
        rotated = np.rot90(keys[key_index]["mask"], rotation)
        r0, c0, r1, c1 = lock["notch_bbox"]
        assert rotated.shape == (r1 - r0 + 1, c1 - c0 + 1)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[rotated] = color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_polyomino(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        target_area = rng.randint(4, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.add(cand)
            options = []
            for cand in frontier:
                grown = _normalize_cells(cells | {cand})
                rows = [r for r, _ in grown]
                cols = [c for _, c in grown]
                if max(rows) + 1 <= 4 and max(cols) + 1 <= 4:
                    options.append(grown)
            if not options:
                break
            cells = rng.choice(options)
        mask = _cells_to_mask(_normalize_cells(cells))
        if mask.sum() != target_area:
            continue
        if 1 in mask.shape or mask.all():
            continue
        if len(_unique_rotations(mask)) != 4:
            continue
        return mask
    return None


def _build_lock_sprite(
    base_mask: np.ndarray,
    side: str,
    ring_count: int,
    ring_colors: list[int],
    rng: random.Random,
) -> dict | None:
    mh, mw = [int(x) for x in base_mask.shape]
    if side in ("top", "bottom"):
        body_h = max(5, mh + 1 + rng.randint(0, 1))
        body_w = max(5, mw + 2 + rng.randint(0, 1))
        if body_w <= mw + 1:
            return None
        notch_r = 0 if side == "top" else body_h - mh
        notch_c = rng.randint(1, body_w - mw - 1)
    else:
        body_h = max(5, mh + 2 + rng.randint(0, 1))
        body_w = max(5, mw + 1 + rng.randint(0, 1))
        if body_h <= mh + 1:
            return None
        notch_r = rng.randint(1, body_h - mh - 1)
        notch_c = 0 if side == "left" else body_w - mw

    total_h = body_h + 2 * ring_count
    total_w = body_w + 2 * ring_count
    sprite = np.full((total_h, total_w), BG_COLOR, dtype=int)

    for layer, color in enumerate(ring_colors):
        sprite[layer : total_h - layer, layer : total_w - layer] = color

    body_top = ring_count
    body_left = ring_count
    sprite[body_top : body_top + body_h, body_left : body_left + body_w] = BLUE
    notch_top = body_top + notch_r
    notch_left = body_left + notch_c
    notch_view = sprite[notch_top : notch_top + mh, notch_left : notch_left + mw]
    notch_view[base_mask] = BG_COLOR

    blue_mask = sprite == BLUE
    if len(_components(blue_mask)) != 1:
        return None

    return {
        "assembly": sprite,
        "notch_mask": base_mask,
        "notch_rel": (notch_top, notch_left),
        "ring_count": ring_count,
    }


def _find_position(
    blocked: np.ndarray,
    obj_h: int,
    obj_w: int,
    pad: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates = []
    for r in range(h - obj_h + 1):
        rr0 = max(0, r - pad)
        rr1 = min(h, r + obj_h + pad)
        for c in range(w - obj_w + 1):
            cc0 = max(0, c - pad)
            cc1 = min(w, c + obj_w + pad)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return rng.choice(candidates)


def _reserve(blocked: np.ndarray, top: int, left: int, obj_h: int, obj_w: int, pad: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, top - pad)
    rr1 = min(h, top + obj_h + pad)
    cc0 = max(0, left - pad)
    cc1 = min(w, left + obj_w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _expected_output(
    grid: np.ndarray,
    specs: list[dict],
    lock_positions: list[tuple[int, int]],
    key_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for spec, (kr, kc) in zip(specs, key_positions):
        key_mask = spec["key_mask"]
        out[kr : kr + key_mask.shape[0], kc : kc + key_mask.shape[1]][key_mask] = BG_COLOR
    for spec, (lr, lc) in zip(specs, lock_positions):
        nr = lr + spec["notch_rel"][0]
        nc = lc + spec["notch_rel"][1]
        color = RECOLOR_BY_RINGS[spec["ring_count"]]
        out[nr : nr + spec["notch_mask"].shape[0], nc : nc + spec["notch_mask"].shape[1]][spec["notch_mask"]] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_locks = 3 if rng.random() < 0.7 else 4
    ring_counts = [1, 2, 3]
    if num_locks == 4:
        ring_counts.append(rng.choice([1, 2, 3]))
    rng.shuffle(ring_counts)

    side_choices = rng.sample(SIDES, 3)
    if num_locks == 4:
        side_choices.append(rng.choice(SIDES))
    rng.shuffle(side_choices)

    used_shape_keys = set()
    specs = []
    for ring_count, side in zip(ring_counts, side_choices):
        mask = _random_polyomino(rng)
        if mask is None:
            return None
        shape_key = _rotation_key(mask)
        if shape_key in used_shape_keys:
            return None
        used_shape_keys.add(shape_key)

        ring_colors = rng.sample(BORDER_COLORS, ring_count)
        built = _build_lock_sprite(mask, side, ring_count, ring_colors, rng)
        if built is None:
            return None

        rotation = rng.randrange(4)
        key_mask = np.rot90(mask, rotation)
        specs.append(
            {
                **built,
                "key_mask": key_mask,
            }
        )

    if num_locks == 3:
        height = rng.randint(22, 29)
        width = rng.randint(24, 30)
    else:
        height = rng.randint(27, 30)
        width = rng.randint(27, 30)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    lock_positions: list[tuple[int, int] | None] = [None] * num_locks
    for index in sorted(range(num_locks), key=lambda i: specs[i]["assembly"].shape[0] * specs[i]["assembly"].shape[1], reverse=True):
        assembly = specs[index]["assembly"]
        pos = _find_position(blocked, assembly.shape[0], assembly.shape[1], PLACEMENT_PAD, rng)
        if pos is None:
            return None
        r, c = pos
        grid[r : r + assembly.shape[0], c : c + assembly.shape[1]] = assembly
        _reserve(blocked, r, c, assembly.shape[0], assembly.shape[1], PLACEMENT_PAD)
        lock_positions[index] = pos

    key_positions: list[tuple[int, int] | None] = [None] * num_locks
    key_order = list(range(num_locks))
    rng.shuffle(key_order)
    for index in key_order:
        key_mask = specs[index]["key_mask"]
        key_h, key_w = key_mask.shape
        pos = _find_position(blocked, key_h, key_w, PLACEMENT_PAD, rng)
        if pos is None:
            return None
        r, c = pos
        grid[r : r + key_h, c : c + key_w][key_mask] = KEY_COLOR
        _reserve(blocked, r, c, key_h, key_w, PLACEMENT_PAD)
        key_positions[index] = pos

    assert all(pos is not None for pos in lock_positions)
    assert all(pos is not None for pos in key_positions)
    expected = _expected_output(
        grid,
        specs,
        [pos for pos in lock_positions if pos is not None],
        [pos for pos in key_positions if pos is not None],
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
