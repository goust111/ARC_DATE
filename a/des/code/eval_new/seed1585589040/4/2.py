import random
from typing import Any

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_components(grid: np.ndarray) -> list[dict[str, Any]]:
    mask = grid != 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, Any]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coord_arr)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            rel_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel_mask[coord_arr[:, 0] - r0, coord_arr[:, 1] - c0] = True
            colors = tuple(sorted(int(x) for x in np.unique(patch[rel_mask])))
            comps.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "mask": rel_mask,
                    "colors": colors,
                }
            )

    comps.sort(key=lambda comp: tuple(comp["bbox"]))
    return comps


def _expected_key_border_mask(side: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[h - 1, :] = True
    elif side == "bottom":
        mask[0, :] = True
    elif side == "left":
        mask[:, w - 1] = True
    else:
        mask[:, 0] = True
    return mask


def _analyze_lock(comp: dict[str, Any]) -> dict[str, Any] | None:
    patch = np.array(comp["patch"], dtype=int)
    mask = np.array(comp["mask"], dtype=bool)
    h, w = patch.shape
    if h != w or h not in (3, 4):
        return None
    if len(comp["colors"]) != 2:
        return None

    cavity = ~mask
    if not cavity.any():
        return None
    cavity_coords = np.argwhere(cavity)
    zr0, zc0, zr1, zc1 = _bbox_from_coords(cavity_coords)
    cavity_rect = np.zeros_like(cavity, dtype=bool)
    cavity_rect[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
    if not np.array_equal(cavity, cavity_rect):
        return None

    touches = []
    if zr0 == 0 and zr1 < h - 1:
        touches.append("top")
    if zr1 == h - 1 and zr0 > 0:
        touches.append("bottom")
    if zc0 == 0 and zc1 < w - 1:
        touches.append("left")
    if zc1 == w - 1 and zc0 > 0:
        touches.append("right")
    if len(touches) != 1:
        return None

    border_mask = np.zeros_like(mask, dtype=bool)
    for rr, cc in cavity_coords:
        for dr, dc in NEIGHBORS4:
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                border_mask[nr, nc] = True
    body_mask = mask & (~border_mask)
    if not body_mask.any() or not border_mask.any():
        return None

    body_colors = np.unique(patch[body_mask])
    border_colors = np.unique(patch[border_mask])
    if len(body_colors) != 1 or len(border_colors) != 1:
        return None
    body_color = int(body_colors[0])
    border_color = int(border_colors[0])
    if body_color == border_color:
        return None

    return {
        "bbox": tuple(comp["bbox"]),
        "patch": patch,
        "mask": mask,
        "colors": tuple(comp["colors"]),
        "side": touches[0],
        "cavity_bbox": (zr0, zc0, zr1, zc1),
        "cavity_shape": (zr1 - zr0 + 1, zc1 - zc0 + 1),
        "body_color": body_color,
        "border_color": border_color,
    }


def _analyze_key(comp: dict[str, Any]) -> dict[str, Any] | None:
    patch = np.array(comp["patch"], dtype=int)
    mask = np.array(comp["mask"], dtype=bool)
    if len(comp["colors"]) != 2:
        return None
    if not mask.all():
        return None
    return {
        "bbox": tuple(comp["bbox"]),
        "patch": patch,
        "mask": mask,
        "colors": tuple(comp["colors"]),
        "shape": tuple(int(x) for x in patch.shape),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each matching key into its lock notch after rotating and swapping its two colors.

    Input structure:
    - The background is black, and every foreground object is a 4-connected two-color component.
    - Locks are 3x3 or 4x4 square components whose occupied cells equal a solid square minus one side-attached rectangular notch.
    - Keys are the remaining two-color components; each key uses the same two colors as exactly one lock.

    Rule:
    - Detect each lock's notch side and notch rectangle, then infer the lock body color and notch-border color from the cells surrounding the notch.
    - Match a key to a lock by shared colors, rotate the key so its shape and border-colored strip align with the notch, and move it into the notch rectangle.
    - After insertion, swap the key's two colors, keep the lock in place, and erase the key from its original location.

    Key insight: The lock is recognizable from its square-minus-notch geometry, while the key's correct rotation is fixed by both the notch shape and which side of the key carries the lock-border color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    components = _extract_components(g)
    locks = []
    keys = []
    for comp in components:
        lock = _analyze_lock(comp)
        if lock is not None:
            locks.append(lock)
            continue
        key = _analyze_key(comp)
        if key is not None:
            keys.append(key)

    assert locks and len(locks) == len(keys)

    used_keys: set[int] = set()
    for lock in locks:
        body_color = int(lock["body_color"])
        border_color = int(lock["border_color"])
        pair_colors = {body_color, border_color}
        target_shape = tuple(lock["cavity_shape"])
        expected_border = _expected_key_border_mask(str(lock["side"]), target_shape)

        found_idx = None
        rotated_patch = None
        for exact_only in (True, False):
            for idx, key in enumerate(keys):
                if idx in used_keys:
                    continue
                key_colors = set(int(c) for c in key["colors"])
                if exact_only and key_colors != pair_colors:
                    continue
                if (not exact_only) and body_color not in key_colors:
                    continue
                for rot in range(4):
                    candidate = np.rot90(np.array(key["patch"], dtype=int), rot)
                    if candidate.shape != target_shape:
                        continue
                    if not np.array_equal(candidate == border_color, expected_border):
                        continue
                    found_idx = idx
                    rotated_patch = candidate
                    break
                if found_idx is not None:
                    break
            if found_idx is not None:
                break

        assert found_idx is not None and rotated_patch is not None
        used_keys.add(found_idx)
        key = keys[found_idx]

        kr0, kc0, kr1, kc1 = key["bbox"]
        out[kr0 : kr1 + 1, kc0 : kc1 + 1][np.array(key["mask"], dtype=bool)] = 0

        swapped = np.where(rotated_patch == body_color, border_color, body_color)
        lr0, lc0, _, _ = lock["bbox"]
        zr0, zc0, zr1, zc1 = lock["cavity_bbox"]
        out[lr0 + zr0 : lr0 + zr1 + 1, lc0 + zc0 : lc0 + zc1 + 1] = swapped

    return out


def _make_cavity_mask(spec: dict[str, int | str]) -> np.ndarray:
    n = int(spec["n"])
    side = str(spec["side"])
    depth = int(spec["depth"])
    span = int(spec["span"])
    offset = int(spec["offset"])
    mask = np.zeros((n, n), dtype=bool)
    if side == "top":
        mask[0:depth, offset : offset + span] = True
    elif side == "bottom":
        mask[n - depth : n, offset : offset + span] = True
    elif side == "left":
        mask[offset : offset + span, 0:depth] = True
    else:
        mask[offset : offset + span, n - depth : n] = True
    return mask


def _build_lock_patch(body_color: int, border_color: int, spec: dict[str, int | str]) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    n = int(spec["n"])
    cavity = _make_cavity_mask(spec)
    occ = ~cavity
    border_mask = np.zeros((n, n), dtype=bool)
    for rr, cc in np.argwhere(cavity):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < n and 0 <= nc < n and occ[nr, nc]:
                border_mask[nr, nc] = True
    body_mask = occ & (~border_mask)
    if not border_mask.any() or not body_mask.any():
        return None
    patch = np.zeros((n, n), dtype=int)
    patch[body_mask] = int(body_color)
    patch[border_mask] = int(border_color)
    cavity_bbox = _bbox_from_coords(np.argwhere(cavity))
    return patch, cavity_bbox


def _build_key_patch(body_color: int, border_color: int, spec: dict[str, int | str]) -> np.ndarray | None:
    cavity = _make_cavity_mask(spec)
    r0, c0, r1, c1 = _bbox_from_coords(np.argwhere(cavity))
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    patch = np.full(shape, int(body_color), dtype=int)
    patch[_expected_key_border_mask(str(spec["side"]), shape)] = int(border_color)
    if len(np.unique(patch)) != 2:
        return None
    return patch


def _build_specs_for_size(n: int) -> list[dict[str, int | str]]:
    depth_options = [2] if n == 3 else [2, 3]
    span_options = [1] if n == 3 else [1, 2]
    specs = []
    for side in SIDES:
        for depth in depth_options:
            for span in span_options:
                for offset in range(1, n - span):
                    spec: dict[str, int | str] = {
                        "n": n,
                        "side": side,
                        "depth": depth,
                        "span": span,
                        "offset": offset,
                    }
                    lock_data = _build_lock_patch(1, 2, spec)
                    key_patch = _build_key_patch(1, 2, spec)
                    if lock_data is not None and key_patch is not None:
                        specs.append(spec)
    return specs


SPECS_BY_SIZE = {3: _build_specs_for_size(3), 4: _build_specs_for_size(4)}


def _can_place(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    bh, bw = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(bh, top + h + 1)
    c1 = min(bw, left + w + 1)
    return not blocked[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    gh, gw = grid.shape
    ph, pw = patch.shape
    tops = list(range(gh - ph + 1))
    lefts = list(range(gw - pw + 1))
    random.shuffle(tops)
    random.shuffle(lefts)
    for top in tops:
        for left in lefts:
            if not _can_place(blocked, top, left, patch.shape):
                continue
            grid[top : top + ph, left : left + pw][patch != 0] = patch[patch != 0]
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(gh, top + ph + 1)
            c1 = min(gw, left + pw + 1)
            blocked[r0:r1, c0:c1] = True
            return top, left
    return None


def _construct_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(1, 3)
    needed_colors = 2 * pair_count
    color_pool = random.sample(PALETTE, needed_colors)
    h = random.randint(max(12, 8 + 3 * pair_count), 30)
    w = random.randint(max(12, 8 + 3 * pair_count), 30)

    objects = []
    for idx in range(pair_count):
        body_color = color_pool[2 * idx]
        border_color = color_pool[2 * idx + 1]
        size = random.choice([3, 4])
        spec = dict(random.choice(SPECS_BY_SIZE[size]))
        lock_data = _build_lock_patch(body_color, border_color, spec)
        key_patch = _build_key_patch(body_color, border_color, spec)
        if lock_data is None or key_patch is None:
            return None
        lock_patch, cavity_bbox = lock_data
        key_rot = random.randrange(4)
        rotated_key = np.rot90(key_patch, key_rot)
        inserted = np.where(key_patch == body_color, border_color, body_color)
        objects.append(
            {
                "body_color": body_color,
                "border_color": border_color,
                "spec": spec,
                "lock_patch": lock_patch,
                "cavity_bbox": cavity_bbox,
                "key_patch": rotated_key,
                "inserted_patch": inserted,
            }
        )

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for item in sorted(objects, key=lambda obj: (-obj["lock_patch"].size, -obj["key_patch"].size)):
        lock_pos = _place_patch(grid, blocked, np.array(item["lock_patch"], dtype=int))
        if lock_pos is None:
            return None
        key_pos = _place_patch(grid, blocked, np.array(item["key_patch"], dtype=int))
        if key_pos is None:
            return None
        item["lock_pos"] = lock_pos
        item["key_pos"] = key_pos

    expected = grid.copy()
    for item in objects:
        key_r, key_c = item["key_pos"]
        key_patch = np.array(item["key_patch"], dtype=int)
        expected[key_r : key_r + key_patch.shape[0], key_c : key_c + key_patch.shape[1]][key_patch != 0] = 0

        lock_r, lock_c = item["lock_pos"]
        zr0, zc0, zr1, zc1 = item["cavity_bbox"]
        expected[lock_r + zr0 : lock_r + zr1 + 1, lock_c + zc0 : lock_c + zc1 + 1] = np.array(item["inserted_patch"], dtype=int)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _construct_case()
        if result is None:
            continue
        grid, expected = result
        if grid.ndim != 2 or max(grid.shape) > 30:
            continue
        if expected.ndim != 2 or max(expected.shape) > 30:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
