from __future__ import annotations

import random

import numpy as np


BG = 0
LOCK = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIDE = 30
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, rot.shape)), rot.tobytes()) for rot in _rotation_variants(mask))


def _find_locks(grid: np.ndarray) -> list[dict[str, object]]:
    locks: list[dict[str, object]] = []
    for comp in _components(grid == LOCK):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[0] < 3 or sub.shape[1] < 3:
            continue
        if np.any((sub != BG) & (sub != LOCK)):
            continue
        if np.any(sub[0, :] != LOCK) or np.any(sub[-1, :] != LOCK):
            continue
        if np.any(sub[:, 0] != LOCK) or np.any(sub[:, -1] != LOCK):
            continue

        hole_mask = sub == BG
        if not hole_mask.any():
            continue
        if np.any(hole_mask[0, :]) or np.any(hole_mask[-1, :]):
            continue
        if np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1]):
            continue

        hole_comps = _components(hole_mask)
        interior = []
        for hole in hole_comps:
            hr0, hc0, hr1, hc1 = _bbox(hole)
            if hr0 == 0 or hc0 == 0 or hr1 == hole_mask.shape[0] - 1 or hc1 == hole_mask.shape[1] - 1:
                continue
            interior.append(hole)
        assert len(interior) == 1, "each lock must contain exactly one interior hole"

        hole = interior[0]
        locks.append(
            {
                "mask": _coords_to_mask(hole),
                "rows": hole[:, 0] + r0,
                "cols": hole[:, 1] + c0,
            }
        )
    assert locks, "expected at least one lock"
    return locks


def _find_keys(grid: np.ndarray) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    for comp in _components((grid != BG) & (grid != LOCK)):
        colors = np.unique(grid[comp[:, 0], comp[:, 1]])
        assert len(colors) == 1, "each key must be monochrome"
        keys.append({"mask": _coords_to_mask(comp), "color": int(colors[0])})
    assert keys, "expected at least one key"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each lock's internal hole with the color of its matching key, then erase the keys.

    Input structure:
    - The background is black.
    - One or more gray rectangular lock regions appear on the grid; each rectangle contains exactly one connected black hole shape fully inside it.
    - Outside the locks are small monochrome connected key objects in non-black, non-gray colors.

    Rule:
    - For each gray rectangle, extract its internal black hole shape.
    - Find the unique outside key whose shape matches that hole up to a 0°, 90°, 180°, or 270° rotation.
    - Recolor the hole cells with that key's color, keep the rest of the gray rectangle unchanged, and erase all key objects back to black.

    Key insight:
    - The meaningful geometry is the black void inside each gray rectangle, not the gray mass itself; the external key contributes only identity and color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    locks = _find_locks(g)
    keys = _find_keys(g)
    assert len(locks) == len(keys), "locks and keys must pair one-to-one"

    color_by_shape: dict[tuple[tuple[int, int], bytes], int] = {}
    for key in keys:
        shape = _rotation_key(key["mask"])
        assert shape not in color_by_shape, "key shapes must be unique up to rotation"
        color_by_shape[shape] = int(key["color"])

    out = g.copy()
    out[(out != BG) & (out != LOCK)] = BG
    for lock in locks:
        shape = _rotation_key(lock["mask"])
        assert shape in color_by_shape, "every lock must have a matching key"
        out[lock["rows"], lock["cols"]] = int(color_by_shape[shape])
    return out


def _random_polyomino(area: int, max_h: int = 5, max_w: int = 5) -> np.ndarray | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if h > max_h or w > max_w:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True
    return mask


def _sample_hole_shape(used_shapes: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(120):
        area = random.randint(3, 7)
        mask = _random_polyomino(area)
        if mask is None:
            continue
        variants = _rotation_variants(mask)
        if random.random() < 0.75 and len(variants) == 1:
            continue
        shape = _rotation_key(mask)
        if shape in used_shapes:
            continue
        used_shapes.add(shape)
        return mask
    return None


def _make_lock_patch(hole_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = map(int, hole_mask.shape)
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    patch = np.full((hh + top + bottom, hw + left + right), LOCK, dtype=int)
    hole_view = patch[top : top + hh, left : left + hw]
    hole_view[hole_mask] = BG
    hole_rows, hole_cols = np.where(hole_mask)
    hole_local = np.column_stack([hole_rows + top, hole_cols + left]).astype(int)
    return patch, hole_local


def _make_key_patch(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, int]:
    rot_k = random.randint(0, 3)
    key_mask = np.rot90(hole_mask, rot_k)
    patch = np.zeros(key_mask.shape, dtype=int)
    patch[key_mask] = int(color)
    return patch, rot_k


def _place_bbox(occupied: np.ndarray, shape: tuple[int, int], clearance: int = 1) -> tuple[int, int] | None:
    h, w = map(int, shape)
    H, W = occupied.shape
    if h + 2 > H or w + 2 > W:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, H - h - 1)
        c = random.randint(1, W - w - 1)
        rr0 = max(0, r - clearance)
        rr1 = min(H, r + h + clearance)
        cc0 = max(0, c - clearance)
        cc1 = min(W, c + w + clearance)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[rr0:rr1, cc0:cc1] = True
        return int(r), int(c)
    return None


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    target = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    target[mask] = patch[mask]


def _crop_bounds_with_margin(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_avail = int(r0)
    left_avail = int(c0)
    bottom_avail = int(grid.shape[0] - 1 - r1)
    right_avail = int(grid.shape[1] - 1 - c1)
    top_pad = random.randint(1, min(3, top_avail))
    left_pad = random.randint(1, min(3, left_avail))
    bottom_pad = random.randint(1, min(3, bottom_avail))
    right_pad = random.randint(1, min(3, right_avail))
    return int(r0 - top_pad), int(c0 - left_pad), int(r1 + bottom_pad), int(c1 + right_pad)


def _gen_attempt() -> np.ndarray | None:
    n_locks = random.choice([1, 2, 3])
    colors = random.sample(KEY_COLORS, n_locks)
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()

    items: list[dict[str, object]] = []
    for color in colors:
        hole_mask = _sample_hole_shape(used_shapes)
        if hole_mask is None:
            return None

        lock_patch, hole_local = _make_lock_patch(hole_mask)
        key_patch, _rot_k = _make_key_patch(hole_mask, color)
        items.append(
            {
                "kind": "lock",
                "patch": lock_patch,
                "hole_local": hole_local,
                "color": int(color),
                "area": int(lock_patch.shape[0] * lock_patch.shape[1]),
            }
        )
        items.append(
            {
                "kind": "key",
                "patch": key_patch,
                "color": int(color),
                "area": int(key_patch.shape[0] * key_patch.shape[1]),
            }
        )

    occupied = np.zeros((MAX_SIDE, MAX_SIDE), dtype=bool)
    placed: list[dict[str, object]] = []
    for item in sorted(items, key=lambda d: int(d["area"]), reverse=True):
        patch = np.asarray(item["patch"], dtype=int)
        pos = _place_bbox(occupied, patch.shape, clearance=1)
        if pos is None:
            return None
        placed_item = dict(item)
        placed_item["top"] = int(pos[0])
        placed_item["left"] = int(pos[1])
        placed.append(placed_item)

    inp = np.zeros((MAX_SIDE, MAX_SIDE), dtype=int)
    expected = np.zeros((MAX_SIDE, MAX_SIDE), dtype=int)
    for item in placed:
        patch = np.asarray(item["patch"], dtype=int)
        top = int(item["top"])
        left = int(item["left"])
        _draw_patch(inp, patch, top, left)
        if item["kind"] == "lock":
            _draw_patch(expected, np.where(patch == LOCK, LOCK, BG), top, left)
            hole_local = np.asarray(item["hole_local"], dtype=int)
            expected[hole_local[:, 0] + top, hole_local[:, 1] + left] = int(item["color"])
        else:
            pass

    cr0, cc0, cr1, cc1 = _crop_bounds_with_margin(inp)
    inp = inp[cr0 : cr1 + 1, cc0 : cc1 + 1]
    expected = expected[cr0 : cr1 + 1, cc0 : cc1 + 1]
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
