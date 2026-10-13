from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _canonical_rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    arr = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(arr, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _color_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        if not coords:
            continue
        mask = grid == int(color)
        comps = _components(mask)
        assert len(comps) == 1, "each object color should form one connected component"
        r0, c0, r1, c1 = _bbox(coords)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "crop": crop,
                "mask": crop == int(color),
            }
        )
    return objects


def _parse_lock(obj: dict, bg: int) -> dict | None:
    crop = np.asarray(obj["crop"], dtype=int)
    color = int(obj["color"])
    mask = np.asarray(obj["mask"], dtype=bool)
    h, w = crop.shape

    if h < 6 or w < 6:
        return None
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    if not bool(np.all((crop == int(bg)) | (crop == color))):
        return None

    bg_comps = _components(crop == int(bg))
    edge_comps: list[list[tuple[int, int]]] = []
    inner_comps: list[list[tuple[int, int]]] = []
    for comp in bg_comps:
        touches = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if touches:
            edge_comps.append(comp)
        else:
            inner_comps.append(comp)

    if len(edge_comps) != 1 or len(inner_comps) != 1:
        return None

    notch = edge_comps[0]
    cavity = inner_comps[0]
    if len(cavity) <= len(notch):
        return None

    r0, c0, _r1, _c1 = obj["bbox"]
    cavity_abs = [(int(r0 + r), int(c0 + c)) for r, c in cavity]
    return {
        "color": color,
        "bbox": obj["bbox"],
        "notch_mask": _normalize_coords(notch),
        "cavity_abs": cavity_abs,
    }


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = BG if bool(np.any(g == BG)) else int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    locks: list[dict] = []
    keys: list[dict] = []
    for obj in _color_objects(g, bg):
        parsed = _parse_lock(obj, bg)
        if parsed is None:
            keys.append(
                {
                    "color": int(obj["color"]),
                    "coords": obj["coords"],
                    "shape_key": _canonical_rot_key(obj["mask"]),
                }
            )
        else:
            parsed["shape_key"] = _canonical_rot_key(parsed["notch_mask"])
            locks.append(parsed)

    lock_by_shape = {lock["shape_key"]: lock for lock in locks}
    assert len(lock_by_shape) == len(locks), "lock notches must be unique up to rotation"
    assert len(keys) == len(locks) and locks, "expected one key per lock"

    for key in keys:
        match = lock_by_shape[key["shape_key"]]
        for r, c in key["coords"]:
            out[r, c] = int(bg)
        for r, c in match["cavity_abs"]:
            out[r, c] = int(key["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key to a lock notch by shape up to rotation, then fill the
    matched lock's enclosed cavity with the key color and erase the key.
    """
    return _transform(grid)


def _shape_has_nontrivial_rotation(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return True
    return False


def _make_notch_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        depth = rng.randint(2, 4)
        width = rng.randint(2, 5)
        heights = [rng.randint(1, depth) for _ in range(width)]
        if max(heights) != depth:
            continue
        if len(set(heights)) == 1:
            continue
        mask = np.zeros((depth, width), dtype=bool)
        for c, h in enumerate(heights):
            mask[:h, c] = True
        if int(mask.sum()) < 4:
            continue
        if _shape_has_nontrivial_rotation(mask):
            continue
        return mask
    return None


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < int(area):
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                frontier.add((nr, nc))
    return mask


def _orient_from_top(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return mask.copy()
    if side == "right":
        return np.rot90(mask, 3)
    if side == "bottom":
        return np.rot90(mask, 2)
    return np.rot90(mask, 1)


def _valid_lock_patch(patch: np.ndarray, color: int) -> bool:
    color_comps = _components(patch == int(color))
    if len(color_comps) != 1:
        return False
    bg_comps = _components(patch == BG)
    h, w = patch.shape
    edge = 0
    inner = 0
    for comp in bg_comps:
        touches = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if touches:
            edge += 1
        else:
            inner += 1
    return edge == 1 and inner == 1


def _make_lock_and_key(lock_color: int, key_color: int, used_shapes: set, rng: random.Random) -> tuple[dict, dict] | None:
    for _ in range(SHAPE_RETRIES):
        base_notch = _make_notch_shape(rng)
        if base_notch is None:
            continue
        shape_key = _canonical_rot_key(base_notch)
        if shape_key in used_shapes:
            continue

        side = rng.choice(SIDES)
        notch_mask = _orient_from_top(base_notch, side)
        nh, nw = map(int, notch_mask.shape)
        lock_h = rng.randint(max(9, nh + 6), 14)
        lock_w = rng.randint(max(9, nw + 6), 14)

        if side in ("top", "bottom"):
            notch_r0 = 0 if side == "top" else lock_h - nh
            notch_c0 = rng.randint(1, lock_w - nw - 1)
        else:
            notch_r0 = rng.randint(1, lock_h - nh - 1)
            notch_c0 = 0 if side == "left" else lock_w - nw

        patch = np.full((lock_h, lock_w), int(lock_color), dtype=int)
        notch_rows, notch_cols = np.argwhere(notch_mask).T
        patch[notch_r0 + notch_rows, notch_c0 + notch_cols] = BG
        notch_area = int(notch_mask.sum())

        cavity_done = False
        cavity_abs_local: list[tuple[int, int]] = []
        for _ in range(SHAPE_RETRIES):
            max_ch = min(6, lock_h - 4)
            max_cw = min(6, lock_w - 4)
            if max_ch < 3 or max_cw < 3:
                break
            cav_h = rng.randint(3, max_ch)
            cav_w = rng.randint(3, max_cw)

            area_min = max(6, notch_area + 2)
            area_max = min(cav_h * cav_w - 1, notch_area + 10)
            if area_min > area_max:
                continue

            r_min, r_max = 1, lock_h - cav_h - 1
            c_min, c_max = 1, lock_w - cav_w - 1
            if side == "top":
                r_min = max(r_min, nh + 1)
            elif side == "bottom":
                r_max = min(r_max, lock_h - cav_h - nh - 1)
            elif side == "left":
                c_min = max(c_min, nw + 1)
            else:
                c_max = min(c_max, lock_w - cav_w - nw - 1)
            if r_min > r_max or c_min > c_max:
                continue

            cav_r0 = rng.randint(r_min, r_max)
            cav_c0 = rng.randint(c_min, c_max)
            cavity_mask = _grow_connected_mask(cav_h, cav_w, rng.randint(area_min, area_max), rng)

            trial = patch.copy()
            cav_rows, cav_cols = np.argwhere(cavity_mask).T
            trial[cav_r0 + cav_rows, cav_c0 + cav_cols] = BG
            if not _valid_lock_patch(trial, int(lock_color)):
                continue

            patch = trial
            cavity_abs_local = [(int(cav_r0 + r), int(cav_c0 + c)) for r, c in np.argwhere(cavity_mask)]
            cavity_done = True
            break

        if not cavity_done:
            continue

        key_rot = rng.randrange(4)
        key_mask = np.rot90(base_notch, key_rot)
        key_patch = np.full(key_mask.shape, BG, dtype=int)
        key_patch[key_mask] = int(key_color)

        used_shapes.add(shape_key)
        return (
            {
                "patch": patch,
                "color": int(lock_color),
                "shape_key": shape_key,
                "cavity_local": cavity_abs_local,
            },
            {
                "patch": key_patch,
                "color": int(key_color),
                "shape_key": shape_key,
            },
        )
    return None


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    bh, bw = blocked.shape
    r1 = max(0, r0 - pad)
    r2 = min(bh, r0 + h + pad)
    c1 = max(0, c0 - pad)
    c2 = min(bw, c0 + w + pad)
    return not bool(blocked[r1:r2, c1:c2].any())


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    pad: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    h, w = patch.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(0, gh - h)
        c0 = rng.randint(0, gw - w)
        if not _can_place(blocked, r0, c0, h, w, pad):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
        blocked[r0:r0 + h, c0:c0 + w] = True
        return int(r0), int(c0)
    return None


def _expected_output(grid: np.ndarray, lock_places: list[dict], key_places: list[dict]) -> np.ndarray:
    out = grid.copy()
    key_by_shape = {item["shape_key"]: item for item in key_places}
    for lock in lock_places:
        key = key_by_shape[lock["shape_key"]]
        for r, c in key["coords"]:
            out[r, c] = BG
        for r, c in lock["cavity_abs"]:
            out[r, c] = int(key["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_locks = rng.randint(2, 4)
    palette = rng.sample(range(1, 10), 2 * num_locks)
    lock_colors = palette[:num_locks]
    key_colors = palette[num_locks:]

    used_shapes: set = set()
    locks: list[dict] = []
    keys: list[dict] = []
    for lock_color, key_color in zip(lock_colors, key_colors):
        made = _make_lock_and_key(int(lock_color), int(key_color), used_shapes, rng)
        if made is None:
            return None
        lock_obj, key_obj = made
        locks.append(lock_obj)
        keys.append(key_obj)

    base = 18 + 3 * num_locks
    h = rng.randint(base, min(32, base + 4))
    w = rng.randint(base, min(32, base + 6))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    lock_places: list[dict] = []
    key_places: list[dict] = []
    for lock in sorted(locks, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        pos = _place_patch(grid, blocked, lock["patch"], pad=2, rng=rng)
        if pos is None:
            return None
        r0, c0 = pos
        lock_places.append(
            {
                "shape_key": lock["shape_key"],
                "color": int(lock["color"]),
                "cavity_abs": [(int(r0 + r), int(c0 + c)) for r, c in lock["cavity_local"]],
            }
        )

    rng.shuffle(keys)
    for key in keys:
        pos = _place_patch(grid, blocked, key["patch"], pad=1, rng=rng)
        if pos is None:
            return None
        r0, c0 = pos
        key_places.append(
            {
                "shape_key": key["shape_key"],
                "color": int(key["color"]),
                "coords": [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(key["patch"] == int(key["color"]))],
            }
        )

    expected = _expected_output(grid, lock_places, key_places)
    solved = _transform(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
