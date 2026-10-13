from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _tight_mask_from_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    return _tight_mask_from_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    return objects


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _rotation_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        for rot in _rotation_variants(mask)
    )


def _matches_under_rotation(key_mask: np.ndarray, hole_mask: np.ndarray) -> bool:
    for rot in _rotation_variants(key_mask):
        if rot.shape == hole_mask.shape and np.array_equal(rot, hole_mask):
            return True
    return False


def _is_lock_patch(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    if not np.all((patch == color) | (patch == bg)):
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    hole = patch == bg
    area = int(hole.sum())
    if area < 1 or area > 4:
        return False
    if hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any():
        return False
    return len(_components(hole)) == 1


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _background_color(grid)
    locks: list[dict] = []
    keys: list[dict] = []
    for obj in _find_color_components(grid, bg):
        r0, c0, r1, c1 = obj["bbox"]
        color = int(obj["color"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if _is_lock_patch(patch, color, bg):
            hole = patch == bg
            hole_coords = np.argwhere(hole)
            locks.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "hole_mask": _tight_mask_from_bool(hole),
                    "hole_coords": hole_coords + np.array([r0, c0], dtype=int),
                }
            )
        else:
            if not (1 <= int(obj["area"]) <= 4):
                raise AssertionError("unexpected non-key component")
            keys.append(
                {
                    "color": color,
                    "coords": obj["coords"],
                    "mask": _tight_mask_from_coords(obj["coords"]),
                }
            )
    return bg, locks, keys


def _match_keys_to_locks(keys: list[dict], locks: list[dict]) -> dict[int, int]:
    candidates: list[list[int]] = []
    for key in keys:
        fit = [
            li
            for li, lock in enumerate(locks)
            if _matches_under_rotation(key["mask"], lock["hole_mask"])
        ]
        assert fit, "every key must fit at least one lock"
        candidates.append(fit)

    order = sorted(range(len(keys)), key=lambda i: (len(candidates[i]), len(keys[i]["coords"])))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        ki = order[pos]
        for li in candidates[ki]:
            if li in used:
                continue
            used.add(li)
            assignment[ki] = li
            if dfs(pos + 1):
                return True
            used.remove(li)
            del assignment[ki]
        return False

    assert len(keys) == len(locks) and keys
    assert dfs(0), "must find a one-to-one key-lock matching"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each key to the uniquely matching lock-hole, recolor the lock to the
    key color, and leave a same-shape color-swapped residue at the key origin.
    """
    g = np.asarray(grid, dtype=int)
    _bg, locks, keys = _extract_scene(g)
    match = _match_keys_to_locks(keys, locks)

    out = g.copy()
    for key_index, lock_index in match.items():
        key = keys[key_index]
        lock = locks[lock_index]
        r0, c0, r1, c1 = lock["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(key["color"])
        out[key["coords"][:, 0], key["coords"][:, 1]] = int(lock["color"])
    return out


_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]

_FOUR_ROTATION_SHAPES = [shape for shape in _SHAPES if len(_rotation_variants(shape)) == 4]

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (16, 29)
NUM_PAIRS_CHOICES = [1, 2, 3, 4]
OUTER_SIZE_CHOICES = [3, 4, 5]
PLACEMENT_BORDER = 1
GEN_ATTEMPTS = 400


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_bbox(
    grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = grid.shape
    r_lo = PLACEMENT_BORDER
    c_lo = PLACEMENT_BORDER
    r_hi = H - h - PLACEMENT_BORDER
    c_hi = W - w - PLACEMENT_BORDER
    if r_hi < r_lo or c_hi < c_lo:
        return None

    blocked = _dilate8(occupied)
    origins = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(origins)
    for r0, c0 in origins:
        if blocked[r0 : r0 + h, c0 : c0 + w].any():
            continue
        mask = patch != bg
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        return r0, c0
    return None


def _build_lock_patch(lock_color: int, bg: int, hole_mask: np.ndarray) -> np.ndarray | None:
    hh, hw = hole_mask.shape
    choices = [size for size in OUTER_SIZE_CHOICES if size - 2 >= hh and size - 2 >= hw]
    if not choices:
        return None
    outer = random.choice(choices)
    inner = outer - 2
    r_off = random.randint(0, inner - hh)
    c_off = random.randint(0, inner - hw)

    patch = np.full((outer, outer), lock_color, dtype=int)
    window = patch[1 + r_off : 1 + r_off + hh, 1 + c_off : 1 + c_off + hw]
    window[hole_mask] = bg
    return patch


def _build_key_patch(key_color: int, shape_mask: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full(shape_mask.shape, bg, dtype=int)
    patch[shape_mask] = key_color
    return patch


def _expected_output(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        lr0, lc0 = pair["lock_origin"]
        ls = int(pair["lock_patch"].shape[0])
        out[lr0 : lr0 + ls, lc0 : lc0 + ls] = int(pair["key_color"])
        kr0, kc0 = pair["key_origin"]
        key_mask = pair["key_patch"] == int(pair["key_color"])
        out[kr0 : kr0 + key_mask.shape[0], kc0 : kc0 + key_mask.shape[1]][key_mask] = int(pair["lock_color"])
    return out


def _sample_pair_shapes(num_pairs: int) -> list[np.ndarray]:
    featured = random.choice(_FOUR_ROTATION_SHAPES)
    remaining = [shape for shape in _SHAPES if _rotation_key(shape) != _rotation_key(featured)]
    chosen = [featured] + random.sample(remaining, num_pairs - 1)
    random.shuffle(chosen)
    return [shape.copy() for shape in chosen]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    if 1 + 2 * num_pairs > 10:
        return None
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    pair_shapes = _sample_pair_shapes(num_pairs)

    need_rot = random.randint(0, 3)
    featured_indices = [i for i, shape in enumerate(pair_shapes) if len(_rotation_variants(shape)) == 4]
    featured_index = random.choice(featured_indices)

    pairs: list[dict] = []
    used_shape_keys = set()
    for i, base_shape in enumerate(pair_shapes):
        key_rot = random.randint(0, 3)
        if i == featured_index:
            hole_rot = (key_rot + need_rot) % 4
        else:
            hole_rot = random.randint(0, 3)

        key_mask = np.rot90(base_shape, key_rot)
        hole_mask = np.rot90(base_shape, hole_rot)
        shape_key = _rotation_key(hole_mask)
        if shape_key in used_shape_keys:
            return None
        used_shape_keys.add(shape_key)

        lock_color = colors[2 * i]
        key_color = colors[2 * i + 1]
        lock_patch = _build_lock_patch(lock_color, bg, hole_mask)
        if lock_patch is None:
            return None
        key_patch = _build_key_patch(key_color, key_mask, bg)
        pairs.append(
            {
                "lock_color": lock_color,
                "key_color": key_color,
                "lock_patch": lock_patch,
                "key_patch": key_patch,
            }
        )

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for index in sorted(range(num_pairs), key=lambda i: -pairs[i]["lock_patch"].shape[0]):
        origin = _place_bbox(grid, occupied, pairs[index]["lock_patch"], bg)
        if origin is None:
            return None
        pairs[index]["lock_origin"] = origin

    for index in sorted(range(num_pairs), key=lambda i: -pairs[i]["key_patch"].size):
        origin = _place_bbox(grid, occupied, pairs[index]["key_patch"], bg)
        if origin is None:
            return None
        pairs[index]["key_origin"] = origin

    if _background_color(grid) != bg:
        return None

    expected = _expected_output(grid, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
