from __future__ import annotations

from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_crop(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _rotation_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best: tuple[int, int, bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, int(k))
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, int(k))
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot.copy())
    return out


def _infer_grey(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    assert counts
    grey, _count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return int(grey)


def _extract_locks(grid: np.ndarray, bg: int, grey: int) -> list[dict]:
    locks: list[dict] = []
    for coords in _components_mask(grid == int(grey)):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all((patch == int(grey)) | (patch == int(bg))))
        hole = patch == int(bg)
        assert bool(hole.any())
        hole_comps = _components_mask(hole)
        assert len(hole_comps) == 1
        touches = []
        if bool(hole[0, :].any()):
            touches.append("top")
        if bool(hole[-1, :].any()):
            touches.append("bottom")
        if bool(hole[:, 0].any()):
            touches.append("left")
        if bool(hole[:, -1].any()):
            touches.append("right")
        assert len(touches) == 1
        hole_mask, hr0, hc0 = _tight_crop(hole)
        locks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "hole_mask": hole_mask,
                "hole_r0": int(r0 + hr0),
                "hole_c0": int(c0 + hc0),
            }
        )
    locks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return locks


def _extract_keys(grid: np.ndarray, bg: int, grey: int) -> list[dict]:
    keys: list[dict] = []
    colors = sorted(set(int(v) for v in grid.ravel()) - {int(bg), int(grey)})
    for color in colors:
        for coords in _components_mask(grid == int(color)):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
            keys.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": patch.copy(),
                    "coords": coords.copy(),
                }
            )
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return keys


def _match_keys_to_locks(keys: list[dict], locks: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    options: dict[int, list[tuple[int, np.ndarray]]] = {}
    for ki, key in enumerate(keys):
        matches: list[tuple[int, np.ndarray]] = []
        for li, lock in enumerate(locks):
            chosen: np.ndarray | None = None
            for rot in _unique_rotations(key["mask"]):
                if rot.shape == lock["hole_mask"].shape and np.array_equal(rot, lock["hole_mask"]):
                    chosen = rot
                    break
            if chosen is not None:
                matches.append((int(li), chosen))
        assert matches
        options[int(ki)] = matches

    order = sorted(range(len(keys)), key=lambda idx: (len(options[idx]), keys[idx]["bbox"]))

    def dfs(pos: int, used_locks: set[int]) -> dict[int, tuple[int, np.ndarray]] | None:
        if pos == len(order):
            return {}
        ki = int(order[pos])
        for li, rot in options[ki]:
            if int(li) in used_locks:
                continue
            sub = dfs(pos + 1, used_locks | {int(li)})
            if sub is None:
                continue
            out = dict(sub)
            out[int(ki)] = (int(li), rot)
            return out
        return None

    matched = dfs(0, set())
    assert matched is not None and len(matched) == len(keys)
    return matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey lock notch with the uniquely matching rotated colored key.

    Input structure:
    - The background is black.
    - Several grey 4-connected lock objects are rectangle-like solids with one black notch cut from exactly one side of their bounding box.
    - Separate colored 4-connected key objects lie elsewhere on the canvas, one per lock, and each key has a distinct non-black, non-grey color.

    Rule:
    - For each grey lock, take its full bounding box and read the black cells inside that box as the notch mask.
    - For each colored key, test its four 90-degree rotations and find the lock whose notch mask matches exactly.
    - Erase the key from its original location.
    - Paint the matching rotated key shape into the lock's notch cells using the key's original color, while leaving the other grey lock pixels unchanged.

    Key insight:
    - The puzzle is a rotation-based shape-matching task: the important objects are the black notch masks inside the grey locks, and each notch inherits the color of the matching free key rather than becoming grey.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey = _infer_grey(g, bg)

    locks = _extract_locks(g, bg, grey)
    keys = _extract_keys(g, bg, grey)
    assert locks and len(locks) == len(keys)

    matched = _match_keys_to_locks(keys, locks)

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = int(bg)

    for ki, (li, rot) in matched.items():
        lock = locks[int(li)]
        rr = int(lock["hole_r0"])
        cc = int(lock["hole_c0"])
        patch = out[rr : rr + rot.shape[0], cc : cc + rot.shape[1]]
        patch[rot] = int(keys[int(ki)]["color"])

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_GEN_BG = 0
_GEN_GREY = 5
_GEN_KEY_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_SIDES = ("top", "right", "bottom", "left")
_GEN_MAX_ATTEMPTS = 400


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


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = canvas[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(bg)
    target[mask] = patch[mask]


def _colorize(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _generate_base_notch() -> np.ndarray:
    depth = int(_RNG.integers(1, 4))
    width = int(_RNG.integers(2, 5))
    mask = np.zeros((depth, width), dtype=bool)
    start = 0
    size = width
    mask[0, :] = True
    for r in range(1, depth):
        new_size = int(_RNG.integers(1, size + 1))
        new_start = int(_RNG.integers(start, start + size - new_size + 1))
        mask[r, new_start : new_start + new_size] = True
        start = new_start
        size = new_size
    return mask


def _orient_notch(base_mask: np.ndarray, side: str) -> np.ndarray:
    rot_k = {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]
    return np.rot90(base_mask, int(rot_k)).copy()


def _build_lock_patch(side: str, hole_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    hole_h, hole_w = hole_mask.shape
    outer_h = int(_RNG.integers(hole_h + 2, min(10, hole_h + 6)))
    outer_w = int(_RNG.integers(hole_w + 2, min(10, hole_w + 6)))
    patch = np.full((outer_h, outer_w), int(_GEN_GREY), dtype=int)

    if side in {"top", "bottom"}:
        hole_r0 = 0 if side == "top" else int(outer_h - hole_h)
        hole_c0 = int(_RNG.integers(1, outer_w - hole_w))
    else:
        hole_r0 = int(_RNG.integers(1, outer_h - hole_h))
        hole_c0 = 0 if side == "left" else int(outer_w - hole_w)

    patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w][hole_mask] = int(_GEN_BG)
    return patch, int(hole_r0), int(hole_c0)


def _place_objects(objects: list[dict], height: int, width: int) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = list(range(len(objects)))
    _RNG.shuffle(order)
    order.sort(
        key=lambda idx: (
            -int(objects[idx]["footprint"].sum()),
            -int(objects[idx]["patch"].shape[0]),
            -int(objects[idx]["patch"].shape[1]),
        )
    )

    for idx in order:
        patch = objects[idx]["patch"]
        footprint = objects[idx]["footprint"]
        h, w = patch.shape
        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - h):
            for c0 in range(1, width - w):
                window = blocked[r0 : r0 + h, c0 : c0 + w]
                if not bool((window & footprint).any()):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        occupied[r0 : r0 + h, c0 : c0 + w] |= footprint
        placements[int(idx)] = (int(r0), int(c0))

    return placements


def _attempt_generate() -> np.ndarray | None:
    n_locks = int(_RNG.integers(2, 4))
    colors = [int(v) for v in _RNG.choice(_GEN_KEY_COLORS, size=n_locks, replace=False)]
    sides = [str(v) for v in _RNG.choice(np.array(_GEN_SIDES), size=n_locks, replace=False)]
    key_rotations = [int(v) for v in _RNG.choice(np.arange(4), size=n_locks, replace=False)]

    pairs: list[dict] = []
    seen_shapes: set[tuple[int, int, bytes]] = set()
    for color, side, key_rot in zip(colors, sides, key_rotations):
        base_mask = None
        for _ in range(100):
            candidate = _generate_base_notch()
            canon = _rotation_key(candidate)
            if canon in seen_shapes:
                continue
            base_mask = candidate
            seen_shapes.add(canon)
            break
        if base_mask is None:
            return None

        hole_mask = _orient_notch(base_mask, side)
        lock_patch, hole_r0, hole_c0 = _build_lock_patch(side, hole_mask)
        key_mask = np.rot90(base_mask, int(key_rot)).copy()
        key_patch = _colorize(key_mask, color, _GEN_BG)
        pairs.append(
            {
                "color": int(color),
                "hole_mask": hole_mask,
                "lock_patch": lock_patch,
                "hole_r0": int(hole_r0),
                "hole_c0": int(hole_c0),
                "key_patch": key_patch,
            }
        )

    objects: list[dict] = []
    for pair_idx, pair in enumerate(pairs):
        objects.append(
            {
                "kind": "lock",
                "pair_idx": int(pair_idx),
                "patch": pair["lock_patch"],
                "footprint": np.ones(pair["lock_patch"].shape, dtype=bool),
            }
        )
        objects.append(
            {
                "kind": "key",
                "pair_idx": int(pair_idx),
                "patch": pair["key_patch"],
                "footprint": np.ones(pair["key_patch"].shape, dtype=bool),
            }
        )

    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    padded_area = sum((int(obj["patch"].shape[0]) + 2) * (int(obj["patch"].shape[1]) + 2) for obj in objects)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(padded_area))) + 1, 18)
    if min_side > 30:
        return None
    height = int(_RNG.integers(min_side, 31))
    width = int(_RNG.integers(min_side, 31))

    placements = _place_objects(objects, height, width)
    if placements is None:
        return None

    inp = np.full((height, width), int(_GEN_BG), dtype=int)
    expected = np.full((height, width), int(_GEN_BG), dtype=int)

    for obj_idx, obj in enumerate(objects):
        r0, c0 = placements[int(obj_idx)]
        pair = pairs[int(obj["pair_idx"])]
        patch = obj["patch"]
        _paint_patch(inp, patch, r0, c0, _GEN_BG)
        if obj["kind"] == "lock":
            _paint_patch(expected, patch, r0, c0, _GEN_BG)
            hole = pair["hole_mask"]
            rr = int(r0 + pair["hole_r0"])
            cc = int(c0 + pair["hole_c0"])
            expected[rr : rr + hole.shape[0], cc : cc + hole.shape[1]][hole] = int(pair["color"])

    counts = Counter(int(v) for v in inp.ravel())
    non_bg_counts = [count for color, count in counts.items() if color != int(_GEN_BG)]
    if not non_bg_counts or counts[int(_GEN_BG)] <= max(non_bg_counts):
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_MAX_ATTEMPTS} attempts")
