from __future__ import annotations

from collections import Counter

import numpy as np


_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in _DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert int(len(pts)) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _canonical_rot_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    cur = np.asarray(mask, dtype=bool)
    best = None
    for _ in range(4):
        key = (cur.tobytes(), tuple(map(int, cur.shape)))
        if best is None or key < best:
            best = key
        cur = np.rot90(cur)
    assert best is not None
    return best


def _has_enclosed_void(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    for comp in _components(~mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        return True
    return False


def _lock_notch_info(lock_mask: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    lock_mask = np.asarray(lock_mask, dtype=bool)
    h, w = lock_mask.shape
    border_voids = []
    for comp in _components(~lock_mask):
        sides = set()
        for r, c in comp:
            if r == 0:
                sides.add("top")
            if r == h - 1:
                sides.add("bottom")
            if c == 0:
                sides.add("left")
            if c == w - 1:
                sides.add("right")
        if sides:
            border_voids.append((comp, sides))
    if len(border_voids) != 1:
        return None
    best, sides = border_voids[0]
    if len(sides) != 1:
        return None
    r0, c0, r1, c1 = _bbox(best)
    trimmed = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in best:
        trimmed[int(r - r0), int(c - c0)] = True
    return best, trimmed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to a grey lock-hole by shape up to rotation, then fill the hole with the key color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = Counter(int(v) for v in g.flat).most_common(1)[0][0]

    non_bg_colors = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    assert non_bg_colors

    lock_color = None
    best_score = None
    for color in non_bg_colors:
        comps = _components(g == int(color))
        notch_count = 0
        largest = 0
        total = sum(len(comp) for comp in comps)
        for comp in comps:
            largest = max(largest, int(len(comp)))
            r0, c0, r1, c1 = _bbox(comp)
            local = np.asarray(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color), dtype=bool)
            if _lock_notch_info(local) is not None:
                notch_count += 1
        score = (int(notch_count), int(largest), int(total))
        if best_score is None or score > best_score:
            best_score = score
            lock_color = int(color)
    assert lock_color is not None

    locks = []
    for comp in _components(g == int(lock_color)):
        r0, c0, r1, c1 = _bbox(comp)
        local = np.asarray(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(lock_color), dtype=bool)
        notch_info = _lock_notch_info(local)
        if notch_info is None:
            continue
        hole_local_coords, hole_mask = notch_info
        hole_global = [(int(r0 + rr), int(c0 + cc)) for rr, cc in hole_local_coords]
        locks.append({"shape_key": _canonical_rot_key(hole_mask), "hole": hole_global})

    assert locks
    keys = []
    for comp in _components((g != int(bg)) & (g != int(lock_color))):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        for r, c in comp:
            local[int(r - r0), int(c - c0)] = True
        keys.append({"shape_key": _canonical_rot_key(local), "color": color, "coords": comp})

    assert len(keys) == len(locks)
    key_by_shape: dict[tuple[bytes, tuple[int, int]], dict[str, object]] = {}
    for key in keys:
        shape_key = key["shape_key"]
        assert shape_key not in key_by_shape
        key_by_shape[shape_key] = key

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[int(r), int(c)] = int(bg)

    for lock in locks:
        matched = key_by_shape[lock["shape_key"]]
        color = int(matched["color"])
        for r, c in lock["hole"]:
            out[int(r), int(c)] = color

    return out


_BG = 0
_LOCK = 5
_KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_GRID_MIN = 18
_GRID_MAX = 30
_MAX_GEN_ATTEMPTS = 400
_PLACE_ATTEMPTS = 80


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in _DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < height and 0 <= nc < width and not bool(mask[nr, nc]):
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < int(area):
        assert frontier
        idx = int(_RNG.integers(len(frontier)))
        cell = list(frontier)[idx]
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)
    return mask


def _random_base_shape(used_keys: set[tuple[bytes, tuple[int, int]]]) -> np.ndarray:
    for _ in range(200):
        height = int(_RNG.integers(2, 7))
        width = int(_RNG.integers(2, 7))
        max_area = min(int(height * width) - 1, 10)
        if max_area < 4:
            continue
        area = int(_RNG.integers(4, max_area + 1))
        mask = _trim_mask(_grow_connected_mask(height, width, area))
        h, w = map(int, mask.shape)
        area = int(mask.sum())
        if min(h, w) < 2:
            continue
        if area == h * w and _RNG.random() < 0.85:
            continue
        if area < 4 or area > 10:
            continue
        if _has_enclosed_void(mask):
            continue
        if _lock_notch_info(mask) is not None:
            continue
        key = _canonical_rot_key(mask)
        if key in used_keys:
            continue
        used_keys.add(key)
        return mask
    raise RuntimeError("failed to sample a unique key shape")


def _build_lock_patch(hole_mask: np.ndarray, side: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    hole_mask = np.asarray(hole_mask, dtype=bool)
    hole_h, hole_w = map(int, hole_mask.shape)
    if side in ("left", "right"):
        top_margin = int(_RNG.integers(1, 4))
        bottom_margin = int(_RNG.integers(1, 4))
        depth = int(_RNG.integers(2, 6))
        body_h = hole_h + top_margin + bottom_margin
        body_w = hole_w + depth
        row0 = top_margin
        col0 = 0 if side == "left" else body_w - hole_w
    else:
        left_margin = int(_RNG.integers(1, 4))
        right_margin = int(_RNG.integers(1, 4))
        depth = int(_RNG.integers(2, 6))
        body_h = hole_h + depth
        body_w = hole_w + left_margin + right_margin
        row0 = 0 if side == "top" else body_h - hole_h
        col0 = left_margin

    patch = np.full((int(body_h), int(body_w)), int(_LOCK), dtype=int)
    hole_coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(hole_mask):
        r = int(row0 + rr)
        c = int(col0 + cc)
        patch[r, c] = int(_BG)
        hole_coords.append((r, c))
    return patch, hole_coords


def _make_key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, int(_BG), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _grid_choices(objects: list[dict[str, object]]) -> list[tuple[int, int]]:
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    reserved_area = sum((int(obj["patch"].shape[0]) + 2) * (int(obj["patch"].shape[1]) + 2) for obj in objects)
    choices: list[tuple[int, int]] = []
    for h in range(max(int(_GRID_MIN), max_h), int(_GRID_MAX) + 1):
        for w in range(max(int(_GRID_MIN), max_w), int(_GRID_MAX) + 1):
            if h * w >= int(np.ceil(reserved_area * 1.1)):
                choices.append((int(h), int(w)))
    return choices


def _reserve_bbox(reserved: np.ndarray, row0: int, col0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = reserved.shape
    r0 = max(0, int(row0 - pad))
    c0 = max(0, int(col0 - pad))
    r1 = min(h, int(row0 + height + pad))
    c1 = min(w, int(col0 + width + pad))
    reserved[r0:r1, c0:c1] = True


def _place_objects(objects: list[dict[str, object]]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    choices = _grid_choices(objects)
    if not choices:
        return None
    for _ in range(_PLACE_ATTEMPTS):
        grid_h, grid_w = choices[int(_RNG.integers(len(choices)))]
        grid = np.full((int(grid_h), int(grid_w)), int(_BG), dtype=int)
        reserved = np.zeros((int(grid_h), int(grid_w)), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(objects)
        order = sorted(range(len(objects)), key=lambda idx: int(np.prod(objects[idx]["patch"].shape)), reverse=True)
        ok = True
        for idx in order:
            patch = np.asarray(objects[idx]["patch"], dtype=int)
            ph, pw = map(int, patch.shape)
            candidates = []
            for r0 in range(int(grid_h - ph + 1)):
                for c0 in range(int(grid_w - pw + 1)):
                    if not bool(reserved[r0 : r0 + ph, c0 : c0 + pw].any()):
                        candidates.append((int(r0), int(c0)))
            if not candidates:
                ok = False
                break
            row0, col0 = candidates[int(_RNG.integers(len(candidates)))]
            fg = patch != int(_BG)
            grid[row0 : row0 + ph, col0 : col0 + pw][fg] = patch[fg]
            _reserve_bbox(reserved, row0, col0, ph, pw, pad=1)
            placements[idx] = (int(row0), int(col0))
        if ok and all(pos is not None for pos in placements):
            return grid, [(int(r), int(c)) for r, c in placements if r is not None]
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(_MAX_GEN_ATTEMPTS):
        pair_count = int(_RNG.integers(1, 5))
        colors = [int(c) for c in _RNG.choice(_KEY_COLORS, size=pair_count, replace=False)]
        used_shapes: set[tuple[bytes, tuple[int, int]]] = set()
        objects: list[dict[str, object]] = []
        pairs: list[dict[str, object]] = []

        try:
            for idx in range(pair_count):
                base = _random_base_shape(used_shapes)
                hole_mask = np.rot90(base, int(_RNG.integers(0, 4)))
                key_mask = np.rot90(base, int(_RNG.integers(0, 4)))
                side = ("left", "right", "top", "bottom")[int(_RNG.integers(0, 4))]
                lock_patch, hole_coords = _build_lock_patch(hole_mask, side)
                key_patch = _make_key_patch(key_mask, colors[idx])
                lock_obj = {"kind": "lock", "pair": idx, "patch": lock_patch, "hole_coords": hole_coords}
                key_obj = {"kind": "key", "pair": idx, "patch": key_patch, "color": colors[idx]}
                pairs.append({"lock_index": len(objects), "key_index": len(objects) + 1, "color": colors[idx]})
                objects.append(lock_obj)
                objects.append(key_obj)
        except RuntimeError:
            continue

        placed = _place_objects(objects)
        if placed is None:
            continue

        inp, positions = placed
        expected = inp.copy()
        for pair in pairs:
            lock = objects[int(pair["lock_index"])]
            key = objects[int(pair["key_index"])]
            lock_r0, lock_c0 = positions[int(pair["lock_index"])]
            key_r0, key_c0 = positions[int(pair["key_index"])]
            key_patch = np.asarray(key["patch"], dtype=int)
            for rr, cc in np.argwhere(key_patch != int(_BG)):
                expected[int(key_r0 + rr), int(key_c0 + cc)] = int(_BG)
            for rr, cc in lock["hole_coords"]:
                expected[int(lock_r0 + rr), int(lock_c0 + cc)] = int(pair["color"])

        if np.array_equal(main(inp), expected):
            return inp, expected

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
