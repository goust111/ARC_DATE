import random
from collections import Counter

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    if arr.size == 0:
        return tuple()
    arr = arr - arr.min(axis=0)
    pts = [tuple(map(int, pt)) for pt in arr]
    pts.sort()
    return tuple(pts)


def _coords_to_array(coords) -> np.ndarray:
    norm = _normalize_coords(coords)
    if not norm:
        return np.zeros((0, 0), dtype=bool)
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    arr = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        arr[r, c] = True
    return arr


def _rotations(coords) -> list[tuple[tuple[int, int], ...]]:
    arr = _coords_to_array(coords)
    rots: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        rots.append(_normalize_coords(np.argwhere(rot)))
    return rots


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    return min(_rotations(coords))


def _border_values(patch: np.ndarray) -> list[int]:
    vals = [int(x) for x in patch[0, :]]
    vals += [int(x) for x in patch[-1, :]]
    vals += [int(x) for x in patch[1:-1, 0]]
    vals += [int(x) for x in patch[1:-1, -1]]
    return vals


def _parse_lock(grid: np.ndarray, comp: list[tuple[int, int]], bg: int):
    r0, c0, r1, c1 = _bbox(comp)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    if np.any(patch[0, :] == bg) or np.any(patch[-1, :] == bg):
        return None
    if np.any(patch[:, 0] == bg) or np.any(patch[:, -1] == bg):
        return None

    comp_colors = {int(grid[r, c]) for r, c in comp}
    if len(comp_colors) != 2:
        return None

    bg_mask = patch == bg
    if not bg_mask.any():
        return None
    holes = _components(bg_mask)
    if len(holes) != 1:
        return None
    hole_rel = holes[0]

    border = [v for v in _border_values(patch) if v != bg]
    if not border:
        return None
    body_color = Counter(border).most_common(1)[0][0]

    neighbor_colors = set()
    for hr, hc in hole_rel:
        for dr, dc in NEIGHBORS4:
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1]:
                val = int(patch[nr, nc])
                if val not in {bg, body_color}:
                    neighbor_colors.add(val)
    if len(neighbor_colors) != 1:
        return None
    strip_color = next(iter(neighbor_colors))

    if comp_colors != {body_color, strip_color}:
        return None

    hole_abs = [(int(r0 + hr), int(c0 + hc)) for hr, hc in hole_rel]
    return {
        "body_color": int(body_color),
        "strip_color": int(strip_color),
        "hole_coords": hole_abs,
        "shape": _normalize_coords(hole_rel),
    }


def _parse_key(grid: np.ndarray, comp: list[tuple[int, int]], bg: int):
    colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != bg}
    if len(colors) != 1:
        return None
    r0, c0, _r1, _c1 = _bbox(comp)
    rel = [(int(r - r0), int(c - c0)) for r, c in comp]
    return {
        "color": next(iter(colors)),
        "coords": [(int(r), int(c)) for r, c in comp],
        "shape": _normalize_coords(rel),
    }


def _assign_keys_to_locks(keys, locks):
    assert len(keys) == len(locks) and keys

    candidates: list[list[tuple[int, int, bool]]] = []
    for key in keys:
        options: list[tuple[int, int, bool]] = []
        key_rots = _rotations(key["shape"])
        for lock_i, lock in enumerate(locks):
            for k, rotated in enumerate(key_rots):
                if rotated == lock["shape"]:
                    options.append((int(lock_i), int(k), key["color"] == lock["body_color"]))
                    break
        color_matched = [opt for opt in options if opt[2]]
        if color_matched:
            options = color_matched
        assert options, "every key must match at least one lock hole"
        candidates.append(options)

    order = sorted(range(len(keys)), key=lambda i: len(candidates[i]))
    used: set[int] = set()
    assignment: dict[int, tuple[int, int]] = {}

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        key_i = order[depth]
        ranked = sorted(candidates[key_i], key=lambda item: (not item[2], item[0]))
        for lock_i, rot_k, _is_color_match in ranked:
            if lock_i in used:
                continue
            used.add(lock_i)
            assignment[key_i] = (int(lock_i), int(rot_k))
            if dfs(depth + 1):
                return True
            used.remove(lock_i)
            del assignment[key_i]
        return False

    assert dfs(0), "matching must be unique enough to assign every key"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each solid key to a congruent lock hole under 90-degree rotations,
    move it into the hole, and recolor the inserted key with the adjacent strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    comps = _components(g != bg)
    locks = []
    keys = []
    for comp in comps:
        lock = _parse_lock(g, comp, bg)
        if lock is not None:
            locks.append(lock)
            continue
        key = _parse_key(g, comp, bg)
        if key is not None:
            keys.append(key)

    assignment = _assign_keys_to_locks(keys, locks)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg
    for key_i, (lock_i, _rot_k) in assignment.items():
        lock = locks[lock_i]
        for r, c in lock["hole_coords"]:
            out[r, c] = lock["strip_color"]

    return out


# --------------- generator ---------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LOCK_COUNT_CHOICES = [1, 2, 3]
GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (22, 32)
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 300
GEN_ATTEMPTS = 400
OBJECT_GAP = 1


def _shape_from_profile(side: str, h: int, w: int, profile: list[int]) -> tuple[tuple[int, int], ...]:
    coords = []
    if side == "top":
        for c, height in enumerate(profile):
            for r in range(height):
                coords.append((int(r), int(c)))
    elif side == "bottom":
        for c, height in enumerate(profile):
            for r in range(h - height, h):
                coords.append((int(r), int(c)))
    elif side == "left":
        for r, width in enumerate(profile):
            for c in range(width):
                coords.append((int(r), int(c)))
    else:
        assert side == "right"
        for r, width in enumerate(profile):
            for c in range(w - width, w):
                coords.append((int(r), int(c)))
    return _normalize_coords(coords)


def _generate_hole_shape(side: str):
    for _ in range(SHAPE_ATTEMPTS):
        if side in {"top", "bottom"}:
            h = random.randint(3, 5)
            w = random.randint(max(4, h + 1), 7)
            profile = [random.randint(1, h) for _ in range(w)]
            if max(profile) != h:
                continue
        else:
            w = random.randint(3, 5)
            h = random.randint(max(4, w + 1), 7)
            profile = [random.randint(1, w) for _ in range(h)]
            if max(profile) != w:
                continue

        if len(set(profile)) < 2:
            continue

        shape = _shape_from_profile(side, h, w, profile)
        area = len(shape)
        if area < 6 or area >= h * w:
            continue
        if len(_rotations(shape)) != 4:
            continue

        return {"side": side, "shape": shape, "height": h, "width": w}
    return None


def _build_lock_patch(shape_info, body_color: int, strip_color: int) -> np.ndarray:
    side = shape_info["side"]
    hole = shape_info["shape"]
    hole_h = int(shape_info["height"])
    hole_w = int(shape_info["width"])

    top = random.randint(2, 3) if side == "top" else random.randint(1, 2)
    bottom = random.randint(2, 3) if side == "bottom" else random.randint(1, 2)
    left = random.randint(2, 3) if side == "left" else random.randint(1, 2)
    right = random.randint(2, 3) if side == "right" else random.randint(1, 2)

    patch = np.full((top + hole_h + bottom, left + hole_w + right), body_color, dtype=int)
    for r, c in hole:
        patch[top + r, left + c] = BG

    if side == "top":
        patch[top - 1, left : left + hole_w] = strip_color
    elif side == "bottom":
        patch[top + hole_h, left : left + hole_w] = strip_color
    elif side == "left":
        patch[top : top + hole_h, left - 1] = strip_color
    else:
        patch[top : top + hole_h, left + hole_w] = strip_color

    return patch


def _build_key_patch(shape_info, body_color: int, rotation_k: int) -> np.ndarray:
    arr = _coords_to_array(shape_info["shape"])
    arr = np.rot90(arr, rotation_k)
    patch = np.full(arr.shape, BG, dtype=int)
    patch[arr] = body_color
    return patch


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(0, gh - h)
        c0 = random.randint(0, gw - w)
        rr0 = max(0, r0 - OBJECT_GAP)
        rr1 = min(gh, r0 + h + OBJECT_GAP)
        cc0 = max(0, c0 - OBJECT_GAP)
        cc1 = min(gw, c0 + w + OBJECT_GAP)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _attempt_grid() -> np.ndarray | None:
    lock_count = random.choice(LOCK_COUNT_CHOICES)
    needed_colors = random.sample(PALETTE, 2 * lock_count)
    body_colors = needed_colors[:lock_count]
    strip_colors = needed_colors[lock_count:]

    shape_infos = []
    seen_shapes = set()
    for _i in range(lock_count):
        ok = False
        for _ in range(SHAPE_ATTEMPTS):
            side = random.choice(["top", "bottom", "left", "right"])
            info = _generate_hole_shape(side)
            if info is None:
                continue
            canon = _canonical_shape(info["shape"])
            if canon in seen_shapes:
                continue
            seen_shapes.add(canon)
            shape_infos.append(info)
            ok = True
            break
        if not ok:
            return None

    patches = []
    for i, info in enumerate(shape_infos):
        rotation_k = random.randint(0, 3)
        lock_patch = _build_lock_patch(info, body_colors[i], strip_colors[i])
        key_patch = _build_key_patch(info, body_colors[i], rotation_k)
        patches.append(lock_patch)
        patches.append(key_patch)

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    grid_h = random.randint(max(22, max_h + 6), GRID_H_RANGE[1])
    grid_w = random.randint(max(22, max_w + 6), GRID_W_RANGE[1])

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        if not _place_patch(grid, reserved, patches[idx]):
            return None

    _, counts = np.unique(grid, return_counts=True)
    if counts.max() != int((grid == BG).sum()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
