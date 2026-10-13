import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
KEY_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]

GEN_RETRIES = 400
SHAPE_RETRIES = 120
LAYOUT_RETRIES = 80
PLACE_RETRIES = 180
LOCK_MIN_SIZE = 8
LOCK_MAX_SIZE = 18
GRID_MIN_SIZE = 18
GRID_MAX_SIZE = 30
SIDE_MARGIN = 2
SIDE_GAP = 2


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((k, rot))
    return out


def _shape_key(mask: np.ndarray):
    keys = []
    for _, rot in _rotation_variants(mask):
        keys.append((int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes()))
    return min(keys)


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    for comp in _components(bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        return True
    return False


def _lock_holes(grid: np.ndarray):
    lock_comps = _components(grid == BLUE)
    assert lock_comps, "expected one blue lock"
    lock_coords = max(lock_comps, key=len)
    h, w = grid.shape
    lock_mask = np.zeros((h, w), dtype=bool)
    for r, c in lock_coords:
        lock_mask[r, c] = True

    r0, c0, r1, c1 = _bbox(lock_coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _components(sub == BLACK):
        if not any(
            rr == 0 or rr == sub.shape[0] - 1 or cc == 0 or cc == sub.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        if not any(
            sub[nr, nc] == BLUE
            for rr, cc in comp
            for nr, nc in _neighbors4(rr, cc, sub.shape[0], sub.shape[1])
        ):
            continue
        mask, (hr0, hc0, hr1, hc1) = _coords_to_mask(comp)
        holes.append(
            {
                "coords": [(r0 + rr, c0 + cc) for rr, cc in comp],
                "mask": mask,
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
            }
        )
    return lock_mask, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to a same-shape border notch in the blue lock, rotate
    the key into the notch, then recolor the directly adjacent blue outline cells
    with that key's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    lock_mask, holes = _lock_holes(g)
    assert holes, "lock must contain at least one notch"

    keys = []
    for color in sorted(int(x) for x in np.unique(g) if x not in (BLACK, BLUE)):
        for comp in _components(g == color):
            mask, bbox = _coords_to_mask(comp)
            keys.append({"color": color, "coords": comp, "mask": mask, "bbox": bbox})
    assert len(keys) == len(holes), "number of keys must match number of notches"

    candidates: list[list[tuple[int, int]]] = []
    for hole in holes:
        opts: list[tuple[int, int]] = []
        for key_idx, key in enumerate(keys):
            for rot_k, rot_mask in _rotation_variants(key["mask"]):
                if rot_mask.shape == hole["mask"].shape and np.array_equal(rot_mask, hole["mask"]):
                    opts.append((key_idx, rot_k))
        assert opts, "each notch must have a matching key"
        candidates.append(opts)

    order = sorted(range(len(holes)), key=lambda idx: (len(candidates[idx]), holes[idx]["bbox"]))
    assignment: dict[int, tuple[int, int]] = {}
    used_keys: set[int] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        hole_idx = order[pos]
        for key_idx, rot_k in sorted(candidates[hole_idx], key=lambda item: (keys[item[0]]["color"], item[1])):
            if key_idx in used_keys:
                continue
            used_keys.add(key_idx)
            assignment[hole_idx] = (key_idx, rot_k)
            if search(pos + 1):
                return True
            used_keys.remove(key_idx)
            del assignment[hole_idx]
        return False

    assert search(0), "expected a one-to-one key-notch matching"

    out = g.copy()
    out[(g != BLACK) & (g != BLUE)] = BLACK

    for hole_idx, hole in enumerate(holes):
        key_idx, _rot_k = assignment[hole_idx]
        color = int(keys[key_idx]["color"])

        shell = set()
        for r, c in hole["coords"]:
            out[r, c] = color
            for nr, nc in _neighbors4(r, c, h, w):
                if lock_mask[nr, nc]:
                    shell.add((nr, nc))

        for r, c in shell:
            out[r, c] = color

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_area = min(h * w, 8)
        min_area = max(3, (h * w + 1) // 2 - 1)
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        mh, mw = mask.shape
        area = int(mask.sum())
        if min(mh, mw) < 2:
            continue
        if area * 10 < 4 * mh * mw:
            continue
        if _has_hole(mask):
            continue
        if len(_rotation_variants(mask)) < 2:
            continue
        if mask.all() and random.random() < 0.8:
            continue
        return mask
    return None


def _required_span(lengths: list[int]) -> int:
    if not lengths:
        return LOCK_MIN_SIZE
    return 2 * SIDE_MARGIN + sum(lengths) + SIDE_GAP * (len(lengths) - 1)


def _random_slack(total: int, slots: int) -> list[int]:
    out = [0] * slots
    for _ in range(total):
        out[random.randrange(slots)] += 1
    return out


def _sample_positions(span: int, lengths: list[int]) -> list[int] | None:
    if not lengths:
        return []
    base = _required_span(lengths)
    extra = span - base
    if extra < 0:
        return None
    pads = _random_slack(extra, len(lengths) + 1)
    pos = []
    cursor = SIDE_MARGIN + pads[0]
    for i, length in enumerate(lengths):
        pos.append(cursor)
        cursor += length
        if i + 1 < len(lengths):
            cursor += SIDE_GAP + pads[i + 1]
    return pos


def _build_lock(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    by_side: dict[str, list[int]] = defaultdict(list)
    for idx, spec in enumerate(specs):
        by_side[spec["side"]].append(idx)
    for idxs in by_side.values():
        random.shuffle(idxs)

    top_depth = max((int(specs[i]["mask"].shape[0]) for i in by_side["top"]), default=0)
    bottom_depth = max((int(specs[i]["mask"].shape[0]) for i in by_side["bottom"]), default=0)
    left_depth = max((int(specs[i]["mask"].shape[1]) for i in by_side["left"]), default=0)
    right_depth = max((int(specs[i]["mask"].shape[1]) for i in by_side["right"]), default=0)

    width_need = max(
        _required_span([int(specs[i]["mask"].shape[1]) for i in by_side["top"]]),
        _required_span([int(specs[i]["mask"].shape[1]) for i in by_side["bottom"]]),
        left_depth + right_depth + 3,
        LOCK_MIN_SIZE,
    )
    height_need = max(
        _required_span([int(specs[i]["mask"].shape[0]) for i in by_side["left"]]),
        _required_span([int(specs[i]["mask"].shape[0]) for i in by_side["right"]]),
        top_depth + bottom_depth + 3,
        LOCK_MIN_SIZE,
    )

    lock_h = height_need + random.randint(0, 4)
    lock_w = width_need + random.randint(0, 4)
    if lock_h > LOCK_MAX_SIZE or lock_w > LOCK_MAX_SIZE:
        return None

    along_positions: dict[int, int] = {}
    for side in ("top", "bottom"):
        idxs = by_side[side]
        starts = _sample_positions(lock_w, [int(specs[i]["mask"].shape[1]) for i in idxs])
        if starts is None:
            return None
        for idx, start in zip(idxs, starts):
            along_positions[idx] = int(start)
    for side in ("left", "right"):
        idxs = by_side[side]
        starts = _sample_positions(lock_h, [int(specs[i]["mask"].shape[0]) for i in idxs])
        if starts is None:
            return None
        for idx, start in zip(idxs, starts):
            along_positions[idx] = int(start)

    lock = np.full((lock_h, lock_w), BLUE, dtype=int)
    built_specs: list[dict] = []
    for idx, spec in enumerate(specs):
        mask = spec["mask"]
        mh, mw = mask.shape
        side = spec["side"]
        if side == "top":
            r0 = 0
            c0 = along_positions[idx]
        elif side == "bottom":
            r0 = lock_h - mh
            c0 = along_positions[idx]
        elif side == "left":
            r0 = along_positions[idx]
            c0 = 0
        else:
            r0 = along_positions[idx]
            c0 = lock_w - mw

        coords = []
        for r in range(mh):
            for c in range(mw):
                if not mask[r, c]:
                    continue
                if lock[r0 + r, c0 + c] != BLUE:
                    return None
                lock[r0 + r, c0 + c] = BLACK
                coords.append((r0 + r, c0 + c))

        built_specs.append(
            {
                "side": side,
                "mask": mask,
                "bbox": (r0, c0, r0 + mh - 1, c0 + mw - 1),
                "coords": coords,
            }
        )

    if not np.any(lock[0] == BLUE) or not np.any(lock[-1] == BLUE):
        return None
    if not np.any(lock[:, 0] == BLUE) or not np.any(lock[:, -1] == BLUE):
        return None
    if len(_components(lock == BLUE)) != 1:
        return None
    if len(_components(lock == BLACK)) != len(specs):
        return None

    shells = []
    for spec in built_specs:
        shell = set()
        for r, c in spec["coords"]:
            for nr, nc in _neighbors4(r, c, lock_h, lock_w):
                if lock[nr, nc] == BLUE:
                    shell.add((nr, nc))
        if not shell:
            return None
        shells.append(shell)
    for i in range(len(shells)):
        for j in range(i + 1, len(shells)):
            if shells[i] & shells[j]:
                return None
    for spec, shell in zip(built_specs, shells):
        spec["shell"] = shell

    return lock, built_specs


def _make_specs(count: int) -> list[dict] | None:
    specs = []
    seen = set()
    for _ in range(count * 20):
        if len(specs) == count:
            return specs
        base = _random_shape()
        if base is None:
            continue
        lock_rot = random.randrange(4)
        notch_mask = np.rot90(base, lock_rot)
        key = _shape_key(notch_mask)
        if key in seen:
            continue
        seen.add(key)
        specs.append({"side": random.choice(SIDES), "mask": notch_mask})
    return None


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = color
    return patch


def _place_scene(lock: np.ndarray, key_patches: list[np.ndarray]):
    lock_h, lock_w = lock.shape
    min_h = max(GRID_MIN_SIZE, lock_h + 4)
    min_w = max(GRID_MIN_SIZE, lock_w + 4)
    if min_h > GRID_MAX_SIZE or min_w > GRID_MAX_SIZE:
        return None

    for _ in range(LAYOUT_RETRIES):
        h = random.randint(min_h, GRID_MAX_SIZE)
        w = random.randint(min_w, GRID_MAX_SIZE)
        if h - lock_h < 2 or w - lock_w < 2:
            continue

        lock_r = random.randint(1, h - lock_h - 1)
        lock_c = random.randint(1, w - lock_w - 1)
        grid = np.full((h, w), BLACK, dtype=int)
        grid[lock_r : lock_r + lock_h, lock_c : lock_c + lock_w] = lock

        blocked = np.zeros((h, w), dtype=bool)
        br0 = max(0, lock_r - 1)
        bc0 = max(0, lock_c - 1)
        br1 = min(h, lock_r + lock_h + 1)
        bc1 = min(w, lock_c + lock_w + 1)
        blocked[br0:br1, bc0:bc1] = True

        positions: dict[int, tuple[int, int]] = {}
        order = list(range(len(key_patches)))
        random.shuffle(order)
        ok = True
        for key_idx in order:
            patch = key_patches[key_idx]
            ph, pw = patch.shape
            placed = False
            for _ in range(PLACE_RETRIES):
                r0 = random.randint(1, h - ph - 1)
                c0 = random.randint(1, w - pw - 1)
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r0 + ph + 1
                cc1 = c0 + pw + 1
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                mask = patch != BLACK
                if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][mask] != BLACK):
                    continue
                sub = grid[r0 : r0 + ph, c0 : c0 + pw]
                sub[mask] = patch[mask]
                blocked[rr0:rr1, cc0:cc1] = True
                positions[key_idx] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid, (lock_r, lock_c), positions
    return None


def _scene_output(grid: np.ndarray, built_specs: list[dict], lock_origin: tuple[int, int], colors: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[(out != BLACK) & (out != BLUE)] = BLACK
    lock_r, lock_c = lock_origin
    for spec, color in zip(built_specs, colors):
        for r, c in spec["coords"]:
            out[lock_r + r, lock_c + c] = color
        for r, c in spec["shell"]:
            out[lock_r + r, lock_c + c] = color
    return out


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(12):
        count = random.randint(2, 4)
        colors = random.sample(KEY_COLORS, count)
        specs = _make_specs(count)
        if specs is None:
            continue

        built = _build_lock(specs)
        if built is None:
            continue
        lock, built_specs = built

        key_patches = []
        for color, spec in zip(colors, built_specs):
            key_rot = random.randrange(4)
            key_mask = np.rot90(spec["mask"], key_rot)
            key_patches.append(_color_patch(key_mask, color))

        placed = _place_scene(lock, key_patches)
        if placed is None:
            continue
        grid, lock_origin, _positions = placed

        expected = _scene_output(grid, built_specs, lock_origin, colors)

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid lock-and-key sample")
