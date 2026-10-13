import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
PAIR_COUNT_CHOICES = [1, 2, 3]
PAIR_COUNT_WEIGHTS = [0.25, 0.4, 0.35]
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 28
HOLE_AREA_RANGE = (3, 7)
LOCK_PAD_RANGE = (1, 3)
GEN_RETRIES = 400
HOLE_RETRIES = 120
PAIR_RETRIES = 120
PLACEMENT_RETRIES = 240
BORDER_MARGIN = 1
LOCK_LOCK_GAP = 2
KEY_KEY_GAP = 2
KEY_LOCK_GAP = 3


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    coords = np.array(shape, dtype=int)
    turns %= 4
    if turns == 0:
        return _normalize(coords)
    h, w = _shape_dims(shape)
    for _ in range(turns):
        coords = np.array([(c, h - 1 - r) for r, c in coords], dtype=int)
        h, w = w, h
        coords = coords - coords.min(axis=0)
    return _normalize(coords)


def _all_rotations(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(_rotate_shape(shape, k) for k in range(4))


def _canonical_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_all_rotations(shape))


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _lock_from_component(grid: np.ndarray, color: int, coords: np.ndarray, bg: int):
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != color) & (sub != bg)):
        return None
    if int(np.count_nonzero(sub == color)) != int(len(coords)):
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros(sub.shape, dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return None

    hole_mask = ~mask
    if not hole_mask.any():
        return None
    holes = _components(hole_mask)
    if len(holes) != 1:
        return None
    hole = holes[0]
    hr0, hc0, hr1, hc1 = _bbox(hole)
    if hr0 == 0 or hc0 == 0 or hr1 == sub.shape[0] - 1 or hc1 == sub.shape[1] - 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "hole_shape": _normalize(hole),
        "lock_color": int(color),
    }


def _solve_matching(options: dict[int, list[int]]) -> dict[int, int] | None:
    used_locks: set[int] = set()
    assignment: dict[int, int] = {}
    keys = list(options)

    def dfs() -> bool:
        if len(assignment) == len(keys):
            return True

        remaining = [k for k in keys if k not in assignment]
        key = min(remaining, key=lambda k: len([lid for lid in options[k] if lid not in used_locks]))
        candidates = [lid for lid in options[key] if lid not in used_locks]
        for lid in candidates:
            assignment[key] = lid
            used_locks.add(lid)
            if dfs():
                return True
            used_locks.remove(lid)
            del assignment[key]
        return False

    return assignment if dfs() else None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(g == color):
            comps.append({"color": color, "coords": coords})

    locks = []
    keys = []
    for comp in comps:
        lock = _lock_from_component(g, comp["color"], comp["coords"], bg)
        if lock is not None:
            locks.append(lock)
        else:
            keys.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "shape": _normalize(comp["coords"]),
                }
            )

    assert locks, "expected at least one lock"
    assert len(locks) == len(keys), "lock/key counts must match"

    options: dict[int, list[int]] = {}
    for ki, key in enumerate(keys):
        key_rots = set(_all_rotations(key["shape"]))
        matches = [li for li, lock in enumerate(locks) if lock["hole_shape"] in key_rots]
        assert matches, "every key must match at least one lock"
        options[ki] = matches

    assignment = _solve_matching(options)
    assert assignment is not None and len(assignment) == len(keys), "matching must be unique and complete"

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = bg
    for ki, li in assignment.items():
        r0, c0, r1, c1 = locks[li]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(keys[ki]["color"])
    return out


def _random_hole_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...] | None:
    for _ in range(HOLE_RETRIES):
        target = int(rng.integers(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1] + 1))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(frontier[int(rng.integers(len(frontier)))])

        shape = _normalize(list(cells))
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        if len(shape) == h * w:
            continue
        if len(set(_all_rotations(shape))) != 4:
            continue
        return shape
    return None


def _make_lock_patch(
    rng: np.random.Generator,
    hole_shape: tuple[tuple[int, int], ...],
    lock_color: int,
    bg: int,
) -> np.ndarray:
    hole_h, hole_w = _shape_dims(hole_shape)
    pad_t = int(rng.integers(LOCK_PAD_RANGE[0], LOCK_PAD_RANGE[1] + 1))
    pad_b = int(rng.integers(LOCK_PAD_RANGE[0], LOCK_PAD_RANGE[1] + 1))
    pad_l = int(rng.integers(LOCK_PAD_RANGE[0], LOCK_PAD_RANGE[1] + 1))
    pad_r = int(rng.integers(LOCK_PAD_RANGE[0], LOCK_PAD_RANGE[1] + 1))
    patch = np.full((hole_h + pad_t + pad_b, hole_w + pad_l + pad_r), int(lock_color), dtype=int)
    for r, c in hole_shape:
        patch[pad_t + r, pad_l + c] = int(bg)
    return patch


def _rect_too_close(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    sub[patch != bg] = patch[patch != bg]


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[vals == bg][0])
    for val, count in zip(vals, counts):
        if int(val) != bg and bg_count <= int(count):
            return False
    return True


def _build_expected(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        key_patch = spec["key_patch"]
        kr, kc = spec["key_pos"]
        kh, kw = key_patch.shape
        sub = out[kr : kr + kh, kc : kc + kw]
        sub[key_patch != bg] = bg

        lr, lc = spec["lock_pos"]
        lh, lw = spec["lock_patch"].shape
        out[lr : lr + lh, lc : lc + lw] = int(spec["key_color"])
    return out


def _generate_case(rng: np.random.Generator):
    bg = int(rng.integers(10))
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES, p=PAIR_COUNT_WEIGHTS))
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    lock_colors = colors[:pair_count]
    key_colors = colors[pair_count : 2 * pair_count]

    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for idx in range(pair_count):
        hole_shape = None
        for _ in range(PAIR_RETRIES):
            cand = _random_hole_shape(rng)
            if cand is None:
                continue
            canon = _canonical_shape(cand)
            if canon in used_shapes:
                continue
            used_shapes.add(canon)
            hole_shape = cand
            break
        if hole_shape is None:
            return None

        rotation = int(rng.integers(4))
        lock_patch = _make_lock_patch(rng, hole_shape, int(lock_colors[idx]), bg)
        key_patch = _shape_patch(_rotate_shape(hole_shape, rotation), int(key_colors[idx]), bg)
        specs.append(
            {
                "kind": "pair",
                "lock_patch": lock_patch,
                "key_patch": key_patch,
                "lock_color": int(lock_colors[idx]),
                "key_color": int(key_colors[idx]),
            }
        )

    footprint = sum(spec["lock_patch"].size + spec["key_patch"].size for spec in specs)
    min_side = max(GRID_SIDE_MIN, int(np.ceil(np.sqrt(footprint * 2.6))))
    if min_side > GRID_SIDE_MAX:
        return None
    h = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    w = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    grid = np.full((h, w), bg, dtype=int)

    placed: list[dict] = []
    objects = []
    for idx, spec in enumerate(specs):
        objects.append(("lock", idx, spec["lock_patch"]))
        objects.append(("key", idx, spec["key_patch"]))
    objects.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    for kind, idx, patch in objects:
        ph, pw = patch.shape
        if ph + 2 * BORDER_MARGIN > h or pw + 2 * BORDER_MARGIN > w:
            return None
        placed_here = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(rng.integers(BORDER_MARGIN, h - ph - BORDER_MARGIN + 1))
            left = int(rng.integers(BORDER_MARGIN, w - pw - BORDER_MARGIN + 1))
            rect = (top, left, top + ph - 1, left + pw - 1)
            ok = True
            for prev in placed:
                if kind == "lock" and prev["kind"] == "lock":
                    gap = LOCK_LOCK_GAP
                elif kind == "key" and prev["kind"] == "key":
                    gap = KEY_KEY_GAP
                else:
                    gap = KEY_LOCK_GAP
                if _rect_too_close(rect, prev["rect"], gap):
                    ok = False
                    break
            if not ok:
                continue

            _place_patch(grid, patch, top, left, bg)
            placed.append({"kind": kind, "rect": rect})
            specs[idx][f"{kind}_pos"] = (top, left)
            placed_here = True
            break
        if not placed_here:
            return None

    if not _bg_is_dominant(grid, bg):
        return None

    expected = _build_expected(grid, specs, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
