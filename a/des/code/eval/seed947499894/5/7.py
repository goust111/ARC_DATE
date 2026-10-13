from __future__ import annotations

import numpy as np


BLACK = 0
GREEN = 3
ORANGE = 7
MAROON = 9

LOCK_FILL_COLORS = (1, 2, 4, 5, 6, 8)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 34
PAIR_COUNT_CHOICES = (2, 3, 4)
LOCK_INNER_MIN = 4
LOCK_INNER_MAX = 9
PLACEMENT_RETRIES = 250

SHAPE_POOL = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_component(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for arr in (rot, np.fliplr(rot)):
            key = ((int(arr.shape[0]), int(arr.shape[1])), arr.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(arr.copy())
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(
        (((int(arr.shape[0]), int(arr.shape[1])), arr.tobytes()) for arr in _variants(mask)),
        key=lambda item: (item[0][0], item[0][1], item[1]),
    )


def _tile_mask(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    h, w = out_shape
    rows = np.arange(h) % int(mask.shape[0])
    cols = np.arange(w) % int(mask.shape[1])
    return mask[rows[:, None], cols[None, :]]


def _balanced_choices(rng: np.random.Generator, items: tuple[str, ...], n: int) -> list[str]:
    out: list[str] = []
    base = list(items)
    while len(out) < n:
        block = base.copy()
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def _rect_free(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _find_locks(grid: np.ndarray) -> list[dict]:
    locks: list[dict] = []
    for comp in _components(grid == ORANGE):
        coords = comp
        outer_mask, (r0, c0, r1, c1) = _crop_component(coords)
        if outer_mask.shape[0] < 4 or outer_mask.shape[1] < 4:
            continue
        if not all(
            int(r) in (0, outer_mask.shape[0] - 1) or int(c) in (0, outer_mask.shape[1] - 1)
            for r, c in coords - np.array([r0, c0], dtype=int)
        ):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        boundary = np.zeros_like(sub, dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True

        if np.any((sub != ORANGE) & (sub != BLACK) & boundary):
            continue
        if np.any((sub == ORANGE) & ~boundary):
            continue

        inner_colors = set(map(int, np.unique(sub))) - {BLACK, ORANGE}
        if len(inner_colors) != 1:
            continue
        fill_color = int(next(iter(inner_colors)))

        hole = sub == BLACK
        if not 2 <= int(hole.sum()) <= 4:
            continue
        if len(_components(hole)) != 1:
            continue

        touched = []
        if bool(hole[0, :].any()):
            touched.append("top")
        if bool(hole[-1, :].any()):
            touched.append("bottom")
        if bool(hole[:, 0].any()):
            touched.append("left")
        if bool(hole[:, -1].any()):
            touched.append("right")
        if len(touched) != 1:
            continue

        hole_coords = np.argwhere(hole)
        hr0, hc0, hr1, hc1 = _bbox(hole_coords)
        hole_mask = hole[hr0 : hr1 + 1, hc0 : hc1 + 1]
        locks.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "fill": fill_color,
                "hole_mask": hole_mask,
                "hole_cells": [(int(r), int(c)) for r, c in hole_coords],
            }
        )
    locks.sort(key=lambda item: (item["r0"], item["c0"]))
    return locks


def _find_keys(grid: np.ndarray) -> list[dict]:
    keys: list[dict] = []
    for comp in _components(grid == GREEN):
        mask, (r0, c0, r1, c1) = _crop_component(comp)
        keys.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "coords": [(int(r), int(c)) for r, c in comp],
                "mask": mask,
                "shape": _shape_key(mask),
            }
        )
    keys.sort(key=lambda item: (item["r0"], item["c0"]))
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match green keys to orange lock notches by shape up to rotation/flip.
    Insert matched keys into the lockholes, remove the original keys, and
    reveal a maroon tiling of the lockhole template inside each lock.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    locks = _find_locks(g)
    keys = _find_keys(g)

    by_shape: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, key in enumerate(keys):
        by_shape.setdefault(key["shape"], []).append(idx)

    used: set[int] = set()
    for lock in locks:
        shape = _shape_key(lock["hole_mask"])
        candidates = by_shape.get(shape, [])
        match_idx = next((idx for idx in candidates if idx not in used), None)
        if match_idx is None:
            continue
        used.add(match_idx)
        key = keys[match_idx]

        for r, c in key["coords"]:
            out[r, c] = BLACK

        for dr, dc in lock["hole_cells"]:
            out[lock["r0"] + dr, lock["c0"] + dc] = GREEN

        inner = out[lock["r0"] + 1 : lock["r1"], lock["c0"] + 1 : lock["c1"]]
        cover = _tile_mask(lock["hole_mask"], inner.shape)
        inner[(inner == lock["fill"]) & cover] = MAROON

    return out


def _choose_hole_mask(
    rng: np.random.Generator, base_mask: np.ndarray, side: str
) -> np.ndarray:
    if side in ("top", "bottom"):
        valid = [arr for arr in _variants(base_mask) if int(arr.shape[0]) >= 2]
    else:
        valid = [arr for arr in _variants(base_mask) if int(arr.shape[1]) >= 2]
    assert valid
    return valid[int(rng.integers(len(valid)))]


def _choose_key_mask(
    rng: np.random.Generator, base_mask: np.ndarray, hole_mask: np.ndarray
) -> np.ndarray:
    variants = _variants(base_mask)
    different = [
        arr
        for arr in variants
        if arr.shape != hole_mask.shape or not np.array_equal(arr, hole_mask)
    ]
    if different and rng.random() < 0.75:
        return different[int(rng.integers(len(different)))]
    return variants[int(rng.integers(len(variants)))]


def _make_pair_spec(
    rng: np.random.Generator, base_mask: np.ndarray, side: str, fill: int
) -> dict:
    hole_mask = _choose_hole_mask(rng, base_mask, side)
    hh, hw = map(int, hole_mask.shape)

    inner_h = int(rng.integers(max(LOCK_INNER_MIN, hh), LOCK_INNER_MAX + 1))
    inner_w = int(rng.integers(max(LOCK_INNER_MIN, hw), LOCK_INNER_MAX + 1))
    lock_h = inner_h + 2
    lock_w = inner_w + 2

    if side == "top":
        hole_r0 = 0
        hole_c0 = int(rng.integers(1, lock_w - hw))
    elif side == "bottom":
        hole_r0 = lock_h - hh
        hole_c0 = int(rng.integers(1, lock_w - hw))
    elif side == "left":
        hole_r0 = int(rng.integers(1, lock_h - hh))
        hole_c0 = 0
    else:
        hole_r0 = int(rng.integers(1, lock_h - hh))
        hole_c0 = lock_w - hw

    hole_cells = [(hole_r0 + int(r), hole_c0 + int(c)) for r, c in np.argwhere(hole_mask)]
    key_mask = _choose_key_mask(rng, base_mask, hole_mask)

    return {
        "fill": int(fill),
        "hole_mask": hole_mask,
        "hole_cells": hole_cells,
        "lock_h": lock_h,
        "lock_w": lock_w,
        "key_mask": key_mask,
        "key_h": int(key_mask.shape[0]),
        "key_w": int(key_mask.shape[1]),
    }


def _place_objects(
    rng: np.random.Generator, specs: list[dict], h: int, w: int
) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["lock_h"] * specs[idx]["lock_w"],
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(0, h - spec["lock_h"] + 1))
            c0 = int(rng.integers(0, w - spec["lock_w"] + 1))
            if not _rect_free(occupied, r0, c0, spec["lock_h"], spec["lock_w"], pad=1):
                continue
            _mark_rect(occupied, r0, c0, spec["lock_h"], spec["lock_w"], pad=1)
            spec["lock_pos"] = (r0, c0)
            placed = True
            break
        if not placed:
            return False

    key_order = order.copy()
    rng.shuffle(key_order)
    for idx in key_order:
        spec = specs[idx]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(0, h - spec["key_h"] + 1))
            c0 = int(rng.integers(0, w - spec["key_w"] + 1))
            if not _rect_free(occupied, r0, c0, spec["key_h"], spec["key_w"], pad=1):
                continue
            _mark_rect(occupied, r0, c0, spec["key_h"], spec["key_w"], pad=1)
            spec["key_pos"] = (r0, c0)
            placed = True
            break
        if not placed:
            return False

    return True


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for spec in specs:
        r0, c0 = spec["lock_pos"]
        lock = grid[r0 : r0 + spec["lock_h"], c0 : c0 + spec["lock_w"]]
        lock[0, :] = ORANGE
        lock[-1, :] = ORANGE
        lock[:, 0] = ORANGE
        lock[:, -1] = ORANGE
        lock[1:-1, 1:-1] = spec["fill"]
        for dr, dc in spec["hole_cells"]:
            lock[dr, dc] = BLACK

        kr0, kc0 = spec["key_pos"]
        key = grid[kr0 : kr0 + spec["key_h"], kc0 : kc0 + spec["key_w"]]
        key[spec["key_mask"]] = GREEN
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        kr0, kc0 = spec["key_pos"]
        key = out[kr0 : kr0 + spec["key_h"], kc0 : kc0 + spec["key_w"]]
        key[spec["key_mask"]] = BLACK

        lr0, lc0 = spec["lock_pos"]
        for dr, dc in spec["hole_cells"]:
            out[lr0 + dr, lc0 + dc] = GREEN

        inner = out[lr0 + 1 : lr0 + spec["lock_h"] - 1, lc0 + 1 : lc0 + spec["lock_w"] - 1]
        cover = _tile_mask(spec["hole_mask"], inner.shape)
        inner[(inner == spec["fill"]) & cover] = MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.choice(PAIR_COUNT_CHOICES))
    shape_ids = rng.choice(len(SHAPE_POOL), size=n_pairs, replace=False)
    sides = _balanced_choices(rng, SIDES, n_pairs)
    fills = rng.choice(LOCK_FILL_COLORS, size=n_pairs, replace=True)

    specs = [
        _make_pair_spec(rng, SHAPE_POOL[int(shape_idx)], side, int(fill))
        for shape_idx, side, fill in zip(shape_ids, sides, fills)
    ]

    area_need = sum(
        (spec["lock_h"] + 2) * (spec["lock_w"] + 2) + (spec["key_h"] + 2) * (spec["key_w"] + 2)
        for spec in specs
    )
    base = int(np.ceil(np.sqrt(area_need * 1.5)))
    min_h = max(base, max(spec["lock_h"] for spec in specs), max(spec["key_h"] for spec in specs))
    min_w = max(base, max(spec["lock_w"] for spec in specs), max(spec["key_w"] for spec in specs))
    min_h = max(min_h, GRID_MIN)
    min_w = max(min_w, GRID_MIN)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = int(rng.integers(min_h, GRID_MAX + 1))
    w = int(rng.integers(min_w, GRID_MAX + 1))
    if not _place_objects(rng, specs, h, w):
        return None

    inp = _render_input(h, w, specs)
    expected = _expected_output(inp, specs)

    if len(_find_locks(inp)) != n_pairs or len(_find_keys(inp)) != n_pairs:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid lock-and-key input")
