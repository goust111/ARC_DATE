from __future__ import annotations

import itertools

import numpy as np


BG = 0
AZURE = 8
MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
PAIR_COUNT_RANGE = (2, 6)
GEN_RETRIES = 500
NOTCH_PLACE_RETRIES = 120
ROOM_PLACE_RETRIES = 160
KEY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(mask, dtype=bool)
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(base, int(k))
        for var in (rot, np.fliplr(rot)):
            key = (int(var.shape[0]), int(var.shape[1]), var.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(np.array(var, copy=True))
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(var.shape[0]), int(var.shape[1]), var.astype(np.uint8).tobytes())
        for var in _dihedral_variants(mask)
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []

    for border_coords in _components4(g == AZURE):
        r0, c0, r1, c1 = _bbox(border_coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all((border_coords[:, 0] == r0) | (border_coords[:, 0] == r1) | (border_coords[:, 1] == c0) | (border_coords[:, 1] == c1)):
            continue
        if not np.all(g[r0, c0 : c1 + 1] == AZURE):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == AZURE):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == AZURE):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == AZURE):
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        key_mask = (interior != BG) & (interior != AZURE)
        key_comps = _components4(key_mask)
        if len(key_comps) != 1:
            continue

        colors = np.unique(interior[key_mask])
        if len(colors) != 1:
            continue

        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": int(colors[0]),
                "mask": _tight_mask(key_comps[0]),
            }
        )

    rooms.sort(key=lambda d: d["bbox"])
    return rooms


def _extract_notches(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    notches: list[dict] = []

    for coords in _components4(g == MAROON):
        if not _touches_border(coords, h, w):
            continue
        notches.append({"bbox": _bbox(coords), "coords": coords, "mask": _tight_mask(coords)})

    notches.sort(key=lambda d: d["bbox"])
    return notches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's key to a border lockhole by shape under rotation and
    horizontal reflection, then recolor the lockhole with the key color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    notches = _extract_notches(g)
    assert rooms and notches and len(rooms) == len(notches)

    color_by_shape: dict[tuple[int, int, bytes], int] = {}
    for room in rooms:
        key = _canonical_shape_key(room["mask"])
        assert key not in color_by_shape
        color_by_shape[key] = int(room["color"])

    out = g.copy()
    used: set[tuple[int, int, bytes]] = set()
    for notch in notches:
        key = _canonical_shape_key(notch["mask"])
        assert key in color_by_shape
        assert key not in used
        used.add(key)
        color = int(color_by_shape[key])
        coords = notch["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _connected_cell_set(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _build_shape_pool() -> tuple[np.ndarray, ...]:
    cells = [(r, c) for r in range(4) for c in range(4)]
    unique: dict[tuple[int, int, bytes], np.ndarray] = {}

    for area in range(3, 7):
        for combo in itertools.combinations(cells, area):
            if not _connected_cell_set(combo):
                continue
            coords = np.asarray(combo, dtype=int)
            mask = _tight_mask(coords)
            key = _canonical_shape_key(mask)
            unique.setdefault(key, mask)

    pool = list(unique.values())
    pool.sort(
        key=lambda mask: (
            int(mask.sum()),
            int(mask.shape[0] * mask.shape[1]),
            int(mask.shape[0]),
            int(mask.shape[1]),
            mask.astype(np.uint8).tobytes(),
        )
    )
    return tuple(pool)


SHAPE_POOL = _build_shape_pool()
IRREGULAR_POOL = tuple(
    mask
    for mask in SHAPE_POOL
    if mask.shape[0] > 1 and mask.shape[1] > 1 and int(mask.sum()) < int(mask.shape[0] * mask.shape[1])
)


def _random_variant(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    variants = _dihedral_variants(mask)
    return np.array(variants[int(rng.integers(0, len(variants)))], copy=True)


def _choose_pair_count(h: int, w: int, rng: np.random.Generator) -> int:
    candidates = []
    for n in range(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1]):
        enough_area = h * w >= 60 * n + 30
        enough_edge = 2 * (h + w) - 4 >= 6 * n + 10
        if enough_area and enough_edge:
            candidates.append(n)
    if not candidates:
        return 2
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _pick_base_shapes(count: int, rng: np.random.Generator) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    used = set()
    while len(chosen) < count:
        pool = IRREGULAR_POOL if IRREGULAR_POOL and rng.random() < 0.75 else SHAPE_POOL
        mask = pool[int(rng.integers(0, len(pool)))]
        key = _canonical_shape_key(mask)
        if key in used:
            continue
        used.add(key)
        chosen.append(np.array(mask, copy=True))
    return chosen


def _rect_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _edge_start_range(limit: int, span: int) -> tuple[int, int] | None:
    if span > limit:
        return None
    low = 1
    high = limit - span - 1
    if high < low:
        low = 0
        high = limit - span
    if high < low:
        return None
    return int(low), int(high)


def _sample_notch_anchor(
    side: str, h: int, w: int, grid_h: int, grid_w: int, rng: np.random.Generator
) -> tuple[int, int] | None:
    if side in ("top", "bottom"):
        rng_range = _edge_start_range(grid_w, w)
        if rng_range is None:
            return None
        c0 = int(rng.integers(rng_range[0], rng_range[1] + 1))
        r0 = 0 if side == "top" else grid_h - h
        return int(r0), int(c0)

    rng_range = _edge_start_range(grid_h, h)
    if rng_range is None:
        return None
    r0 = int(rng.integers(rng_range[0], rng_range[1] + 1))
    c0 = 0 if side == "left" else grid_w - w
    return int(r0), int(c0)


def _place_notches(
    notch_specs: list[dict], grid_h: int, grid_w: int, rng: np.random.Generator
) -> tuple[list[dict], np.ndarray] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict | None] = [None] * len(notch_specs)
    order = sorted(
        range(len(notch_specs)),
        key=lambda idx: int(notch_specs[idx]["mask"].shape[0] * notch_specs[idx]["mask"].shape[1]),
        reverse=True,
    )

    for idx in order:
        spec = notch_specs[idx]
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        success = False
        for _ in range(NOTCH_PLACE_RETRIES):
            side = SIDES[int(rng.integers(0, len(SIDES)))]
            anchor = _sample_notch_anchor(side, h, w, grid_h, grid_w, rng)
            if anchor is None:
                continue
            r0, c0 = anchor
            if not _rect_clear(occupied, r0, c0, h, w, pad=1):
                continue
            coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            placed[idx] = {
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "coords": coords.astype(int),
                "color": int(spec["color"]),
                "mask": np.array(mask, copy=True),
            }
            _mark_rect(occupied, r0, c0, h, w, pad=1)
            success = True
            break
        if not success:
            return None

    return [item for item in placed if item is not None], occupied


def _sample_room_spec(mask: np.ndarray, color: int, rng: np.random.Generator) -> dict:
    kh, kw = map(int, mask.shape)
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    return {
        "key_mask": np.array(mask, copy=True),
        "color": int(color),
        "top_margin": int(top),
        "left_margin": int(left),
        "room_h": int(kh + top + bottom + 2),
        "room_w": int(kw + left + right + 2),
    }


def _sample_room_anchor(limit: int, span: int, rng: np.random.Generator) -> int | None:
    low = 1
    high = limit - span - 1
    if high < low:
        return None
    return int(rng.integers(low, high + 1))


def _place_rooms(
    room_specs: list[dict], occupied: np.ndarray, grid_h: int, grid_w: int, rng: np.random.Generator
) -> list[dict] | None:
    occ = np.array(occupied, copy=True)
    placed: list[dict | None] = [None] * len(room_specs)
    order = sorted(
        range(len(room_specs)),
        key=lambda idx: int(room_specs[idx]["room_h"] * room_specs[idx]["room_w"]),
        reverse=True,
    )

    for idx in order:
        spec = room_specs[idx]
        room_h = int(spec["room_h"])
        room_w = int(spec["room_w"])
        success = False
        for _ in range(ROOM_PLACE_RETRIES):
            r0 = _sample_room_anchor(grid_h, room_h, rng)
            c0 = _sample_room_anchor(grid_w, room_w, rng)
            if r0 is None or c0 is None:
                return None
            if not _rect_clear(occ, r0, c0, room_h, room_w, pad=1):
                continue
            _mark_rect(occ, r0, c0, room_h, room_w, pad=1)
            placed[idx] = {
                "bbox": (int(r0), int(c0), int(r0 + room_h - 1), int(c0 + room_w - 1)),
                "key_r0": int(r0 + 1 + spec["top_margin"]),
                "key_c0": int(c0 + 1 + spec["left_margin"]),
                "key_mask": np.array(spec["key_mask"], copy=True),
                "color": int(spec["color"]),
            }
            success = True
            break
        if not success:
            return None

    return [item for item in placed if item is not None]


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = AZURE
    grid[r1, c0 : c1 + 1] = AZURE
    grid[r0 : r1 + 1, c0] = AZURE
    grid[r0 : r1 + 1, c1] = AZURE

    mask = room["key_mask"]
    coords = np.argwhere(mask)
    rr = coords[:, 0] + int(room["key_r0"])
    cc = coords[:, 1] + int(room["key_c0"])
    grid[rr, cc] = int(room["color"])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(*GRID_H_RANGE))
    grid_w = int(rng.integers(*GRID_W_RANGE))
    pair_count = _choose_pair_count(grid_h, grid_w, rng)
    if pair_count > len(KEY_COLORS):
        return None

    base_shapes = _pick_base_shapes(pair_count, rng)
    colors = [int(v) for v in rng.choice(KEY_COLORS, size=pair_count, replace=False).tolist()]

    notch_specs = []
    room_specs = []
    expected_pairs = []
    for base_mask, color in zip(base_shapes, colors):
        room_mask = _random_variant(base_mask, rng)
        notch_mask = _random_variant(base_mask, rng)
        notch_specs.append({"mask": notch_mask, "color": int(color)})
        room_specs.append(_sample_room_spec(room_mask, int(color), rng))
        expected_pairs.append((_canonical_shape_key(base_mask), int(color)))

    notches_res = _place_notches(notch_specs, grid_h, grid_w, rng)
    if notches_res is None:
        return None
    placed_notches, occupied = notches_res

    placed_rooms = _place_rooms(room_specs, occupied, grid_h, grid_w, rng)
    if placed_rooms is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for room in placed_rooms:
        _draw_room(grid, room)
    for notch in placed_notches:
        coords = notch["coords"]
        grid[coords[:, 0], coords[:, 1]] = MAROON

    expected = grid.copy()
    for notch in placed_notches:
        coords = notch["coords"]
        expected[coords[:, 0], coords[:, 1]] = int(notch["color"])

    if np.array_equal(expected, grid):
        return None

    rooms = _extract_rooms(grid)
    notches = _extract_notches(grid)
    if len(rooms) != pair_count or len(notches) != pair_count:
        return None

    room_keys = {_canonical_shape_key(room["mask"]) for room in rooms}
    notch_keys = {_canonical_shape_key(notch["mask"]) for notch in notches}
    if room_keys != notch_keys or len(room_keys) != pair_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
