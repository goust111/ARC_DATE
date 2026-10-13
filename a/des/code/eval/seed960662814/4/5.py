from __future__ import annotations

import random
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GREY = 5
WALL = 2
GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 33)
LOCK_COUNT_CHOICES = (2, 3)
PLACEMENT_RETRIES = 250
GENERATION_RETRIES = 400

_RAW_GROOVE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: np.ndarray | tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    norm = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm, h, _ = _normalize(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize(rot)[0]


def _rotations(coords: np.ndarray | tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    cur = _normalize(coords)[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        cur = _normalize(cur)[0]
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rot90(cur)
    return out


def _shape_key(coords: np.ndarray | tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_rotations(coords))


def _coords_to_mask(coords: np.ndarray | tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> np.ndarray:
    norm, h, w = _normalize(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_lock_component(grid: np.ndarray, bg: int, lock_color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    rel = coords.copy()
    rel[:, 0] -= int(r0)
    rel[:, 1] -= int(c0)

    # Lock borders span the tight bbox and contain one enclosed colored region plus one edge-touching groove.
    if not (
        np.any(rel[:, 0] == 0)
        and np.any(rel[:, 0] == sub.shape[0] - 1)
        and np.any(rel[:, 1] == 0)
        and np.any(rel[:, 1] == sub.shape[1] - 1)
    ):
        return None

    colored = _components((sub != lock_color) & (sub != bg))
    interiors = [comp for comp in colored if not _touches_edge(comp, sub.shape)]
    if len(interiors) != 1 or sum(len(comp) for comp in colored) != len(interiors[0]):
        return None

    interior_rel = interiors[0]
    interior_colors = set(map(int, sub[interior_rel[:, 0], interior_rel[:, 1]].tolist()))
    if len(interior_colors) != 1:
        return None

    bg_comps = _components(sub == bg)
    grooves = [comp for comp in bg_comps if _touches_edge(comp, sub.shape)]
    if len(grooves) != 1:
        return None

    groove_rel = grooves[0]
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "interior_abs": _shift(interior_rel, r0, c0),
        "groove_abs": _shift(groove_rel, r0, c0),
        "shape_key": _shape_key(groove_rel),
    }


def _find_locks(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    best_color = None
    best_locks: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        locks = []
        for comp in _components(grid == color):
            parsed = _parse_lock_component(grid, bg, int(color), comp)
            if parsed is not None:
                locks.append(parsed)
        if len(locks) > len(best_locks):
            best_color = int(color)
            best_locks = locks
    assert best_color is not None and best_locks, "expected at least one lock"
    return int(best_color), best_locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each solid key to the unique lock groove with the same shape up to rotation.
    Fill the groove with the key color, recolor the lock's enclosed region with that
    color, and erase the key from its original position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    lock_color, locks = _find_locks(g, bg)

    interior_mask = np.zeros(g.shape, dtype=bool)
    for lock in locks:
        interior_mask[lock["interior_abs"][:, 0], lock["interior_abs"][:, 1]] = True

    lock_by_shape = {}
    for lock in locks:
        assert lock["shape_key"] not in lock_by_shape, "lock grooves must be uniquely matchable"
        lock_by_shape[lock["shape_key"]] = lock

    key_components = _components((g != bg) & (g != lock_color) & (~interior_mask))
    assert len(key_components) == len(locks), "key count must equal lock count"

    assignments = []
    seen_shapes = set()
    for comp in key_components:
        colors = set(map(int, g[comp[:, 0], comp[:, 1]].tolist()))
        assert len(colors) == 1, "each key must be monochrome"
        color = colors.pop()
        shape = _shape_key(comp)
        assert shape in lock_by_shape, "every key must match a lock groove"
        assert shape not in seen_shapes, "key shapes must be unique"
        seen_shapes.add(shape)
        assignments.append((comp, int(color), lock_by_shape[shape]))

    out = g.copy()
    for key_coords, color, lock in assignments:
        out[key_coords[:, 0], key_coords[:, 1]] = bg
        out[lock["groove_abs"][:, 0], lock["groove_abs"][:, 1]] = int(color)
        out[lock["interior_abs"][:, 0], lock["interior_abs"][:, 1]] = int(color)
    return out


def _unique_groove_shapes() -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for shape in _RAW_GROOVE_SHAPES:
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        out.append(_normalize(shape)[0])
    return out


GROOVE_SHAPES = _unique_groove_shapes()
assert len(GROOVE_SHAPES) >= 6


def _paint_shape(coords: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    mask = _coords_to_mask(coords)
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _orientations_for_side(shape: tuple[tuple[int, int], ...], side: str) -> list[tuple[tuple[int, int], ...]]:
    out = []
    for ori in _rotations(shape):
        _, h, w = _normalize(ori)
        if side in {"top", "bottom"} and h <= WALL:
            out.append(ori)
        if side in {"left", "right"} and w <= WALL:
            out.append(ori)
    assert out
    return out


def _make_lock(shape: tuple[tuple[int, int], ...], interior_color: int, bg: int) -> dict:
    side = random.choice(["top", "bottom", "left", "right"])
    groove_norm = random.choice(_orientations_for_side(shape, side))
    _, gh, gw = _normalize(groove_norm)

    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 7)
    outer_h = inner_h + 2 * WALL
    outer_w = inner_w + 2 * WALL

    if side in {"top", "bottom"} and outer_w - gw < 2:
        outer_w = gw + 2
        inner_w = outer_w - 2 * WALL
    if side in {"left", "right"} and outer_h - gh < 2:
        outer_h = gh + 2
        inner_h = outer_h - 2 * WALL

    patch = np.full((outer_h, outer_w), GREY, dtype=int)
    patch[WALL : WALL + inner_h, WALL : WALL + inner_w] = int(interior_color)

    if side in {"top", "bottom"}:
        c0 = random.randint(1, outer_w - gw - 1)
        r0 = 0 if side == "top" else outer_h - gh
    else:
        r0 = random.randint(1, outer_h - gh - 1)
        c0 = 0 if side == "left" else outer_w - gw

    groove_abs = np.array([(r0 + r, c0 + c) for r, c in groove_norm], dtype=int)
    patch[groove_abs[:, 0], groove_abs[:, 1]] = int(bg)

    rr, cc = np.indices((inner_h, inner_w))
    interior_abs = np.column_stack((rr.ravel() + WALL, cc.ravel() + WALL)).astype(int)

    return {
        "patch": patch,
        "groove_abs": groove_abs,
        "interior_abs": interior_abs,
        "groove_norm": groove_norm,
        "shape_key": _shape_key(groove_norm),
    }


def _make_key(shape: tuple[tuple[int, int], ...], color: int, bg: int, orientation: tuple[tuple[int, int], ...]) -> dict:
    return {
        "patch": _paint_shape(orientation, color, bg),
        "coords": np.array(_normalize(orientation)[0], dtype=int),
        "color": int(color),
        "shape_key": _shape_key(shape),
        "orientation": _normalize(orientation)[0],
    }


def _choose_colors(bg: int, count: int) -> list[int]:
    pool = [c for c in range(10) if c not in {int(bg), GREY}]
    return random.sample(pool, count)


def _place_objects(objects: list[dict], h: int, w: int) -> dict[str, tuple[int, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    placements: dict[str, tuple[int, int]] = {}
    order = sorted(objects, key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True)

    for obj in order:
        ph, pw = obj["patch"].shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements[obj["name"]] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None
    return placements


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _expected_output(inp: np.ndarray, bg: int, pairs: list[dict], placements: dict[str, tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        lock_r, lock_c = placements[pair["lock_name"]]
        key_r, key_c = placements[pair["key_name"]]

        key_abs = _shift(pair["key"]["coords"], key_r, key_c)
        groove_abs = _shift(pair["lock"]["groove_abs"], lock_r, lock_c)
        interior_abs = _shift(pair["lock"]["interior_abs"], lock_r, lock_c)
        color = int(pair["key"]["color"])

        out[key_abs[:, 0], key_abs[:, 1]] = int(bg)
        out[groove_abs[:, 0], groove_abs[:, 1]] = color
        out[interior_abs[:, 0], interior_abs[:, 1]] = color
    return out


def _build_instance() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != GREY])
    lock_count = random.choice(LOCK_COUNT_CHOICES)
    shapes = random.sample(GROOVE_SHAPES, lock_count)
    colors = _choose_colors(bg, 2 * lock_count)
    random.shuffle(colors)
    interior_colors = colors[:lock_count]
    key_colors = colors[lock_count:]

    pairs: list[dict] = []
    for idx, shape in enumerate(shapes):
        lock = _make_lock(shape, interior_colors[idx], bg)
        key_orientation = random.choice(_rotations(shape))
        pairs.append(
            {
                "lock": lock,
                "key": _make_key(shape, key_colors[idx], bg, key_orientation),
                "lock_name": f"lock_{idx}",
                "key_name": f"key_{idx}",
                "rotations": _rotations(shape),
            }
        )

    # Encourage visible rotation reasoning when an alternative orientation exists.
    if all(pair["key"]["orientation"] == pair["lock"]["groove_norm"] for pair in pairs):
        candidates = [pair for pair in pairs if len(pair["rotations"]) > 1]
        if candidates:
            pair = random.choice(candidates)
            alts = [ori for ori in pair["rotations"] if ori != pair["lock"]["groove_norm"]]
            if alts:
                pair["key"] = _make_key(pair["lock"]["shape_key"], pair["key"]["color"], bg, random.choice(alts))

    objects = []
    for pair in pairs:
        objects.append({"name": pair["lock_name"], "patch": pair["lock"]["patch"]})
        objects.append({"name": pair["key_name"], "patch": pair["key"]["patch"]})

    h_lo = max(GRID_H_RANGE[0], max(obj["patch"].shape[0] for obj in objects) + 2)
    w_lo = max(GRID_W_RANGE[0], max(obj["patch"].shape[1] for obj in objects) + 2)
    if h_lo > GRID_H_RANGE[1] or w_lo > GRID_W_RANGE[1]:
        return None
    h = random.randint(h_lo, GRID_H_RANGE[1])
    w = random.randint(w_lo, GRID_W_RANGE[1])

    placements = _place_objects(objects, h, w)
    if placements is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for obj in objects:
        r0, c0 = placements[obj["name"]]
        _paste(grid, obj["patch"], r0, c0, bg)

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != int(bg):
        return None
    if np.sum(counts == counts.max()) != 1:
        return None

    expected = _expected_output(grid, bg, pairs, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC input")
