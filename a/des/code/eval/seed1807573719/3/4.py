from __future__ import annotations

import numpy as np


GREY = 5
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OTHER_COLORS = [0, 1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
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
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return tuple(map(int, trimmed.shape)), trimmed.tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_shape_key(np.rot90(mask, k)) for k in range(4))


def _rotation_match(src: np.ndarray, dst: np.ndarray) -> int | None:
    for k in range(4):
        rot = _trim_mask(np.rot90(src, k))
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            return int(k)
    return None


def _touches_bbox_edge(coords: np.ndarray, h: int, w: int) -> bool:
    for r, c in coords:
        if int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1:
            return True
    return False


def _arrow_side(lock_bbox: tuple[int, int, int, int], arrow_bbox: tuple[int, int, int, int]) -> str | None:
    lr0, lc0, lr1, lc1 = lock_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox
    if ar1 == lr0 - 1 and max(lc0, ac0) <= min(lc1, ac1):
        return "top"
    if ar0 == lr1 + 1 and max(lc0, ac0) <= min(lc1, ac1):
        return "bottom"
    if ac1 == lc0 - 1 and max(lr0, ar0) <= min(lr1, ar1):
        return "left"
    if ac0 == lc1 + 1 and max(lr0, ar0) <= min(lr1, ar1):
        return "right"
    return None


def _find_keys(grid: np.ndarray, bg: int) -> list[dict]:
    key_components = _components(grid == GREY)
    keys: list[dict] = []
    used_markers: set[tuple[int, int]] = set()
    for coords in key_components:
        marker_cells: dict[tuple[int, int], int] = {}
        for r, c in coords:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                val = int(grid[nr, nc])
                if val in {bg, GREY, RED}:
                    continue
                marker_cells[(nr, nc)] = val
        assert len(marker_cells) == 1, "each grey key must have exactly one adjacent marker"
        (mr, mc), marker_color = next(iter(marker_cells.items()))
        assert (mr, mc) not in used_markers, "marker cells must be unique"
        used_markers.add((mr, mc))
        keys.append(
            {
                "coords": coords,
                "mask": _tight_mask_from_coords(coords),
                "marker": (int(mr), int(mc)),
                "marker_color": int(marker_color),
            }
        )
    assert keys, "at least one key is required"
    return keys


def _find_locks(grid: np.ndarray, bg: int) -> list[dict]:
    arrow_components = []
    for coords in _components(grid == RED):
        r0, c0, r1, c1 = _bbox(coords)
        if int(coords.shape[0]) == 3 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            arrow_components.append({"coords": coords, "bbox": (r0, c0, r1, c1)})

    locks: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, GREY, RED}):
        for coords in _components(grid == color):
            if int(coords.shape[0]) < 8:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((sub != int(color)) & (sub != int(bg))):
                continue
            if not np.all(sub[0, :] == int(color)):
                continue
            if not np.all(sub[-1, :] == int(color)):
                continue
            if not np.all(sub[:, 0] == int(color)):
                continue
            if not np.all(sub[:, -1] == int(color)):
                continue

            holes = []
            for hole_coords in _components(sub == bg):
                if not _touches_bbox_edge(hole_coords, sub.shape[0], sub.shape[1]):
                    holes.append(hole_coords)
            if len(holes) != 1:
                continue

            matching_arrows = []
            for arrow in arrow_components:
                side = _arrow_side((r0, c0, r1, c1), arrow["bbox"])
                if side is not None:
                    matching_arrows.append((arrow, side))
            assert len(matching_arrows) == 1, "each lock must have exactly one adjacent arrow"
            arrow, side = matching_arrows[0]
            hole_coords = holes[0]
            hole_global = hole_coords + np.array([r0, c0], dtype=int)
            locks.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "hole_coords": hole_global,
                    "hole_mask": _tight_mask_from_coords(hole_global),
                    "arrow_coords": arrow["coords"],
                    "side": side,
                }
            )
    assert locks, "at least one lock is required"
    return locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey key to the unique lock hole with the same shape up to rotation,
    remove the original key and red arrow, fill the hole with grey, and repaint the
    lock frame using the key's adjacent marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    keys = _find_keys(g, bg)
    locks = _find_locks(g, bg)
    assert len(keys) == len(locks), "the number of keys and locks must match"

    matches: list[tuple[int, int, int]] = []
    for key_idx, key in enumerate(keys):
        options: list[tuple[int, int]] = []
        for lock_idx, lock in enumerate(locks):
            rot = _rotation_match(key["mask"], lock["hole_mask"])
            if rot is not None:
                options.append((int(lock_idx), int(rot)))
        assert len(options) == 1, "each key must match exactly one lock"
        lock_idx, rot = options[0]
        matches.append((int(key_idx), int(lock_idx), int(rot)))

    matched_locks = {lock_idx for _key_idx, lock_idx, _rot in matches}
    assert len(matched_locks) == len(locks), "matching must be one-to-one"

    out = g.copy()
    for key_idx, lock_idx, _rot in matches:
        key = keys[key_idx]
        lock = locks[lock_idx]

        for r, c in key["coords"]:
            out[int(r), int(c)] = bg
        mr, mc = key["marker"]
        out[int(mr), int(mc)] = bg

        for r, c in lock["arrow_coords"]:
            out[int(r), int(c)] = bg
        for r, c in lock["coords"]:
            out[int(r), int(c)] = int(key["marker_color"])
        for r, c in lock["hole_coords"]:
            out[int(r), int(c)] = GREY

    return out


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(4, 8))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if abs(nr) > 3 or abs(nc) > 3 or (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add((int(pick[0]), int(pick[1])))

        coords = np.array(sorted(cells), dtype=int)
        mask = _tight_mask_from_coords(coords)
        h, w = map(int, mask.shape)
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if int(mask.sum()) == h * w:
            continue
        if len({_shape_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric key shape")


def _build_key_patch(shape: np.ndarray, marker_color: int, bg: int, rng: np.random.Generator) -> dict:
    h, w = map(int, shape.shape)
    canvas = np.full((h + 2, w + 2), bg, dtype=int)
    canvas[1 : 1 + h, 1 : 1 + w][shape] = GREY

    choices: set[tuple[int, int]] = set()
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if int(canvas[r, c]) != GREY:
                continue
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if int(canvas[nr, nc]) == bg:
                    choices.add((nr, nc))
    marker = list(sorted(choices))[int(rng.integers(len(choices)))]
    canvas[int(marker[0]), int(marker[1])] = int(marker_color)

    coords = np.argwhere(canvas != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    key_coords = np.argwhere(patch == GREY)
    marker_coords = np.argwhere(patch == int(marker_color))
    assert marker_coords.shape == (1, 2)
    return {
        "patch": patch,
        "key_coords": key_coords,
        "marker": tuple(map(int, marker_coords[0])),
    }


def _outward_missing_choices(side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, 0), (0, 1)]
    if side == "bottom":
        return [(1, 0), (1, 1)]
    if side == "left":
        return [(0, 0), (1, 0)]
    return [(0, 1), (1, 1)]


def _build_lock_patch(shape: np.ndarray, lock_color: int, bg: int, side: str, rng: np.random.Generator) -> dict:
    hh, hw = map(int, shape.shape)
    pad_top = int(rng.integers(1, 4))
    pad_bottom = int(rng.integers(1, 4))
    pad_left = int(rng.integers(1, 4))
    pad_right = int(rng.integers(1, 4))
    outer_h = int(hh + pad_top + pad_bottom)
    outer_w = int(hw + pad_left + pad_right)
    assert outer_h >= 4 and outer_w >= 4

    if side == "top":
        patch = np.full((outer_h + 2, outer_w), bg, dtype=int)
        rect_r0, rect_c0 = 2, 0
        arrow_r0 = 0
        arrow_c0 = int(rng.integers(1, outer_w - 2))
    elif side == "bottom":
        patch = np.full((outer_h + 2, outer_w), bg, dtype=int)
        rect_r0, rect_c0 = 0, 0
        arrow_r0 = outer_h
        arrow_c0 = int(rng.integers(1, outer_w - 2))
    elif side == "left":
        patch = np.full((outer_h, outer_w + 2), bg, dtype=int)
        rect_r0, rect_c0 = 0, 2
        arrow_r0 = int(rng.integers(1, outer_h - 2))
        arrow_c0 = 0
    else:
        patch = np.full((outer_h, outer_w + 2), bg, dtype=int)
        rect_r0, rect_c0 = 0, 0
        arrow_r0 = int(rng.integers(1, outer_h - 2))
        arrow_c0 = outer_w

    patch[rect_r0 : rect_r0 + outer_h, rect_c0 : rect_c0 + outer_w] = int(lock_color)
    hole_r0 = int(rect_r0 + pad_top)
    hole_c0 = int(rect_c0 + pad_left)
    patch[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + hw][shape] = bg

    patch[arrow_r0 : arrow_r0 + 2, arrow_c0 : arrow_c0 + 2] = RED
    miss_r, miss_c = _outward_missing_choices(side)[int(rng.integers(2))]
    patch[int(arrow_r0 + miss_r), int(arrow_c0 + miss_c)] = bg

    frame_coords = np.argwhere(patch == int(lock_color))
    hole_coords = np.argwhere(patch == bg)
    inside_hole_box = (
        (hole_coords[:, 0] >= hole_r0)
        & (hole_coords[:, 0] < hole_r0 + hh)
        & (hole_coords[:, 1] >= hole_c0)
        & (hole_coords[:, 1] < hole_c0 + hw)
    )
    hole_coords = hole_coords[inside_hole_box]
    hole_coords = hole_coords[shape[hole_coords[:, 0] - hole_r0, hole_coords[:, 1] - hole_c0]]
    arrow_coords = np.argwhere(patch == RED)
    return {
        "patch": patch,
        "frame_coords": frame_coords,
        "hole_coords": hole_coords,
        "arrow_coords": arrow_coords,
    }


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    view = canvas[int(r0) : int(r0) + patch.shape[0], int(c0) : int(c0) + patch.shape[1]]
    mask = patch != bg
    view[mask] = patch[mask]


def _can_place(occupied: np.ndarray, patch_shape: tuple[int, int], r0: int, c0: int, padding: int) -> bool:
    h, w = map(int, patch_shape)
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, int(r0 - padding))
    cc0 = max(0, int(c0 - padding))
    rr1 = min(H, int(r0 + h + padding))
    cc1 = min(W, int(c0 + w + padding))
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_items(
    items: list[dict], H: int, W: int, bg: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[tuple[str, int], tuple[int, int]]] | None:
    canvas = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}
    order = list(range(len(items)))
    rng.shuffle(order)

    for item_idx in order:
        item = items[item_idx]
        ph, pw = map(int, item["patch"].shape)
        candidates = [
            (int(r), int(c))
            for r in range(H - ph + 1)
            for c in range(W - pw + 1)
            if _can_place(occupied, (ph, pw), int(r), int(c), padding=1)
        ]
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        _paste(canvas, item["patch"], int(r0), int(c0), bg)
        occupied[int(r0) : int(r0) + ph, int(c0) : int(c0) + pw] = True
        placements[(item["kind"], int(item["index"]))] = (int(r0), int(c0))
    return canvas, placements


def _build_expected(
    inp: np.ndarray,
    lock_specs: list[dict],
    key_specs: list[dict],
    placements: dict[tuple[str, int], tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for idx, key in enumerate(key_specs):
        kr0, kc0 = placements[("key", idx)]
        for r, c in key["key_coords"]:
            out[int(kr0 + r), int(kc0 + c)] = bg
        mr, mc = key["marker"]
        out[int(kr0 + mr), int(kc0 + mc)] = bg

    for idx, lock in enumerate(lock_specs):
        lr0, lc0 = placements[("lock", idx)]
        for r, c in lock["arrow_coords"]:
            out[int(lr0 + r), int(lc0 + c)] = bg
        for r, c in lock["frame_coords"]:
            out[int(lr0 + r), int(lc0 + c)] = int(key_specs[idx]["marker_color"])
        for r, c in lock["hole_coords"]:
            out[int(lr0 + r), int(lc0 + c)] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_pairs = int(rng.integers(1, 4))
    bg = OTHER_COLORS[int(rng.integers(len(OTHER_COLORS)))]

    usable = [c for c in OTHER_COLORS if c != bg]
    chosen = list(rng.choice(np.array(usable, dtype=int), size=2 * n_pairs, replace=False))
    rng.shuffle(chosen)
    lock_colors = [int(c) for c in chosen[:n_pairs]]
    marker_colors = [int(c) for c in chosen[n_pairs:]]

    shapes: list[np.ndarray] = []
    seen_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    while len(shapes) < n_pairs:
        shape = _random_shape(rng)
        key = _canonical_shape_key(shape)
        if key in seen_shape_keys:
            continue
        seen_shape_keys.add(key)
        shapes.append(shape)

    sides = ["top", "bottom", "left", "right"]
    lock_specs: list[dict] = []
    key_specs: list[dict] = []
    for idx in range(n_pairs):
        shape = shapes[idx]
        side = sides[int(rng.integers(4))]
        lock_specs.append(_build_lock_patch(shape, lock_colors[idx], bg, side, rng))

        rotation = int(rng.integers(4))
        rotated = _trim_mask(np.rot90(shape, rotation))
        key_info = _build_key_patch(rotated, marker_colors[idx], bg, rng)
        key_info["marker_color"] = int(marker_colors[idx])
        key_specs.append(key_info)

    items = []
    for idx, lock in enumerate(lock_specs):
        items.append({"kind": "lock", "index": idx, "patch": lock["patch"]})
    for idx, key in enumerate(key_specs):
        items.append({"kind": "key", "index": idx, "patch": key["patch"]})

    total_area = int(sum(item["patch"].shape[0] * item["patch"].shape[1] for item in items))
    max_h = max(int(item["patch"].shape[0]) for item in items)
    max_w = max(int(item["patch"].shape[1]) for item in items)
    side_base = max(max_h + 6, max_w + 6, int(np.ceil(np.sqrt(total_area * 2.0))))

    for _ in range(30):
        H = int(rng.integers(side_base, side_base + 8))
        W = int(rng.integers(side_base, side_base + 8))
        placed = _place_items(items, H, W, bg, rng)
        if placed is None:
            continue
        inp, placements = placed

        colors, counts = np.unique(inp, return_counts=True)
        color_count = {int(c): int(n) for c, n in zip(colors, counts)}
        if color_count.get(bg, 0) <= max(n for c, n in color_count.items() if c != bg):
            continue

        expected = _build_expected(inp, lock_specs, key_specs, placements, bg)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
