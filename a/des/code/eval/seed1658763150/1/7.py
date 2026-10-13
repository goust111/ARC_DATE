import random

import numpy as np


BG = 0
LOCK_COLOR = 5
KEY_COLOR = 9
CHEST_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_RANGE = (20, 28)
CHEST_COUNT_RANGE = (2, 4)
CHEST_SIZE_RANGE = (4, 6)
MAX_EXTRA_KEYS = 1
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_for_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(coords, dtype=int))
    comps.sort(key=lambda pts: tuple(np.min(pts, axis=0)))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_for_mask(mask.astype(bool))
    return len(comps) == 1


def _is_tight(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _all_small_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for h, w in ((2, 3), (3, 2)):
        for bits in range(1, 1 << (h * w)):
            mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=bool).reshape(h, w)
            cells = int(mask.sum())
            if cells < 3 or cells >= h * w:
                continue
            if not _is_tight(mask) or not _is_connected(mask):
                continue
            shapes.append(mask)
    shapes.sort(key=lambda m: (int(m.shape[0]), int(m.shape[1]), int(m.sum()), _mask_key(m)))
    return shapes


def _lock_shape_specs() -> list[dict]:
    specs: list[dict] = []
    for mask in _all_small_shapes():
        hflip = np.fliplr(mask)
        vflip = np.flipud(mask)
        if np.array_equal(mask, hflip) or np.array_equal(mask, vflip):
            continue
        if np.array_equal(hflip, vflip):
            continue
        specs.append(
            {
                "mask": mask,
                "h_key": hflip,
                "v_key": vflip,
                "targets": {_mask_key(hflip), _mask_key(vflip)},
            }
        )
    return specs


ALL_SMALL_SHAPES = _all_small_shapes()
LOCK_SPECS = _lock_shape_specs()
SIDES = ("left", "right", "top", "bottom")


def _adjacent_side(chest_bbox: tuple[int, int, int, int], lock_bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = chest_bbox
    lr0, lc0, lr1, lc1 = lock_bbox
    if lc1 == c0 - 1 and r0 <= lr0 and lr1 <= r1:
        return "left"
    if lc0 == c1 + 1 and r0 <= lr0 and lr1 <= r1:
        return "right"
    if lr1 == r0 - 1 and c0 <= lc0 and lc1 <= c1:
        return "top"
    if lr0 == r1 + 1 and c0 <= lc0 and lc1 <= c1:
        return "bottom"
    return None


def _is_filled_rectangle(g: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))


def _find_chests(g: np.ndarray) -> list[dict]:
    chests: list[dict] = []
    for color in CHEST_COLORS:
        for coords in _components_for_mask(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            if _is_filled_rectangle(g, coords, color):
                chests.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    chests.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return chests


def _find_locks(g: np.ndarray, chests: list[dict]) -> list[dict]:
    locks: list[dict] = []
    for coords in _components_for_mask(g == LOCK_COLOR):
        mask = _tight_mask(coords)
        box = _bbox(coords)
        if mask.shape not in ((2, 3), (3, 2)):
            continue
        attached = []
        for chest_idx, chest in enumerate(chests):
            side = _adjacent_side(chest["bbox"], box)
            if side is not None:
                attached.append((chest_idx, side))
        if len(attached) != 1:
            continue
        chest_idx, side = attached[0]
        locks.append(
            {
                "coords": coords,
                "bbox": box,
                "mask": mask,
                "chest_idx": chest_idx,
                "side": side,
            }
        )
    locks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return locks


def _find_keys(g: np.ndarray) -> list[dict]:
    keys: list[dict] = []
    for coords in _components_for_mask(g == KEY_COLOR):
        keys.append({"coords": coords, "bbox": _bbox(coords), "mask": _tight_mask(coords)})
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock chest rectangles whose attached grey lock is mirrored by a maroon key.

    Matched chests keep their border color, turn their interior grey, and remove
    both the lock and the matching key. Unmatched objects remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    chests = _find_chests(g)
    locks = _find_locks(g, chests)
    keys = _find_keys(g)

    used_locks: set[int] = set()
    pairings: list[tuple[int, int]] = []
    for key_idx, key in enumerate(keys):
        chosen = None
        for pass_name in ("h", "v"):
            for lock_idx, lock in enumerate(locks):
                if lock_idx in used_locks:
                    continue
                target = np.fliplr(lock["mask"]) if pass_name == "h" else np.flipud(lock["mask"])
                if key["mask"].shape == target.shape and np.array_equal(key["mask"], target):
                    chosen = lock_idx
                    break
            if chosen is not None:
                break
        if chosen is not None:
            used_locks.add(chosen)
            pairings.append((chosen, key_idx))

    for lock_idx, key_idx in pairings:
        chest = chests[locks[lock_idx]["chest_idx"]]
        r0, c0, r1, c1 = chest["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = LOCK_COLOR
        out[locks[lock_idx]["coords"][:, 0], locks[lock_idx]["coords"][:, 1]] = BG
        out[keys[key_idx]["coords"][:, 0], keys[key_idx]["coords"][:, 1]] = BG

    return out


def _valid_positions(occupied: np.ndarray, ph: int, pw: int, padding: int = 1) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        rr0 = max(0, r0 - padding)
        rr1 = min(h, r0 + ph + padding)
        for c0 in range(w - pw + 1):
            cc0 = max(0, c0 - padding)
            cc1 = min(w, c0 + pw + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    sub[mask] = patch[mask]


def _compose_chest_patch(
    chest_color: int,
    chest_h: int,
    chest_w: int,
    lock_mask: np.ndarray,
    side: str,
    offset: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    lh, lw = map(int, lock_mask.shape)
    if side in ("left", "right"):
        patch = np.full((chest_h, chest_w + lw), BG, dtype=int)
        chest_r0, chest_c0 = 0, (lw if side == "left" else 0)
        lock_r0, lock_c0 = offset, (0 if side == "left" else chest_w)
    else:
        patch = np.full((chest_h + lh, chest_w), BG, dtype=int)
        chest_r0, chest_c0 = (lh if side == "top" else 0), 0
        lock_r0, lock_c0 = (0 if side == "top" else chest_h), offset

    patch[chest_r0 : chest_r0 + chest_h, chest_c0 : chest_c0 + chest_w] = chest_color
    patch[lock_r0 : lock_r0 + lh, lock_c0 : lock_c0 + lw][lock_mask] = LOCK_COLOR
    chest_bbox = (chest_r0, chest_c0, chest_r0 + chest_h - 1, chest_c0 + chest_w - 1)
    return patch, chest_bbox, np.argwhere(lock_mask) + np.array([lock_r0, lock_c0])


def _choose_lock_specs(count: int) -> list[dict] | None:
    specs = LOCK_SPECS.copy()
    random.shuffle(specs)
    chosen: list[dict] = []
    used_targets: set[tuple[int, int, bytes]] = set()
    for spec in specs:
        if spec["targets"] & used_targets:
            continue
        chosen.append(spec)
        used_targets |= spec["targets"]
        if len(chosen) == count:
            return chosen
    return None


def _safe_noise_shapes(forbidden: set[tuple[int, int, bytes]]) -> list[np.ndarray]:
    return [mask for mask in ALL_SMALL_SHAPES if _mask_key(mask) not in forbidden]


def _build_expected(inp: np.ndarray, paired: list[tuple[dict, dict]]) -> np.ndarray:
    out = inp.copy()
    for chest, key in paired:
        r0, c0, r1, c1 = chest["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = LOCK_COLOR
        out[chest["lock_coords"][:, 0], chest["lock_coords"][:, 1]] = BG
        out[key["coords"][:, 0], key["coords"][:, 1]] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    n_chests = random.randint(*CHEST_COUNT_RANGE)
    lock_specs = _choose_lock_specs(n_chests)
    if lock_specs is None:
        return None

    colors = random.sample(CHEST_COLORS, n_chests)
    unlocked_count = random.randint(1, n_chests)
    if n_chests > 1 and random.random() < 0.5:
        unlocked_count = random.randint(1, n_chests - 1)
    unmatched_locks = n_chests - unlocked_count
    extra_keys = random.randint(0, min(MAX_EXTRA_KEYS, unlocked_count))

    modes: list[str] = []
    if unlocked_count == 1:
        modes = [random.choice(("h", "v"))]
    else:
        if random.random() < 0.5:
            modes = ["h", "v"] + [random.choice(("h", "v")) for _ in range(unlocked_count - 2)]
            random.shuffle(modes)
        else:
            modes = [random.choice(("h", "v")) for _ in range(unlocked_count)]

    chest_plan = []
    order = list(range(n_chests))
    random.shuffle(order)
    unlocked_set = set(order[:unlocked_count])

    for idx in range(n_chests):
        chest_h = random.randint(*CHEST_SIZE_RANGE)
        chest_w = random.randint(*CHEST_SIZE_RANGE)
        spec = lock_specs[idx]
        lock_mask = spec["mask"]
        side = random.choice(SIDES)
        if side in ("left", "right"):
            offset = random.randint(0, chest_h - lock_mask.shape[0])
        else:
            offset = random.randint(0, chest_w - lock_mask.shape[1])
        chest_plan.append(
            {
                "color": colors[idx],
                "h": chest_h,
                "w": chest_w,
                "lock_mask": lock_mask,
                "side": side,
                "offset": offset,
                "paired": idx in unlocked_set,
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    chests: list[dict] = []
    for item in chest_plan:
        patch, local_bbox, local_lock = _compose_chest_patch(
            item["color"], item["h"], item["w"], item["lock_mask"], item["side"], item["offset"]
        )
        positions = _valid_positions(occupied, patch.shape[0], patch.shape[1], padding=1)
        if not positions:
            return None
        pr0, pc0 = random.choice(positions)
        _paste_nonzero(grid, patch, pr0, pc0)
        occupied[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]] = True
        chest_bbox = (
            local_bbox[0] + pr0,
            local_bbox[1] + pc0,
            local_bbox[2] + pr0,
            local_bbox[3] + pc0,
        )
        chests.append(
            {
                "bbox": chest_bbox,
                "lock_mask": item["lock_mask"],
                "paired": item["paired"],
                "lock_coords": local_lock + np.array([pr0, pc0]),
            }
        )

    key_specs: list[dict] = []
    mode_idx = 0
    used_targets = {_mask_key(np.fliplr(spec["mask"])) for spec in lock_specs}
    used_targets |= {_mask_key(np.flipud(spec["mask"])) for spec in lock_specs}
    for chest in chests:
        if not chest["paired"]:
            continue
        mode = modes[mode_idx]
        mode_idx += 1
        key_mask = np.fliplr(chest["lock_mask"]) if mode == "h" else np.flipud(chest["lock_mask"])
        key_specs.append({"mask": key_mask, "match_bbox": chest["bbox"]})

    if extra_keys:
        noise_shapes = _safe_noise_shapes(used_targets)
        if len(noise_shapes) < extra_keys:
            return None
        for mask in random.sample(noise_shapes, extra_keys):
            key_specs.append({"mask": mask, "match_bbox": None})

    random.shuffle(key_specs)
    keys: list[dict] = []
    for item in key_specs:
        mask = item["mask"]
        ph, pw = mask.shape
        positions = _valid_positions(occupied, ph, pw, padding=1)
        if not positions:
            return None
        pr0, pc0 = random.choice(positions)
        patch = np.full((ph, pw), BG, dtype=int)
        patch[mask] = KEY_COLOR
        _paste_nonzero(grid, patch, pr0, pc0)
        occupied[pr0 : pr0 + ph, pc0 : pc0 + pw] = True
        keys.append({"coords": np.argwhere(mask) + np.array([pr0, pc0]), "match_bbox": item["match_bbox"]})

    matched_pairs: list[tuple[dict, dict]] = []
    paired_chests = {tuple(chest["bbox"]): chest for chest in chests if chest["paired"]}
    for key in keys:
        if key["match_bbox"] is not None:
            matched_pairs.append((paired_chests[tuple(key["match_bbox"])], key))

    if unmatched_locks == 0 and extra_keys == 0 and unlocked_count == 1:
        return None

    expected = _build_expected(grid, matched_pairs)
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
    raise RuntimeError("failed to generate a valid unlock-the-chest input")
