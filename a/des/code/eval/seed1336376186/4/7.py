import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (rot.shape, np.ascontiguousarray(rot.astype(np.uint8)).tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot.copy())
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    best = None
    for rot in _rotation_variants(mask):
        key = (int(rot.shape[0]), int(rot.shape[1]), np.ascontiguousarray(rot.astype(np.uint8)).tobytes())
        if best is None or key < best:
            best = key
    return best


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(arr)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _detect_lock(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != color) & (sub != bg)):
        return None
    missing = sub == bg
    if int(missing.sum()) < 3:
        return None
    if int((~missing).sum()) != int(comp["area"]):
        return None
    missing_comps = _components(missing)
    if len(missing_comps) != 1:
        return None
    notch_coords = missing_comps[0]
    sides = set()
    if np.any(notch_coords[:, 0] == 0):
        sides.add("top")
    if np.any(notch_coords[:, 0] == h - 1):
        sides.add("bottom")
    if np.any(notch_coords[:, 1] == 0):
        sides.add("left")
    if np.any(notch_coords[:, 1] == w - 1):
        sides.add("right")
    if len(sides) != 1:
        return None
    if int(sub[0, 0]) != color or int(sub[0, -1]) != color or int(sub[-1, 0]) != color or int(sub[-1, -1]) != color:
        return None
    notch_mask = missing
    notch_crop = _crop_mask(notch_mask)
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "area": int(comp["area"]),
        "side": next(iter(sides)),
        "notch_mask": notch_mask.copy(),
        "notch_crop": notch_crop,
        "signature": _shape_signature(notch_crop),
    }


def _apply_unlock(out: np.ndarray, lock: dict, key: dict, bg: int) -> None:
    r0, c0, r1, c1 = lock["bbox"]
    notch = lock["notch_mask"]
    key_color = int(key["color"])
    out[r0 : r1 + 1, c0 : c1 + 1][notch] = key_color

    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if not notch[rr - r0, cc - c0]:
                out[rr, cc] = bg

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    key_mask = key["mask"]
    scale = min(inner_w // int(key_mask.shape[1]), inner_h // int(key_mask.shape[0]))
    if scale < 1:
        return
    scaled = np.kron(key_mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
    sh, sw = scaled.shape
    top = r0 + 1 + (inner_h - sh) // 2
    left = c0 + 1 + (inner_w - sw) // 2
    if top < r0 + 1 or left < c0 + 1 or top + sh > r1 or left + sw > c1:
        return
    window = out[top : top + sh, left : left + sw]
    window[scaled] = key_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each small key shape to the background notch of a large lock rectangle,
    repair the notch using the key color, remove the key from its original location,
    and reveal a centered enlarged copy of the key inside the matched lock.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    components = _color_components(g, bg)
    locks = []
    keys = []
    for comp in components:
        lock = _detect_lock(g, bg, comp)
        if lock is not None:
            locks.append(lock)
        else:
            keys.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "mask": comp["mask"].copy(),
                    "bbox": comp["bbox"],
                    "signature": _shape_signature(comp["mask"]),
                }
            )

    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = bg

    locks_by_sig: dict[tuple[int, int, bytes], list[dict]] = {}
    keys_by_sig: dict[tuple[int, int, bytes], list[dict]] = {}
    for lock in locks:
        locks_by_sig.setdefault(lock["signature"], []).append(lock)
    for key in keys:
        keys_by_sig.setdefault(key["signature"], []).append(key)

    for sig, key_list in keys_by_sig.items():
        lock_list = locks_by_sig.get(sig, [])
        if len(key_list) != 1 or len(lock_list) != 1:
            continue
        _apply_unlock(out, lock_list[0], key_list[0], bg)

    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        ["11", "10"],
        ["111", "010"],
        ["110", "011"],
        ["111", "100"],
        ["111", "001"],
        ["110", "010", "010"],
        ["011", "110", "100"],
        ["111", "110"],
        ["111", "010", "010"],
        ["1110", "0111"],
        ["111", "101"],
        ["111", "011", "001"],
    ]
    shapes = []
    seen = set()
    for rows in raw:
        mask = _mask_from_rows(rows)
        sig = _shape_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _build_lock_spec(
    rng: random.Random,
    bg: int,
    lock_color: int,
    shape: np.ndarray,
    matched: bool,
    key_color: int | None = None,
) -> dict | None:
    for _ in range(80):
        key_mask = np.rot90(shape, rng.randrange(4))
        notch_mask = np.rot90(shape, rng.randrange(4))
        side = rng.choice(SIDES)

        scale = rng.choice([2, 3])
        key_h, key_w = map(int, key_mask.shape)
        notch_h, notch_w = map(int, notch_mask.shape)

        extra_w = rng.randrange(key_w)
        extra_h = rng.randint(2, max(3, key_h + 2))
        inner_w = key_w * scale + extra_w
        inner_h = max(key_h * scale + extra_h, notch_h + 1)
        if inner_w > 12 or inner_h > 12:
            continue
        if side in ("top", "bottom") and inner_w < notch_w:
            continue
        if side in ("left", "right") and inner_h < notch_h:
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        notch_full = np.zeros((outer_h, outer_w), dtype=bool)

        if side == "top":
            start_r = 0
            start_c = 1 + rng.randrange(inner_w - notch_w + 1)
        elif side == "bottom":
            start_r = outer_h - notch_h
            start_c = 1 + rng.randrange(inner_w - notch_w + 1)
        elif side == "left":
            start_r = 1 + rng.randrange(inner_h - notch_h + 1)
            start_c = 0
        else:
            start_r = 1 + rng.randrange(inner_h - notch_h + 1)
            start_c = outer_w - notch_w

        notch_full[start_r : start_r + notch_h, start_c : start_c + notch_w] |= notch_mask

        input_patch = np.full((outer_h, outer_w), lock_color, dtype=int)
        input_patch[notch_full] = bg
        output_patch = input_patch.copy()

        if matched:
            if key_color is None:
                continue
            output_patch[notch_full] = key_color
            for rr in range(1, outer_h - 1):
                for cc in range(1, outer_w - 1):
                    if not notch_full[rr, cc]:
                        output_patch[rr, cc] = bg

            scaled = np.kron(key_mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
            sh, sw = scaled.shape
            top = 1 + (inner_h - sh) // 2
            left = 1 + (inner_w - sw) // 2
            if np.any(notch_full[top : top + sh, left : left + sw] & scaled):
                continue
            output_patch[top : top + sh, left : left + sw][scaled] = key_color

        return {
            "kind": "lock",
            "matched": matched,
            "input_patch": input_patch,
            "output_patch": output_patch,
            "key_mask": key_mask.copy(),
        }
    return None


def _build_key_spec(bg: int, key_color: int, key_mask: np.ndarray) -> dict:
    patch = np.full(key_mask.shape, bg, dtype=int)
    patch[key_mask] = key_color
    return {"kind": "key", "input_patch": patch, "output_patch": None}


def _place_objects(
    rng: random.Random,
    shape: tuple[int, int],
    objects: list[dict],
    pad: int = 1,
) -> list[tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(objects)), key=lambda i: objects[i]["input_patch"].shape[0] * objects[i]["input_patch"].shape[1], reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)

    for idx in order:
        patch = objects[idx]["input_patch"]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(250):
            r = rng.randrange(h - ph + 1)
            c = rng.randrange(w - pw + 1)
            rr0 = max(0, r - pad)
            cc0 = max(0, c - pad)
            rr1 = min(h, r + ph + pad)
            cc1 = min(w, c + pw + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            placements[idx] = (r, c)
            occupied[r : r + ph, c : c + pw] = True
            placed = True
            break
        if not placed:
            return None
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _render_scene(shape: tuple[int, int], bg: int, objects: list[dict], placements: list[tuple[int, int]], output: bool) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj, (r, c) in zip(objects, placements):
        if output and obj["kind"] == "key":
            continue
        patch = obj["output_patch"] if output and obj["kind"] == "lock" else obj["input_patch"]
        ph, pw = patch.shape
        grid[r : r + ph, c : c + pw] = patch
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    if rng.random() < 0.5:
        n_unmatched = 0
        n_matched = rng.choice([1, 2, 3])
    else:
        n_unmatched = 1
        n_matched = rng.choice([1, 2])

    total_locks = n_matched + n_unmatched
    if total_locks > len(SHAPE_LIBRARY):
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    lock_colors = [colors.pop() for _ in range(total_locks)]
    key_colors = [colors.pop() for _ in range(n_matched)]

    chosen_shapes = rng.sample(SHAPE_LIBRARY, total_locks)
    objects: list[dict] = []

    for i in range(n_matched):
        lock = _build_lock_spec(rng, bg, lock_colors[i], chosen_shapes[i], matched=True, key_color=key_colors[i])
        if lock is None:
            return None
        objects.append(lock)
        objects.append(_build_key_spec(bg, key_colors[i], lock["key_mask"]))

    for i in range(n_unmatched):
        lock = _build_lock_spec(
            rng,
            bg,
            lock_colors[n_matched + i],
            chosen_shapes[n_matched + i],
            matched=False,
        )
        if lock is None:
            return None
        objects.append(lock)

    max_h = max(int(obj["input_patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["input_patch"].shape[1]) for obj in objects)
    for _ in range(30):
        grid_h = rng.randint(max(22, max_h + 2), 31)
        grid_w = rng.randint(max(22, max_w + 2), 31)
        placements = _place_objects(rng, (grid_h, grid_w), objects, pad=1)
        if placements is None:
            continue
        inp = _render_scene((grid_h, grid_w), bg, objects, placements, output=False)
        vals, counts = np.unique(inp, return_counts=True)
        if int(vals[np.argmax(counts)]) != bg:
            continue
        expected = _render_scene((grid_h, grid_w), bg, objects, placements, output=True)
        if np.array_equal(main(inp), expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
