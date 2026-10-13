from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
SIDES = ("left", "right", "top", "bottom")


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue
            color = int(g[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            br0, bc0, br1, bc1 = _bbox(coords)
            patch = np.full((br1 - br0 + 1, bc1 - bc0 + 1), int(bg), dtype=int)
            for r, c in coords:
                patch[r - br0, c - bc0] = color
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(br0), int(bc0), int(br1), int(bc1)),
                    "patch": patch,
                    "area": len(coords),
                }
            )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _boundary_connected_bg(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _side_profile(patch: np.ndarray, bg: int, side: str) -> tuple[int, ...]:
    h, w = patch.shape
    prof: list[int] = []

    if side == "left":
        for r in range(h):
            depth = 0
            while depth < w and int(patch[r, depth]) == int(bg):
                depth += 1
            prof.append(int(depth))
        return tuple(prof)

    if side == "right":
        for r in range(h):
            depth = 0
            while depth < w and int(patch[r, w - 1 - depth]) == int(bg):
                depth += 1
            prof.append(int(depth))
        return tuple(prof)

    if side == "top":
        for c in range(w):
            depth = 0
            while depth < h and int(patch[depth, c]) == int(bg):
                depth += 1
            prof.append(int(depth))
        return tuple(prof)

    for c in range(w):
        depth = 0
        while depth < h and int(patch[h - 1 - depth, c]) == int(bg):
            depth += 1
        prof.append(int(depth))
    return tuple(prof)


def _row_contiguous(mask: np.ndarray, row: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(mask[row])
    if cols.size == 0:
        return None
    c0 = int(cols[0])
    c1 = int(cols[-1])
    if c1 - c0 + 1 != int(cols.size):
        return None
    return c0, c1


def _col_contiguous(mask: np.ndarray, col: int) -> tuple[int, int] | None:
    rows = np.flatnonzero(mask[:, col])
    if rows.size == 0:
        return None
    r0 = int(rows[0])
    r1 = int(rows[-1])
    if r1 - r0 + 1 != int(rows.size):
        return None
    return r0, r1


def _extract_lock(patch: np.ndarray, bg: int) -> dict | None:
    bg_mask = patch == int(bg)
    boundary_bg = _boundary_connected_bg(bg_mask)
    cavity = bg_mask & ~boundary_bg
    if not bool(cavity.any()):
        return None

    candidates = []
    for side in SIDES:
        prof = _side_profile(patch, bg, side)
        if max(prof, default=0) > 0:
            candidates.append((side, prof))

    if len(candidates) != 1:
        return None

    side, prof = candidates[0]
    return {
        "kind": "lock",
        "side": side,
        "profile": prof,
        "cavity": [(int(r), int(c)) for r, c in np.argwhere(cavity)],
    }


def _extract_key(patch: np.ndarray, bg: int) -> dict | None:
    mask = patch != int(bg)
    h, w = patch.shape
    candidates = []

    lengths = []
    ok = True
    for r in range(h):
        span = _row_contiguous(mask, r)
        if span is None or span[0] != 0:
            ok = False
            break
        lengths.append(int(span[1] + 1))
    if ok:
        base = min(lengths)
        if max(lengths) > base:
            candidates.append(("right", tuple(int(v - base) for v in lengths)))

    lengths = []
    ok = True
    for r in range(h):
        span = _row_contiguous(mask, r)
        if span is None or span[1] != w - 1:
            ok = False
            break
        lengths.append(int(w - span[0]))
    if ok:
        base = min(lengths)
        if max(lengths) > base:
            candidates.append(("left", tuple(int(v - base) for v in lengths)))

    lengths = []
    ok = True
    for c in range(w):
        span = _col_contiguous(mask, c)
        if span is None or span[0] != 0:
            ok = False
            break
        lengths.append(int(span[1] + 1))
    if ok:
        base = min(lengths)
        if max(lengths) > base:
            candidates.append(("bottom", tuple(int(v - base) for v in lengths)))

    lengths = []
    ok = True
    for c in range(w):
        span = _col_contiguous(mask, c)
        if span is None or span[1] != h - 1:
            ok = False
            break
        lengths.append(int(h - span[0]))
    if ok:
        base = min(lengths)
        if max(lengths) > base:
            candidates.append(("top", tuple(int(v - base) for v in lengths)))

    if len(candidates) != 1:
        return None

    side, prof = candidates[0]
    return {"kind": "key", "side": side, "profile": prof}


def _profiles_match(lock_side: str, lock_prof: tuple[int, ...], key_side: str, key_prof: tuple[int, ...]) -> bool:
    if OPPOSITE[key_side] != lock_side:
        return False
    if len(key_prof) > len(lock_prof):
        return False
    pad = len(lock_prof) - len(key_prof)
    for offset in range(pad + 1):
        if any(lock_prof[i] != 0 for i in range(offset)):
            continue
        if any(lock_prof[offset + len(key_prof) + i] != 0 for i in range(pad - offset)):
            continue
        if tuple(lock_prof[offset : offset + len(key_prof)]) == tuple(key_prof):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each protruding key with the complementary notched lock, then fill the lock's
    enclosed cavity using the key's color. Matched keys disappear; unmatched objects stay.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _find_components(g, bg)

    locks = []
    keys = []
    for comp in comps:
        lock = _extract_lock(comp["patch"], bg)
        if lock is not None:
            lock.update(comp)
            locks.append(lock)
            continue

        key = _extract_key(comp["patch"], bg)
        if key is not None:
            key.update(comp)
            keys.append(key)

    out = g.copy()
    if not locks or not keys:
        return out

    candidates = []
    for ki, key in enumerate(keys):
        matches = []
        for li, lock in enumerate(locks):
            if _profiles_match(lock["side"], lock["profile"], key["side"], key["profile"]):
                matches.append(int(li))
        candidates.append(matches)

    lock_to_key: dict[int, int] = {}

    def _augment(ki: int, seen: set[int]) -> bool:
        for li in candidates[ki]:
            if li in seen:
                continue
            seen.add(li)
            if li not in lock_to_key or _augment(lock_to_key[li], seen):
                lock_to_key[li] = int(ki)
                return True
        return False

    order = sorted(range(len(keys)), key=lambda idx: (len(candidates[idx]), keys[idx]["bbox"]))
    for ki in order:
        _augment(int(ki), set())

    for li, ki in lock_to_key.items():
        lock = locks[int(li)]
        key = keys[int(ki)]
        kr0, kc0, _kr1, _kc1 = key["bbox"]
        for r, c in key["coords"]:
            out[int(r), int(c)] = int(bg)
        for rr, cc in lock["cavity"]:
            out[int(lock["bbox"][0] + rr), int(lock["bbox"][1] + cc)] = int(key["color"])

    return out


# --------------- generator ---------------

BG = 0
COLORS = list(range(1, 10))
PAIR_COUNT_RANGE = (1, 3)
EXTRA_KEY_CHOICES = (0, 1)
LOCK_ORTH_RANGE = (7, 11)
LOCK_NORMAL_RANGE = (8, 12)
KEY_BASE_RANGE = (2, 4)
PROFILE_DEPTH_MAX = 3
GRID_H_RANGE = (18, 32)
GRID_W_RANGE = (18, 32)
PLACEMENT_PADDING = 1
BUILD_RETRIES = 500
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120


def _random_profile(length: int, max_depth: int) -> tuple[int, ...]:
    assert length >= 3
    while True:
        prof = [0] * length
        span_len = random.randint(1, length - 1)
        start = random.randint(0, length - span_len)
        indices = list(range(start, start + span_len))
        chosen_n = random.randint(1, len(indices))
        for idx in random.sample(indices, chosen_n):
            prof[int(idx)] = random.randint(1, max_depth)
        if 0 < max(prof) and min(prof) == 0:
            return tuple(int(v) for v in prof)


def _grow_polyomino(height: int, width: int, area: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(40):
        seed = random.choice(cells)
        chosen = {seed}
        frontier = {seed}
        while len(chosen) < area and frontier:
            base_r, base_c = random.choice(tuple(frontier))
            nbrs = []
            for dr, dc in DIR4:
                nr = int(base_r + dr)
                nc = int(base_c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.discard((base_r, base_c))
                continue
            pick = random.choice(nbrs)
            chosen.add(pick)
            frontier.add(pick)
        if len(chosen) == area:
            mask = np.zeros((height, width), dtype=bool)
            for r, c in chosen:
                mask[int(r), int(c)] = True
            return mask
    return None


def _make_cavity(lock_h: int, lock_w: int, side: str, max_depth: int) -> list[tuple[int, int]] | None:
    top = 1
    bottom = lock_h - 2
    left = 1
    right = lock_w - 2
    if side == "left":
        left = max(left, max_depth + 1)
    elif side == "right":
        right = min(right, lock_w - max_depth - 2)
    elif side == "top":
        top = max(top, max_depth + 1)
    else:
        bottom = min(bottom, lock_h - max_depth - 2)

    avail_h = bottom - top + 1
    avail_w = right - left + 1
    if avail_h < 2 or avail_w < 2:
        return None

    box_h = random.randint(2, min(4, avail_h))
    box_w = random.randint(2, min(4, avail_w))
    box_r = random.randint(top, bottom - box_h + 1)
    box_c = random.randint(left, right - box_w + 1)
    max_area = min(box_h * box_w - 1, 7)
    if max_area < 3:
        return None
    area = random.randint(3, max_area)
    mask = _grow_polyomino(box_h, box_w, area)
    if mask is None:
        return None
    return [(int(box_r + r), int(box_c + c)) for r, c in np.argwhere(mask)]


def _render_lock(lock_h: int, lock_w: int, side: str, profile: tuple[int, ...], cavity: list[tuple[int, int]], color: int) -> np.ndarray:
    patch = np.full((lock_h, lock_w), int(color), dtype=int)
    if side == "left":
        for r, depth in enumerate(profile):
            patch[int(r), : int(depth)] = BG
    elif side == "right":
        for r, depth in enumerate(profile):
            if depth:
                patch[int(r), lock_w - int(depth) :] = BG
    elif side == "top":
        for c, depth in enumerate(profile):
            patch[: int(depth), int(c)] = BG
    else:
        for c, depth in enumerate(profile):
            if depth:
                patch[lock_h - int(depth) :, int(c)] = BG
    for r, c in cavity:
        patch[int(r), int(c)] = BG
    return patch


def _render_key(side: str, profile: tuple[int, ...], base: int, color: int) -> np.ndarray:
    max_depth = max(profile)
    if side in ("left", "right"):
        h = len(profile)
        w = base + max_depth
        patch = np.full((h, w), BG, dtype=int)
        for r, depth in enumerate(profile):
            span = base + int(depth)
            if side == "right":
                patch[int(r), :span] = color
            else:
                patch[int(r), w - span :] = color
        return patch

    w = len(profile)
    h = base + max_depth
    patch = np.full((h, w), BG, dtype=int)
    for c, depth in enumerate(profile):
        span = base + int(depth)
        if side == "bottom":
            patch[:span, int(c)] = color
        else:
            patch[h - span :, int(c)] = color
    return patch


def _make_pair(lock_color: int, key_color: int) -> tuple[dict, dict] | None:
    for _ in range(SHAPE_RETRIES):
        side = random.choice(SIDES)
        orth_len = random.randint(*LOCK_ORTH_RANGE)
        key_len = random.randint(3, orth_len)
        max_depth = random.randint(1, PROFILE_DEPTH_MAX)
        key_prof = _random_profile(key_len, max_depth)
        offset = random.randint(0, orth_len - key_len)
        lock_prof = tuple([0] * offset + list(key_prof) + [0] * (orth_len - key_len - offset))

        normal_min = max(LOCK_NORMAL_RANGE[0], max(lock_prof) + 6)
        if normal_min > LOCK_NORMAL_RANGE[1]:
            continue
        normal_len = random.randint(normal_min, LOCK_NORMAL_RANGE[1])
        if side in ("left", "right"):
            lock_h, lock_w = orth_len, normal_len
        else:
            lock_h, lock_w = normal_len, orth_len

        cavity = _make_cavity(lock_h, lock_w, side, max(lock_prof))
        if cavity is None:
            continue

        lock_patch = _render_lock(lock_h, lock_w, side, lock_prof, cavity, lock_color)
        key_side = OPPOSITE[side]
        key_base = random.randint(*KEY_BASE_RANGE)
        key_patch = _render_key(key_side, key_prof, key_base, key_color)

        lock_meta = {
            "kind": "lock",
            "side": side,
            "profile": lock_prof,
            "cavity": cavity,
            "color": int(lock_color),
            "patch": lock_patch,
            "matched_key_color": int(key_color),
        }
        key_meta = {
            "kind": "key",
            "side": key_side,
            "profile": key_prof,
            "color": int(key_color),
            "patch": key_patch,
            "matched": True,
        }
        return lock_meta, key_meta
    return None


def _make_extra_key(color: int, locks: list[dict]) -> dict | None:
    for _ in range(SHAPE_RETRIES):
        side = random.choice(SIDES)
        prof = _random_profile(random.randint(3, 8), random.randint(1, PROFILE_DEPTH_MAX))
        if any(_profiles_match(lock["side"], lock["profile"], side, prof) for lock in locks):
            continue
        patch = _render_key(side, prof, random.randint(*KEY_BASE_RANGE), color)
        return {
            "kind": "key",
            "side": side,
            "profile": prof,
            "color": int(color),
            "patch": patch,
            "matched": False,
        }
    return None


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    target[mask] = patch[mask]


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    total_area = sum(int(np.count_nonzero(obj["patch"])) for obj in objects)
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    min_h = max(GRID_H_RANGE[0], max_h + 2, int(np.ceil(np.sqrt(total_area * 2.4))))
    min_w = max(GRID_W_RANGE[0], max_w + 2, int(np.ceil(np.sqrt(total_area * 2.4))))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        H = random.randint(min_h, GRID_H_RANGE[1])
        W = random.randint(min_w, GRID_W_RANGE[1])
        grid = np.full((H, W), BG, dtype=int)
        ok = True
        placed: list[dict] = []

        for obj in sorted(objects, key=lambda item: (-(item["patch"] != BG).sum(), -item["patch"].shape[0] * item["patch"].shape[1])):
            patch = obj["patch"]
            h, w = patch.shape
            positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            random.shuffle(positions)
            chosen = None
            for r0, c0 in positions:
                if _can_place(grid, patch, int(r0), int(c0), PLACEMENT_PADDING):
                    chosen = (int(r0), int(c0))
                    break
            if chosen is None:
                ok = False
                break
            _paste(grid, patch, chosen[0], chosen[1])
            placed_obj = dict(obj)
            placed_obj["top"] = chosen[0]
            placed_obj["left"] = chosen[1]
            placed.append(placed_obj)

        if ok and int(np.sum(grid == BG)) * 2 > int(grid.size):
            objects[:] = placed
            return grid
    return None


def _expected_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for obj in objects:
        if obj["kind"] != "key" or not obj.get("matched", False):
            continue
        patch = obj["patch"]
        for dr, dc in np.argwhere(patch != BG):
            out[int(obj["top"] + dr), int(obj["left"] + dc)] = BG

    for obj in objects:
        if obj["kind"] != "lock":
            continue
        fill = int(obj["matched_key_color"])
        for dr, dc in obj["cavity"]:
            out[int(obj["top"] + dr), int(obj["left"] + dc)] = fill
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    n_extra = random.choice(EXTRA_KEY_CHOICES)
    need_colors = n_pairs * 2 + n_extra
    if need_colors > len(COLORS):
        return None
    palette = random.sample(COLORS, need_colors)

    locks: list[dict] = []
    keys: list[dict] = []
    color_idx = 0
    for _ in range(n_pairs):
        pair = _make_pair(palette[color_idx], palette[color_idx + 1])
        color_idx += 2
        if pair is None:
            return None
        lock, key = pair
        if any(_profiles_match(prev["side"], prev["profile"], key["side"], key["profile"]) for prev in locks):
            return None
        if any(_profiles_match(lock["side"], lock["profile"], prev["side"], prev["profile"]) for prev in keys):
            return None
        locks.append(lock)
        keys.append(key)

    for _ in range(n_extra):
        extra = _make_extra_key(palette[color_idx], locks)
        color_idx += 1
        if extra is None:
            return None
        keys.append(extra)

    for i, key in enumerate(keys[:n_pairs]):
        matches = [j for j, lock in enumerate(locks) if _profiles_match(lock["side"], lock["profile"], key["side"], key["profile"])]
        if matches != [i]:
            return None
    for key in keys[n_pairs:]:
        if any(_profiles_match(lock["side"], lock["profile"], key["side"], key["profile"]) for lock in locks):
            return None

    objects = locks + keys
    inp = _place_objects(objects)
    if inp is None:
        return None
    return inp, _expected_output(inp, objects)


def generate_input() -> np.ndarray:
    for _ in range(BUILD_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid lock-key ARC input")
