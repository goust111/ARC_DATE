from __future__ import annotations

import random
from typing import Optional

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120
PLACEMENT_GAP = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _crop_true(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _boolean_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return comps


def _lock_side(gap: tuple[int, int], h: int, w: int) -> str:
    r, c = gap
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def _classify_lock(comp: dict) -> Optional[dict]:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 5 or w < 5:
        return None

    border_gaps: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            on_border = r == 0 or r == h - 1 or c == 0 or c == w - 1
            if on_border and not mask[r, c]:
                border_gaps.append((r, c))
    if len(border_gaps) != 1:
        return None

    gap = border_gaps[0]
    if (gap[0] in (0, h - 1)) and (gap[1] in (0, w - 1)):
        return None

    cutout = ~mask
    cutout_comps = _boolean_components(cutout)
    if len(cutout_comps) != 1:
        return None

    cutout_pts = cutout_comps[0]
    border_touch = [pt for pt in cutout_pts if pt[0] in (0, h - 1) or pt[1] in (0, w - 1)]
    if border_touch != [gap]:
        return None
    if not any(0 < r < h - 1 and 0 < c < w - 1 for r, c in cutout_pts):
        return None

    cutout_crop, (off_r, off_c) = _crop_true(cutout)
    r0, c0, _r1, _c1 = comp["bbox"]
    return {
        "kind": "lock",
        "color": comp["color"],
        "bbox": comp["bbox"],
        "cutout_full": cutout,
        "cutout_crop": cutout_crop,
        "cutout_offset": (off_r, off_c),
        "gap_side": _lock_side(gap, h, w),
        "abs_origin": (r0, c0),
    }


def _classify_key(comp: dict) -> dict:
    return {
        "kind": "key",
        "color": comp["color"],
        "bbox": comp["bbox"],
        "mask": comp["mask"],
        "coords": comp["coords"],
    }


def _shape_match(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _candidate_key_ids(locks: list[dict], keys: list[dict]) -> list[list[int]]:
    out: list[list[int]] = []
    for lock in locks:
        want = lock["cutout_crop"]
        candidates: list[int] = []
        for kid, key in enumerate(keys):
            kmask = key["mask"]
            if any(_shape_match(np.rot90(kmask, rot), want) for rot in range(4)):
                candidates.append(kid)
        out.append(candidates)
    return out


def _solve_assignment(options: list[list[int]]) -> Optional[list[int]]:
    order = sorted(range(len(options)), key=lambda i: len(options[i]))
    used: set[int] = set()
    assign = [-1] * len(options)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        lid = order[pos]
        for kid in options[lid]:
            if kid in used:
                continue
            used.add(kid)
            assign[lid] = kid
            if backtrack(pos + 1):
                return True
            used.remove(kid)
            assign[lid] = -1
        return False

    if backtrack(0):
        return assign
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify lock rectangles with an internal cavity plus a border gap, match each
    to a rotatable key shape, remove the key, and fill the lock's missing region
    with the key color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    comps = _monochrome_components(g)
    locks: list[dict] = []
    keys: list[dict] = []
    for comp in comps:
        lock = _classify_lock(comp)
        if lock is not None:
            locks.append(lock)
        else:
            keys.append(_classify_key(comp))

    if not locks or not keys:
        return out

    options = _candidate_key_ids(locks, keys)
    if any(not candidates for candidates in options):
        raise AssertionError("every lock must have at least one matching key")
    assignment = _solve_assignment(options)
    if assignment is None:
        raise AssertionError("lock-key matching is ambiguous or impossible")

    for key in keys:
        coords = key["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for lock_id, key_id in enumerate(assignment):
        lock = locks[lock_id]
        key = keys[key_id]
        lr0, lc0 = lock["abs_origin"]
        fill = lock["cutout_full"]
        rows, cols = np.where(fill)
        out[lr0 + rows, lc0 + cols] = int(key["color"])

    return out


def _mask_has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bg[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and bg[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                return True
    return False


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (rng.randrange(h), rng.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        present = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if mask[nr, nc] or (nr, nc) in present:
                    continue
                present.add((nr, nc))
                frontier.append((nr, nc))
        pick_r, pick_c = rng.choice(frontier)
        mask[pick_r, pick_c] = True
    cropped, _ = _crop_true(mask)
    return cropped


def _sample_body_mask(area: int, rng: random.Random) -> np.ndarray:
    candidates: list[tuple[int, int]] = []
    for h in range(2, 6):
        for w in range(2, 6):
            if area > h * w:
                continue
            candidates.append((h, w))

    for _ in range(120):
        h, w = rng.choice(candidates)
        body = _grow_connected_mask(h, w, area, rng)
        bh, bw = body.shape
        if bh < 2 or bw < 2:
            continue
        if _mask_has_hole(body):
            continue
        if int(body.sum()) == bh * bw and rng.random() < 0.8:
            continue
        return body

    # Guaranteed fallback.
    h = max(2, min(4, area // 2 + 1))
    w = max(2, min(4, area))
    return _grow_connected_mask(h, w, area, rng)


def _attachment_point(body: np.ndarray, side: str) -> tuple[int, int]:
    pts = np.argwhere(body)
    if side == "left":
        edge = pts[:, 1].min()
        cand = pts[pts[:, 1] == edge]
        idx = np.argmin(np.abs(cand[:, 0] - (body.shape[0] - 1) / 2))
        return int(cand[idx, 0]), int(cand[idx, 1])
    if side == "right":
        edge = pts[:, 1].max()
        cand = pts[pts[:, 1] == edge]
        idx = np.argmin(np.abs(cand[:, 0] - (body.shape[0] - 1) / 2))
        return int(cand[idx, 0]), int(cand[idx, 1])
    if side == "top":
        edge = pts[:, 0].min()
        cand = pts[pts[:, 0] == edge]
        idx = np.argmin(np.abs(cand[:, 1] - (body.shape[1] - 1) / 2))
        return int(cand[idx, 0]), int(cand[idx, 1])
    edge = pts[:, 0].max()
    cand = pts[pts[:, 0] == edge]
    idx = np.argmin(np.abs(cand[:, 1] - (body.shape[1] - 1) / 2))
    return int(cand[idx, 0]), int(cand[idx, 1])


def _make_lock_and_key(
    body: np.ndarray,
    lock_color: int,
    key_color: int,
    side: str,
    handle_len: int,
    rng: random.Random,
) -> dict:
    bh, bw = body.shape

    if side in ("left", "right"):
        top = rng.randint(1, 3)
        bottom = max(1, top + rng.choice([-1, 0, 1]))
        opposite = rng.randint(1, 3)
        left = handle_len if side == "left" else opposite
        right = handle_len if side == "right" else opposite
    else:
        left = rng.randint(1, 3)
        right = max(1, left + rng.choice([-1, 0, 1]))
        opposite = rng.randint(1, 3)
        top = handle_len if side == "top" else opposite
        bottom = handle_len if side == "bottom" else opposite

    outer_h = top + bh + bottom
    outer_w = left + bw + right
    if outer_h < 5:
        bottom += 5 - outer_h
        outer_h = 5
    if outer_w < 5:
        right += 5 - outer_w
        outer_w = 5

    cutout = np.zeros((outer_h, outer_w), dtype=bool)
    body_r0 = top
    body_c0 = left
    cutout[body_r0 : body_r0 + bh, body_c0 : body_c0 + bw] = body

    ar, ac = _attachment_point(body, side)
    abs_r = body_r0 + ar
    abs_c = body_c0 + ac
    if side == "left":
        cutout[abs_r, :abs_c] = True
    elif side == "right":
        cutout[abs_r, abs_c + 1 :] = True
    elif side == "top":
        cutout[:abs_r, abs_c] = True
    else:
        cutout[abs_r + 1 :, abs_c] = True

    lock_patch = np.full((outer_h, outer_w), lock_color, dtype=int)
    lock_patch[cutout] = BG

    key_mask, _ = _crop_true(cutout)
    input_rot = rng.randrange(4)
    key_mask_in = np.rot90(key_mask, input_rot)
    key_patch = np.full(key_mask_in.shape, BG, dtype=int)
    key_patch[key_mask_in] = key_color

    return {
        "lock_patch": lock_patch,
        "lock_cutout": cutout,
        "key_patch": key_patch,
        "key_mask": key_mask_in,
        "key_color": key_color,
        "lock_color": lock_color,
        "body_area": int(body.sum()),
    }


def _rect_conflicts(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + PLACEMENT_GAP < pr0 or pr1 + PLACEMENT_GAP < r0 or c1 + PLACEMENT_GAP < pc0 or pc1 + PLACEMENT_GAP < c0):
            return True
    return False


def _place_items(items: list[dict], h: int, w: int, rng: random.Random) -> Optional[list[tuple[int, int]]]:
    order = list(range(len(items)))
    rng.shuffle(order)
    order.sort(key=lambda i: items[i]["patch"].shape[0] * items[i]["patch"].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    pos = [(-1, -1)] * len(items)
    for idx in order:
        patch = items[idx]["patch"]
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _rect_conflicts(rect, rects):
                continue
            rects.append(rect)
            pos[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return pos


def _build_expected(grid: np.ndarray, pairs: list[dict], item_positions: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for pair_id, pair in enumerate(pairs):
        key_r, key_c = item_positions[2 * pair_id + 1]
        key_mask = pair["key_mask"]
        kr, kc = np.where(key_mask)
        out[key_r + kr, key_c + kc] = BG

    for pair_id, pair in enumerate(pairs):
        lock_r, lock_c = item_positions[2 * pair_id]
        cutout = pair["lock_cutout"]
        rr, cc = np.where(cutout)
        out[lock_r + rr, lock_c + cc] = int(pair["key_color"])
    return out


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    num_pairs = rng.randint(1, 3)
    body_areas = rng.sample(list(range(4, 11)), num_pairs)
    colors = rng.sample(list(range(1, 10)), 2 * num_pairs)

    pairs: list[dict] = []
    for i in range(num_pairs):
        body = _sample_body_mask(body_areas[i], rng)
        pair = _make_lock_and_key(
            body=body,
            lock_color=colors[2 * i],
            key_color=colors[2 * i + 1],
            side=rng.choice(SIDES),
            handle_len=rng.choice([1, 2]),
            rng=rng,
        )
        pairs.append(pair)

    items: list[dict] = []
    for pair in pairs:
        items.append({"patch": pair["lock_patch"], "kind": "lock"})
        items.append({"patch": pair["key_patch"], "kind": "key"})

    max_h = max(item["patch"].shape[0] for item in items)
    max_w = max(item["patch"].shape[1] for item in items)
    min_h = max(max_h + 2, 11 + 3 * num_pairs)
    min_w = max(max_w + 2, 11 + 3 * num_pairs)
    if min_h > 30 or min_w > 30:
        return None

    positions = None
    h = w = 0
    for _ in range(60):
        h = rng.randint(min_h, min(30, min_h + 8))
        w = rng.randint(min_w, min(30, min_w + 8))
        positions = _place_items(items, h, w, rng)
        if positions is not None:
            break
    if positions is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for item, (r0, c0) in zip(items, positions):
        patch = item["patch"]
        ph, pw = patch.shape
        nz = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][nz] = patch[nz]

    expected = _build_expected(grid, pairs, positions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC sample")
