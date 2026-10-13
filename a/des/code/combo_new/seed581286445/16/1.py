from __future__ import annotations

import random

import numpy as np


BG = 0
SIDE_ORDER = ("top", "left", "bottom", "right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _full_ring(h: int, w: int, thickness: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    dist = np.minimum.reduce((rows, cols, h - 1 - rows, w - 1 - cols))
    return dist < thickness


def _side_band(h: int, w: int, thickness: int, side: str) -> np.ndarray:
    band = np.zeros((h, w), dtype=bool)
    if side == "top":
        band[:thickness, :] = True
    elif side == "bottom":
        band[h - thickness :, :] = True
    elif side == "left":
        band[:, :thickness] = True
    else:
        band[:, w - thickness :] = True
    return band


def _is_contiguous(indexes: np.ndarray) -> bool:
    if indexes.size == 0:
        return False
    return int(indexes[-1] - indexes[0] + 1) == int(indexes.size)


def _side_opening_ok(gap: np.ndarray, side: str) -> bool:
    h, w = gap.shape
    if side == "top":
        opening = np.flatnonzero(gap[0])
        return (
            opening.size >= 2
            and _is_contiguous(opening)
            and not gap[h - 1].any()
            and not gap[:, 0].any()
            and not gap[:, w - 1].any()
        )
    if side == "bottom":
        opening = np.flatnonzero(gap[h - 1])
        return (
            opening.size >= 2
            and _is_contiguous(opening)
            and not gap[0].any()
            and not gap[:, 0].any()
            and not gap[:, w - 1].any()
        )
    if side == "left":
        opening = np.flatnonzero(gap[:, 0])
        return (
            opening.size >= 2
            and _is_contiguous(opening)
            and not gap[:, w - 1].any()
            and not gap[0].any()
            and not gap[h - 1].any()
        )
    opening = np.flatnonzero(gap[:, w - 1])
    return (
        opening.size >= 2
        and _is_contiguous(opening)
        and not gap[:, 0].any()
        and not gap[0].any()
        and not gap[h - 1].any()
    )


def _matching_rotation(key_mask: np.ndarray, target_mask: np.ndarray) -> int | None:
    target = _crop_tight(target_mask)
    seen: set[bytes] = set()
    for k in range(4):
        rotated = _crop_tight(np.rot90(key_mask, k))
        sig = rotated.tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            return int(k)
    return None


def _infer_lock(lock_mask: np.ndarray, key_mask: np.ndarray) -> dict | None:
    h, w = lock_mask.shape
    max_thickness = min(h, w) // 2
    for thickness in range(1, max_thickness + 1):
        if h <= 2 * thickness or w <= 2 * thickness:
            continue
        ring = _full_ring(h, w, thickness)
        if np.any(lock_mask & ~ring):
            continue
        gap = ring & ~lock_mask
        if not gap.any():
            continue
        if len(_components(gap)) != 1:
            continue
        valid_sides = []
        for side in SIDE_ORDER:
            band = _side_band(h, w, thickness, side)
            if np.any(gap & ~band):
                continue
            if _side_opening_ok(gap, side):
                valid_sides.append(side)
        if len(valid_sides) != 1:
            continue
        if _matching_rotation(key_mask, gap) is None:
            continue
        return {
            "thickness": int(thickness),
            "side": valid_sides[0],
            "ring": ring,
            "gap": gap,
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored key and insert it into the matching border gap of its lock frame.

    Input structure:
    - Black is the background.
    - Each non-background color appears as one hollow rectangular border frame with a single side gap and one detached solid key sprite.
    - The gap is a connected missing region carved from exactly one side band of the frame.

    Rule:
    - For each color, pair the detached key with the frame of the same color.
    - Infer the frame's full rectangular ring, locate its unique side gap, and find the 0/90/180/270 rotation of the key whose shape matches that gap exactly.
    - Fill the gap with the matched key shape and erase the original scattered key, leaving only the completed locks.

    Key insight:
    - Color gives the pairing, but the missing side-band shape determines which rotation of the key completes the lock.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    for color in [int(x) for x in np.unique(g) if int(x) != bg]:
        comps = _components(g == color)
        parts = []
        for coords in comps:
            mask, bbox = _tight_mask_from_coords(coords)
            area = int(mask.sum())
            bbox_area = int(mask.shape[0] * mask.shape[1])
            parts.append({"coords": coords, "mask": mask, "bbox": bbox, "area": area, "bbox_area": bbox_area})

        best = None
        for i, lock_part in enumerate(parts):
            for j, key_part in enumerate(parts):
                if i == j:
                    continue
                inferred = _infer_lock(lock_part["mask"], key_part["mask"])
                if inferred is None:
                    continue
                score = (int(inferred["ring"].sum()), int(lock_part["bbox_area"]), -int(key_part["bbox_area"]))
                if best is None or score > best[0]:
                    best = (score, lock_part, inferred)

        if best is None:
            continue

        _, lock_part, inferred = best
        r0, c0, r1, c1 = lock_part["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[inferred["ring"]] = color

    return out


def _top_gap_shape(depth: int, length: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((depth, length), dtype=bool)
    left = 0
    right = length - 1
    mask[0, :] = True
    changed = False
    for row in range(1, depth):
        span = right - left
        if span > 0:
            shrink_left = rng.randint(0, min(1, span))
            shrink_right = rng.randint(0, min(1, span - shrink_left))
            left += shrink_left
            right -= shrink_right
            changed = changed or shrink_left > 0 or shrink_right > 0
        mask[row, left : right + 1] = True
    if not changed and depth >= 2 and length >= 3:
        if rng.random() < 0.5:
            mask[-1, 0] = False
        else:
            mask[-1, -1] = False
    return mask


def _oriented_gap(shape_top: np.ndarray, side: str) -> np.ndarray:
    turns = {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]
    return np.rot90(shape_top, turns)


def _build_pair(color: int, rng: random.Random) -> dict | None:
    for _ in range(80):
        thickness = rng.choice((2, 3))
        outer_h = rng.randint(2 * thickness + 3, 13)
        outer_w = rng.randint(2 * thickness + 3, 13)
        side = rng.choice(SIDE_ORDER)
        side_extent = outer_w if side in ("top", "bottom") else outer_h
        max_gap_len = min(6, side_extent - 2)
        if max_gap_len < 2:
            continue
        gap_len = rng.randint(2, max_gap_len)
        gap_top = _top_gap_shape(thickness, gap_len, rng)
        gap = _oriented_gap(gap_top, side)

        ring = _full_ring(outer_h, outer_w, thickness)
        lock = ring.copy()
        if side == "top":
            start = rng.randint(1, outer_w - gap_len - 1)
            window = lock[:thickness, start : start + gap_len]
            window[gap_top] = False
        elif side == "bottom":
            start = rng.randint(1, outer_w - gap_len - 1)
            window = lock[outer_h - thickness :, start : start + gap_len]
            window[gap] = False
        elif side == "left":
            start = rng.randint(1, outer_h - gap_len - 1)
            window = lock[start : start + gap_len, :thickness]
            window[gap] = False
        else:
            start = rng.randint(1, outer_h - gap_len - 1)
            window = lock[start : start + gap_len, outer_w - thickness :]
            window[gap] = False

        if not _components(lock):
            continue
        if len(_components(lock)) != 1:
            continue
        key = np.rot90(gap, rng.randrange(4))
        if _infer_lock(lock, key) is None:
            continue
        return {"color": color, "lock": lock, "ring": ring, "key": key}
    return None


def _find_slot(blocked: np.ndarray, shape: tuple[int, int], rng: random.Random, margin: int = 1) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = shape
    if ph + 2 * margin > h or pw + 2 * margin > w:
        return None
    for _ in range(300):
        r0 = rng.randint(margin, h - ph - margin)
        c0 = rng.randint(margin, w - pw - margin)
        rr0 = r0 - margin
        rr1 = r0 + ph + margin
        cc0 = c0 - margin
        cc1 = c0 + pw + margin
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        blocked[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)
    return None


def _paint(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _generate_candidate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    num_pairs = rng.randint(1, 3)
    colors = rng.sample(list(range(1, 10)), num_pairs)

    pairs = []
    for color in colors:
        pair = _build_pair(color, rng)
        if pair is None:
            return None
        pairs.append(pair)

    total_bbox_area = sum(
        int(pair["lock"].shape[0] * pair["lock"].shape[1] + pair["key"].shape[0] * pair["key"].shape[1])
        for pair in pairs
    )
    max_h = max(max(pair["lock"].shape[0], pair["key"].shape[0]) for pair in pairs)
    max_w = max(max(pair["lock"].shape[1], pair["key"].shape[1]) for pair in pairs)

    dims = None
    min_h = max(16, max_h + 4)
    min_w = max(16, max_w + 4)
    for _ in range(80):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        if h * w >= total_bbox_area * 3:
            dims = (h, w)
            break
    if dims is None:
        return None

    h, w = dims
    blocked = np.zeros((h, w), dtype=bool)
    items = []
    for pair in pairs:
        items.append({"kind": "lock", "color": pair["color"], "mask": pair["lock"], "full": pair["ring"]})
        items.append({"kind": "key", "color": pair["color"], "mask": pair["key"], "full": None})
    items.sort(key=lambda item: int(item["mask"].shape[0] * item["mask"].shape[1]), reverse=True)

    placed = []
    for item in items:
        pos = _find_slot(blocked, item["mask"].shape, rng)
        if pos is None:
            return None
        placed.append((item, pos))

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for item, (top, left) in placed:
        _paint(inp, item["mask"], int(item["color"]), top, left)
        if item["kind"] == "lock":
            _paint(expected, item["full"], int(item["color"]), top, left)

    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        built = _generate_candidate(rng)
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
