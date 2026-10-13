from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
SIDES = ("left", "right", "top", "bottom")
NON_BG_COLORS = tuple(range(1, 10))
CANVAS_SIZE = 30
PAIR_COUNT_CHOICES = (2, 3)
INNER_SIZE_RANGE = (3, 6)
PLACE_TRIES = 200
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    by_color: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            by_color[color].append(coords)
    return by_color


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def _analyze_lock(grid: np.ndarray, coords: list[tuple[int, int]], bg: int):
    bbox, mask = _coords_to_mask(coords)
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    if h < 4 or w < 4:
        return None

    rr, cc = np.nonzero(mask)
    on_boundary = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
    if not np.all(on_boundary):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    side_vectors = {
        "top": mask[0, 1:-1],
        "bottom": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
        "right": mask[1:-1, -1],
    }
    gap_sides = [side for side, vec in side_vectors.items() if not np.all(vec)]
    if len(gap_sides) != 1:
        return None
    gap_side = gap_sides[0]
    for side, vec in side_vectors.items():
        if side != gap_side and not np.all(vec):
            return None

    profile = tuple(bool(x) for x in (~side_vectors[gap_side]).tolist())
    if not any(profile) or all(profile):
        return None

    patch = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    if patch.size == 0 or not np.any(patch != bg):
        return None

    return {
        "bbox": bbox,
        "coords": coords,
        "side": gap_side,
        "profile": profile,
        "patch": patch,
    }


def _analyze_key(coords: list[tuple[int, int]]):
    bbox, mask = _coords_to_mask(coords)
    h, w = mask.shape

    if w == 2 and h >= 3:
        left_full = bool(np.all(mask[:, 0]))
        right_full = bool(np.all(mask[:, 1]))
        if left_full == right_full:
            return None
        if left_full:
            profile = tuple(bool(x) for x in mask[:, 1].tolist())
            side = "right"
        else:
            profile = tuple(bool(x) for x in mask[:, 0].tolist())
            side = "left"
        if not any(profile) or all(profile):
            return None
        return {"bbox": bbox, "coords": coords, "side": side, "profile": profile}

    if h == 2 and w >= 3:
        top_full = bool(np.all(mask[0, :]))
        bottom_full = bool(np.all(mask[1, :]))
        if top_full == bottom_full:
            return None
        if top_full:
            profile = tuple(bool(x) for x in mask[1, :].tolist())
            side = "bottom"
        else:
            profile = tuple(bool(x) for x in mask[0, :].tolist())
            side = "top"
        if not any(profile) or all(profile):
            return None
        return {"bbox": bbox, "coords": coords, "side": side, "profile": profile}

    return None


def _find_pairs(grid: np.ndarray, bg: int):
    pairs = []
    for color, comps in _components_by_color(grid, bg).items():
        locks = []
        keys = []
        for comp in comps:
            lock = _analyze_lock(grid, comp, bg)
            if lock is not None:
                locks.append(lock)
                continue
            key = _analyze_key(comp)
            if key is not None:
                keys.append(key)
        if len(locks) == 1 and len(keys) == 1:
            pairs.append((int(color), locks[0], keys[0]))
    return pairs


def _release_position(lock) -> tuple[int, int]:
    r0, c0, r1, c1 = lock["bbox"]
    patch = lock["patch"]
    inner_h, inner_w = patch.shape
    side = lock["side"]
    if side == "left":
        return r0 + 1, c0 - inner_w
    if side == "right":
        return r0 + 1, c1 + 1
    if side == "top":
        return r0 - inner_h, c0 + 1
    return r1 + 1, c0 + 1


def _apply_unlock(out: np.ndarray, lock, key, bg: int) -> None:
    h, w = out.shape
    dst_r, dst_c = _release_position(lock)
    patch = lock["patch"]
    inner_h, inner_w = patch.shape
    if not (0 <= dst_r and 0 <= dst_c and dst_r + inner_h <= h and dst_c + inner_w <= w):
        return

    r0, c0, r1, c1 = lock["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = bg
    for r, c in key["coords"]:
        out[r, c] = bg

    target = out[dst_r : dst_r + inner_h, dst_c : dst_c + inner_w]
    mask = patch != bg
    target[mask] = patch[mask]


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    for _color, lock, key in _find_pairs(g, bg):
        if lock["side"] != key["side"]:
            continue
        if lock["profile"] != key["profile"]:
            continue
        _apply_unlock(out, lock, key, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock each colored frame with the key whose tooth profile exactly matches the frame's notched side.

    Input structure:
    - Black background with several colored lock-and-key groups.
    - Each lock is a single-color rectangular border enclosing a multicolor interior pattern.
    - Exactly one side of each lock border has one or more missing runs that form a tooth-slot profile.
    - Each key is a small same-color 2-cell-thick bar whose protrusions on one side define a tooth profile.

    Rule:
    - Group same-color monochrome components and identify the frame-like component as the lock and the thin bar as the key.
    - Read the lock profile from the missing cells on its unique notched side, and read the key profile from the protruding cells on its toothed side.
    - If the two profiles match exactly, remove the key, clear the lock interior, and copy the enclosed pattern rigidly to the outside of the lock in the notch direction so it sits flush against the border.
    - If the profile does not match, leave both the lock and key unchanged.

    Key insight:
    - Color only proposes a candidate pair; the actual action depends on exact geometric equality between the lock's gap pattern and the key's tooth pattern.
    """
    return _transform(np.asarray(grid, dtype=int))


def _bbox_clear(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = occ.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_bbox(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = occ.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _random_profile(span: int, rng: random.Random) -> tuple[bool, ...]:
    for _ in range(100):
        profile = [False] * span
        seg_count = rng.randint(1, min(3, span))
        for _ in range(seg_count):
            seg_len = rng.randint(1, max(1, span - 1))
            start = rng.randint(0, span - seg_len)
            for i in range(start, start + seg_len):
                profile[i] = True
        if any(profile) and not all(profile):
            return tuple(profile)
    raise RuntimeError("failed to build gap profile")


def _different_profile(span: int, original: tuple[bool, ...], rng: random.Random) -> tuple[bool, ...]:
    for _ in range(100):
        candidate = _random_profile(span, rng)
        if candidate != original:
            return candidate
    raise RuntimeError("failed to build mismatching profile")


def _make_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    color_count = rng.randint(2, min(4, len(palette)))
    colors = rng.sample(palette, color_count)
    style = rng.choice(["checker", "stripes_h", "stripes_v", "quadrants", "diag"])
    patch = np.zeros((h, w), dtype=int)

    if style == "checker":
        for r in range(h):
            for c in range(w):
                patch[r, c] = colors[(r + c) % color_count]
    elif style == "stripes_h":
        for r in range(h):
            patch[r, :] = colors[r % color_count]
    elif style == "stripes_v":
        for c in range(w):
            patch[:, c] = colors[c % color_count]
    elif style == "quadrants":
        for r in range(h):
            for c in range(w):
                idx = (r >= h // 2) * 2 + (c >= w // 2)
                patch[r, c] = colors[idx % color_count]
    else:
        for r in range(h):
            for c in range(w):
                patch[r, c] = colors[(2 * r + c) % color_count]

    if len(set(int(x) for x in patch.ravel())) < 2:
        patch[0, 0] = colors[0]
        patch[-1, -1] = colors[-1]
    return patch


def _zone_geometry(side: str, inner_h: int, inner_w: int):
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if side == "left":
        zone_h, zone_w = outer_h, outer_w + inner_w
        lock_off = (0, inner_w)
    elif side == "right":
        zone_h, zone_w = outer_h, outer_w + inner_w
        lock_off = (0, 0)
    elif side == "top":
        zone_h, zone_w = outer_h + inner_h, outer_w
        lock_off = (inner_h, 0)
    else:
        zone_h, zone_w = outer_h + inner_h, outer_w
        lock_off = (0, 0)
    return zone_h, zone_w, lock_off


def _draw_lock(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    color: int,
    side: str,
    profile: tuple[bool, ...],
    patch: np.ndarray,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top, left : left + outer_w] = color
    grid[top + outer_h - 1, left : left + outer_w] = color
    grid[top : top + outer_h, left] = color
    grid[top : top + outer_h, left + outer_w - 1] = color

    if side == "left":
        for i, is_gap in enumerate(profile):
            if is_gap:
                grid[top + 1 + i, left] = BG
    elif side == "right":
        for i, is_gap in enumerate(profile):
            if is_gap:
                grid[top + 1 + i, left + outer_w - 1] = BG
    elif side == "top":
        for i, is_gap in enumerate(profile):
            if is_gap:
                grid[top, left + 1 + i] = BG
    else:
        for i, is_gap in enumerate(profile):
            if is_gap:
                grid[top + outer_h - 1, left + 1 + i] = BG

    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = patch


def _draw_key(
    grid: np.ndarray,
    top: int,
    left: int,
    color: int,
    side: str,
    profile: tuple[bool, ...],
) -> None:
    span = len(profile)
    if side == "left":
        grid[top : top + span, left + 1] = color
        for i, bit in enumerate(profile):
            if bit:
                grid[top + i, left] = color
    elif side == "right":
        grid[top : top + span, left] = color
        for i, bit in enumerate(profile):
            if bit:
                grid[top + i, left + 1] = color
    elif side == "top":
        grid[top + 1, left : left + span] = color
        for i, bit in enumerate(profile):
            if bit:
                grid[top, left + i] = color
    else:
        grid[top, left : left + span] = color
        for i, bit in enumerate(profile):
            if bit:
                grid[top + 1, left + i] = color


def _place_lock_zone(
    occ: np.ndarray,
    side: str,
    inner_h: int,
    inner_w: int,
    rng: random.Random,
):
    zone_h, zone_w, (lock_dr, lock_dc) = _zone_geometry(side, inner_h, inner_w)
    for _ in range(PLACE_TRIES):
        top = rng.randint(1, CANVAS_SIZE - zone_h - 1)
        left = rng.randint(1, CANVAS_SIZE - zone_w - 1)
        zone_bbox = (top, left, top + zone_h - 1, left + zone_w - 1)
        if not _bbox_clear(occ, zone_bbox, pad=1):
            continue
        lock_top = top + lock_dr
        lock_left = left + lock_dc
        return zone_bbox, lock_top, lock_left
    return None


def _place_key_bbox(
    occ: np.ndarray,
    side: str,
    span: int,
    rng: random.Random,
):
    key_h, key_w = (span, 2) if side in ("left", "right") else (2, span)
    for _ in range(PLACE_TRIES):
        top = rng.randint(1, CANVAS_SIZE - key_h - 1)
        left = rng.randint(1, CANVAS_SIZE - key_w - 1)
        bbox = (top, left, top + key_h - 1, left + key_w - 1)
        if _bbox_clear(occ, bbox, pad=1):
            return bbox
    return None


def _crop_canvas(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> np.ndarray:
    r0, c0, r1, c1 = _union_bbox(boxes)
    top_margin = 1 + rng.randint(0, 1)
    bottom_margin = 1 + rng.randint(0, 1)
    left_margin = 1 + rng.randint(0, 1)
    right_margin = 1 + rng.randint(0, 1)
    rr0 = max(0, r0 - top_margin)
    cc0 = max(0, c0 - left_margin)
    rr1 = min(grid.shape[0] - 1, r1 + bottom_margin)
    cc1 = min(grid.shape[1] - 1, c1 + right_margin)
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    occ = np.zeros_like(grid, dtype=bool)
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    lock_colors = rng.sample(list(NON_BG_COLORS), pair_count)
    pattern_palette = [c for c in NON_BG_COLORS if c not in lock_colors]
    if len(pattern_palette) < 2:
        return None

    matched_count = rng.randint(1, pair_count - 1)
    matched_indices = set(rng.sample(range(pair_count), matched_count))

    boxes: list[tuple[int, int, int, int]] = []
    expected_matches = 0

    for idx, color in enumerate(lock_colors):
        side = rng.choice(SIDES)
        inner_h = rng.randint(*INNER_SIZE_RANGE)
        inner_w = rng.randint(*INNER_SIZE_RANGE)
        span = inner_h if side in ("left", "right") else inner_w

        profile = _random_profile(span, rng)
        key_profile = profile if idx in matched_indices else _different_profile(span, profile, rng)
        patch = _make_pattern(inner_h, inner_w, pattern_palette, rng)

        placed = _place_lock_zone(occ, side, inner_h, inner_w, rng)
        if placed is None:
            return None
        zone_bbox, lock_top, lock_left = placed
        _draw_lock(grid, lock_top, lock_left, inner_h, inner_w, color, side, profile, patch)
        _mark_bbox(occ, zone_bbox, pad=1)
        boxes.append(zone_bbox)

        key_bbox = _place_key_bbox(occ, side, span, rng)
        if key_bbox is None:
            return None
        key_top, key_left = key_bbox[0], key_bbox[1]
        _draw_key(grid, key_top, key_left, color, side, key_profile)
        _mark_bbox(occ, key_bbox, pad=1)
        boxes.append(key_bbox)

        if idx in matched_indices:
            expected_matches += 1

    cropped = _crop_canvas(grid, boxes, rng)
    pairs = _find_pairs(cropped, BG)
    if len(pairs) != pair_count:
        return None
    actual_matches = sum(
        1
        for _color, lock, key in pairs
        if lock["side"] == key["side"] and lock["profile"] == key["profile"]
    )
    if actual_matches != expected_matches:
        return None

    out = _transform(cropped)
    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key puzzle")
