from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, RED, GREEN)

LOCK_H_RANGE = (10, 14)
LOCK_W_RANGE = (8, 12)
GRID_H_RANGE = (15, 23)
LEFT_MARGIN_RANGE = (1, 4)
RIGHT_MARGIN_RANGE = (1, 4)
KEY_GAP_RANGE = (3, 6)
SLOT_COUNT_CHOICES = (3, 4)
SLOT_HEIGHT_CHOICES = (2, 2, 3, 3, 4)
SLOT_WIDTH_CHOICES = (2, 2, 3)
MAX_GENERATION_ATTEMPTS = 400

_PROFILE_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _profile_mask(profile: tuple[int, ...]) -> np.ndarray:
    width = max(profile)
    mask = np.zeros((len(profile), width), dtype=bool)
    for r, depth in enumerate(profile):
        mask[r, width - depth : width] = True
    return mask


def _paint_profile(
    grid: np.ndarray, top: int, left: int, profile: tuple[int, ...], color: int
) -> None:
    width = max(profile)
    for dr, depth in enumerate(profile):
        grid[top + dr, left + width - depth : left + width] = color


def _extract_key(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int], np.ndarray]:
    comps = _components(grid == YELLOW)
    assert comps, "expected a yellow key"
    key = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(key)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in key:
        mask[r - r0, c - c0] = True
    return key, (r0, c0, r1, c1), mask


def _find_lock_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    lock_mask = (grid != BLACK) & (grid != YELLOW)
    comps = _components(lock_mask)
    assert comps, "expected a lock structure"
    lock = max(comps, key=len)
    return _bbox(lock)


def _extract_slots(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = bbox
    depths: list[int] = []
    for r in range(r0, r1 + 1):
        depth = 0
        while c1 - depth >= c0 and int(grid[r, c1 - depth]) == BLACK:
            depth += 1
        depths.append(depth)

    slots: list[dict[str, object]] = []
    idx = 0
    while idx < len(depths):
        if depths[idx] == 0:
            idx += 1
            continue
        start = idx
        profile: list[int] = []
        while idx < len(depths) and depths[idx] > 0:
            profile.append(int(depths[idx]))
            idx += 1
        profile_t = tuple(profile)
        slots.append(
            {
                "top": r0 + start,
                "bottom": r0 + idx - 1,
                "profile": profile_t,
                "mask": _profile_mask(profile_t),
                "width": max(profile_t),
            }
        )
    assert slots, "expected at least one slot"
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the yellow key to the unique right-edge slot in the grey lock, insert it,
    and extend the colored pixels on the activated slot rows leftward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    key_cells, _key_bbox, key_mask = _extract_key(g)
    lock_bbox = _find_lock_bbox(g)
    lr0, lc0, _lr1, lc1 = lock_bbox
    slots = _extract_slots(g, lock_bbox)

    matches = [
        slot
        for slot in slots
        if slot["mask"].shape == key_mask.shape and np.array_equal(slot["mask"], key_mask)
    ]
    assert len(matches) == 1, "expected a unique matching slot"
    slot = matches[0]

    out = g.copy()
    for r, c in key_cells:
        out[r, c] = BLACK

    slot_top = int(slot["top"])
    slot_mask = slot["mask"]
    slot_left = lc1 - int(slot["width"]) + 1
    for dr in range(slot_mask.shape[0]):
        for dc in range(slot_mask.shape[1]):
            if slot_mask[dr, dc]:
                out[slot_top + dr, slot_left + dc] = YELLOW

    for r in range(int(slot["top"]), int(slot["bottom"]) + 1):
        for c in range(lc0, lc1 + 1):
            value = int(g[r, c])
            if value not in (BLACK, GREY, YELLOW):
                out[r, lc0 : c + 1] = value

    return out


def _profile_pool(height: int, width: int) -> list[tuple[int, ...]]:
    key = (height, width)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]

    profiles: list[tuple[int, ...]] = []

    def rec(prefix: list[int]) -> None:
        if len(prefix) == height:
            if max(prefix) == width:
                profiles.append(tuple(prefix))
            return
        for depth in range(1, width + 1):
            prefix.append(depth)
            rec(prefix)
            prefix.pop()

    rec([])
    _PROFILE_CACHE[key] = profiles
    return profiles


def _sample_profiles(slot_count: int) -> tuple[list[tuple[int, ...]], int, int] | None:
    for _ in range(200):
        match_h = random.choice(SLOT_HEIGHT_CHOICES)
        match_w = random.choice(SLOT_WIDTH_CHOICES)
        match_pool = _profile_pool(match_h, match_w)
        if len(match_pool) < 2:
            continue

        match_profile, foil_profile = random.sample(match_pool, 2)
        profiles = [match_profile, foil_profile]
        used = {(match_h, match_w, match_profile), (match_h, match_w, foil_profile)}

        while len(profiles) < slot_count:
            h = random.choice(SLOT_HEIGHT_CHOICES)
            w = random.choice(SLOT_WIDTH_CHOICES)
            profile = random.choice(_profile_pool(h, w))
            sig = (h, w, profile)
            if sig in used:
                continue
            used.add(sig)
            profiles.append(profile)

        random.shuffle(profiles)
        match_idx = profiles.index(match_profile)
        min_lock_h = sum(len(profile) for profile in profiles) + slot_count + 1
        if min_lock_h > LOCK_H_RANGE[1]:
            continue
        lock_h = random.randint(max(LOCK_H_RANGE[0], min_lock_h), LOCK_H_RANGE[1])
        return profiles, match_idx, lock_h
    return None


def _slot_starts(profiles: list[tuple[int, ...]], lock_h: int) -> list[int]:
    gap_count = len(profiles) + 1
    gaps = [1] * gap_count
    extra = lock_h - (sum(len(profile) for profile in profiles) + gap_count)
    for _ in range(extra):
        gaps[random.randrange(gap_count)] += 1

    starts: list[int] = []
    row = gaps[0]
    for i, profile in enumerate(profiles):
        starts.append(row)
        row += len(profile) + gaps[i + 1]
    return starts


def _build_instance() -> np.ndarray | None:
    slot_count = random.choice(SLOT_COUNT_CHOICES)
    sampled = _sample_profiles(slot_count)
    if sampled is None:
        return None
    profiles, match_idx, lock_h = sampled
    slot_rows = _slot_starts(profiles, lock_h)

    max_depth = max(max(profile) for profile in profiles)
    min_lock_w = max(LOCK_W_RANGE[0], max_depth + 5)
    if min_lock_w > LOCK_W_RANGE[1]:
        return None
    lock_w = random.randint(min_lock_w, LOCK_W_RANGE[1])

    key_profile = profiles[match_idx]
    key_h = len(key_profile)
    key_w = max(key_profile)
    min_grid_h = max(GRID_H_RANGE[0], lock_h + 2, key_h + 2)
    if min_grid_h > GRID_H_RANGE[1]:
        return None
    grid_h = random.randint(min_grid_h, GRID_H_RANGE[1])

    lock_top = random.randint(1, grid_h - lock_h - 1)
    key_top = random.randint(1, grid_h - key_h - 1)
    lock_left = random.randint(*LEFT_MARGIN_RANGE)
    key_gap = random.randint(*KEY_GAP_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    grid_w = lock_left + lock_w + key_gap + key_w + right_margin
    key_left = lock_left + lock_w + key_gap

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[lock_top : lock_top + lock_h, lock_left : lock_left + lock_w] = GREY

    for profile, start in zip(profiles, slot_rows):
        slot_top = lock_top + start
        slot_left = lock_left + lock_w - max(profile)
        _paint_profile(grid, slot_top, slot_left, profile, BLACK)

    ray_col_min = lock_left + 2
    ray_col_max = lock_left + lock_w - max_depth - 2
    if ray_col_min > ray_col_max:
        return None

    markers: list[tuple[int, int, int]] = []
    for profile, start in zip(profiles, slot_rows):
        for dr in range(len(profile)):
            row = lock_top + start + dr
            col = random.randint(ray_col_min, ray_col_max)
            color = random.choice(MARKER_COLORS)
            markers.append((row, col, color))

    if len({color for _, _, color in markers}) == 1:
        idx = random.randrange(len(markers))
        row, col, color = markers[idx]
        alt = random.choice([value for value in MARKER_COLORS if value != color])
        markers[idx] = (row, col, alt)

    for row, col, color in markers:
        grid[row, col] = color

    _paint_profile(grid, key_top, key_left, key_profile, YELLOW)
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        key_cells, _key_bbox, key_mask = _extract_key(grid)
        slots = _extract_slots(grid, _find_lock_bbox(grid))
        matches = [
            slot
            for slot in slots
            if slot["mask"].shape == key_mask.shape and np.array_equal(slot["mask"], key_mask)
        ]
        if len(matches) != 1:
            return False
        out = main(grid)
    except Exception:
        return False

    slot = matches[0]
    key_area = len(key_cells)
    inserted = int(np.count_nonzero((out == YELLOW) & (grid != YELLOW)))
    removed = int(np.count_nonzero((grid == YELLOW) & (out == BLACK)))
    rays = int(np.count_nonzero((grid == GREY) & (out != GREY)))

    return (
        inserted == key_area
        and removed == key_area
        and rays >= 2 * len(slot["profile"])
        and not np.array_equal(out, grid)
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
