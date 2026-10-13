from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120

PROFILE_LIBRARY = [
    (2, 3, 2),
    (2, 4, 2),
    (3, 2, 3),
    (2, 3, 2, 1),
    (1, 2, 3, 2),
    (2, 3, 3, 2),
    (2, 4, 3, 2),
    (2, 3, 4, 2),
    (1, 2, 3, 2, 1),
    (2, 2, 3, 2),
    (2, 3, 2, 2),
    (3, 3, 2, 1),
    (1, 3, 3, 2),
    (2, 4, 2, 1),
    (1, 2, 4, 2, 1),
]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in components_mask(grid == GREY):
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 7 or w < 7:
            continue
        if coords.shape[0] != 2 * (h + w) - 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_mask_from_bbox(sub.shape, (0, 0, h - 1, w - 1))
        if not np.all(sub[border] == GREY):
            continue
        if np.any(sub[1:-1, 1:-1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    assert 2 <= len(frames) <= 3, "expected 2-3 grey hollow frames"
    return frames


def _settle_mask(red_mask: np.ndarray) -> np.ndarray:
    h, w = red_mask.shape
    settled = np.zeros((h, w), dtype=bool)
    counts = red_mask.sum(axis=0)
    for c, count in enumerate(counts):
        n = int(count)
        if n:
            settled[h - n :, c] = True
    return settled


def _classify_mask(mask: np.ndarray) -> np.ndarray:
    assert mask.any(), "shape must be non-empty"
    comps = components_mask(mask)
    assert len(comps) == 1, "settled red pixels must form one 4-connected shape"

    out = np.zeros(mask.shape, dtype=int)
    shape = {(int(r), int(c)) for r, c in comps[0]}
    for r, c in shape:
        degree = sum((nr, nc) in shape for nr, nc in neighbors4(r, c))
        if degree == 4:
            out[r, c] = RED
        elif degree == 2:
            out[r, c] = BLUE
        else:
            out[r, c] = YELLOW
    return out


def _trimmed_profile(mask: np.ndarray) -> tuple[int, ...]:
    cols = tuple(int(v) for v in mask.sum(axis=0).tolist())
    nz = [i for i, v in enumerate(cols) if v > 0]
    assert nz, "expected non-empty profile"
    return cols[nz[0] : nz[-1] + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle red pixels inside each grey frame and recolor the emerged shape by local position.

    Input structure:
    - A black background contains 2-3 non-overlapping grey hollow rectangular frames.
    - Each frame interior has scattered red pixels and may also contain empty black cells.
    - The frame borders are one pixel thick and remain fixed while only interior red pixels participate.

    Rule:
    - For each frame independently, apply stable downward gravity to the red pixels inside the frame: each column keeps its red count and packs against the inner bottom edge.
    - Treat the settled red pixels in that frame as one 4-connected shape and classify each settled pixel by local 4-neighborhood inside that shape.
    - Boundary pixels with exactly 2 same-shape 4-neighbors become blue, other boundary pixels become yellow, interior pixels with 4 same-shape neighbors stay red; frame and background stay unchanged.

    Key insight:
    - The colored object to classify does not exist in the input as a ready-made shape; it first emerges from column-wise gravity settling inside each frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        red_mask = inner == RED
        assert red_mask.any(), "each frame must contain red pixels"

        settled = _settle_mask(red_mask)
        recolored = _classify_mask(settled)

        view = out[r0 + 1 : r1, c0 + 1 : c1]
        view[view == RED] = BLACK
        view[recolored > 0] = recolored[recolored > 0]

    return out


def _has_interior(mask: np.ndarray) -> bool:
    classified = _classify_mask(mask)
    return bool(np.any(classified == RED))


def _valid_profile(profile: tuple[int, ...], inner_h: int) -> bool:
    if len(profile) < 3 or max(profile) > inner_h:
        return False
    if min(profile) <= 0:
        return False
    if len(set(profile)) == 1:
        return False
    total = sum(profile)
    if not (5 <= total <= 12):
        return False
    if max(profile) < 2:
        return False
    mask = np.zeros((inner_h, len(profile)), dtype=bool)
    for c, n in enumerate(profile):
        mask[inner_h - n :, c] = True
    if len(components_mask(mask)) != 1:
        return False
    if mask.sum(axis=0).astype(int).ptp() == 0:
        return False
    return _has_interior(mask)


def _choose_profile(inner_h: int, inner_w: int, used: set[tuple[int, ...]]) -> tuple[int, ...] | None:
    candidates = []
    for base in PROFILE_LIBRARY:
        if len(base) <= inner_w - 1 and _valid_profile(base, inner_h):
            candidates.append(tuple(base))
    random.shuffle(candidates)

    for profile in candidates:
        if profile not in used:
            return profile
    return candidates[0] if candidates else None


def _profile_to_counts(profile: tuple[int, ...], inner_w: int) -> tuple[int, ...]:
    left_margin = random.randint(0, inner_w - len(profile))
    counts = [0] * inner_w
    for i, value in enumerate(profile):
        counts[left_margin + i] = int(value)
    return tuple(counts)


def _scatter_column(inner_h: int, count: int) -> list[int]:
    settled_rows = list(range(inner_h - count, inner_h))
    for _ in range(20):
        rows = sorted(random.sample(range(inner_h), count))
        if rows != settled_rows:
            return rows
    moved = list(settled_rows)
    moved[0] = max(0, settled_rows[0] - 1)
    moved = sorted(set(moved))
    while len(moved) < count:
        candidate = random.randrange(inner_h)
        if candidate not in moved:
            moved.append(candidate)
    return sorted(moved)


def _scatter_from_counts(inner_h: int, counts: tuple[int, ...]) -> np.ndarray:
    inner = np.zeros((inner_h, len(counts)), dtype=int)
    moved_any = False
    for c, count in enumerate(counts):
        if count == 0:
            continue
        rows = _scatter_column(inner_h, int(count))
        settled_rows = list(range(inner_h - int(count), inner_h))
        moved_any |= rows != settled_rows
        inner[rows, c] = RED
    assert moved_any, "at least one pixel must move under gravity"
    return inner


def _place_frames(
    canvas: np.ndarray, frame_specs: list[tuple[int, int, int, tuple[int, ...]]]
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, tuple[int, ...]]]] | None:
    h, w = canvas.shape
    placed: list[tuple[tuple[int, int, int, int], tuple[int, int, tuple[int, ...]]]] = []
    occupied = np.zeros((h, w), dtype=bool)
    for spec in sorted(frame_specs, key=lambda item: (item[0] * item[1], item[0], item[1]), reverse=True):
        fh, fw, inner_h, profile = spec
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed.append(((r0, c0, r1, c1), (inner_h, fw - 2, profile)))
            success = True
            break
        if not success:
            return None
    placed.sort(key=lambda item: item[0])
    return placed


def _render_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _frame_semantics_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        frames = _find_frames(inp)
    except Exception:
        return False

    if not np.array_equal(out, main(inp)):
        return False

    moved_frames = 0
    trimmed_profiles: list[tuple[int, ...]] = []
    for r0, c0, r1, c1 in frames:
        inner_in = inp[r0 + 1 : r1, c0 + 1 : c1]
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        red_mask = inner_in == RED
        if not (5 <= int(red_mask.sum()) <= 12):
            return False

        settled = _settle_mask(red_mask)
        profile = _trimmed_profile(settled)
        trimmed_profiles.append(profile)
        if len(profile) < 3 or max(profile) < 2:
            return False
        if not _has_interior(settled):
            return False
        if np.array_equal(settled.astype(int) * RED, inner_in):
            return False
        moved_frames += 1

        classified = _classify_mask(settled)
        if not (np.any(classified == BLUE) and np.any(classified == YELLOW) and np.any(classified == RED)):
            return False
        expected_inner = np.where(classified > 0, classified, BLACK)
        if not np.array_equal(inner_out, expected_inner):
            return False

    if moved_frames != len(frames):
        return False
    if len(set(trimmed_profiles)) != len(trimmed_profiles):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    frame_count = random.choice([2, 3])
    min_side = 18 if frame_count == 2 else 21
    max_side = 26 if frame_count == 2 else 29
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    grid = np.zeros((h, w), dtype=int)

    frame_specs: list[tuple[int, int, int, tuple[int, ...]]] = []
    used_profiles: set[tuple[int, ...]] = set()
    for _ in range(frame_count):
        inner_h = random.randint(5, 8)
        inner_w = random.randint(5, 8)
        profile = _choose_profile(inner_h, inner_w, used_profiles)
        if profile is None:
            return None
        used_profiles.add(profile)
        frame_specs.append((inner_h + 2, inner_w + 2, inner_h, profile))

    placed = _place_frames(grid, frame_specs)
    if placed is None:
        return None

    for bbox, (inner_h, inner_w, profile) in placed:
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 - 1, c1 - c0 - 1) == (inner_h, inner_w)
        _render_frame(grid, bbox)
        counts = _profile_to_counts(profile, inner_w)
        inner = _scatter_from_counts(inner_h, counts)
        grid[r0 + 1 : r1, c0 + 1 : c1] = inner

    out = main(grid)
    if not _frame_semantics_ok(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-settling frame puzzle")
