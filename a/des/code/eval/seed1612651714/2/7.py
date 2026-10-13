from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
KEY = 1
LOCK = 2
FIT = 3
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PAIR_COUNT_OPTIONS = (2, 3, 4)
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
GENERATE_RETRIES = 300
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 400
MARGIN = 1
PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        raise AssertionError("mask must be non-empty")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _coords_to_cropped_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k).copy()
        sig = (tuple(rot.shape), rot.tobytes())
        if sig not in seen:
            seen.add(sig)
            variants.append(rot)
    return variants


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(rot.shape), rot.tobytes()) for rot in _rotations(mask))


def _opening_sides(missing: np.ndarray) -> list[str]:
    sides: list[str] = []
    if missing[0, :].any():
        sides.append("up")
    if missing[-1, :].any():
        sides.append("down")
    if missing[:, 0].any():
        sides.append("left")
    if missing[:, -1].any():
        sides.append("right")
    return sides


def _parse_key(coords: list[tuple[int, int]]) -> dict:
    mask, (r0, c0) = _coords_to_cropped_mask(coords)
    return {
        "coords": coords,
        "mask": mask,
        "bbox": (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1),
    }


def _parse_lock(coords: list[tuple[int, int]]) -> dict:
    lock_mask, (r0, c0) = _coords_to_cropped_mask(coords)
    missing = ~lock_mask
    assert missing.any(), "each lock must have a missing notch inside its bounding box"
    missing_comps = _components(missing)
    assert len(missing_comps) == 1, "lock bbox should contain a single connected notch"
    sides = _opening_sides(missing)
    assert len(sides) == 1, "notch must open to exactly one side of the lock"

    nr0, nc0, nr1, nc1 = _bbox(missing_comps[0])
    notch = np.zeros((nr1 - nr0 + 1, nc1 - nc0 + 1), dtype=bool)
    for r, c in missing_comps[0]:
        notch[r - nr0, c - nc0] = True

    return {
        "coords": coords,
        "coords_set": set(coords),
        "mask": lock_mask,
        "bbox": (r0, c0, r0 + lock_mask.shape[0] - 1, c0 + lock_mask.shape[1] - 1),
        "opening": sides[0],
        "notch_mask": notch,
        "notch_origin": (r0 + nr0, c0 + nc0),
    }


def _match_assignment(keys: list[dict], locks: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    candidates: dict[int, list[tuple[int, np.ndarray]]] = {}
    for li, lock in enumerate(locks):
        cand: list[tuple[int, np.ndarray]] = []
        for ki, key in enumerate(keys):
            for rot in _rotations(key["mask"]):
                if rot.shape == lock["notch_mask"].shape and np.array_equal(rot, lock["notch_mask"]):
                    cand.append((ki, rot))
                    break
        assert cand, "every lock notch must match at least one key up to rotation"
        candidates[li] = cand

    order = sorted(range(len(locks)), key=lambda li: (len(candidates[li]), -int(locks[li]["notch_mask"].sum())))

    def dfs(pos: int, used: set[int]) -> dict[int, tuple[int, np.ndarray]] | None:
        if pos == len(order):
            return {}
        li = order[pos]
        for ki, rot in candidates[li]:
            if ki in used:
                continue
            used.add(ki)
            rest = dfs(pos + 1, used)
            if rest is not None:
                rest[li] = (ki, rot)
                return rest
            used.remove(ki)
        return None

    result = dfs(0, set())
    assert result is not None, "keys and locks must admit a one-to-one shape matching"
    return result


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    key_comps = _components(g == KEY)
    lock_comps = _components(g == LOCK)
    assert key_comps and lock_comps, "grid must contain keys and locks"
    assert len(key_comps) == len(lock_comps), "keys and locks should pair one-to-one"

    keys = [_parse_key(comp) for comp in key_comps]
    locks = [_parse_lock(comp) for comp in lock_comps]
    assignment = _match_assignment(keys, locks)

    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BG

    for li, lock in enumerate(locks):
        _ki, rot = assignment[li]
        top, left = lock["notch_origin"]
        for rr, cc in np.argwhere(rot):
            r = int(top + rr)
            c = int(left + cc)
            touching = any((r + dr, c + dc) in lock["coords_set"] for dr, dc in DIRS4)
            out[r, c] = FIT if touching else KEY

    return out


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w))
        cells = {(random.randrange(h), random.randrange(w))}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if len(_rotations(mask)) < 2:
            continue
        return mask

    raise RuntimeError("failed to sample a connected key shape")


def _choose_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen = set()
    for _ in range(SHAPE_ATTEMPTS * 4):
        mask = _random_shape()
        sig = _canonical_shape(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    return None


def _build_lock(shape: np.ndarray, opening: str) -> np.ndarray | None:
    sh, sw = shape.shape
    if opening in {"up", "down"}:
        rect_h = sh + random.randint(2, 4)
        rect_w = sw + random.randint(2, 5)
        rect_w = max(rect_w, sw + 2)
        offset = random.randint(1, rect_w - sw - 1)
        notch_top = 0 if opening == "up" else rect_h - sh
        notch_left = offset
    else:
        rect_h = sh + random.randint(2, 5)
        rect_w = sw + random.randint(2, 4)
        rect_h = max(rect_h, sh + 2)
        offset = random.randint(1, rect_h - sh - 1)
        notch_top = offset
        notch_left = 0 if opening == "left" else rect_w - sw

    lock = np.ones((rect_h, rect_w), dtype=bool)
    sub = lock[notch_top : notch_top + sh, notch_left : notch_left + sw].copy()
    sub[shape] = False
    lock[notch_top : notch_top + sh, notch_left : notch_left + sw] = sub

    if len(_components(lock)) != 1:
        return None
    missing = ~lock
    if _opening_sides(missing) != [opening]:
        return None
    return lock


def _place_bbox(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    if H - h - MARGIN < MARGIN or W - w - MARGIN < MARGIN:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(MARGIN, H - h - MARGIN)
        left = random.randint(MARGIN, W - w - MARGIN)
        r0 = max(0, top - PADDING)
        r1 = min(H, top + h + PADDING)
        c0 = max(0, left - PADDING)
        c1 = min(W, left + w + PADDING)
        if occupied[r0:r1, c0:c1].any():
            continue
        occupied[r0:r1, c0:c1] = True
        return top, left
    return None


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = color


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    shapes = _choose_shapes(pair_count)
    if shapes is None:
        return None

    directions = ["up", "down", "left", "right"]
    random.shuffle(directions)
    openings = directions[:pair_count]

    rotations = [0, 1, 2, 3]
    random.shuffle(rotations)
    key_rots = rotations[:pair_count]

    specs = []
    for shape, opening, rot_k in zip(shapes, openings, key_rots):
        lock_mask = _build_lock(shape, opening)
        if lock_mask is None:
            return None
        key_mask = np.rot90(shape, rot_k).copy()
        specs.append({"lock": lock_mask, "key": key_mask})

    padded_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for spec in specs for obj in (spec["lock"], spec["key"]))
    max_h = max(max(spec["lock"].shape[0], spec["key"].shape[0]) for spec in specs)
    max_w = max(max(spec["lock"].shape[1], spec["key"].shape[1]) for spec in specs)
    min_side = max(
        GRID_SIDE_MIN,
        max_h + 2 * MARGIN + 1,
        max_w + 2 * MARGIN + 1,
        int(math.ceil(math.sqrt(padded_area * 1.35))),
    )
    if min_side > GRID_SIDE_MAX:
        return None

    H = random.randint(min_side, GRID_SIDE_MAX)
    W = random.randint(min_side, GRID_SIDE_MAX)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    tasks = []
    for idx, spec in enumerate(specs):
        tasks.append(("lock", idx, spec["lock"]))
        tasks.append(("key", idx, spec["key"]))
    tasks.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, mask in tasks:
        pos = _place_bbox(occupied, mask.shape[0], mask.shape[1])
        if pos is None:
            return None
        placements[(kind, idx)] = pos

    for idx, spec in enumerate(specs):
        lock_top, lock_left = placements[("lock", idx)]
        key_top, key_left = placements[("key", idx)]
        _paint(grid, spec["lock"], lock_top, lock_left, LOCK)
        _paint(grid, spec["key"], key_top, key_left, KEY)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.count_nonzero(grid == KEY) != np.count_nonzero((out == KEY) | (out == FIT)):
        return None
    if np.count_nonzero(grid == LOCK) != np.count_nonzero(out == LOCK):
        return None
    if not np.any(out == FIT):
        return None
    if len(_components(grid == KEY)) != pair_count or len(_components(grid == LOCK)) != pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key sample")
