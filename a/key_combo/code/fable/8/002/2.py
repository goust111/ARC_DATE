from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = list(range(1, 10))
GEN_ATTEMPTS = 500
GEN_SHAPE_ATTEMPTS = 200
GRID_BOUNDS = (18, 30)
PAIR_COUNT_OPTIONS = (1, 2, 3)  # total objects: 5, 7, 9
AREA_BOUNDS = (5, 8)
BBOX_SIDE_BOUNDS = (3, 5)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    return _crop_bbox(mask.astype(bool), box).astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in base)


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    return min(_mask_key(np.rot90(base, k)) for k in range(4))


def _dihedral_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_chiral_asymmetric(mask: np.ndarray) -> bool:
    base = _tight_mask(mask)
    base_key = _mask_key(base)
    seen = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        seen.add(_mask_key(rotated))
        seen.add(_mask_key(np.fliplr(rotated)))
    if len(seen) != 8:
        return False
    return _rotation_signature(base) != _rotation_signature(np.fliplr(base)) and base_key in seen


def _random_corner_path(h: int, w: int) -> np.ndarray:
    starts = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    sr, sc = random.choice(starts)
    er, ec = h - 1 - sr, w - 1 - sc
    steps = [(1 if er > sr else -1, 0)] * abs(er - sr)
    steps += [(0, 1 if ec > sc else -1)] * abs(ec - sc)
    random.shuffle(steps)

    mask = np.zeros((h, w), dtype=bool)
    r, c = sr, sc
    mask[r, c] = True
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True
    return mask


def _growth_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    choices: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            touches = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    touches += 1
            if touches in (1, 2):
                choices.append((r, c))
    random.shuffle(choices)
    return choices


def _random_shape() -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = random.randint(*BBOX_SIDE_BOUNDS)
        w = random.randint(*BBOX_SIDE_BOUNDS)
        area_lo = max(AREA_BOUNDS[0], h + w - 1)
        area_hi = min(AREA_BOUNDS[1], h * w - 1)
        if area_lo > area_hi:
            continue
        target_area = random.randint(area_lo, area_hi)
        mask = _random_corner_path(h, w)
        while int(mask.sum()) < target_area:
            choices = _growth_candidates(mask)
            if not choices:
                break
            r, c = choices[0]
            mask[r, c] = True
        if int(mask.sum()) != target_area:
            continue
        if not _is_connected(mask):
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_chiral_asymmetric(mask):
            continue
        return _tight_mask(mask)
    return None


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return _crop_bbox(grid == int(comp["color"]), (r0, c0, r1, c1)).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the only shape whose class has no pure-rotation partner.

    Input structure:
    - A black background contains an odd number of pairwise non-touching monochrome sprites.
    - Every sprite has its own color and none is self-symmetric under rotation or reflection.
    - All but one sprites belong to size-2 classes when grouped by exact shape up to 0/90/180/270 degree rotation.
    - The remaining singleton has a mirror-relative decoy elsewhere in the grid, but that decoy belongs to a different rotation class.

    Rule:
    - Split the foreground into connected monochrome sprites and crop each sprite to its tight bounding box.
    - Compare sprite masks only up to rotation, not reflection.
    - Find the unique rotation-equivalence class of size 1.
    - Return that sprite's tight crop, preserving its original color and orientation.

    Key insight:
    - A reflected cousin can make every sprite look paired at first glance, but only one sprite lacks a partner under pure rotation alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g)
    assert len(comps) % 2 == 1 and len(comps) >= 3

    groups: dict[tuple[str, ...], list[dict]] = {}
    for comp in comps:
        mask = _component_mask(g, comp)
        groups.setdefault(_rotation_signature(mask), []).append(comp)

    singletons = [members[0] for members in groups.values() if len(members) == 1]
    assert len(singletons) == 1
    assert all(len(members) == 2 for members in groups.values() if len(members) != 1)

    orphan = singletons[0]
    return _crop_bbox(g, orphan["bbox"])


def _distinct_rotations(mask: np.ndarray, count: int) -> list[np.ndarray]:
    variants = [_tight_mask(np.rot90(mask, k)) for k in range(4)]
    keys: dict[tuple[str, ...], np.ndarray] = {}
    for variant in variants:
        keys.setdefault(_mask_key(variant), variant)
    assert len(keys) == 4
    chosen = random.sample(list(keys.values()), count)
    return [np.asarray(v, dtype=bool).copy() for v in chosen]


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _try_place(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    row_range = list(range(1, h - mh)) if h - mh > 1 else list(range(h - mh + 1))
    col_range = list(range(1, w - mw)) if w - mw > 1 else list(range(w - mw + 1))
    positions = [(r0, c0) for r0 in row_range for c0 in col_range]
    random.shuffle(positions)
    fg = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for r0, c0 in positions:
        placed = [(r0 + dr, c0 + dc) for dr, dc in fg]
        if any(blocked[r, c] for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = int(color)
        _mark_blocked(blocked, placed)
        return True
    return False


def _sample_base_masks(pair_count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    orphan_base = _random_shape()
    if orphan_base is None:
        return None
    orphan_dihedral = _dihedral_signature(orphan_base)

    others: list[np.ndarray] = []
    used_dihedral = {orphan_dihedral}
    for _ in range(pair_count):
        for _attempt in range(GEN_SHAPE_ATTEMPTS):
            candidate = _random_shape()
            if candidate is None:
                continue
            sig = _dihedral_signature(candidate)
            if sig in used_dihedral:
                continue
            used_dihedral.add(sig)
            others.append(candidate)
            break
        else:
            return None
    return orphan_base, others


def _analyze_semantics(grid: np.ndarray) -> dict:
    comps = _find_components(grid)
    rot_groups: dict[tuple[str, ...], list[int]] = {}
    dihedral_groups: dict[tuple[str, ...], list[int]] = {}
    masks: list[np.ndarray] = []
    rot_sigs: list[tuple[str, ...]] = []
    dihedral_sigs: list[tuple[str, ...]] = []
    for idx, comp in enumerate(comps):
        mask = _component_mask(grid, comp)
        masks.append(mask)
        rot_sig = _rotation_signature(mask)
        di_sig = _dihedral_signature(mask)
        rot_sigs.append(rot_sig)
        dihedral_sigs.append(di_sig)
        rot_groups.setdefault(rot_sig, []).append(idx)
        dihedral_groups.setdefault(di_sig, []).append(idx)

    singleton_ids = [members[0] for members in rot_groups.values() if len(members) == 1]
    assert len(singleton_ids) == 1
    orphan_id = singleton_ids[0]
    orphan_dihedral = dihedral_sigs[orphan_id]

    assert all(_is_chiral_asymmetric(mask) for mask in masks)
    assert all(len(members) == 2 for members in rot_groups.values() if len(members) != 1)
    assert len(dihedral_groups[orphan_dihedral]) == 3

    other_ids = [idx for idx in dihedral_groups[orphan_dihedral] if idx != orphan_id]
    assert len(other_ids) == 2
    assert len({rot_sigs[idx] for idx in other_ids}) == 1
    assert rot_sigs[other_ids[0]] != rot_sigs[orphan_id]
    assert sum(len(members) == 3 for members in dihedral_groups.values()) == 1
    assert all(len(members) == 2 for members in dihedral_groups.values() if len(members) != 3)

    orphan_box = comps[orphan_id]["bbox"]
    expected = _crop_bbox(grid, orphan_box)
    return {"orphan_id": orphan_id, "expected": expected}


def _build_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    sampled = _sample_base_masks(pair_count)
    if sampled is None:
        return None
    orphan_base, other_bases = sampled

    objects: list[np.ndarray] = []
    objects.extend(_distinct_rotations(orphan_base, 1))
    objects.extend(_distinct_rotations(np.fliplr(orphan_base), 2))
    for base in other_bases:
        objects.extend(_distinct_rotations(base, 2))
    random.shuffle(objects)

    grid_h = random.randint(*GRID_BOUNDS)
    grid_w = random.randint(*GRID_BOUNDS)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    colors = random.sample(FOREGROUND_COLORS, len(objects))

    for mask, color in zip(objects, colors):
        if not _try_place(grid, blocked, mask, color):
            return None

    try:
        analysis = _analyze_semantics(grid)
        if not np.array_equal(main(grid), analysis["expected"]):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chirality-orphan input")
