from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    return _crop_bbox(mask.astype(bool), _bbox_from_coords(coords))


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes()


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_signature(np.rot90(mask, k=k)) for k in range(4))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = np.where(patch != BG, patch, dst[r0 : r0 + h, c0 : c0 + w])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _count_mask_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _is_lock_component(grid: np.ndarray, comp: dict) -> bool:
    color = comp["color"]
    r0, c0, r1, c1 = comp["bbox"]
    patch = _crop_bbox(grid, comp["bbox"])
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    if np.count_nonzero(patch == color) != comp["area"]:
        return False
    if not np.all((patch == BG) | (patch == color)):
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    interior = patch[1:-1, 1:-1]
    return bool(np.any(interior == BG) and np.any(interior == color))


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    locks: list[dict] = []
    keys: list[dict] = []

    for comp in _find_color_components(g):
        r0, c0, r1, c1 = comp["bbox"]
        patch = _crop_bbox(g, comp["bbox"])
        color = comp["color"]
        if _is_lock_component(g, comp):
            hole_full = patch == BG
            hole_abs = np.argwhere(hole_full) + np.array([r0, c0], dtype=int)
            locks.append(
                {
                    **comp,
                    "hole_mask": _tight_mask(hole_full),
                    "hole_abs": hole_abs,
                    "hole_area": int(hole_abs.shape[0]),
                }
            )
        else:
            key_mask = patch == color
            keys.append(
                {
                    **comp,
                    "mask": _tight_mask(key_mask),
                }
            )
    return locks, keys


def _find_unique_matches(locks: list[dict], keys: list[dict]) -> dict[int, tuple[int, int]]:
    assert len(locks) == len(keys) and locks, "expected equal non-zero numbers of locks and keys"
    matches: dict[int, tuple[int, int]] = {}
    lock_to_keys: dict[int, list[int]] = {}

    for key_idx, key in enumerate(keys):
        options: list[tuple[int, int]] = []
        for lock_idx, lock in enumerate(locks):
            if key["area"] != lock["hole_area"]:
                continue
            for rot in range(4):
                rotated = np.rot90(key["mask"], k=rot)
                if rotated.shape == lock["hole_mask"].shape and np.array_equal(rotated, lock["hole_mask"]):
                    options.append((lock_idx, rot))
        assert len(options) == 1, f"key {key_idx} must have exactly one fit, got {options}"
        matches[key_idx] = options[0]
        lock_to_keys.setdefault(options[0][0], []).append(key_idx)

    assert len(lock_to_keys) == len(locks), "every lock must be claimed exactly once"
    assert all(len(key_ids) == 1 for key_ids in lock_to_keys.values()), "lock match must be unique"
    return matches


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored key to fill the unique lock hole it exactly complements, then repaint that hole with the key color.

    Input structure:
    - A black background contains several monochrome rectangular lock objects whose outer border is complete.
    - Each lock has some same-color interior teeth and some black interior hole cells.
    - The scene also contains the same number of monochrome 4-connected key sprites in distinct colors.

    Rule:
    - For every key, test the four right-angle rotations and find the unique lock whose black hole cells match the rotated key shape exactly.
    - Recolor that lock's hole cells to the key color.
    - Erase the original key location to black while leaving lock borders and fixed teeth unchanged.

    Key insight:
    - The key matches the lock's missing black complement inside the frame, not the colored teeth or the nearest frame, so exact hole-shape matching under rotation is required.
    """
    g = np.asarray(grid, dtype=int)
    locks, keys = _extract_scene(g)
    matches = _find_unique_matches(locks, keys)

    out = g.copy()
    for key_idx, key in enumerate(keys):
        coords = key["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for key_idx, (lock_idx, _rot) in matches.items():
        hole_abs = locks[lock_idx]["hole_abs"]
        out[hole_abs[:, 0], hole_abs[:, 1]] = keys[key_idx]["color"]
    return out


def _normalize_coords(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _random_polyomino(area: int, max_dim: int = 4) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        coords = _normalize_coords(cells)
        mask = _coords_to_mask(coords)
        h, w = mask.shape
        if h > max_dim or w > max_dim:
            continue
        if area == h * w:
            continue
        if len({_mask_signature(np.rot90(mask, k=k)) for k in range(4)}) < 4:
            continue
        return mask
    return None


def _choose_shapes(count: int) -> list[np.ndarray] | None:
    shared_area = random.choice((5, 6))
    shape_a = _random_polyomino(shared_area)
    if shape_a is None:
        return None
    canon_a = _canonical_shape(shape_a)

    shape_b = None
    for _ in range(200):
        cand = _random_polyomino(shared_area)
        if cand is not None and _canonical_shape(cand) != canon_a:
            shape_b = cand
            break
    if shape_b is None:
        return None

    shapes = [shape_a, shape_b]
    if count == 3:
        existing = {_canonical_shape(shape) for shape in shapes}
        for _ in range(200):
            cand = _random_polyomino(random.choice((4, 5, 6)))
            if cand is not None and _canonical_shape(cand) not in existing:
                shapes.append(cand)
                break
        if len(shapes) != 3:
            return None
    return shapes


def _build_lock_patch(hole_mask: np.ndarray, color: int, max_extra: int) -> tuple[np.ndarray, np.ndarray] | None:
    hole_h, hole_w = hole_mask.shape
    for _ in range(120):
        top = random.randint(0, max_extra)
        bottom = random.randint(0, max_extra)
        left = random.randint(0, max_extra)
        right = random.randint(0, max_extra)
        interior_h = hole_h + top + bottom
        interior_w = hole_w + left + right
        patch = np.full((interior_h + 2, interior_w + 2), color, dtype=int)
        hole_view = patch[1 + top : 1 + top + hole_h, 1 + left : 1 + left + hole_w]
        hole_view[hole_mask] = BG
        if np.count_nonzero(patch[1:-1, 1:-1] == color) < 1:
            continue
        if _count_mask_components(patch == color) != 1:
            continue
        return patch, np.argwhere(patch == BG)
    return None


def _build_key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _random_derangement(n: int) -> list[int]:
    if n == 2:
        return [1, 0]
    perms = [perm for perm in itertools.permutations(range(n)) if all(perm[i] != i for i in range(n))]
    return list(random.choice(perms))


def _layout_bands(lock_patches: list[np.ndarray], key_patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    count = len(lock_patches)
    max_lock_h = max(int(patch.shape[0]) for patch in lock_patches)
    max_lock_w = max(int(patch.shape[1]) for patch in lock_patches)
    max_key_h = max(int(patch.shape[0]) for patch in key_patches)
    max_key_w = max(int(patch.shape[1]) for patch in key_patches)

    col_gap = 2 if count == 2 else 1
    slot_w = max(max_lock_w, max_key_w) + 2
    min_w = count * slot_w + (count + 1) * col_gap

    row_gap = 2
    lock_band_h = max_lock_h + 2
    key_band_h = max_key_h + 2
    min_h = lock_band_h + key_band_h + 3 * row_gap
    if min_w > 30 or min_h > 30:
        return None

    for _ in range(120):
        width = random.randint(min_w, 30)
        height = random.randint(min_h, 30)
        actual_slot_w = (width - (count + 1) * col_gap) // count
        if actual_slot_w < max(max_lock_w, max_key_w):
            continue
        col_starts = [col_gap + idx * (actual_slot_w + col_gap) for idx in range(count)]
        lock_top = row_gap
        key_top = height - row_gap - key_band_h
        perm = _random_derangement(count)

        lock_pos: list[tuple[int, int]] = []
        key_pos: list[tuple[int, int]] = []
        for idx, patch in enumerate(lock_patches):
            h, w = patch.shape
            r = lock_top + random.randint(0, lock_band_h - h)
            c = col_starts[idx] + random.randint(0, actual_slot_w - w)
            lock_pos.append((r, c))
        for idx, patch in enumerate(key_patches):
            h, w = patch.shape
            slot = perm[idx]
            r = key_top + random.randint(0, key_band_h - h)
            c = col_starts[slot] + random.randint(0, actual_slot_w - w)
            key_pos.append((r, c))

        lock_centers = [_bbox_center((r, c, r + patch.shape[0] - 1, c + patch.shape[1] - 1)) for (r, c), patch in zip(lock_pos, lock_patches)]
        nearest_wrong = False
        for idx, ((r, c), patch) in enumerate(zip(key_pos, key_patches)):
            key_center = _bbox_center((r, c, r + patch.shape[0] - 1, c + patch.shape[1] - 1))
            dists = [(key_center[0] - lr) ** 2 + (key_center[1] - lc) ** 2 for lr, lc in lock_centers]
            if int(np.argmin(dists)) != idx:
                nearest_wrong = True
                break
        if not nearest_wrong:
            continue

        grid = np.zeros((height, width), dtype=int)
        for patch, (r, c) in zip(lock_patches, lock_pos):
            _paste(grid, patch, r, c)
        for patch, (r, c) in zip(key_patches, key_pos):
            _paste(grid, patch, r, c)
        return grid, lock_pos, key_pos
    return None


def _expected_from_construction(
    grid: np.ndarray,
    key_patches: list[np.ndarray],
    key_pos: list[tuple[int, int]],
    key_colors: list[int],
    hole_locals: list[np.ndarray],
    lock_pos: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for patch, (r0, c0) in zip(key_patches, key_pos):
        coords = np.argwhere(patch != BG)
        out[coords[:, 0] + r0, coords[:, 1] + c0] = BG
    for color, hole_local, (r0, c0) in zip(key_colors, hole_locals, lock_pos):
        out[hole_local[:, 0] + r0, hole_local[:, 1] + c0] = color
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        locks, keys = _extract_scene(grid)
        if len(locks) != len(keys) or len(locks) < 2:
            return False
        matches = _find_unique_matches(locks, keys)
    except AssertionError:
        return False

    if all(rot == 0 for _lock_idx, rot in matches.values()):
        return False

    has_same_area_distractor = False
    for key_idx, key in enumerate(keys):
        matched_lock = matches[key_idx][0]
        for lock_idx, lock in enumerate(locks):
            if lock_idx != matched_lock and key["area"] == lock["hole_area"]:
                has_same_area_distractor = True
                break
        if has_same_area_distractor:
            break
    if not has_same_area_distractor:
        return False

    lock_centers = [_bbox_center(lock["bbox"]) for lock in locks]
    nearest_wrong = False
    for key_idx, key in enumerate(keys):
        key_center = _bbox_center(key["bbox"])
        dists = [(key_center[0] - lr) ** 2 + (key_center[1] - lc) ** 2 for lr, lc in lock_centers]
        if int(np.argmin(dists)) != matches[key_idx][0]:
            nearest_wrong = True
            break
    if not nearest_wrong:
        return False

    out = main(grid)
    for key in keys:
        coords = key["coords"]
        if np.any(out[coords[:, 0], coords[:, 1]] != BG):
            return False
    for key_idx, (lock_idx, _rot) in matches.items():
        hole_abs = locks[lock_idx]["hole_abs"]
        if not np.all(out[hole_abs[:, 0], hole_abs[:, 1]] == keys[key_idx]["color"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 2, 3))
    base_shapes = _choose_shapes(count)
    if base_shapes is None:
        return None

    all_colors = random.sample(PALETTE, 2 * count)
    lock_colors = all_colors[:count]
    key_colors = all_colors[count:]

    lock_rots = [random.randrange(4) for _ in range(count)]
    key_rots = [random.randrange(4) for _ in range(count)]
    if all((lock_rots[idx] - key_rots[idx]) % 4 == 0 for idx in range(count)):
        bump = random.randrange(count)
        key_rots[bump] = (key_rots[bump] + random.choice((1, 2, 3))) % 4

    max_extra = 1 if count == 3 else 2
    lock_patches: list[np.ndarray] = []
    hole_locals: list[np.ndarray] = []
    key_patches: list[np.ndarray] = []
    for idx, base in enumerate(base_shapes):
        hole_mask = np.rot90(base, k=lock_rots[idx])
        key_mask = np.rot90(base, k=key_rots[idx])
        built = _build_lock_patch(hole_mask, lock_colors[idx], max_extra=max_extra)
        if built is None:
            return None
        lock_patch, hole_local = built
        lock_patches.append(lock_patch)
        hole_locals.append(hole_local)
        key_patches.append(_build_key_patch(key_mask, key_colors[idx]))

    laid_out = _layout_bands(lock_patches, key_patches)
    if laid_out is None:
        return None
    grid, lock_pos, key_pos = laid_out

    expected = _expected_from_construction(grid, key_patches, key_pos, key_colors, hole_locals, lock_pos)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-hole ARC instance")
