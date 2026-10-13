from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GEN_MAX_ATTEMPTS = 300
GEN_GROUP_RETRIES = 200
GEN_MASK_RETRIES = 200
GEN_PLACEMENT_RETRIES = 80
GEN_PADDING = 1

GEN_GRID_MIN = 18
GEN_GRID_MAX = 30

GEN_ASYM_SIZE = ((2, 5), (2, 6))
GEN_SYM_SIZE = ((2, 5), (3, 6))
GEN_AREA_MIN = 3
GEN_AREA_MAX = 11


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_grid(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "mask must be non-empty"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(int(x) for x in coords[0])
    q = deque([start])
    seen[start] = True
    count = 0

    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return count == int(mask.sum())


def _mask_signature(mask: np.ndarray) -> tuple[str, ...]:
    cropped = _crop_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in cropped)


def _mirror_signature(sig: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row[::-1] for row in sig)


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox_from_coords(coords_arr),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair objects by horizontal-mirror equivalence and return the tight crop of the unique orphan.

    Input structure:
    - A black background contains several separated monochrome foreground objects.
    - Objects may repeat in color, orientation, and shape, and internal holes may appear inside a tight bounding box.
    - Exactly one object cannot be included in any maximum set of disjoint left-right mirror pairs.

    Rule:
    - Treat two objects as a pair when one becomes exactly the other after a horizontal flip and a translation.
    - A pure translation match without the flip does not count unless the object is itself horizontally symmetric.
    - Form as many disjoint mirror pairs as possible, then output the remaining object's minimum bounding-box crop.

    Key insight:
    - The decisive feature is mirror equivalence of the shape mask, not mere equality up to translation; asymmetric duplicate copies are decoys, while symmetric shapes can pair with translated copies of themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert components, "expected at least one foreground object"

    by_sig: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for comp in components:
        sig = _mask_signature(_component_mask(comp))
        comp["sig"] = sig
        by_sig[sig].append(comp)

    leftovers: list[tuple[str, ...]] = []
    handled: set[tuple[str, ...]] = set()
    for sig, comps in by_sig.items():
        if sig in handled:
            continue

        mirror_sig = _mirror_signature(sig)
        count = len(comps)
        if mirror_sig == sig:
            if count % 2 == 1:
                leftovers.append(sig)
            handled.add(sig)
            continue

        other_count = len(by_sig.get(mirror_sig, []))
        diff = count - other_count
        if diff > 0:
            leftovers.extend([sig] * diff)
        elif diff < 0:
            leftovers.extend([mirror_sig] * (-diff))
        handled.add(sig)
        handled.add(mirror_sig)

    assert len(leftovers) == 1, f"expected exactly one orphan, got {len(leftovers)}"
    orphan_sig = leftovers[0]
    orphan = by_sig[orphan_sig][0]
    return _crop_grid(g, orphan["bbox"])


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    mask[random.randrange(height), random.randrange(width)] = True

    while int(mask.sum()) < target_area:
        frontier = _frontier_cells(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    return _crop_mask(mask)


def _orbit(cell: tuple[int, int], width: int) -> tuple[tuple[int, int], ...]:
    r, c = cell
    mc = width - 1 - c
    if mc == c:
        return ((r, c),)
    if c < mc:
        return ((r, c), (r, mc))
    return ((r, mc), (r, c))


def _frontier_orbits(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    frontier: set[tuple[tuple[int, int], ...]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                orbit = _orbit((nr, nc), w)
                if all(not mask[or_r, or_c] for or_r, or_c in orbit):
                    frontier.add(orbit)
    return list(frontier)


def _gen_asymmetric_mask() -> np.ndarray:
    for _ in range(GEN_MASK_RETRIES):
        h = random.randint(*GEN_ASYM_SIZE[0])
        w = random.randint(*GEN_ASYM_SIZE[1])
        target = random.randint(GEN_AREA_MIN, min(GEN_AREA_MAX, h * w))
        mask = _grow_connected_mask(h, w, target)
        if mask is None:
            continue
        if mask.shape[1] < 2:
            continue
        if not _is_connected(mask):
            continue
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric mask")


def _gen_symmetric_mask() -> np.ndarray:
    for _ in range(GEN_MASK_RETRIES):
        h = random.randint(*GEN_SYM_SIZE[0])
        w = random.randint(*GEN_SYM_SIZE[1])
        target = random.randint(GEN_AREA_MIN, min(GEN_AREA_MAX, h * w))

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange((w + 1) // 2))
        for r, c in _orbit(seed, w):
            mask[r, c] = True

        while int(mask.sum()) < target:
            candidates = _frontier_orbits(mask)
            if not candidates:
                break
            random.shuffle(candidates)
            remaining = target - int(mask.sum())
            placed = False
            for orbit in candidates:
                if len(orbit) <= remaining:
                    for r, c in orbit:
                        mask[r, c] = True
                    placed = True
                    break
            if not placed:
                break

        mask = _crop_mask(mask)
        area = int(mask.sum())
        if area < GEN_AREA_MIN or area > GEN_AREA_MAX:
            continue
        if mask.shape[1] < 2:
            continue
        if not np.array_equal(mask, mask[:, ::-1]):
            continue
        if not _is_connected(mask):
            continue
        if np.all(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a symmetric mask")


def _take_color(pool: list[int]) -> int:
    if not pool:
        raise RuntimeError("ran out of colors")
    return int(pool.pop())


def _make_asym_group(
    used_sigs: set[tuple[str, ...]],
    color_pool: list[int],
    *,
    orphan: bool,
    pair_count: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    for _ in range(GEN_GROUP_RETRIES):
        mask = _gen_asymmetric_mask()
        sig = _mask_signature(mask)
        mirror_mask = mask[:, ::-1].copy()
        mirror_sig = _mask_signature(mirror_mask)
        if sig == mirror_sig:
            continue
        if sig in used_sigs or mirror_sig in used_sigs:
            continue

        pairs = pair_count if pair_count is not None else random.randint(1, 2)
        majority_is_base = random.choice((True, False))
        base_count = pairs + (1 if orphan and majority_is_base else 0)
        mirror_count = pairs + (1 if orphan and not majority_is_base else 0)

        base_patch = _render_patch(mask, _take_color(color_pool))
        mirror_patch = _render_patch(mirror_mask, _take_color(color_pool))

        used_sigs.add(sig)
        used_sigs.add(mirror_sig)

        objects = [base_patch.copy() for _ in range(base_count)] + [mirror_patch.copy() for _ in range(mirror_count)]
        orphan_patch = None
        if orphan:
            orphan_patch = base_patch.copy() if base_count > mirror_count else mirror_patch.copy()
        return objects, orphan_patch

    raise RuntimeError("failed to build an asymmetric group")


def _make_sym_group(
    used_sigs: set[tuple[str, ...]],
    color_pool: list[int],
    *,
    orphan: bool,
    count: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    for _ in range(GEN_GROUP_RETRIES):
        mask = _gen_symmetric_mask()
        sig = _mask_signature(mask)
        if sig in used_sigs:
            continue

        copies = count
        if copies is None:
            copies = random.choice((3, 3, 5)) if orphan else random.choice((2, 2, 4))

        patch = _render_patch(mask, _take_color(color_pool))
        used_sigs.add(sig)
        objects = [patch.copy() for _ in range(copies)]
        return objects, patch.copy() if orphan else None

    raise RuntimeError("failed to build a symmetric group")


def _bbox_area(patch: np.ndarray) -> int:
    h, w = patch.shape
    return int(h * w)


def _can_place(reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = patch.shape
    grid_h, grid_w = reserved.shape
    if top + h > grid_h or left + w > grid_w:
        return False
    r0 = max(0, top - GEN_PADDING)
    c0 = max(0, left - GEN_PADDING)
    r1 = min(grid_h, top + h + GEN_PADDING)
    c1 = min(grid_w, left + w + GEN_PADDING)
    return not reserved[r0:r1, c0:c1].any()


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _reserve(reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    grid_h, grid_w = reserved.shape
    r0 = max(0, top - GEN_PADDING)
    c0 = max(0, left - GEN_PADDING)
    r1 = min(grid_h, top + h + GEN_PADDING)
    c1 = min(grid_w, left + w + GEN_PADDING)
    reserved[r0:r1, c0:c1] = True


def _place_objects(patches: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(GEN_PLACEMENT_RETRIES):
        h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)

        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda idx: _bbox_area(patches[idx]), reverse=True)

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            candidates = [
                (r, c)
                for r in range(h - ph + 1)
                for c in range(w - pw + 1)
                if _can_place(reserved, r, c, patch)
            ]
            if not candidates:
                ok = False
                break
            top, left = random.choice(candidates)
            _stamp(grid, patch, top, left)
            _reserve(reserved, top, left, patch)

        if ok:
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    used_sigs: set[tuple[str, ...]] = set()

    orphan_asymmetric = random.random() < 0.65
    patches: list[np.ndarray] = []

    if orphan_asymmetric:
        orphan_group, orphan_patch = _make_asym_group(used_sigs, color_pool, orphan=True)
        opposite_group, _ = _make_sym_group(used_sigs, color_pool, orphan=False, count=2)
    else:
        orphan_group, orphan_patch = _make_sym_group(used_sigs, color_pool, orphan=True, count=random.choice((3, 5)))
        opposite_group, _ = _make_asym_group(used_sigs, color_pool, orphan=False, pair_count=2)

    patches.extend(orphan_group)
    patches.extend(opposite_group)

    if random.random() < 0.5:
        if random.random() < 0.5:
            extra_group, _ = _make_sym_group(used_sigs, color_pool, orphan=False, count=2)
        else:
            extra_group, _ = _make_asym_group(used_sigs, color_pool, orphan=False, pair_count=1)
        patches.extend(extra_group)

    assert orphan_patch is not None
    grid = _place_objects(patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out, orphan_patch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
