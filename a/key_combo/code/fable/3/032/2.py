from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLORS = list(range(1, 10))
GEN_MAX_ATTEMPTS = 400
GEN_BASE_ATTEMPTS = 300
GEN_PAIR_COUNT_CHOICES = [2, 2, 3, 3, 4]
GEN_AREA_RANGE = (4, 8)
GEN_SHAPE_SIDE_RANGE = (2, 5)
GEN_DENSITY_RANGE = (0.35, 0.8)
GEN_STRETCH_CHOICES = [1, 1, 2, 2, 3]
GEN_MAX_PATCH_SIDE = 9
GEN_MAX_PATCH_AREA = 48


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            objects.append({"color": color, "coords": arr, "bbox": _bbox(arr), "area": int(arr.shape[0])})
    return objects


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 2 and bool(mask.any())
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    reduced = mask[row_keep, :]
    col_keep = [0]
    for c in range(1, reduced.shape[1]):
        if not np.array_equal(reduced[:, c], reduced[:, c - 1]):
            col_keep.append(c)
    return reduced[:, col_keep]


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return (int(mask.shape[0]), int(mask.shape[1]), tuple(int(v) for v in mask.reshape(-1)))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    best = None
    best_key = None
    for variant in _dihedral_variants(mask):
        key = _mask_key(variant)
        if best_key is None or key < best_key:
            best_key = key
            best = variant
    assert best is not None
    return best.copy()


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return _mask_key(_canonical_mask(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """Find the unique sprite without a stretched twin and output its run-compressed canonical mask.

    Input structure:
    The grid has a black background and several separated solid monochromatic sprites.
    Except for one sprite, every sprite belongs to a pair whose cropped masks become
    rotation/reflection-equivalent after collapsing repeated adjacent rows and columns.

    Rule:
    Extract each 4-connected monochromatic sprite, crop it to its bbox, and run-compress
    identical neighboring rows and columns. Group sprites by the canonical dihedral form
    of that compressed mask, then output the only unpaired sprite as its canonical
    compressed mask using the sprite's own color.

    Key insight:
    Raw size, orientation, and color are distractions: matching must compare the topology
    that remains after undoing row/column stretch, and the answer is the only topology
    class with exactly one leftover member.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects

    groups: dict[tuple[int, int, tuple[int, ...]], list[dict]] = {}
    for obj in objects:
        raw_mask = _coords_to_mask(obj["coords"])
        compressed = _compress_runs(raw_mask)
        canonical = _canonical_mask(compressed)
        info = {"color": obj["color"], "canonical": canonical}
        groups.setdefault(_mask_key(canonical), []).append(info)

    odd_groups = [members for members in groups.values() if len(members) % 2 == 1]
    assert len(odd_groups) == 1
    assert len(odd_groups[0]) == 1

    answer = odd_groups[0][0]
    out = np.zeros(answer["canonical"].shape, dtype=int)
    out[answer["canonical"]] = int(answer["color"])
    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    arr = np.array(shifted, dtype=int)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1) + 1, int(c1) + 1), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    h, w = empty.shape
    seen = np.zeros_like(empty, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(empty & ~seen))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    pts = {tuple(map(int, p)) for p in coords}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _random_base_mask() -> np.ndarray | None:
    for _ in range(GEN_BASE_ATTEMPTS):
        target_area = random.randint(*GEN_AREA_RANGE)
        coords = {(0, 0)}
        while len(coords) < target_area:
            frontier = []
            for r, c in coords:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in coords:
                        frontier.append(nxt)
            coords.add(random.choice(frontier))
        mask = _normalize_coords(coords)
        h, w = mask.shape
        if not (GEN_SHAPE_SIDE_RANGE[0] <= h <= GEN_SHAPE_SIDE_RANGE[1]):
            continue
        if not (GEN_SHAPE_SIDE_RANGE[0] <= w <= GEN_SHAPE_SIDE_RANGE[1]):
            continue
        if not _is_connected(mask):
            continue
        if _has_hole(mask):
            continue
        if not (GEN_DENSITY_RANGE[0] <= float(mask.mean()) <= GEN_DENSITY_RANGE[1]):
            continue
        if not np.array_equal(_compress_runs(mask), mask):
            continue
        canonical = _canonical_mask(mask)
        if canonical.shape[0] < 2 or canonical.shape[1] < 2:
            continue
        return canonical
    return None


def _sample_stretch_factors(length: int) -> list[int]:
    while True:
        factors = [random.choice(GEN_STRETCH_CHOICES) for _ in range(length)]
        if max(factors) > 1:
            return factors


def _stretch_mask(mask: np.ndarray) -> np.ndarray | None:
    for _ in range(100):
        row_factors = _sample_stretch_factors(mask.shape[0])
        col_factors = _sample_stretch_factors(mask.shape[1])
        stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
        if stretched.shape == mask.shape:
            continue
        if max(stretched.shape) > GEN_MAX_PATCH_SIDE:
            continue
        if stretched.shape[0] * stretched.shape[1] > GEN_MAX_PATCH_AREA:
            continue
        return stretched
    return None


def _random_object_patch(base_mask: np.ndarray, color: int) -> dict | None:
    stretched = _stretch_mask(base_mask)
    if stretched is None:
        return None
    variant = random.choice(_dihedral_variants(stretched))
    patch = np.zeros(variant.shape, dtype=int)
    patch[variant] = int(color)
    return {"raw_mask": variant, "canonical": base_mask, "color": int(color), "patch": patch}


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _candidate_positions(occupied: np.ndarray, shape: tuple[int, int], border: int = 1, padding: int = 1) -> list[tuple[int, int]]:
    h, w = shape
    H, W = occupied.shape
    if h + 2 * border > H or w + 2 * border > W:
        return []
    out = []
    for r0 in range(border, H - border - h + 1):
        for c0 in range(border, W - border - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc1 = min(W, c0 + w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def _place_objects(objects: list[dict], height: int, width: int) -> np.ndarray | None:
    occupied = np.zeros((height, width), dtype=bool)
    grid = np.zeros((height, width), dtype=int)
    order = sorted(range(len(objects)), key=lambda i: objects[i]["patch"].shape[0] * objects[i]["patch"].shape[1], reverse=True)
    for idx in order:
        patch = objects[idx]["patch"]
        positions = _candidate_positions(occupied, patch.shape, border=1, padding=1)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste_patch(grid, patch, r0, c0)
        h, w = patch.shape
        occupied[r0 : r0 + h, c0 : c0 + w] = True
    return grid


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(GEN_PAIR_COUNT_CHOICES)
    base_count = pair_count + 1

    bases = []
    seen = set()
    for _ in range(2000):
        if len(bases) == base_count:
            break
        base = _random_base_mask()
        if base is None:
            return None
        key = _canonical_signature(base)
        if key in seen:
            continue
        seen.add(key)
        bases.append(base)
    if len(bases) != base_count:
        return None

    answer_index = random.randrange(base_count)
    object_count = pair_count * 2 + 1
    palette_size = random.randint(3, min(6, object_count, len(COLORS)))
    palette = random.sample(COLORS, palette_size)

    objects = []
    for i, base in enumerate(bases):
        copies = 1 if i == answer_index else 2
        variants = []
        for _ in range(copies):
            obj = None
            for _ in range(80):
                color = random.choice(palette)
                candidate = _random_object_patch(base, color)
                if candidate is None:
                    continue
                if any(np.array_equal(candidate["raw_mask"], prev["raw_mask"]) and candidate["color"] == prev["color"] for prev in variants):
                    continue
                obj = candidate
                break
            if obj is None:
                return None
            variants.append(obj)
        objects.extend(variants)

    answer_obj = next(obj for obj in objects if obj["canonical"] is bases[answer_index])
    if random.random() < 0.7:
        same_color_candidates = [obj for obj in objects if obj is not answer_obj]
        target = random.choice(same_color_candidates)
        target["color"] = answer_obj["color"]
        target["patch"] = np.where(target["raw_mask"], answer_obj["color"], BG).astype(int)

    rect_area = sum(obj["patch"].shape[0] * obj["patch"].shape[1] for obj in objects)
    max_h = max(obj["patch"].shape[0] for obj in objects)
    max_w = max(obj["patch"].shape[1] for obj in objects)

    for _ in range(40):
        min_h = max(max_h + 2, 16)
        min_w = max(max_w + 2, 16)
        max_h_grid = min(30, max(min_h, min_h + 10))
        max_w_grid = min(30, max(min_w, min_w + 10))
        height = random.randint(min_h, max_h_grid)
        width = random.randint(min_w, max_w_grid)
        if height * width < rect_area * 2:
            continue
        grid = _place_objects(objects, height, width)
        if grid is None:
            continue
        expected = np.zeros(answer_obj["canonical"].shape, dtype=int)
        expected[answer_obj["canonical"]] = answer_obj["color"]
        if np.array_equal(main(grid), expected):
            return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input grid")
