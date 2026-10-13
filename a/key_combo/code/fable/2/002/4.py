from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 300
MAX_MASK_SIDE = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for transformed in (rotated, np.fliplr(rotated)):
            key = _shape_key(transformed)
            if key in seen:
                continue
            seen.add(key)
            transforms.append(transformed.copy())
    return transforms


def _rotation_orbit(mask: np.ndarray) -> set[tuple[tuple[int, int], tuple[int, ...]]]:
    return {_shape_key(np.rot90(mask, k)) for k in range(4)}


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_shape_key(transformed) for transformed in _unique_transforms(mask))


def _has_reflection_only_variant(mask: np.ndarray) -> bool:
    rot_keys = _rotation_orbit(mask)
    return any(_shape_key(transformed) not in rot_keys for transformed in _unique_transforms(mask))


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == bg or visited[r0, c0]:
                continue

            queue = deque([(r0, c0)])
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            bbox = _bbox_coords(coord_array)
            mask = _crop_bbox(grid == color, bbox)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": bbox,
                    "area": int(coord_array.shape[0]),
                    "mask": mask,
                    "canonical_key": _canonical_key(mask),
                }
            )

    return components


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair geometrically congruent objects under rotation/reflection, swap each pair's colors, and erase the rest.

    Input structure:
    - A black background contains several separated single-color 4-connected foreground objects.
    - Some objects come in pairs whose tight shape masks match after a rotation and/or mirror reflection.
    - Every matched pair uses two different colors, while every unmatched object's shape is unique in the whole grid.

    Rule:
    - Extract each object's tight boolean shape mask and group objects by equality up to all rotations and mirror reflections.
    - If a shape class contains exactly two objects, keep both objects in place and recolor each one with the other member's color.
    - If a shape class contains only one object, erase it to the black background.

    Key insight:
    - Pairing depends only on dihedral shape equivalence; color reuse, object area, and spatial proximity are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    components = _find_components(g, bg=BG)
    assert components, "expected at least one foreground object"

    grouped: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for comp in components:
        grouped[comp["canonical_key"]].append(comp)

    out = np.zeros_like(g)
    for group in grouped.values():
        assert len(group) in {1, 2}, "shape classes must be either unique or paired"
        if len(group) == 1:
            continue
        first, second = group
        assert first["color"] != second["color"], "paired objects must have different colors"
        out[first["coords"][:, 0], first["coords"][:, 1]] = second["color"]
        out[second["coords"][:, 0], second["coords"][:, 1]] = first["color"]

    return out


def _random_connected_mask(
    area: int,
    *,
    min_h: int,
    min_w: int,
    max_h: int = MAX_MASK_SIDE,
    max_w: int = MAX_MASK_SIDE,
) -> np.ndarray | None:
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells or (nr, nc) in seen:
                        continue
                    trial = cells | {(nr, nc)}
                    trial_mask = _mask_from_cells(trial)
                    if trial_mask.shape[0] <= max_h and trial_mask.shape[1] <= max_w:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = _mask_from_cells(cells)
        h, w = mask.shape
        if h < min_h or w < min_w or h > max_h or w > max_w:
            continue
        return mask

    return None


def _sample_shape(
    used_keys: set[tuple[tuple[int, int], tuple[int, ...]]],
    *,
    area: int | None = None,
    require_chiral: bool = False,
    min_h: int = 1,
    min_w: int = 1,
    min_unique_transforms: int = 1,
) -> np.ndarray | None:
    areas = [area] if area is not None else list(range(max(3, min_h * min_w), 9))
    if require_chiral and area is None:
        areas = [5, 6, 7, 8]

    for _ in range(MAX_SHAPE_ATTEMPTS):
        mask = _random_connected_mask(
            area=random.choice(areas),
            min_h=min_h,
            min_w=min_w,
        )
        if mask is None:
            continue

        canonical = _canonical_key(mask)
        if canonical in used_keys:
            continue
        if len(_unique_transforms(mask)) < min_unique_transforms:
            continue
        if require_chiral and not _has_reflection_only_variant(mask):
            continue
        return mask

    return None


def _choose_pair_transforms(mask: np.ndarray, *, force_mirror: bool) -> tuple[np.ndarray, np.ndarray, bool]:
    transforms = _unique_transforms(mask)
    first = random.choice(transforms).copy()

    if force_mirror:
        orbit = _rotation_orbit(first)
        reflected_only = [transformed for transformed in transforms if _shape_key(transformed) not in orbit]
        if reflected_only:
            return first, random.choice(reflected_only).copy(), True

    if len(transforms) > 1 and random.random() < 0.9:
        different = [transformed for transformed in transforms if _shape_key(transformed) != _shape_key(first)]
        if different:
            return first, random.choice(different).copy(), False

    return first, first.copy(), False


def _grid_shape_for(object_count: int) -> tuple[int, int]:
    base = max(16, 2 * object_count + 9)
    return random.randint(base, 30), random.randint(base, 30)


def _place_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict | None] = [None] * len(objects)
    anchors: list[tuple[float, float]] = []

    order = sorted(range(len(objects)), key=lambda idx: (objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1], objects[idx]["area"]), reverse=True)
    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        mh, mw = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - mh + 1):
            for c0 in range(w - mw + 1):
                if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw]):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        if anchors:
            scored = []
            for r0, c0 in candidates:
                center = (r0 + (mh - 1) / 2.0, c0 + (mw - 1) / 2.0)
                spacing = min(abs(center[0] - ar) + abs(center[1] - ac) for ar, ac in anchors)
                scored.append((spacing, random.random(), r0, c0))
            scored.sort(reverse=True)
            keep = max(1, min(20, len(scored) // 4 or 1))
            _, _, r0, c0 = random.choice(scored[:keep])
        else:
            r0, c0 = random.choice(candidates)

        br0 = max(0, r0 - 1)
        bc0 = max(0, c0 - 1)
        br1 = min(h, r0 + mh + 1)
        bc1 = min(w, c0 + mw + 1)
        blocked[br0:br1, bc0:bc1] = True
        anchors.append((r0 + (mh - 1) / 2.0, c0 + (mw - 1) / 2.0))

        placed[idx] = {
            **obj,
            "r0": int(r0),
            "c0": int(c0),
        }

    return [obj for obj in placed if obj is not None]


def _build_grids(shape: tuple[int, int], placed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)

    by_pair: dict[int, list[dict]] = defaultdict(list)
    for obj in placed:
        _paint_mask(inp, obj["mask"], obj["r0"], obj["c0"], obj["color"])
        if obj["pair_id"] is not None:
            by_pair[obj["pair_id"]].append(obj)

    for members in by_pair.values():
        first, second = members
        _paint_mask(expected, first["mask"], first["r0"], first["c0"], second["color"])
        _paint_mask(expected, second["mask"], second["r0"], second["c0"], first["color"])

    return inp, expected


def _validate_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    placed: list[dict],
    *,
    require_area_decoy: bool,
    require_mirror_pair: bool,
) -> bool:
    components = _find_components(inp, bg=BG)
    if len(components) != len(placed):
        return False

    planned_counts = Counter(obj["canonical_key"] for obj in placed)
    actual_counts = Counter(comp["canonical_key"] for comp in components)
    if actual_counts != planned_counts:
        return False
    if any(count not in {1, 2} for count in actual_counts.values()):
        return False
    if 1 not in actual_counts.values() or 2 not in actual_counts.values():
        return False

    by_pair: dict[int, list[dict]] = defaultdict(list)
    pair_areas: list[int] = []
    singleton_areas: list[int] = []
    mirror_pair_found = False
    for obj in placed:
        if obj["pair_id"] is None:
            singleton_areas.append(int(obj["area"]))
        else:
            by_pair[obj["pair_id"]].append(obj)

    for members in by_pair.values():
        if len(members) != 2:
            return False
        first, second = members
        if first["color"] == second["color"]:
            return False
        if first["canonical_key"] != second["canonical_key"]:
            return False
        pair_areas.append(int(first["area"]))
        if _shape_key(second["mask"]) not in _rotation_orbit(first["mask"]):
            mirror_pair_found = True

    if require_area_decoy and not any(area in pair_areas for area in singleton_areas):
        return False
    if require_mirror_pair and not mirror_pair_found:
        return False
    if np.array_equal(inp, expected):
        return False
    return np.array_equal(main(inp), expected)


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    singleton_count = random.randint(1, 2)

    used_keys: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    pair_bases: list[np.ndarray] = []
    for pair_idx in range(pair_count):
        mask = _sample_shape(
            used_keys,
            require_chiral=(pair_idx == 0),
            min_h=2,
            min_w=2,
            min_unique_transforms=4,
        )
        if mask is None:
            return None
        used_keys.add(_canonical_key(mask))
        pair_bases.append(mask)

    pair_areas = [int(mask.sum()) for mask in pair_bases]
    singleton_bases: list[np.ndarray] = []
    for single_idx in range(singleton_count):
        target_area = random.choice(pair_areas) if single_idx == 0 or random.random() < 0.5 else None
        mask = _sample_shape(
            used_keys,
            area=target_area,
            min_h=1,
            min_w=1,
            min_unique_transforms=1,
        )
        if mask is None:
            return None
        used_keys.add(_canonical_key(mask))
        singleton_bases.append(mask)

    pair_colors_pool: list[int] = []
    objects: list[dict] = []
    mirror_pair_used = False
    for pair_id, base in enumerate(pair_bases):
        first_mask, second_mask, used_mirror = _choose_pair_transforms(base, force_mirror=(pair_id == 0))
        color_a = random.choice(PALETTE)
        color_b = random.choice([color for color in PALETTE if color != color_a])
        pair_colors_pool.extend([color_a, color_b])
        canonical = _canonical_key(base)
        area = int(base.sum())
        objects.append(
            {
                "mask": first_mask,
                "color": color_a,
                "pair_id": pair_id,
                "canonical_key": canonical,
                "area": area,
            }
        )
        objects.append(
            {
                "mask": second_mask,
                "color": color_b,
                "pair_id": pair_id,
                "canonical_key": canonical,
                "area": area,
            }
        )
        mirror_pair_used = mirror_pair_used or used_mirror

    for base in singleton_bases:
        color = random.choice(pair_colors_pool) if pair_colors_pool and random.random() < 0.7 else random.choice(PALETTE)
        objects.append(
            {
                "mask": random.choice(_unique_transforms(base)).copy(),
                "color": color,
                "pair_id": None,
                "canonical_key": _canonical_key(base),
                "area": int(base.sum()),
            }
        )

    shape = _grid_shape_for(len(objects))
    placed = _place_objects(objects, shape)
    if placed is None:
        return None
    if not mirror_pair_used:
        return None

    inp, expected = _build_grids(shape, placed)
    if not _validate_sample(
        inp,
        expected,
        placed,
        require_area_decoy=True,
        require_mirror_pair=True,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
