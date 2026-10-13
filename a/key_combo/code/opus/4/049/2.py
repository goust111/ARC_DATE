from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


def _mask_from_relative_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _exact_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _shape_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    base = np.asarray(mask, dtype=bool)
    for rot in range(4):
        turned = np.rot90(base, rot)
        for variant in (turned, np.fliplr(turned)):
            key = _exact_shape_key(variant)
            if key not in seen:
                seen[key] = variant.copy()
    return list(seen.values())


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_exact_shape_key(variant) for variant in _shape_transforms(mask))


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites: list[dict] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = np.asarray(comp["coords"], dtype=int) - np.array([[r0, c0]], dtype=int)
        mask = _mask_from_relative_coords(rel, patch.shape)
        isolated = np.full_like(patch, bg)
        isolated[mask] = int(comp["color"])
        sprites.append(
            {
                "id": idx,
                "color": int(comp["color"]),
                "r0": r0,
                "c0": c0,
                "mask": mask,
                "patch": isolated,
                "area": int(mask.sum()),
                "shape_key": _canonical_shape_key(mask),
                "exact_key": _exact_shape_key(mask),
            }
        )
    return bg, sprites


def _scaled_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(patch, dtype=int), factor, axis=0), factor, axis=1)


def _compose_survivors(survivors: list[dict], bg: int, factor: int, sort_key) -> np.ndarray:
    ordered = sorted(survivors, key=sort_key)
    scaled = [_scaled_patch(sprite["patch"], factor) for sprite in ordered]
    height = max(int(p.shape[0]) for p in scaled)
    width = sum(int(p.shape[1]) for p in scaled) + max(0, len(scaled) - 1)
    out = np.full((height, width), int(bg), dtype=int)
    c0 = 0
    for patch in scaled:
        paste(out, patch, 0, c0)
        c0 += int(patch.shape[1]) + 1
    return out


def _solve(
    grid: np.ndarray,
    *,
    group_mode: str = "canonical",
    sort_mode: str = "row_major",
    scale_mode: str = "survivor_count",
) -> np.ndarray:
    bg, sprites = _extract_sprites(grid)
    assert sprites, "expected at least one sprite"

    if group_mode == "canonical":
        group_keys = [sprite["shape_key"] for sprite in sprites]
    elif group_mode == "exact":
        group_keys = [sprite["exact_key"] for sprite in sprites]
    elif group_mode == "area":
        group_keys = [sprite["area"] for sprite in sprites]
    else:
        raise ValueError(f"unknown group_mode: {group_mode}")

    freq = Counter(group_keys)
    survivors = [sprite for sprite, key in zip(sprites, group_keys) if int(freq[key]) == 1]
    assert survivors, "expected at least one surviving sprite"

    if scale_mode == "survivor_count":
        factor = len(survivors)
    elif scale_mode == "object_count":
        factor = len(sprites)
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")

    if sort_mode == "row_major":
        sort_key = lambda sprite: (int(sprite["r0"]), int(sprite["c0"]))
    elif sort_mode == "col_major":
        sort_key = lambda sprite: (int(sprite["c0"]), int(sprite["r0"]))
    else:
        raise ValueError(f"unknown sort_mode: {sort_mode}")

    return _compose_survivors(survivors, bg, int(factor), sort_key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove shape-duplicate sprites, then enlarge the unique ones by their own survivor count and concatenate them.

    Input structure:
    - A black background contains 5 to 7 single-color 4-connected sprites.
    - Each sprite uses a distinct non-black color and occupies 3 to 6 cells.
    - Some sprites share the same binary shape up to rotation or reflection, while the others are shape-unique.

    Rule:
    - Extract each sprite's binary mask inside its own bounding box and canonicalize that mask over the 8 rotation/reflection variants.
    - Count how many sprites belong to each canonical shape class, then erase every sprite whose class occurs at least twice.
    - Let N be the number of surviving unique-shape sprites.
    - Scale every surviving sprite's cropped patch by N in both dimensions, preserving black holes as black.
    - Sort survivors by original position using top-to-bottom row order and left-to-right column order as the tiebreaker, then place the scaled sprites left-to-right with one black column between neighbors.

    Key insight:
    - Shape frequency and output scale are coupled: only singleton shape classes survive, and the number of those survivors is exactly the pixel-enlargement factor applied to each remaining sprite.
    """
    return _solve(grid, group_mode="canonical", sort_mode="row_major", scale_mode="survivor_count")


_RNG = np.random.default_rng()
_PALETTE = (1, 2, 3, 4, 5, 6, 7)
_GRID_MIN = 18
_GRID_MAX = 24
_MAX_RETRIES = 400


def _shape(mask_rows: list[list[int]]) -> np.ndarray:
    return np.array(mask_rows, dtype=bool)


def _build_shape_library() -> list[dict]:
    raw = {
        "line3": _shape([[1, 1, 1]]),
        "ell3": _shape([[1, 0], [1, 1]]),
        "line4": _shape([[1, 1, 1, 1]]),
        "square4": _shape([[1, 1], [1, 1]]),
        "tee4": _shape([[1, 1, 1], [0, 1, 0]]),
        "ell4": _shape([[1, 0], [1, 0], [1, 1]]),
        "zig4": _shape([[0, 1, 1], [1, 1, 0]]),
        "plus5": _shape([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        "u5": _shape([[1, 0, 1], [1, 1, 1]]),
        "p5": _shape([[1, 1], [1, 1], [1, 0]]),
        "tee5": _shape([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
    }
    seen = set()
    out = []
    for name, mask in raw.items():
        key = _canonical_shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        transforms = _shape_transforms(mask)
        out.append(
            {
                "name": name,
                "mask": mask,
                "key": key,
                "area": int(mask.sum()),
                "transforms": transforms,
                "min_width": min(int(v.shape[1]) for v in transforms),
                "min_height": min(int(v.shape[0]) for v in transforms),
            }
        )
    return out


_SHAPES = _build_shape_library()
_SHAPES_BY_AREA: dict[int, list[dict]] = defaultdict(list)
for _entry in _SHAPES:
    _SHAPES_BY_AREA[int(_entry["area"])].append(_entry)


def _duplicate_partitions(duplicate_total: int) -> list[tuple[int, ...]]:
    parts = set()
    for first in (2, 3):
        if first == duplicate_total:
            parts.add((first,))
        for second in (2, 3):
            if first + second == duplicate_total:
                parts.add(tuple(sorted((first, second))))
    return sorted(parts)


def _choose_shape_classes(unique_count: int, duplicate_groups: tuple[int, ...]) -> tuple[list[dict], list[dict]] | None:
    width_budget = (30 - (unique_count - 1)) // unique_count
    areas = [area for area, entries in _SHAPES_BY_AREA.items() if len(entries) >= 2]
    _RNG.shuffle(areas)
    for area in areas:
        entries = list(_SHAPES_BY_AREA[area])
        anchor_pairs = [(u, d) for u in entries for d in entries if u["key"] != d["key"] and len(d["transforms"]) > 1]
        _RNG.shuffle(anchor_pairs)
        for unique_anchor, duplicate_anchor in anchor_pairs:
            remaining = [shape for shape in _SHAPES if shape["key"] not in {unique_anchor["key"], duplicate_anchor["key"]}]
            unique_tail_options = [
                combo
                for combo in combinations(remaining, unique_count - 1)
                if int(unique_anchor["min_width"]) + sum(int(shape["min_width"]) for shape in combo) <= width_budget
            ]
            _RNG.shuffle(unique_tail_options)
            for unique_tail in unique_tail_options:
                unique_shapes = [unique_anchor, *unique_tail]
                used_unique = {shape["key"] for shape in unique_shapes}
                remaining_dups = [shape for shape in remaining if shape["key"] not in used_unique]
                extra_needed = len(duplicate_groups) - 1
                if len(remaining_dups) < extra_needed:
                    continue
                if extra_needed == 0:
                    return unique_shapes, [duplicate_anchor]
                dup_options = list(combinations(remaining_dups, extra_needed))
                if not dup_options:
                    continue
                choice = dup_options[int(_RNG.integers(len(dup_options)))]
                return unique_shapes, [duplicate_anchor, *choice]
    return None


def _sample_object_variants(unique_shapes: list[dict], duplicate_shapes: list[dict], duplicate_groups: tuple[int, ...]) -> list[dict] | None:
    unique_count = len(unique_shapes)
    width_budget = (30 - (unique_count - 1)) // unique_count
    for _ in range(160):
        objects: list[dict] = []
        width_sum = 0
        max_height = 0
        for idx, shape in enumerate(unique_shapes):
            mask = shape["transforms"][int(_RNG.integers(len(shape["transforms"])))]
            width_sum += int(mask.shape[1])
            max_height = max(max_height, int(mask.shape[0]))
            objects.append({"kind": "unique", "group": idx, "shape": shape, "mask": mask.copy()})
        if width_sum > width_budget or unique_count * max_height > 30:
            continue

        duplicate_exact_keys: dict[int, set] = defaultdict(set)
        for group_id, (shape, count) in enumerate(zip(duplicate_shapes, duplicate_groups)):
            transforms = shape["transforms"]
            if group_id == 0:
                picks = list(_RNG.choice(len(transforms), size=min(len(transforms), count), replace=False))
                while len(picks) < count:
                    picks.append(int(_RNG.integers(len(transforms))))
            else:
                picks = [int(_RNG.integers(len(transforms))) for _ in range(count)]
            _RNG.shuffle(picks)
            for pick in picks:
                mask = transforms[pick].copy()
                duplicate_exact_keys[group_id].add(_exact_shape_key(mask))
                objects.append({"kind": "duplicate", "group": group_id, "shape": shape, "mask": mask})
        if len(duplicate_exact_keys[0]) < 2:
            continue
        return objects
    return None


def _valid_positions(mask: np.ndarray, forbidden: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, forbidden.shape)
    mh, mw = map(int, mask.shape)
    rel = np.argwhere(mask)
    spots = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            abs_coords = rel + np.array([[r0, c0]], dtype=int)
            if not forbidden[abs_coords[:, 0], abs_coords[:, 1]].any():
                spots.append((r0, c0))
    return spots


def _place_objects(objects: list[dict], height: int, width: int) -> list[dict] | None:
    forbidden = np.zeros((height, width), dtype=bool)
    placed = []
    order = sorted(range(len(objects)), key=lambda idx: (-int(objects[idx]["mask"].sum()), -int(objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1])))
    for idx in order:
        obj = objects[idx]
        options = _valid_positions(obj["mask"], forbidden)
        if not options:
            return None
        r0, c0 = options[int(_RNG.integers(len(options)))]
        rel = np.argwhere(obj["mask"])
        abs_coords = rel + np.array([[r0, c0]], dtype=int)
        for r, c in abs_coords:
            forbidden[max(0, int(r) - 1) : min(height, int(r) + 2), max(0, int(c) - 1) : min(width, int(c) + 2)] = True
        placed_obj = dict(obj)
        placed_obj["r0"] = int(r0)
        placed_obj["c0"] = int(c0)
        placed.append(placed_obj)
    return placed


def _paint_grid(objects: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    colors = list(_RNG.choice(_PALETTE, size=len(objects), replace=False))
    _RNG.shuffle(colors)
    for obj, color in zip(objects, colors):
        mask = obj["mask"]
        mh, mw = map(int, mask.shape)
        patch = np.zeros((mh, mw), dtype=int)
        patch[mask] = int(color)
        paste(grid, patch, int(obj["r0"]), int(obj["c0"]))
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg, sprites = _extract_sprites(grid)
        if bg != 0 or not (5 <= len(sprites) <= 7):
            return False
        if len({int(sprite["color"]) for sprite in sprites}) != len(sprites):
            return False
        if any(not (3 <= int(sprite["area"]) <= 6) for sprite in sprites):
            return False

        freq = Counter(sprite["shape_key"] for sprite in sprites)
        survivors = [sprite for sprite in sprites if int(freq[sprite["shape_key"]]) == 1]
        duplicates = [sprite for sprite in sprites if int(freq[sprite["shape_key"]]) >= 2]
        if not (2 <= len(survivors) <= 4) or not duplicates:
            return False

        groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
        for sprite in sprites:
            groups[sprite["shape_key"]].append(sprite)
        if not any(len({member["exact_key"] for member in members}) >= 2 for members in groups.values() if len(members) >= 2):
            return False

        unique_areas = {int(sprite["area"]) for sprite in survivors}
        duplicate_areas = {int(sprite["area"]) for sprite in duplicates}
        if not (unique_areas & duplicate_areas):
            return False

        row_order = [int(sprite["color"]) for sprite in sorted(survivors, key=lambda sprite: (int(sprite["r0"]), int(sprite["c0"])))]
        col_order = [int(sprite["color"]) for sprite in sorted(survivors, key=lambda sprite: (int(sprite["c0"]), int(sprite["r0"])))]
        if row_order == col_order:
            return False

        correct = main(grid)
        if correct.ndim != 2 or max(correct.shape) > 30:
            return False
        if np.array_equal(correct, _solve(grid, group_mode="area", sort_mode="row_major", scale_mode="survivor_count")):
            return False
        if np.array_equal(correct, _solve(grid, group_mode="exact", sort_mode="row_major", scale_mode="survivor_count")):
            return False
        if np.array_equal(correct, _solve(grid, group_mode="canonical", sort_mode="col_major", scale_mode="survivor_count")):
            return False
        if np.array_equal(correct, _solve(grid, group_mode="canonical", sort_mode="row_major", scale_mode="object_count")):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    unique_count = int(_RNG.choice([2, 3, 4]))
    total_objects = int(_RNG.integers(max(5, unique_count + 2), 8))
    duplicate_total = total_objects - unique_count
    duplicate_groups = _duplicate_partitions(duplicate_total)
    if not duplicate_groups:
        return None
    duplicate_groups = duplicate_groups[int(_RNG.integers(len(duplicate_groups)))]

    chosen = _choose_shape_classes(unique_count, duplicate_groups)
    if chosen is None:
        return None
    unique_shapes, duplicate_shapes = chosen

    objects = _sample_object_variants(unique_shapes, duplicate_shapes, duplicate_groups)
    if objects is None:
        return None

    _RNG.shuffle(objects)
    height = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    width = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    placed = _place_objects(objects, height, width)
    if placed is None:
        return None

    grid = _paint_grid(placed, height, width)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 2")
