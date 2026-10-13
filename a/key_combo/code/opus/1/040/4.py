from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2 = rr + dr
                    c2 = cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and not seen[r2, c2]:
                        seen[r2, c2] = True
                        stack.append((r2, c2))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 2 and mask.any()

    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _mask_key(_compress_runs(mask))


def _signature_variants(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    compressed = _compress_runs(mask)
    return {_signature(variant) for variant in _unique_variants(compressed)}


def _choose_prototype(halves: list[np.ndarray]) -> np.ndarray:
    assert len(halves) == 2

    def rank(coords: np.ndarray) -> tuple[int, int]:
        box = _bbox(coords)
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        return int(coords.shape[0]), int(area)

    return max(halves, key=rank)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a cracked green object, compress the larger half into a prototype, and recolor matching blue candidates.

    Input structure:
    - The background is black.
    - One green object on the left contains a full red crack line across its shorter bbox axis.
    - Several separate blue candidate objects sit elsewhere on the grid.

    Rule:
    - Find the single 4-connected component made from the green body plus the red crack.
    - Erase the red crack to black, leaving two green halves in place; crop those halves and choose the larger one as the prototype, breaking size ties by larger bbox area.
    - Run-compress the prototype and every blue candidate by removing consecutive duplicate rows, then consecutive duplicate columns; recolor each candidate green iff its compressed mask matches the prototype up to rotation or reflection.

    Key insight:
    - The template is not directly given: it must first be extracted as the larger half of the cracked object, and matching must compare compressed topology rather than raw stretched geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    cracked_mask = (g == GREEN) | (g == RED)
    cracked_components = [coords for coords in _components(cracked_mask) if np.any(g[coords[:, 0], coords[:, 1]] == RED)]
    assert len(cracked_components) == 1, "expected one cracked green object"
    cracked_coords = cracked_components[0]

    out = g.copy()
    out[g == RED] = BG

    green_only_mask = np.zeros_like(cracked_mask)
    green_coords = cracked_coords[g[cracked_coords[:, 0], cracked_coords[:, 1]] == GREEN]
    green_only_mask[green_coords[:, 0], green_coords[:, 1]] = True
    halves = _components(green_only_mask)
    assert len(halves) == 2, "erasing the crack must split the object into two halves"

    prototype_coords = _choose_prototype(halves)
    prototype_mask = _crop_component(prototype_coords)
    prototype_signatures = _signature_variants(prototype_mask)

    for candidate_coords in _components(g == BLUE):
        candidate_mask = _crop_component(candidate_coords)
        if _signature(candidate_mask) in prototype_signatures:
            out[candidate_coords[:, 0], candidate_coords[:, 1]] = GREEN

    return out


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.asarray([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _build_shape_library() -> list[dict[str, object]]:
    raw_shapes = [
        _parse_mask(["110", "011"]),
        _parse_mask(["111", "010"]),
        _parse_mask(["101", "111"]),
        _parse_mask(["111", "101"]),
        _parse_mask(["110", "011", "001"]),
        _parse_mask(["100", "110", "011"]),
        _parse_mask(["111", "110", "010"]),
        _parse_mask(["010", "111", "010"]),
        _parse_mask(["011", "111", "100"]),
        _parse_mask(["011", "110", "010"]),
        _parse_mask(["010", "111", "001"]),
        _parse_mask(["010", "111", "100"]),
    ]

    library: list[dict[str, object]] = []
    seen = set()
    for mask in raw_shapes:
        canon = min(_mask_key(variant) for variant in _unique_variants(mask))
        if canon in seen:
            continue
        seen.add(canon)
        variants = _unique_variants(mask)
        wide = [variant for variant in variants if variant.shape[1] == 3]
        tall = [variant for variant in variants if variant.shape[0] == 3]
        library.append(
            {
                "mask": mask,
                "variants": variants,
                "wide": wide,
                "tall": tall,
                "canon_sig": min(_signature(variant) for variant in variants),
            }
        )
    return library


_SHAPES = _build_shape_library()
_WIDE_SHAPES = [item for item in _SHAPES if item["wide"]]
_TALL_SHAPES = [item for item in _SHAPES if item["tall"]]


def _stretch(mask: np.ndarray, row_factors: tuple[int, ...], col_factors: tuple[int, ...]) -> np.ndarray:
    stretched = np.repeat(mask, row_factors, axis=0)
    stretched = np.repeat(stretched, col_factors, axis=1)
    return stretched


def _factor_options(length: int, values: tuple[int, ...], sum_range: tuple[int, int]) -> list[tuple[int, ...]]:
    out = []
    lo, hi = sum_range
    for factors in itertools.product(values, repeat=length):
        total = sum(factors)
        if lo <= total <= hi:
            out.append(tuple(int(v) for v in factors))
    return out


_LONG_FACTORS = _factor_options(3, (1, 2, 3), (6, 8))
_SHORT_FACTORS_2 = _factor_options(2, (1, 2), (2, 4))
_SHORT_FACTORS_3 = _factor_options(3, (1, 2), (2, 4))
_CAND_FACTORS_2 = _factor_options(2, (1, 2, 3), (2, 5))
_CAND_FACTORS_3 = _factor_options(3, (1, 2, 3), (3, 5))


def _half_factors(length: int) -> list[tuple[int, ...]]:
    return _SHORT_FACTORS_2 if length == 2 else _SHORT_FACTORS_3


def _candidate_factors(length: int) -> list[tuple[int, ...]]:
    return _CAND_FACTORS_2 if length == 2 else _CAND_FACTORS_3


def _sample_cracked_object() -> dict[str, object] | None:
    wide_object = random.choice([False, True])
    pool = _WIDE_SHAPES if wide_object else _TALL_SHAPES

    for _ in range(300):
        first, second = random.sample(pool, 2)
        first_base = random.choice(first["wide"] if wide_object else first["tall"]).copy()
        second_base = random.choice(second["wide"] if wide_object else second["tall"]).copy()

        if wide_object:
            long_factors = random.choice(_LONG_FACTORS)
            first_mask = _stretch(first_base, random.choice(_half_factors(first_base.shape[0])), long_factors)
            second_mask = _stretch(second_base, random.choice(_half_factors(second_base.shape[0])), long_factors)
            object_h = first_mask.shape[0] + 1 + second_mask.shape[0]
            object_w = first_mask.shape[1]
            crack_len = object_w
            if not (6 <= object_w <= 8 and 5 <= object_h <= 7 and object_w > object_h):
                continue
        else:
            long_factors = random.choice(_LONG_FACTORS)
            first_mask = _stretch(first_base, long_factors, random.choice(_half_factors(first_base.shape[1])))
            second_mask = _stretch(second_base, long_factors, random.choice(_half_factors(second_base.shape[1])))
            object_h = first_mask.shape[0]
            object_w = first_mask.shape[1] + 1 + second_mask.shape[1]
            crack_len = object_h
            if not (6 <= object_h <= 8 and 5 <= object_w <= 7 and object_h > object_w):
                continue

        halves = [
            {"item": first, "mask": first_mask},
            {"item": second, "mask": second_mask},
        ]
        halves.sort(key=lambda item: int(item["mask"].sum()), reverse=True)
        big = halves[0]
        small = halves[1]
        big_area = int(big["mask"].sum())
        small_area = int(small["mask"].sum())
        if big_area == small_area:
            continue
        if not (7 <= big_area <= 14 and 4 <= small_area <= 10):
            continue
        if not (17 <= big_area + small_area + crack_len <= 25):
            continue

        patch = np.zeros((object_h, object_w), dtype=int)
        if wide_object:
            top_is_big = random.choice([False, True])
            top_mask = big["mask"] if top_is_big else small["mask"]
            bottom_mask = small["mask"] if top_is_big else big["mask"]
            top_slice = patch[: top_mask.shape[0], :]
            top_slice[top_mask] = GREEN
            patch[top_mask.shape[0], :] = RED
            start = top_mask.shape[0] + 1
            bottom_slice = patch[start : start + bottom_mask.shape[0], :]
            bottom_slice[bottom_mask] = GREEN
        else:
            left_is_big = random.choice([False, True])
            left_mask = big["mask"] if left_is_big else small["mask"]
            right_mask = small["mask"] if left_is_big else big["mask"]
            left_slice = patch[:, : left_mask.shape[1]]
            left_slice[left_mask] = GREEN
            patch[:, left_mask.shape[1]] = RED
            start = left_mask.shape[1] + 1
            right_slice = patch[:, start : start + right_mask.shape[1]]
            right_slice[right_mask] = GREEN

        return {
            "patch": patch,
            "wide": wide_object,
            "big_mask": big["mask"],
            "small_mask": small["mask"],
            "big_item": big["item"],
            "small_item": small["item"],
        }
    return None


def _sample_candidate_mask(
    item: dict[str, object],
    *,
    raw_forbidden: set[tuple[tuple[int, ...], ...]] | None = None,
    require_nonraw: bool = False,
) -> np.ndarray | None:
    variants = list(item["variants"])
    for _ in range(300):
        base = random.choice(variants)
        row_factors = random.choice(_candidate_factors(base.shape[0]))
        col_factors = random.choice(_candidate_factors(base.shape[1]))
        mask = _stretch(base, row_factors, col_factors)
        area = int(mask.sum())
        if not (4 <= area <= 12):
            continue
        if max(mask.shape) > 5:
            continue
        key = _mask_key(mask)
        if require_nonraw and raw_forbidden is not None and key in raw_forbidden:
            continue
        return mask
    return None


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    color: int,
    col_lo: int,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w or col_lo > w - pw:
        return None

    choices = [(r, c) for r in range(0, h - ph + 1) for c in range(col_lo, w - pw + 1)]
    random.shuffle(choices)
    for r0, c0 in choices:
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(not _boxes_separated(box, old) for old in boxes):
            continue
        if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] != BG):
            continue
        coords = np.argwhere(patch != 0) + np.array([r0, c0])
        grid[coords[:, 0], coords[:, 1]] = color
        return coords, box
    return None


def _build_candidate_specs(scene: dict[str, object]) -> list[dict[str, object]] | None:
    total = random.randint(4, 6)
    if total == 4:
        match_count = random.choice([1, 2])
    elif total == 5:
        match_count = random.choice([2, 2, 3])
    else:
        match_count = random.choice([2, 3])

    proto_mask = np.asarray(scene["big_mask"], dtype=bool)
    proto_item = dict(scene["big_item"])
    small_item = dict(scene["small_item"])
    proto_raw_keys = {_mask_key(variant) for variant in _unique_variants(proto_mask)}

    specs: list[dict[str, object]] = []
    special_match = _sample_candidate_mask(proto_item, raw_forbidden=proto_raw_keys, require_nonraw=True)
    if special_match is None:
        return None
    specs.append({"mask": special_match, "match": True, "special": True})

    while sum(1 for spec in specs if spec["match"]) < match_count:
        mask = _sample_candidate_mask(proto_item)
        if mask is None:
            return None
        specs.append({"mask": mask, "match": True, "special": False})

    small_decoy = _sample_candidate_mask(small_item)
    if small_decoy is None:
        return None
    specs.append({"mask": small_decoy, "match": False, "small_decoy": True})

    others = [item for item in _SHAPES if item["canon_sig"] not in {proto_item["canon_sig"], small_item["canon_sig"]}]
    while len(specs) < total:
        item = random.choice(others)
        mask = _sample_candidate_mask(item)
        if mask is None:
            return None
        specs.append({"mask": mask, "match": False})

    random.shuffle(specs)
    return specs


def _render_scene(scene: dict[str, object], candidate_specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    object_patch = np.asarray(scene["patch"], dtype=int)
    candidate_max_h = max(mask["mask"].shape[0] for mask in candidate_specs)
    candidate_max_w = max(mask["mask"].shape[1] for mask in candidate_specs)

    min_h = max(16, object_patch.shape[0] + 4, candidate_max_h + 4)
    min_w = max(20, object_patch.shape[1] + candidate_max_w + 9)
    if min_h > 24 or min_w > 28:
        return None

    height = random.randint(min_h, 24)
    width = random.randint(min_w, 28)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    candidate_coords: list[np.ndarray] = []

    obj_r = random.randint(1, height - object_patch.shape[0] - 1)
    obj_c = random.randint(1, max(1, width // 4))
    obj_box = (obj_r, obj_c, obj_r + object_patch.shape[0] - 1, obj_c + object_patch.shape[1] - 1)
    boxes.append(obj_box)

    obj_coords = np.argwhere(object_patch != 0) + np.array([obj_r, obj_c])
    grid[obj_coords[:, 0], obj_coords[:, 1]] = object_patch[object_patch != 0]
    expected[obj_coords[:, 0], obj_coords[:, 1]] = object_patch[object_patch != 0]
    expected[expected == RED] = BG

    col_lo = max(width // 2, obj_box[3] + 3)
    for spec in candidate_specs:
        placed = _place_patch(grid, spec["mask"].astype(int), color=BLUE, col_lo=col_lo, boxes=boxes)
        if placed is None:
            return None
        coords, box = placed
        boxes.append(box)
        candidate_coords.append(coords)
        if spec["match"]:
            expected[coords[:, 0], coords[:, 1]] = GREEN
        else:
            expected[coords[:, 0], coords[:, 1]] = BLUE

    return grid, expected, candidate_coords


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    scene: dict[str, object],
    candidate_specs: list[dict[str, object]],
) -> bool:
    try:
        if not np.array_equal(main(grid), expected):
            return False
    except AssertionError:
        return False

    green_halves = _components(grid == GREEN)
    if len(green_halves) != 2:
        return False
    chosen = _choose_prototype(green_halves)
    if _signature(_crop_component(chosen)) != _signature(np.asarray(scene["big_mask"], dtype=bool)):
        return False

    prototype_signatures = _signature_variants(np.asarray(scene["big_mask"], dtype=bool))
    smaller_signatures = _signature_variants(np.asarray(scene["small_mask"], dtype=bool))

    derived_matches = []
    derived_smaller = []
    compression_needed = False
    proto_raw_keys = {_mask_key(variant) for variant in _unique_variants(np.asarray(scene["big_mask"], dtype=bool))}
    saw_small_decoy = False
    for idx, spec in enumerate(candidate_specs):
        mask = np.asarray(spec["mask"], dtype=bool)
        sig = _signature(mask)
        is_proto = sig in prototype_signatures
        is_small = sig in smaller_signatures
        if is_proto:
            derived_matches.append(idx)
        if is_small:
            derived_smaller.append(idx)
        if spec["match"] and _mask_key(mask) not in proto_raw_keys:
            compression_needed = True
        if spec.get("small_decoy") and is_small and not is_proto:
            saw_small_decoy = True

    if derived_matches != [idx for idx, spec in enumerate(candidate_specs) if spec["match"]]:
        return False
    if not saw_small_decoy:
        return False
    if set(derived_matches) == set(derived_smaller):
        return False
    if not compression_needed:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    scene = _sample_cracked_object()
    if scene is None:
        return None

    candidate_specs = _build_candidate_specs(scene)
    if candidate_specs is None:
        return None

    rendered = _render_scene(scene, candidate_specs)
    if rendered is None:
        return None
    grid, expected, _coords = rendered

    if not _validate_scene(grid, expected, scene, candidate_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cracked-prototype scene")
