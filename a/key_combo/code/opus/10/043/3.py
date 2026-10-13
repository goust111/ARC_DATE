from __future__ import annotations

from collections import Counter, defaultdict
import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
BLOCK_SIZE = 3
MACRO_SIDE = 6
GRID_SHAPE = (MACRO_SIDE * BLOCK_SIZE, MACRO_SIDE * BLOCK_SIZE)
GENERATION_RETRIES = 400
_RNG = np.random.default_rng()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return ()
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not key:
        return np.zeros((0, 0), dtype=bool)
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[int(r), int(c)] = True
    return mask


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _crop_mask(np.asarray(mask, dtype=bool))
    if cropped.size == 0:
        return ()
    variants: list[tuple[tuple[int, int], ...]] = []
    for turns in range(4):
        rotated = np.rot90(cropped, turns)
        for variant in (rotated, np.fliplr(rotated)):
            variants.append(_shape_key(variant))
    return min(variants)


def _is_connected4(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    points = set(coords)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in points and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(points)


def _embed_variant(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    top = int(_RNG.integers(0, BLOCK_SIZE - h + 1))
    left = int(_RNG.integers(0, BLOCK_SIZE - w + 1))
    block[top : top + h, left : left + w][mask] = BLUE
    return block


def _extract_block_mask(grid: np.ndarray, macro_r: int, macro_c: int) -> np.ndarray:
    r0 = macro_r * BLOCK_SIZE
    c0 = macro_c * BLOCK_SIZE
    return np.asarray(grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] == BLUE, dtype=bool)


def _analyze_grid(grid: np.ndarray) -> tuple[Counter, dict, dict, np.ndarray]:
    freq: Counter = Counter()
    area_by_key: dict[tuple[tuple[int, int], ...], int] = {}
    variant_by_key: dict[tuple[tuple[int, int], ...], set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    block_keys: list[list[tuple[tuple[int, int], ...]]] = []

    for macro_r in range(MACRO_SIDE):
        row_keys = []
        for macro_c in range(MACRO_SIDE):
            block_mask = _extract_block_mask(grid, macro_r, macro_c)
            area = int(block_mask.sum())
            assert 2 <= area <= 5, "each macro block must contain a 2-5 pixel sprite"
            assert _is_connected4(block_mask), "each macro block sprite must be 4-connected"
            cropped = _crop_mask(block_mask)
            raw_key = _shape_key(cropped)
            canon_key = _canonical_shape_key(cropped)
            freq[canon_key] += 1
            area_by_key[canon_key] = area
            variant_by_key[canon_key].add(raw_key)
            row_keys.append(canon_key)
        block_keys.append(row_keys)

    out = np.zeros((MACRO_SIDE, MACRO_SIDE), dtype=int)
    for macro_r in range(MACRO_SIDE):
        for macro_c in range(MACRO_SIDE):
            if freq[block_keys[macro_r][macro_c]] >= 2:
                out[macro_r, macro_c] = RED
    return freq, area_by_key, variant_by_key, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each 3x3 macro-block into a 6x6 activation map based on shape sharing up to rotation/flip.

    Input structure:
    - The input is an 18x18 black-background grid partitioned into a 6x6 array of 3x3 macro-blocks.
    - Every macro-block contains exactly one blue 4-connected sprite of area 2 to 5 pixels.
    - Sprites may be translated inside their 3x3 block, and some blocks contain rotated/flipped copies of the same shape.

    Rule:
    - For each macro-block, extract the blue sprite shape and crop it to its tight bounding box.
    - Canonicalize that cropped binary shape under the 8 rotation/flip transforms, then count how many macro-blocks share each canonical form.
    - Mark every macro-block whose canonical shape appears at least twice as activated.
    - Output the 6x6 compressed grid with activated blocks colored red and all others black.

    Key insight:
    - A block is red only if its local sprite has a global shape partner somewhere else after rotation/flip normalization, so neither raw 3x3 pattern matching nor pixel count alone is sufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert tuple(map(int, g.shape)) == GRID_SHAPE, "expected a 6x6 array of 3x3 macro-blocks"
    _freq, _areas, _variants, out = _analyze_grid(g)
    return out


def _enumerate_shape_families() -> list[dict]:
    families: dict[tuple[tuple[int, int], ...], dict] = {}
    for area in range(2, 6):
        for cells in itertools.combinations(range(BLOCK_SIZE * BLOCK_SIZE), area):
            mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
            for cell in cells:
                mask[cell // BLOCK_SIZE, cell % BLOCK_SIZE] = True
            if not _is_connected4(mask):
                continue
            cropped = _crop_mask(mask)
            canon_key = _canonical_shape_key(cropped)
            record = families.setdefault(
                canon_key,
                {
                    "key": canon_key,
                    "area": int(area),
                    "variant_keys": set(),
                },
            )
            for turns in range(4):
                rotated = np.rot90(cropped, turns)
                for variant in (rotated, np.fliplr(rotated)):
                    record["variant_keys"].add(_shape_key(variant))

    out = []
    for key, record in families.items():
        variant_keys = sorted(record["variant_keys"])
        out.append(
            {
                "key": key,
                "area": int(record["area"]),
                "variant_keys": variant_keys,
                "variants": [_mask_from_key(variant_key) for variant_key in variant_keys],
            }
        )
    out.sort(key=lambda family: (family["area"], family["key"]))
    return out


SHAPE_FAMILIES = _enumerate_shape_families()
FAMILIES_BY_AREA: dict[int, list[int]] = defaultdict(list)
ASYMMETRIC_FAMILY_IDS: list[int] = []
for family_id, family in enumerate(SHAPE_FAMILIES):
    FAMILIES_BY_AREA[int(family["area"])].append(family_id)
    if len(family["variant_keys"]) >= 2:
        ASYMMETRIC_FAMILY_IDS.append(family_id)


def _sample_family_plan() -> tuple[list[dict], int]:
    max_unique = min(10, len(SHAPE_FAMILIES) - 3)
    unique_count = int(_RNG.integers(4, max_unique + 1))
    shared_count = int(_RNG.integers(3, min(5, len(SHAPE_FAMILIES) - unique_count) + 1))

    counterexample_area = int(_RNG.choice([area for area, ids in FAMILIES_BY_AREA.items() if len(ids) >= 2]))
    area_pair = list(map(int, _RNG.choice(FAMILIES_BY_AREA[counterexample_area], size=2, replace=False)))
    shared_area_family, unique_area_family = area_pair

    orientation_choices = [family_id for family_id in ASYMMETRIC_FAMILY_IDS if family_id not in area_pair]
    if not orientation_choices:
        return [], counterexample_area
    orientation_family = int(_RNG.choice(orientation_choices))

    used = {shared_area_family, unique_area_family, orientation_family}
    remaining_ids = [family_id for family_id in range(len(SHAPE_FAMILIES)) if family_id not in used]

    if len(remaining_ids) < (shared_count - 2) + (unique_count - 1):
        return [], counterexample_area

    _RNG.shuffle(remaining_ids)
    shared_ids = [orientation_family, shared_area_family]
    shared_ids.extend(remaining_ids[: shared_count - 2])
    unique_start = shared_count - 2
    unique_ids = [unique_area_family]
    unique_ids.extend(remaining_ids[unique_start : unique_start + unique_count - 1])

    shared_sizes = np.full(shared_count, 2, dtype=int)
    remaining_blocks = MACRO_SIDE * MACRO_SIDE - unique_count - int(shared_sizes.sum())
    for _ in range(remaining_blocks):
        shared_sizes[int(_RNG.integers(0, shared_count))] += 1

    groups = []
    for family_id, size in zip(shared_ids, shared_sizes):
        groups.append(
            {
                "family_id": int(family_id),
                "count": int(size),
                "shared": True,
                "force_variant_diversity": int(family_id) == orientation_family,
            }
        )
    for family_id in unique_ids:
        groups.append(
            {
                "family_id": int(family_id),
                "count": 1,
                "shared": False,
                "force_variant_diversity": False,
            }
        )
    return groups, counterexample_area


def _choose_unique_positions(unique_count: int) -> set[tuple[int, int]] | None:
    positions = [(r, c) for r in range(MACRO_SIDE) for c in range(MACRO_SIDE)]
    for _ in range(100):
        pick_ids = list(map(int, _RNG.choice(len(positions), size=unique_count, replace=False)))
        chosen = {positions[idx] for idx in pick_ids}
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if any(r < 3 for r in rows) and any(r >= 3 for r in rows) and any(c < 3 for c in cols) and any(c >= 3 for c in cols):
            return chosen
    return None


def _sample_occurrences(groups: list[dict]) -> tuple[list[dict], Counter]:
    occurrences = []
    planned_freq: Counter = Counter()

    for group in groups:
        family = SHAPE_FAMILIES[int(group["family_id"])]
        variants = family["variants"]
        variant_keys = family["variant_keys"]
        chosen_variant_ids = [int(_RNG.integers(0, len(variants))) for _ in range(int(group["count"]))]
        if group["force_variant_diversity"]:
            distinct_pair = list(map(int, _RNG.choice(len(variants), size=2, replace=False)))
            chosen_variant_ids[0] = distinct_pair[0]
            chosen_variant_ids[1] = distinct_pair[1]
        for variant_id in chosen_variant_ids:
            occurrences.append(
                {
                    "shared": bool(group["shared"]),
                    "family_key": family["key"],
                    "area": int(family["area"]),
                    "variant_key": variant_keys[int(variant_id)],
                    "block": _embed_variant(variants[int(variant_id)]),
                }
            )
            planned_freq[family["key"]] += 1

    _RNG.shuffle(occurrences)
    return occurrences, planned_freq


def _build_grid_from_occurrences(occurrences: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    unique_count = sum(1 for occ in occurrences if not occ["shared"])
    unique_positions = _choose_unique_positions(unique_count)
    if unique_positions is None:
        raise ValueError("failed to spread unique positions")

    shared_occurrences = [occ for occ in occurrences if occ["shared"]]
    unique_occurrences = [occ for occ in occurrences if not occ["shared"]]
    _RNG.shuffle(shared_occurrences)
    _RNG.shuffle(unique_occurrences)

    grid = np.zeros(GRID_SHAPE, dtype=int)
    expected = np.full((MACRO_SIDE, MACRO_SIDE), RED, dtype=int)
    unique_iter = iter(unique_occurrences)
    shared_iter = iter(shared_occurrences)

    for macro_r in range(MACRO_SIDE):
        for macro_c in range(MACRO_SIDE):
            if (macro_r, macro_c) in unique_positions:
                occ = next(unique_iter)
                expected[macro_r, macro_c] = BLACK
            else:
                occ = next(shared_iter)
            r0 = macro_r * BLOCK_SIZE
            c0 = macro_c * BLOCK_SIZE
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = occ["block"]

    return grid, expected


def _validate_generated(grid: np.ndarray, expected: np.ndarray, planned_freq: Counter, counterexample_area: int) -> bool:
    if tuple(map(int, grid.shape)) != GRID_SHAPE:
        return False
    if tuple(map(int, expected.shape)) != (MACRO_SIDE, MACRO_SIDE):
        return False

    actual_freq, area_by_key, variant_by_key, analyzed_output = _analyze_grid(grid)
    if actual_freq != planned_freq:
        return False
    if not np.array_equal(analyzed_output, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    shared_keys = [key for key, count in actual_freq.items() if int(count) >= 2]
    unique_keys = [key for key, count in actual_freq.items() if int(count) == 1]
    if len(shared_keys) < 2 or len(unique_keys) < 3:
        return False
    if not np.any(expected == BLACK) or not np.any(expected == RED):
        return False

    if not any(area_by_key[key] == counterexample_area for key in shared_keys):
        return False
    if not any(area_by_key[key] == counterexample_area for key in unique_keys):
        return False
    if not any(len(variant_by_key[key]) >= 2 for key in shared_keys):
        return False

    inactive = np.argwhere(expected == BLACK)
    if inactive.size == 0:
        return False
    if not np.any(inactive[:, 0] < 3) or not np.any(inactive[:, 0] >= 3):
        return False
    if not np.any(inactive[:, 1] < 3) or not np.any(inactive[:, 1] >= 3):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    groups, counterexample_area = _sample_family_plan()
    if not groups:
        return None
    occurrences, planned_freq = _sample_occurrences(groups)
    if len(occurrences) != MACRO_SIDE * MACRO_SIDE:
        return None
    try:
        grid, expected = _build_grid_from_occurrences(occurrences)
    except ValueError:
        return None
    if not _validate_generated(grid, expected, planned_freq, counterexample_area):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
