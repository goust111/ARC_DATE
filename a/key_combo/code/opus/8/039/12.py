from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
PROTOTYPE_COLOR = 2
CANDIDATE_COLORS = (1, 3, 4, 5, 6, 7)

GRID_ROWS_RANGE = (15, 19)
GRID_COLS_RANGE = (15, 19)
PROTOTYPE_AREA_RANGE = (3, 5)
DISTRACTOR_AREA_RANGE = (3, 6)
MATCH_COUNT_RANGE = (3, 5)
TOTAL_CANDIDATE_RANGE = (5, 6)
PLACEMENT_PADDING = 1
MAX_PATCH_SIDE = 5
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _normalize_shape(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape_to_mask(shape: frozenset[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _enumerate_polyominoes(max_area: int) -> dict[int, list[np.ndarray]]:
    shapes_by_area: dict[int, set[frozenset[tuple[int, int]]]] = {
        1: {frozenset({(0, 0)})}
    }
    for area in range(2, max_area + 1):
        next_shapes: set[frozenset[tuple[int, int]]] = set()
        for shape in shapes_by_area[area - 1]:
            cells = set(shape)
            frontier: set[tuple[int, int]] = set()
            for r, c in shape:
                frontier.update(_neighbors4(r, c))
            for cell in frontier - cells:
                next_shapes.add(_normalize_shape(cells | {cell}))
        shapes_by_area[area] = next_shapes
    return {
        area: [_shape_to_mask(shape) for shape in sorted(shapes, key=lambda s: (len(s), sorted(s)))]
        for area, shapes in shapes_by_area.items()
    }


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    rows = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            rows.append(r)
    reduced = mask[rows, :]

    cols = [0]
    for c in range(1, reduced.shape[1]):
        if not np.array_equal(reduced[:, c], reduced[:, c - 1]):
            cols.append(c)
    return reduced[:, cols]


def _signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _mask_key(_compress_runs(mask))


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_signature(variant) for variant in _dihedral_masks(mask)}


def _concentric_output(matches: list[tuple[int, int]]) -> np.ndarray:
    ranked = sorted(matches, key=lambda item: item[0], reverse=True)
    side = len(ranked)
    out = np.zeros((side, side), dtype=int)
    for layer in range((side + 1) // 2):
        color = int(ranked[layer % len(ranked)][1])
        out[layer : side - layer, layer : side - layer] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify run-compressed candidates against a red prototype, then convert the matching set into concentric square layers.

    Input structure:
    - The background is black, with one small red 4-connected prototype object near the upper-left.
    - The other foreground objects are separated monochrome 4-connected candidates in distinct colors.
    - Some candidates are stretched and/or rotated/reflected versions of the prototype after run compression, while others are not.

    Rule:
    - Crop each object to its tight bbox and collapse adjacent identical rows and identical columns to obtain a run-compressed mask.
    - Compare every candidate mask with the prototype mask up to rotation and reflection.
    - Keep only the matching candidates, let their count be M, sort them by original pixel area from large to small, and build an M x M output.
    - Fill the output from outside to inside with concentric square layers using the sorted matching colors in that order.

    Key insight:
    - Stretching changes raw size but not the compressed topology, so hidden shape equivalence can be detected first and then reinterpreted as size and palette parameters for a concentric-layer output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    prototype_components = _component_coords(g == PROTOTYPE_COLOR)
    assert len(prototype_components) == 1
    prototype_mask = _coords_to_mask(prototype_components[0])
    prototype_signatures = _signature_set(prototype_mask)

    matches: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, PROTOTYPE_COLOR)):
        comps = _component_coords(g == color)
        assert len(comps) == 1
        coords = comps[0]
        if _signature(_coords_to_mask(coords)) in prototype_signatures:
            matches.append((int(coords.shape[0]), int(color)))

    assert matches
    return _concentric_output(matches)


_POLYOMINOES = _enumerate_polyominoes(DISTRACTOR_AREA_RANGE[1])


def _prototype_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for area in range(PROTOTYPE_AREA_RANGE[0], PROTOTYPE_AREA_RANGE[1] + 1):
        for mask in _POLYOMINOES[area]:
            h, w = mask.shape
            if min(h, w) < 2 or max(h, w) > 4:
                continue
            if np.array_equal(_compress_runs(mask), mask) and not (h == 1 or w == 1):
                pool.append(mask.copy())
    return pool


def _distractor_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for area in range(DISTRACTOR_AREA_RANGE[0], DISTRACTOR_AREA_RANGE[1] + 1):
        for mask in _POLYOMINOES[area]:
            h, w = mask.shape
            if min(h, w) < 2 or max(h, w) > 4:
                continue
            pool.append(mask.copy())
    return pool


PROTOTYPE_POOL = _prototype_pool()
DISTRACTOR_POOL = _distractor_pool()
_FACTOR_CACHE: dict[int, list[tuple[int, ...]]] = {}
_MATCH_CACHE: dict[tuple[tuple[int, ...], ...], list[dict[str, object]]] = {}


def _stretch_mask(mask: np.ndarray, row_factors: tuple[int, ...], col_factors: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)


def _factor_options(length: int) -> list[tuple[int, ...]]:
    if length not in _FACTOR_CACHE:
        options = []
        for factors in itertools.product((1, 2, 3), repeat=length):
            if sum(factors) <= MAX_PATCH_SIDE:
                options.append(tuple(int(v) for v in factors))
        _FACTOR_CACHE[length] = options
    return _FACTOR_CACHE[length]


def _match_variant_pool(prototype: np.ndarray) -> list[dict[str, object]]:
    cache_key = _mask_key(prototype)
    if cache_key in _MATCH_CACHE:
        return _MATCH_CACHE[cache_key]
    variants: list[dict[str, object]] = []
    seen_shapes: set[tuple[tuple[int, ...], ...]] = set()
    for oriented in _dihedral_masks(prototype):
        row_options = _factor_options(oriented.shape[0])
        col_options = _factor_options(oriented.shape[1])
        for row_factors in row_options:
            for col_factors in col_options:
                stretched = _stretch_mask(oriented, row_factors, col_factors)
                if max(stretched.shape) > MAX_PATCH_SIDE:
                    continue
                key = _mask_key(stretched)
                if key in seen_shapes:
                    continue
                seen_shapes.add(key)
                variants.append(
                    {
                        "mask": stretched,
                        "area": int(stretched.sum()),
                    }
                )
    _MATCH_CACHE[cache_key] = variants
    return variants


def _sample_distractor_masks(
    prototype_signatures: set[tuple[tuple[int, ...], ...]], count: int
) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    seen_shapes: set[tuple[tuple[int, ...], ...]] = set()
    seen_signatures: set[tuple[tuple[int, ...], ...]] = set()
    for _ in range(400):
        if len(chosen) == count:
            return chosen
        base = random.choice(DISTRACTOR_POOL)
        oriented = random.choice(_dihedral_masks(base))
        row_factors = random.choice(_factor_options(oriented.shape[0]))
        col_factors = random.choice(_factor_options(oriented.shape[1]))
        stretched = _stretch_mask(oriented, row_factors, col_factors)
        if max(stretched.shape) > MAX_PATCH_SIDE:
            continue
        signature = _signature(stretched)
        if signature in prototype_signatures or signature in seen_signatures:
            continue
        key = _mask_key(stretched)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        seen_signatures.add(signature)
        chosen.append(stretched)
    return None


def _candidate_coords(mask: np.ndarray, r0: int, c0: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    return coords[:, 0] + r0, coords[:, 1] + c0


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    top_left_only: bool = False,
) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1 if top_left_only else 0, gh - h + 1):
        for c0 in range(1 if top_left_only else 0, gw - w + 1):
            if top_left_only and (r0 > 4 or c0 > 4):
                continue
            if not top_left_only and r0 < 6 and c0 < 6:
                continue
            rr, cc = _candidate_coords(mask, r0, c0)
            if blocked[rr, cc].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    rr, cc = _candidate_coords(mask, r0, c0)
    grid[rr, cc] = int(color)
    for r, c in zip(rr, cc):
        r1 = max(0, int(r) - PLACEMENT_PADDING)
        r2 = min(grid.shape[0], int(r) + PLACEMENT_PADDING + 1)
        c1 = max(0, int(c) - PLACEMENT_PADDING)
        c2 = min(grid.shape[1], int(c) + PLACEMENT_PADDING + 1)
        blocked[r1:r2, c1:c2] = True
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS_RANGE)
    width = random.randint(*GRID_COLS_RANGE)
    prototype = random.choice(PROTOTYPE_POOL)
    prototype_signatures = _signature_set(prototype)

    match_pool = _match_variant_pool(prototype)
    matches_by_area: dict[int, list[np.ndarray]] = {}
    for item in match_pool:
        area = int(item["area"])
        matches_by_area.setdefault(area, []).append(np.array(item["mask"], dtype=bool))
    if len(matches_by_area) < MATCH_COUNT_RANGE[0]:
        return None

    max_matches = min(MATCH_COUNT_RANGE[1], len(matches_by_area))
    match_count = random.randint(MATCH_COUNT_RANGE[0], max_matches)
    total_min = max(TOTAL_CANDIDATE_RANGE[0], match_count + 1)
    total_max = TOTAL_CANDIDATE_RANGE[1]
    if total_min > total_max:
        return None
    total_candidates = random.randint(total_min, total_max)
    distractor_count = total_candidates - match_count

    distractor_masks = _sample_distractor_masks(prototype_signatures, distractor_count)
    if distractor_masks is None:
        return None

    chosen_areas = random.sample(list(matches_by_area), match_count)
    match_masks = [random.choice(matches_by_area[area]).copy() for area in chosen_areas]
    colors = random.sample(CANDIDATE_COLORS, total_candidates)

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    if not _place_object(grid, blocked, prototype, PROTOTYPE_COLOR, top_left_only=True):
        return None

    objects: list[dict[str, object]] = []
    for mask, color in zip(match_masks, colors[:match_count]):
        objects.append({"mask": mask, "color": int(color), "area": int(mask.sum()), "match": True})
    for mask, color in zip(distractor_masks, colors[match_count:]):
        objects.append({"mask": mask, "color": int(color), "area": int(mask.sum()), "match": False})

    random.shuffle(objects)
    objects.sort(key=lambda item: int(np.prod(np.array(item["mask"]).shape)), reverse=True)
    for obj in objects:
        if not _place_object(grid, blocked, np.array(obj["mask"], dtype=bool), int(obj["color"])):
            return None

    expected_matches = [
        (int(obj["area"]), int(obj["color"])) for obj in objects if bool(obj["match"])
    ]
    expected = _concentric_output(expected_matches)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated placement and oracle retries")
