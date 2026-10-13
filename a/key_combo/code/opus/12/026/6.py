from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_GRID_RANGE = (17, 23)
_GEN_TOTAL_CANDIDATES = (4, 6)
_GEN_MATCH_RANGE = (2, 4)
_GEN_MAX_ATTEMPTS = 400
_GEN_HALO = 2
_GEN_MAX_BBOX = 6
_GEN_MAX_AREA = 12


@dataclass(frozen=True)
class _PlacedObject:
    kind: str
    mask: np.ndarray
    top: int
    left: int
    grew: bool = False


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _dedupe_adjacent_rows(mask: np.ndarray) -> np.ndarray:
    keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            keep.append(r)
    return mask[keep, :]


def _dedupe_adjacent_cols(mask: np.ndarray) -> np.ndarray:
    keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            keep.append(c)
    return mask[:, keep]


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    while True:
        reduced = _dedupe_adjacent_cols(_dedupe_adjacent_rows(out))
        if reduced.shape == out.shape and np.array_equal(reduced, out):
            return reduced
        out = reduced


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
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


def _signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(variant) for variant in _mask_variants(mask)}


def _paint_outline(out: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if (r in (er0, er1) or c in (ec0, ec1)) and out[r, c] == BLACK:
                out[r, c] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor and outline the candidates whose run-compressed shape matches the red prototype, and erase the others.

    Input structure
    - A black background grid contains exactly one red prototype object and several blue candidate objects.
    - Every foreground object is 4-connected, objects are mutually separated, and the prototype is not adjacent to any candidate.

    Rule
    - Crop the red prototype to its bbox and run-compress it by repeatedly merging adjacent identical rows and columns.
    - For each blue candidate, crop to its bbox, run-compress the mask the same way, and compare it to the prototype under rotation/reflection.
    - If a candidate matches, recolor its original blue pixels to red and draw a green border on the perimeter of its original bbox expanded by one cell, only where the grid is black.
    - If a candidate does not match, erase its blue pixels to black.
    - Leave the original prototype unchanged.

    Key insight
    - Matching is decided only after removing repeated row/column runs, so stretched copies of the prototype are accepted while unrelated shapes are deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    prototype_components = _connected_components(g == RED)
    candidate_components = _connected_components(g == BLUE)
    assert len(prototype_components) == 1
    assert len(candidate_components) >= 1

    prototype_mask = _coords_to_mask(prototype_components[0])
    prototype_compressed = _compress_runs(prototype_mask)
    prototype_signatures = _signature_set(prototype_compressed)

    out = g.copy()
    for coords in candidate_components:
        candidate_mask = _coords_to_mask(coords)
        compressed = _compress_runs(candidate_mask)
        if _mask_key(compressed) in prototype_signatures:
            out[coords[:, 0], coords[:, 1]] = RED
            _paint_outline(out, _bbox_coords(coords))
        else:
            out[coords[:, 0], coords[:, 1]] = BLACK
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _touches_bbox_edges(mask: np.ndarray) -> bool:
    return mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()


def _enumerate_base_shapes() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen = set()
    for h in range(2, 5):
        for w in range(2, 5):
            cells = h * w
            for area in range(3, min(6, cells) + 1):
                for combo in itertools.combinations(range(cells), area):
                    mask = np.zeros((h, w), dtype=bool)
                    flat = mask.reshape(-1)
                    flat[list(combo)] = True
                    if not _touches_bbox_edges(mask):
                        continue
                    if not _is_connected(mask):
                        continue
                    if not np.array_equal(_compress_runs(mask), mask):
                        continue
                    canon = min(_mask_key(v) for v in _mask_variants(mask))
                    if canon in seen:
                        continue
                    seen.add(canon)
                    pool.append(mask.copy())
    assert pool
    return pool


_BASE_SHAPES = _enumerate_base_shapes()


def _stretch_options(mask: np.ndarray, require_growth: bool) -> list[tuple[np.ndarray, bool]]:
    options: list[tuple[np.ndarray, bool]] = []
    seen = set()
    for row_factors in itertools.product((1, 2, 3), repeat=mask.shape[0]):
        for col_factors in itertools.product((1, 2, 3), repeat=mask.shape[1]):
            grew = max(row_factors) > 1 or max(col_factors) > 1
            if require_growth and not grew:
                continue
            stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
            if stretched.shape[0] > _GEN_MAX_BBOX or stretched.shape[1] > _GEN_MAX_BBOX:
                continue
            if int(stretched.sum()) > _GEN_MAX_AREA:
                continue
            key = _mask_key(stretched)
            if key in seen:
                continue
            seen.add(key)
            options.append((stretched.copy(), grew))
    return options


def _random_match_mask(rng: np.random.Generator, prototype: np.ndarray, require_growth: bool) -> tuple[np.ndarray, bool] | None:
    prototype_compressed = _compress_runs(prototype)
    orientations = _mask_variants(prototype_compressed)
    rng.shuffle(orientations)
    for oriented in orientations:
        options = _stretch_options(oriented, require_growth=require_growth)
        if not options:
            continue
        idx = int(rng.integers(0, len(options)))
        return options[idx]
    return None


def _random_distractor_mask(
    rng: np.random.Generator,
    prototype_signatures: set[tuple[tuple[int, ...], ...]],
) -> tuple[np.ndarray, bool] | None:
    order = list(range(len(_BASE_SHAPES)))
    rng.shuffle(order)
    for base_idx in order:
        base = _BASE_SHAPES[base_idx]
        orientations = _mask_variants(base)
        rng.shuffle(orientations)
        for oriented in orientations:
            require_growth = bool(rng.integers(0, 2))
            options = _stretch_options(oriented, require_growth=require_growth)
            rng.shuffle(options)
            for stretched, grew in options:
                if _mask_key(_compress_runs(stretched)) not in prototype_signatures:
                    return stretched, grew
    return None


def _candidate_counts(rng: np.random.Generator) -> tuple[int, int]:
    total = int(rng.integers(_GEN_TOTAL_CANDIDATES[0], _GEN_TOTAL_CANDIDATES[1] + 1))
    match_lo = _GEN_MATCH_RANGE[0]
    match_hi = min(_GEN_MATCH_RANGE[1], total - 1)
    match_count = int(rng.integers(match_lo, match_hi + 1))
    distractor_count = total - match_count
    return match_count, distractor_count


def _place_objects(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    objects: list[tuple[str, np.ndarray, int, bool]],
) -> tuple[np.ndarray, list[_PlacedObject]] | None:
    h, w = grid_shape
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[_PlacedObject | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: objects[i][1].shape[0] * objects[i][1].shape[1], reverse=True)

    for obj_idx in order:
        kind, mask, color, grew = objects[obj_idx]
        mh, mw = mask.shape
        positions = []
        for r in range(h - mh + 1):
            for c in range(w - mw + 1):
                if not blocked[r : r + mh, c : c + mw].any():
                    positions.append((r, c))
        if not positions:
            return None
        r, c = positions[int(rng.integers(0, len(positions)))]
        patch = grid[r : r + mh, c : c + mw]
        patch[mask] = color
        blocked[max(0, r - _GEN_HALO) : min(h, r + mh + _GEN_HALO), max(0, c - _GEN_HALO) : min(w, c + mw + _GEN_HALO)] = True
        placed[obj_idx] = _PlacedObject(kind=kind, mask=mask.copy(), top=int(r), left=int(c), grew=bool(grew))

    return grid, [obj for obj in placed if obj is not None]


def _render_expected(inp: np.ndarray, placed: list[_PlacedObject]) -> np.ndarray:
    out = inp.copy()
    for obj in placed:
        coords = np.argwhere(obj.mask) + np.array([obj.top, obj.left])
        if obj.kind == "match":
            out[coords[:, 0], coords[:, 1]] = RED
            _paint_outline(out, _bbox_coords(coords))
        elif obj.kind == "distractor":
            out[coords[:, 0], coords[:, 1]] = BLACK
    return out


def _scene_ok(inp: np.ndarray, expected: np.ndarray, placed: list[_PlacedObject]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    prototype_objects = [obj for obj in placed if obj.kind == "prototype"]
    matched_objects = [obj for obj in placed if obj.kind == "match"]
    distractor_objects = [obj for obj in placed if obj.kind == "distractor"]
    if len(prototype_objects) != 1 or len(matched_objects) < 2 or len(distractor_objects) < 1:
        return False
    if not any(obj.grew for obj in matched_objects):
        return False

    prototype_mask = prototype_objects[0].mask
    prototype_signatures = _signature_set(_compress_runs(prototype_mask))
    raw_prototype_signatures = _signature_set(prototype_mask)
    compression_needed = False

    blue_components = _connected_components(inp == BLUE)
    red_components = _connected_components(inp == RED)
    if len(red_components) != 1 or not (4 <= len(blue_components) <= 6):
        return False

    measured_match_count = 0
    measured_distractor_count = 0
    for coords in blue_components:
        candidate_mask = _coords_to_mask(coords)
        compressed = _compress_runs(candidate_mask)
        if _mask_key(compressed) in prototype_signatures:
            measured_match_count += 1
            if _mask_key(candidate_mask) not in raw_prototype_signatures:
                compression_needed = True
        else:
            measured_distractor_count += 1
    if measured_match_count < 2 or measured_distractor_count < 1 or not compression_needed:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    if np.count_nonzero(expected == GREEN) == 0:
        return False
    if np.any(expected == BLUE):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(_GEN_GRID_RANGE[0], _GEN_GRID_RANGE[1]))
    w = int(rng.integers(_GEN_GRID_RANGE[0], _GEN_GRID_RANGE[1]))
    prototype = _BASE_SHAPES[int(rng.integers(0, len(_BASE_SHAPES)))].copy()
    prototype_signatures = _signature_set(_compress_runs(prototype))
    match_count, distractor_count = _candidate_counts(rng)

    objects: list[tuple[str, np.ndarray, int, bool]] = [("prototype", prototype, RED, False)]
    forced_growth_index = int(rng.integers(0, match_count))
    for idx in range(match_count):
        require_growth = idx == forced_growth_index or bool(rng.integers(0, 2))
        sample = _random_match_mask(rng, prototype, require_growth=require_growth)
        if sample is None:
            return None
        mask, grew = sample
        if require_growth and not grew:
            return None
        objects.append(("match", mask, BLUE, grew))

    for _ in range(distractor_count):
        sample = _random_distractor_mask(rng, prototype_signatures)
        if sample is None:
            return None
        mask, grew = sample
        objects.append(("distractor", mask, BLUE, grew))

    placed_scene = _place_objects(rng, (h, w), objects)
    if placed_scene is None:
        return None
    inp, placed = placed_scene
    expected = _render_expected(inp, placed)
    if not _scene_ok(inp, expected, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
