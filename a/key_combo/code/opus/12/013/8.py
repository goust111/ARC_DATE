from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8

DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

Shape = tuple[tuple[int, int], ...]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS8 if connectivity == 8 else ((-1, 0), (1, 0), (0, -1), (0, 1))
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray | Shape) -> Shape:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _transform(shape: Shape, rotation: int, flip: bool) -> Shape:
    pts = np.asarray(shape, dtype=int)
    for _ in range(int(rotation) % 4):
        pts = np.column_stack((-pts[:, 1], pts[:, 0]))
    if flip:
        pts[:, 1] = -pts[:, 1]
    return _normalize(pts)


def _all_transforms(shape: Shape) -> list[tuple[int, bool, Shape]]:
    return [
        (rotation, flip, _transform(shape, rotation, flip))
        for rotation in range(4)
        for flip in (False, True)
    ]


def _translate(shape: Shape, r0: int, c0: int) -> frozenset[tuple[int, int]]:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in shape)


def _selected_matches(grid: np.ndarray, template_shape: Shape) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    by_anchor: dict[tuple[int, int], dict] = {}

    for rotation, flip, variant in _all_transforms(template_shape):
        vh, vw = _shape_size(variant)
        for r0 in range(h - vh + 1):
            for c0 in range(w - vw + 1):
                cells = _translate(variant, r0, c0)
                if all(int(grid[r, c]) == GREEN for r, c in cells):
                    key = (r0, c0)
                    candidate = {
                        "r": r0,
                        "c": c0,
                        "rotation": rotation,
                        "flip": flip,
                        "shape": variant,
                        "height": vh,
                        "width": vw,
                        "cells": cells,
                    }
                    if key in by_anchor:
                        prev = by_anchor[key]
                        assert (
                            prev["cells"] == candidate["cells"]
                            and prev["rotation"] == candidate["rotation"]
                            and prev["flip"] == candidate["flip"]
                        ), "template match must be unambiguous at each anchor"
                    else:
                        by_anchor[key] = candidate
                        candidates.append(candidate)

    candidates.sort(key=lambda item: (item["r"], item["c"]))
    occupied: set[tuple[int, int]] = set()
    selected: list[dict] = []
    for candidate in candidates:
        if occupied.isdisjoint(candidate["cells"]):
            occupied.update(candidate["cells"])
            selected.append(candidate)
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each matched green template copy and stamp a synchronized red sprite beneath it.

    Input structure:
    - A black background contains one small teal template component and one small red stamp component.
    - Elsewhere, there are 3-4 separated green components; some are rotated or reflected copies of the teal template and the rest are distractors.
    - The template and stamp are both small 8-connected sprites, and the matched-copy drop zones below them are in bounds.

    Rule:
    - Normalize the teal template by its tight bounding box and enumerate its eight rotation/reflection variants.
    - Scan the grid in row-major order; whenever a variant's foreground cells are all green at some top-left anchor, record that anchor and variant, skipping later matches that overlap already selected matches.
    - For every selected match, recolor its green template cells to teal, apply the same rotation/reflection to the red stamp sprite, and place that transformed stamp directly below the matched bounding box with one blank row of spacing.
    - Leave unmatched green distractors, the original teal template, and the original red stamp unchanged.

    Key insight:
    - The important information is not only where the green template copies are, but which specific dihedral transform each copy uses, because that same transform controls the orientation of the stamped red sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    teal_components = _components(g == TEAL, connectivity=8)
    red_components = _components(g == RED, connectivity=8)
    assert len(teal_components) == 1, "expected exactly one teal template"
    assert len(red_components) == 1, "expected exactly one red stamp sprite"

    tr0, tc0, _tr1, _tc1 = _bbox_coords(teal_components[0])
    sr0, sc0, _sr1, _sc1 = _bbox_coords(red_components[0])
    template_shape = _normalize(teal_components[0] - np.array([tr0, tc0]))
    stamp_shape = _normalize(red_components[0] - np.array([sr0, sc0]))

    matches = _selected_matches(g, template_shape)
    out = g.copy()
    for match in matches:
        for r, c in match["cells"]:
            out[r, c] = TEAL

        transformed_stamp = _transform(stamp_shape, match["rotation"], match["flip"])
        stamp_r0 = match["r"] + match["height"] + 1
        stamp_c0 = match["c"]
        for dr, dc in transformed_stamp:
            rr = stamp_r0 + dr
            cc = stamp_c0 + dc
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            out[rr, cc] = RED
    return out


def _family_key(shape: Shape) -> Shape:
    return min(_transform(shape, rotation, flip) for rotation in range(4) for flip in (False, True))


def _is_connected(shape: Shape) -> bool:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return len(_components(mask, connectivity=8)) == 1


def _enumerate_shape_families() -> dict[int, list[Shape]]:
    families: dict[int, dict[Shape, Shape]] = {3: {}, 4: {}, 5: {}}
    for size in (3, 4, 5):
        for combo in combinations(range(9), size):
            coords = [(idx // 3, idx % 3) for idx in combo]
            shape = _normalize(coords)
            h, w = _shape_size(shape)
            if h > 3 or w > 3 or not _is_connected(shape):
                continue
            families[size].setdefault(_family_key(shape), shape)
    return {size: list(items.values()) for size, items in families.items()}


SHAPE_FAMILIES = _enumerate_shape_families()
ASYMMETRIC_FAMILIES = [
    shape
    for size in (4, 5)
    for shape in SHAPE_FAMILIES[size]
    if len({variant for _, _, variant in _all_transforms(shape)}) == 8
]


def _random_display_shape(family_shape: Shape) -> Shape:
    rotation = random.randrange(4)
    flip = bool(random.randrange(2))
    return _transform(family_shape, rotation, flip)


def _sample_match_transforms(count: int) -> list[tuple[int, bool]]:
    all_ids = [(rotation, flip) for rotation in range(4) for flip in (False, True)]
    if count <= 1:
        return [random.choice(all_ids)]
    base = [(random.randrange(4), False), (random.randrange(4), True)]
    remaining = [item for item in all_ids if item not in base]
    random.shuffle(remaining)
    return base + remaining[: count - 2]


def _reserve_bbox_cells(
    r0: int, c0: int, r1: int, c1: int, height: int, width: int, pad: int = 1
) -> set[tuple[int, int]]:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _try_place_match(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    template_shape: Shape,
    stamp_shape: Shape,
    rotation: int,
    flip: bool,
) -> dict | None:
    h, w = grid.shape
    variant = _transform(template_shape, rotation, flip)
    stamp_variant = _transform(stamp_shape, rotation, flip)
    vh, vw = _shape_size(variant)
    sh, sw = _shape_size(stamp_variant)
    max_r = h - (vh + 1 + sh)
    max_c = w - max(vw, sw)
    if max_r < 0 or max_c < 0:
        return None

    anchors = [(r0, c0) for r0 in range(max_r + 1) for c0 in range(max_c + 1)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        match_cells = _translate(variant, r0, c0)
        stamp_r0 = r0 + vh + 1
        stamp_cells = _translate(stamp_variant, stamp_r0, c0)
        all_cells = match_cells | stamp_cells
        if any(cell in occupied for cell in all_cells):
            continue
        r1 = stamp_r0 + sh - 1
        c1 = c0 + max(vw, sw) - 1
        reserve = _reserve_bbox_cells(r0, c0, r1, c1, h, w, pad=1)
        if any(cell in occupied for cell in reserve):
            continue
        for r, c in match_cells:
            grid[r, c] = GREEN
        occupied.update(reserve)
        return {
            "r": r0,
            "c": c0,
            "rotation": rotation,
            "flip": flip,
            "shape": variant,
            "height": vh,
            "width": vw,
            "cells": match_cells,
            "stamp_cells": stamp_cells,
        }
    return None


def _try_place_shape(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    shape: Shape,
    color: int,
) -> dict | None:
    h, w = grid.shape
    sh, sw = _shape_size(shape)
    anchors = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        cells = _translate(shape, r0, c0)
        reserve = _reserve_bbox_cells(r0, c0, r0 + sh - 1, c0 + sw - 1, h, w, pad=1)
        if any(cell in occupied for cell in reserve):
            continue
        for r, c in cells:
            grid[r, c] = color
        occupied.update(reserve)
        return {"r": r0, "c": c0, "cells": cells, "shape": shape}
    return None


def _build_expected_output(
    grid: np.ndarray,
    matches: list[dict],
    stamp_shape: Shape,
) -> np.ndarray:
    out = grid.copy()
    for match in matches:
        for r, c in match["cells"]:
            out[r, c] = TEAL
        transformed_stamp = _transform(stamp_shape, match["rotation"], match["flip"])
        stamp_r0 = match["r"] + match["height"] + 1
        for dr, dc in transformed_stamp:
            out[stamp_r0 + dr, match["c"] + dc] = RED
    return out


def _validate_instance(
    grid: np.ndarray,
    expected_matches: list[dict],
    stamp_shape: Shape,
    total_groups: int,
) -> bool:
    green_components = _components(grid == GREEN, connectivity=8)
    if len(green_components) != total_groups:
        return False

    teal_components = _components(grid == TEAL, connectivity=8)
    red_components = _components(grid == RED, connectivity=8)
    if len(teal_components) != 1 or len(red_components) != 1:
        return False

    tr0, tc0, _tr1, _tc1 = _bbox_coords(teal_components[0])
    template_shape = _normalize(teal_components[0] - np.array([tr0, tc0]))
    observed_matches = _selected_matches(grid, template_shape)
    expected_pairs = sorted(
        ((m["r"], m["c"], m["rotation"], m["flip"], tuple(sorted(m["cells"]))) for m in expected_matches)
    )
    observed_pairs = sorted(
        ((m["r"], m["c"], m["rotation"], m["flip"], tuple(sorted(m["cells"]))) for m in observed_matches)
    )
    if expected_pairs != observed_pairs:
        return False

    expected_output = _build_expected_output(grid, expected_matches, stamp_shape)
    return np.array_equal(main(grid), expected_output)


def _random_distractor_shape(template_shape: Shape, used_same_size: bool) -> Shape:
    template_size = len(template_shape)
    template_key = _family_key(template_shape)
    if not used_same_size:
        same_size_pool = [
            shape
            for shape in SHAPE_FAMILIES[template_size]
            if _family_key(shape) != template_key
        ]
        if same_size_pool:
            return _random_display_shape(random.choice(same_size_pool))

    smaller_sizes = [size for size in (3, 4) if size <= template_size]
    pool = [shape for size in smaller_sizes for shape in SHAPE_FAMILIES[size]]
    return _random_display_shape(random.choice(pool))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 23)
    w = random.randint(18, 23)
    grid = np.full((h, w), BLACK, dtype=int)

    template_family = random.choice(ASYMMETRIC_FAMILIES)
    template_key = _family_key(template_family)
    stamp_candidates = [shape for shape in ASYMMETRIC_FAMILIES if _family_key(shape) != template_key]
    stamp_family = random.choice(stamp_candidates)
    template_shape = _random_display_shape(template_family)
    stamp_shape = _random_display_shape(stamp_family)

    total_groups = random.choice((3, 4))
    if total_groups == 3:
        match_count, distractor_count = random.choices([(2, 1), (1, 2)], weights=[3, 1], k=1)[0]
    else:
        match_count, distractor_count = random.choice(((3, 1), (2, 2)))

    occupied: set[tuple[int, int]] = set()
    matches: list[dict] = []
    for rotation, flip in _sample_match_transforms(match_count):
        placed = _try_place_match(grid, occupied, template_shape, stamp_shape, rotation, flip)
        if placed is None:
            return None
        matches.append(placed)

    used_same_size_distractor = False
    for _ in range(distractor_count):
        distractor = _random_distractor_shape(template_shape, used_same_size_distractor)
        if len(distractor) == len(template_shape) and _family_key(distractor) != _family_key(template_shape):
            used_same_size_distractor = True
        placed = _try_place_shape(grid, occupied, distractor, GREEN)
        if placed is None:
            return None

    if not used_same_size_distractor:
        return None

    if _try_place_shape(grid, occupied, template_shape, TEAL) is None:
        return None
    if _try_place_shape(grid, occupied, stamp_shape, RED) is None:
        return None

    if not _validate_instance(grid, matches, stamp_shape, total_groups):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-stamp synchronization instance")
