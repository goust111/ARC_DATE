from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 4
CLASSIFIER_COLOR = 5
SPRITE_PALETTE = (1, 2, 3, 6, 7, 8, 9)
ROTATION_BY_CORNER_COUNT = {0: 0, 1: 3, 2: 2, 4: 1}  # np.rot90 ccw steps

GRID_ROW_RANGE = (18, 27)
GRID_COL_RANGE = (18, 27)
CLASSIFIER_COUNT_OPTIONS = (2, 3, 4)
CLASSIFIER_BOX_SIZE = 3
CLASSIFIER_GAP = 3
INPUT_GAP = 1
STAMP_GAP = 1
MARKER_DISTANCE_RANGE = (4, 15)
MAX_GENERATION_ATTEMPTS = 500

_GEN_CALL_COUNT = 0


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=bool)


def _rect(r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r + h), int(c + w)


def _rect_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif connectivity == 8:
        dirs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    else:
        raise ValueError(f"unsupported connectivity: {connectivity}")

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _crop_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch != BACKGROUND)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        dst[int(top + rr), int(left + cc)] = int(patch[int(rr), int(cc)])


def _rotate_sprite(sprite: np.ndarray, corner_count: int) -> np.ndarray:
    return np.rot90(sprite, k=ROTATION_BY_CORNER_COUNT[int(corner_count)]).copy()


def _nearest_classifier_indices(
    markers: list[tuple[int, int]],
    classifier_centers: list[tuple[int, int]],
) -> list[int]:
    mapping: list[int] = []
    for mr, mc in markers:
        distances = [abs(mr - cr) + abs(mc - cc) for cr, cc in classifier_centers]
        best = min(distances)
        winners = [i for i, dist in enumerate(distances) if dist == best]
        assert len(winners) == 1, "each marker must have a unique nearest classifier"
        mapping.append(winners[0])
    return mapping


def _detect_sprite(grid: np.ndarray) -> np.ndarray:
    sprite_mask = ~np.isin(grid, [BACKGROUND, MARKER_COLOR, CLASSIFIER_COLOR])
    comps = _components(sprite_mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one sprite"
    coords = comps[0]
    assert 4 <= len(coords) <= 8, "sprite area must be 4..8"
    patch = _crop_patch(grid, coords)
    assert len({int(v) for v in patch.ravel() if int(v) != BACKGROUND}) >= 2, "sprite must be multicolor"
    rotations = [np.rot90(patch, k=i) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(rotations[i], rotations[j]), "sprite rotations must be distinguishable"
    return patch


def _detect_classifiers(grid: np.ndarray) -> list[dict]:
    gray_mask = grid == CLASSIFIER_COLOR
    comps = _components(gray_mask, connectivity=8)
    assert 2 <= len(comps) <= 4, "expected 2..4 classifiers"
    classifiers: list[dict] = []
    for coords in comps:
        assert len(coords) == 5, "classifier must contain 5 gray pixels"
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 == 2 and c1 - c0 == 2, "classifier bbox must be 3x3"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert set(map(int, np.unique(patch))) <= {BACKGROUND, CLASSIFIER_COLOR}, "classifier bbox must be black/gray only"
        local = np.zeros((3, 3), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        corner_count = int(local[0, 0]) + int(local[0, 2]) + int(local[2, 0]) + int(local[2, 2])
        assert corner_count in ROTATION_BY_CORNER_COUNT, "corner count must be 0, 1, 2, or 4"
        classifiers.append(
            {
                "top_left": (int(r0), int(c0)),
                "center": (int(r0 + 1), int(c0 + 1)),
                "corner_count": corner_count,
            }
        )
    return classifiers


def _detect_markers(grid: np.ndarray, expected_count: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, xy)) for xy in np.argwhere(grid == MARKER_COLOR)]
    assert len(coords) == expected_count, "marker count must match classifier count"
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of the corner sprite at yellow markers based on nearby gray 3x3 classifiers.

    Input structure:
    - A black background contains one small multicolor 4-connected sprite placed near a corner.
    - The grid also contains 2..4 gray 3x3 classifier patterns, each with exactly 5 gray cells.
    - The number of gray corners occupied in each classifier is always 0, 1, 2, or 4.
    - The grid contains the same number of yellow single-pixel placement markers as classifiers.

    Rule:
    - Crop the multicolor sprite to its tight bounding box.
    - For each gray classifier, count how many of its four corners are gray:
      0 -> keep the sprite, 1 -> rotate it 90 degrees clockwise,
      2 -> rotate it 180 degrees, 4 -> rotate it 270 degrees clockwise.
    - Associate each yellow marker with the classifier whose 3x3 center has the smallest Manhattan distance
      to that marker; ties do not occur.
    - Stamp the associated rotated sprite with the marker as the stamp's top-left corner.
    - Erase the original sprite, all gray classifiers, and all yellow markers, leaving only the stamped copies.

    Key insight:
    - The gray 3x3 pattern is not copied directly: its corner occupancy count encodes a rotation parameter
      that must be transferred to a separate stamp operation chosen by nearest-marker geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert set(map(int, np.unique(g))) <= set(range(10)), "colors must be ARC digits"

    sprite = _detect_sprite(g)
    classifiers = _detect_classifiers(g)
    markers = _detect_markers(g, expected_count=len(classifiers))
    mapping = _nearest_classifier_indices(markers, [entry["center"] for entry in classifiers])

    out = np.zeros_like(g)
    for marker, classifier_index in zip(markers, mapping):
        rotated = _rotate_sprite(sprite, classifiers[classifier_index]["corner_count"])
        top, left = marker
        h, w = rotated.shape
        assert top + h <= g.shape[0] and left + w <= g.shape[1], "stamp must stay in bounds"
        if np.any((out[top : top + h, left : left + w] != BACKGROUND) & (rotated != BACKGROUND)):
            raise AssertionError("stamps must not overlap")
        _paste_nonzero(out, rotated, top, left)
    return out


# --------------- generator ---------------

_BASE_PATTERN_MASKS = {
    0: [_mask_from_rows(".#.", "###", ".#.")],
    1: [
        _mask_from_rows("##.", ".##", ".#."),
        _mask_from_rows("#..", "###", ".#."),
    ],
    2: [
        _mask_from_rows("#.#", "##.", ".#."),
        _mask_from_rows("##.", ".#.", ".##"),
    ],
    4: [_mask_from_rows("#.#", ".#.", "#.#")],
}


def _all_pattern_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    cur = mask
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            key = candidate.astype(np.uint8).tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
        cur = np.rot90(cur)
    return variants


_PATTERN_LIBRARY = {
    count: [variant for base in bases for variant in _all_pattern_variants(base)]
    for count, bases in _BASE_PATTERN_MASKS.items()
}


def _choose_classifier_counts() -> list[int]:
    global _GEN_CALL_COUNT
    presets = [
        [0, 1, 2, 4],
        [0, 1, 4],
        [0, 2, 4],
        [1, 2, 4],
    ]
    if _GEN_CALL_COUNT < len(presets):
        return presets[_GEN_CALL_COUNT][:]
    k = random.choice(CLASSIFIER_COUNT_OPTIONS)
    return random.sample([0, 1, 2, 4], k=k)


def _generate_sprite() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(4, min(8, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        palette = random.sample(SPRITE_PALETTE, k=random.choice((2, 3)))
        assigned = [random.choice(palette) for _ in range(len(coords))]
        if len(set(assigned)) < 2:
            assigned[0] = palette[0]
            assigned[-1] = palette[1]
        for (rr, cc), color in zip(coords.tolist(), assigned):
            patch[int(rr), int(cc)] = int(color)

        rotations = [np.rot90(patch, k=i) for i in range(4)]
        if any(np.array_equal(rotations[i], rotations[j]) for i in range(4) for j in range(i + 1, 4)):
            continue
        return patch
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _place_sprite(grid_shape: tuple[int, int], sprite: np.ndarray) -> tuple[int, int]:
    h, w = grid_shape
    sh, sw = sprite.shape
    corner = random.randrange(4)
    row_pad = random.randint(0, 2)
    col_pad = random.randint(0, 2)
    if corner == 0:
        return row_pad, col_pad
    if corner == 1:
        return row_pad, w - sw - col_pad
    if corner == 2:
        return h - sh - row_pad, col_pad
    return h - sh - row_pad, w - sw - col_pad


def _candidate_classifier_boxes(
    grid_shape: tuple[int, int],
    sprite_box: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - CLASSIFIER_BOX_SIZE + 1):
        for c in range(w - CLASSIFIER_BOX_SIZE + 1):
            box = _rect(r, c, CLASSIFIER_BOX_SIZE, CLASSIFIER_BOX_SIZE)
            if _rect_overlap(box, sprite_box, pad=CLASSIFIER_GAP):
                continue
            if any(_rect_overlap(box, other, pad=CLASSIFIER_GAP) for other in existing):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_classifiers(
    grid_shape: tuple[int, int],
    sprite_box: tuple[int, int, int, int],
    corner_counts: list[int],
) -> list[dict] | None:
    classifiers: list[dict] = []
    counts = corner_counts[:]
    random.shuffle(counts)
    for corner_count in counts:
        candidates = _candidate_classifier_boxes(
            grid_shape,
            sprite_box,
            [entry["box"] for entry in classifiers],
        )
        if not candidates:
            return None
        placed = False
        for top, left in candidates:
            center = (top + 1, left + 1)
            if classifiers and min(abs(center[0] - cr) + abs(center[1] - cc) for cr, cc in [e["center"] for e in classifiers]) < 6:
                continue
            classifiers.append(
                {
                    "top_left": (top, left),
                    "center": center,
                    "corner_count": corner_count,
                    "box": _rect(top, left, 3, 3),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return classifiers


def _marker_candidates(
    grid_shape: tuple[int, int],
    sprite_box: tuple[int, int, int, int],
    classifier_boxes: list[tuple[int, int, int, int]],
    classifier_centers: list[tuple[int, int]],
    target_index: int,
    rotation_extent: tuple[int, int],
    existing_stamp_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = rotation_extent
    candidates: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            stamp_box = _rect(r, c, ph, pw)
            if _rect_overlap(stamp_box, sprite_box, pad=INPUT_GAP):
                continue
            if any(_rect_overlap(stamp_box, box, pad=INPUT_GAP) for box in classifier_boxes):
                continue
            if any(_rect_overlap(stamp_box, box, pad=STAMP_GAP) for box in existing_stamp_boxes):
                continue
            distances = [abs(r - cr) + abs(c - cc) for cr, cc in classifier_centers]
            best = min(distances)
            if distances[target_index] != best or distances.count(best) != 1:
                continue
            if not (MARKER_DISTANCE_RANGE[0] <= best <= MARKER_DISTANCE_RANGE[1]):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _reading_order_mapping(
    markers: list[tuple[int, int]],
    classifiers: list[dict],
) -> dict[int, int]:
    marker_order = sorted(range(len(markers)), key=lambda i: markers[i])
    classifier_order = sorted(range(len(classifiers)), key=lambda i: classifiers[i]["center"])
    return {marker_idx: classifier_idx for marker_idx, classifier_idx in zip(marker_order, classifier_order)}


def _has_alternative_distance_counterexample(
    markers: list[tuple[int, int]],
    classifiers: list[dict],
    true_mapping: list[int],
) -> bool:
    centers = [entry["center"] for entry in classifiers]
    top_lefts = [entry["top_left"] for entry in classifiers]
    for marker_idx, (mr, mc) in enumerate(markers):
        true_idx = true_mapping[marker_idx]

        top_left_distances = [abs(mr - r0) + abs(mc - c0) for r0, c0 in top_lefts]
        top_left_best = min(top_left_distances)
        if top_left_distances.count(top_left_best) == 1 and top_left_distances.index(top_left_best) != true_idx:
            return True

        euclid_distances = [(mr - cr) ** 2 + (mc - cc) ** 2 for cr, cc in centers]
        euclid_best = min(euclid_distances)
        if euclid_distances.count(euclid_best) == 1 and euclid_distances.index(euclid_best) != true_idx:
            return True
    return False


def _expected_from_metadata(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    classifiers: list[dict],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    mapping = _nearest_classifier_indices(markers, [entry["center"] for entry in classifiers])
    for marker, classifier_index in zip(markers, mapping):
        rotated = _rotate_sprite(sprite, classifiers[classifier_index]["corner_count"])
        _paste_nonzero(out, rotated, marker[0], marker[1])
    return out


def _paint_classifier(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    grid[top : top + 3, left : left + 3][pattern] = CLASSIFIER_COLOR


def _semantic_counterfactual_checks(
    expected: np.ndarray,
    sprite: np.ndarray,
    classifiers: list[dict],
    markers: list[tuple[int, int]],
) -> None:
    true_mapping = _nearest_classifier_indices(markers, [entry["center"] for entry in classifiers])
    assert len(set(true_mapping)) == len(classifiers), "each classifier must control exactly one marker"

    distinct_rotations = {
        count: _rotate_sprite(sprite, count)
        for count in {entry["corner_count"] for entry in classifiers}
    }
    for i, entry in enumerate(classifiers):
        other_counts = [count for count in distinct_rotations if count != entry["corner_count"]]
        assert other_counts, "need at least two distinct classifier labels"
        alt_classifiers = [dict(item) for item in classifiers]
        alt_classifiers[i]["corner_count"] = other_counts[0]
        alt_expected = _expected_from_metadata(expected.shape, sprite, alt_classifiers, markers)
        assert not np.array_equal(alt_expected, expected), "every classifier must affect the output"

    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            if classifiers[true_mapping[i]]["corner_count"] == classifiers[true_mapping[j]]["corner_count"]:
                continue
            swapped = [dict(item) for item in classifiers]
            a = swapped[true_mapping[i]]["corner_count"]
            b = swapped[true_mapping[j]]["corner_count"]
            swapped[true_mapping[i]]["corner_count"] = b
            swapped[true_mapping[j]]["corner_count"] = a
            alt_expected = _expected_from_metadata(expected.shape, sprite, swapped, markers)
            assert not np.array_equal(alt_expected, expected), "marker-to-classifier pairing must matter"
            return
    raise AssertionError("need at least one pair of markers with different rotations")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    global _GEN_CALL_COUNT

    grid_shape = (
        random.randint(*GRID_ROW_RANGE),
        random.randint(*GRID_COL_RANGE),
    )
    sprite = _generate_sprite()
    sprite_top, sprite_left = _place_sprite(grid_shape, sprite)
    sprite_box = _rect(sprite_top, sprite_left, sprite.shape[0], sprite.shape[1])

    corner_counts = _choose_classifier_counts()
    classifiers = _place_classifiers(grid_shape, sprite_box, corner_counts)
    if classifiers is None:
        raise ValueError("failed to place classifiers")

    classifier_boxes = [entry["box"] for entry in classifiers]
    classifier_centers = [entry["center"] for entry in classifiers]
    all_rotations = {count: _rotate_sprite(sprite, count) for count in ROTATION_BY_CORNER_COUNT}
    rotation_extent = (
        max(patch.shape[0] for patch in all_rotations.values()),
        max(patch.shape[1] for patch in all_rotations.values()),
    )

    markers: list[tuple[int, int]] = []
    stamp_boxes: list[tuple[int, int, int, int]] = []
    classifier_order = list(range(len(classifiers)))
    random.shuffle(classifier_order)
    for idx in classifier_order:
        candidates = _marker_candidates(
            grid_shape,
            sprite_box,
            classifier_boxes,
            classifier_centers,
            idx,
            rotation_extent,
            stamp_boxes,
        )
        if not candidates:
            raise ValueError("failed to place markers")
        top, left = random.choice(candidates)
        markers.append((top, left))
        stamp_boxes.append(_rect(top, left, rotation_extent[0], rotation_extent[1]))

    true_mapping = _nearest_classifier_indices(markers, classifier_centers)
    if len(set(true_mapping)) != len(classifiers):
        raise ValueError("markers must cover all classifiers exactly once")

    order_mapping = _reading_order_mapping(markers, classifiers)
    has_order_inversion = any(order_mapping[i] != true_mapping[i] for i in range(len(markers)))
    has_distance_counterexample = _has_alternative_distance_counterexample(markers, classifiers, true_mapping)
    require_counterexample = _GEN_CALL_COUNT < 6 or random.random() < 0.7
    if require_counterexample and not (has_order_inversion or has_distance_counterexample):
        raise ValueError("missing shortcut-breaking geometry")

    pattern_cycle = _GEN_CALL_COUNT
    grid = np.zeros(grid_shape, dtype=int)
    _paste_nonzero(grid, sprite, sprite_top, sprite_left)
    for idx, entry in enumerate(classifiers):
        variants = _PATTERN_LIBRARY[entry["corner_count"]]
        pattern = variants[(pattern_cycle + idx) % len(variants)]
        _paint_classifier(grid, entry["top_left"][0], entry["top_left"][1], pattern)
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    expected = _expected_from_metadata(grid_shape, sprite, classifiers, markers)
    _semantic_counterfactual_checks(expected, sprite, classifiers, markers)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("solver output does not match generation metadata")

    _GEN_CALL_COUNT += 1
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _expected = _build_case()
        except (AssertionError, ValueError):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
