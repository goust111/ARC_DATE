from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    packed = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
    return tuple(map(int, mask.shape)), packed


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    _, w = grid.shape
    candidates = [
        c
        for c in range(1, w - 1)
        if np.all(grid[:, c] == bg)
        and np.any(grid[:, :c] != bg)
        and np.any(grid[:, c + 1 :] != bg)
    ]
    mid = w // 2
    if w % 2 == 1 and mid in candidates:
        return int(mid)
    assert len(candidates) == 1, "expected one panel separator column"
    return int(candidates[0])


def _extract_holes(left: np.ndarray, bg: int) -> list[dict]:
    holes = []
    for coords in _components_from_mask(left == bg):
        if _touches_border(coords, left.shape):
            continue
        box = _bbox_coords(coords)
        mask = _crop_bbox(left == bg, box)
        holes.append({"coords": coords, "bbox": box, "mask": mask})
    assert len(holes) >= 2, "expected at least two enclosed holes"
    return holes


def _extract_objects(right: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for coords in _components_from_mask(right != bg):
        box = _bbox_coords(coords)
        patch = _crop_bbox(right, box)
        mask = patch != bg
        colors, counts = np.unique(patch[mask], return_counts=True)
        assert len(colors) == 2, "each object must use one main color plus one marker color"
        singletons = colors[counts == 1]
        assert len(singletons) == 1, "each object must have exactly one marker pixel"
        marker_color = int(singletons[0])
        main_color = int(colors[colors != marker_color][0])
        objects.append(
            {
                "coords": coords,
                "bbox": box,
                "patch": patch,
                "mask": mask,
                "main_color": main_color,
                "marker_color": marker_color,
            }
        )
    assert len(objects) >= 2, "expected at least two right-panel objects"
    return objects


def _match_edges(holes: list[dict], objects: list[dict]) -> list[list[tuple[int, int]]]:
    edges: list[list[tuple[int, int]]] = [[] for _ in holes]
    scaled_masks = [
        {factor: _scale_sprite(obj["mask"].astype(int), factor).astype(bool) for factor in (1, 2)}
        for obj in objects
    ]
    for h_idx, hole in enumerate(holes):
        for o_idx, obj_masks in enumerate(scaled_masks):
            for factor in (1, 2):
                scaled_mask = obj_masks[factor]
                if scaled_mask.shape == hole["mask"].shape and np.array_equal(scaled_mask, hole["mask"]):
                    edges[h_idx].append((o_idx, factor))
    return edges


def _assignment_solutions(
    holes: list[dict],
    objects: list[dict],
    *,
    limit: int = 2,
) -> list[tuple[dict[int, tuple[int, int]], int, int]]:
    n_holes = len(holes)
    n_objects = len(objects)
    assert n_holes == n_objects, "the puzzle requires one leftover hole and one leftover object"
    edges = _match_edges(holes, objects)
    order = sorted(range(n_holes), key=lambda idx: len(edges[idx]))
    solutions: list[tuple[dict[int, tuple[int, int]], int, int]] = []

    def rec(
        pos: int,
        used_objects: set[int],
        skipped_hole: int | None,
        assignment: dict[int, tuple[int, int]],
    ) -> None:
        if len(solutions) >= limit:
            return
        if pos == n_holes:
            if skipped_hole is None or len(used_objects) != n_objects - 1:
                return
            unused = [idx for idx in range(n_objects) if idx not in used_objects]
            if len(unused) == 1:
                solutions.append((assignment.copy(), skipped_hole, int(unused[0])))
            return

        remaining_holes = n_holes - pos
        if len(used_objects) + remaining_holes < n_objects - 1:
            return

        hole_idx = order[pos]
        for object_idx, factor in edges[hole_idx]:
            if object_idx in used_objects:
                continue
            used_objects.add(object_idx)
            assignment[hole_idx] = (object_idx, factor)
            rec(pos + 1, used_objects, skipped_hole, assignment)
            del assignment[hole_idx]
            used_objects.remove(object_idx)

        if skipped_hole is None:
            rec(pos + 1, used_objects, hole_idx, assignment)

    rec(0, set(), None, {})
    return solutions


def _solve_instance(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], dict[int, tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _find_separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    left_colors = sorted(set(map(int, np.unique(left))) - {bg})
    assert len(left_colors) == 1, "left panel must be a single-color template on the background"

    holes = _extract_holes(left, bg)
    objects = _extract_objects(right, bg)

    solutions = _assignment_solutions(holes, objects, limit=2)
    assert len(solutions) == 1, "shape pairing must be uniquely determined"
    assignment, leftover_hole, leftover_object = solutions[0]

    out = left.copy()
    for hole_idx, (object_idx, factor) in assignment.items():
        hole = holes[hole_idx]
        obj = objects[object_idx]
        scaled_patch = _scale_sprite(obj["patch"], factor)
        box = hole["bbox"]
        assert scaled_patch.shape == (box[2] - box[0] + 1, box[3] - box[1] + 1)
        assert np.array_equal(scaled_patch != bg, hole["mask"])
        _paste(out, scaled_patch, box[0], box[1], transparent=bg)

    coords = holes[leftover_hole]["coords"]
    out[coords[:, 0], coords[:, 1]] = objects[leftover_object]["main_color"]
    return out, holes, objects, assignment, leftover_hole, leftover_object


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each matching right-side object into the left template hole at its required integer scale, then fill the single leftover hole with the unused object's main color.

    Input structure:
    - The grid is split into left and right panels by a blank separator column on the shared background color.
    - The left panel contains one solid template color with several disconnected enclosed background holes.
    - The right panel contains disconnected small objects; each object is mostly one color and has exactly one differently colored marker pixel.

    Rule:
    - For every left-panel hole whose mask exactly matches one right-panel object mask scaled by factor 1 or 2, stamp the whole colored object into that hole at that factor, so the marker grows into a same-factor square block.
    - Exactly one hole matches no object and exactly one object is unused; fill that leftover hole uniformly with the unused object's main color.
    - Return only the transformed left panel.

    Key insight:
    - Matching is determined by precise cross-scale mask equality rather than same-size comparison, and the final hole has no local shape evidence at all: it is solved only by global elimination after every other object is used once.
    """
    out, _, _, _, _, _ = _solve_instance(grid)
    return out


MASK_CATALOG = []


def _build_mask_catalog() -> list[np.ndarray]:
    base_masks = [
        np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [0, 1, 0, 1]], dtype=bool),
    ]

    catalog: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mask in base_masks:
        variants = []
        for k in range(4):
            rot = np.rot90(mask, k)
            variants.append(rot)
            variants.append(np.fliplr(rot))
        for variant in variants:
            key = _mask_key(variant)
            if key in seen:
                continue
            seen.add(key)
            catalog.append(variant.copy())
    return catalog


MASK_CATALOG = _build_mask_catalog()
MASKS_BY_SHAPE: dict[tuple[int, int], list[np.ndarray]] = {}
for _mask in MASK_CATALOG:
    MASKS_BY_SHAPE.setdefault(tuple(map(int, _mask.shape)), []).append(_mask)


def _rectangles_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_rectangles(
    container_h: int,
    container_w: int,
    sizes: list[tuple[int, int]],
    rng: np.random.Generator,
    *,
    border: int,
    gap: int,
) -> list[tuple[int, int]] | None:
    indices = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placements: dict[int, tuple[int, int, int, int]] = {}

    def rec(pos: int) -> bool:
        if pos == len(indices):
            return True
        idx = indices[pos]
        h, w = sizes[idx]
        candidates = [
            (r, c)
            for r in range(border, container_h - h - border + 1)
            for c in range(border, container_w - w - border + 1)
        ]
        if not candidates:
            return False
        order = rng.permutation(len(candidates))
        for cand_idx in order:
            r, c = candidates[int(cand_idx)]
            rect = (r, c, r + h - 1, c + w - 1)
            if all(_rectangles_separated(rect, other, gap) for other in placements.values()):
                placements[idx] = rect
                if rec(pos + 1):
                    return True
                del placements[idx]
        return False

    if not rec(0):
        return None
    return [(placements[idx][0], placements[idx][1]) for idx in range(len(sizes))]


def _choose_distinct_masks(
    rng: np.random.Generator,
    n_objects: int,
) -> list[np.ndarray] | None:
    object_indices = rng.choice(len(MASK_CATALOG), size=n_objects, replace=False)
    masks = [MASK_CATALOG[int(idx)] for idx in object_indices]
    scaled_keys = [_mask_key(_scale_sprite(mask.astype(int), factor).astype(bool)) for mask in masks for factor in (1, 2)]
    if len(set(scaled_keys)) != len(scaled_keys):
        return None
    return [mask.copy() for mask in masks]


def _choose_shape_set(
    rng: np.random.Generator,
    n_objects: int,
) -> tuple[list[np.ndarray], list[int], np.ndarray, int] | None:
    n_matched = n_objects - 1
    for _ in range(200):
        object_masks = _choose_distinct_masks(rng, n_objects)
        if object_masks is None:
            continue

        factors = [1, 2]
        if n_matched == 3:
            factors.append(int(rng.choice([1, 2])))
        factors = [int(x) for x in factors]
        rng.shuffle(factors)

        used_scaled = {
            _mask_key(_scale_sprite(mask.astype(int), factor).astype(bool))
            for mask in object_masks
            for factor in (1, 2)
        }
        target_idx = int(rng.integers(0, n_matched))
        target_shape = tuple(map(int, object_masks[target_idx].shape))
        target_factor = factors[target_idx]
        candidates = [
            mask
            for mask in MASKS_BY_SHAPE[target_shape]
            if _mask_key(mask) != _mask_key(object_masks[target_idx])
            and _mask_key(_scale_sprite(mask.astype(int), target_factor).astype(bool)) not in used_scaled
        ]
        if not candidates:
            continue

        unmatched_hole_mask = candidates[int(rng.integers(0, len(candidates)))].copy()
        return object_masks, factors, unmatched_hole_mask, target_factor
    return None


def _make_object_patch(
    mask: np.ndarray,
    main_color: int,
    marker_color: int,
    rng: np.random.Generator,
) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(main_color)
    coords = np.argwhere(mask)
    marker_idx = int(rng.integers(0, len(coords)))
    mr, mc = coords[marker_idx]
    patch[mr, mc] = int(marker_color)
    return patch


def _compose_expected_left(
    left: np.ndarray,
    holes: list[dict],
    objects: list[dict],
    assignment: dict[int, tuple[int, int]],
    leftover_hole: int,
    leftover_object: int,
    bg: int,
) -> np.ndarray:
    out = left.copy()
    for hole_idx, (object_idx, factor) in assignment.items():
        box = holes[hole_idx]["bbox"]
        scaled_patch = _scale_sprite(objects[object_idx]["patch"], factor)
        _paste(out, scaled_patch, box[0], box[1], transparent=bg)
    coords = holes[leftover_hole]["coords"]
    out[coords[:, 0], coords[:, 1]] = int(objects[leftover_object]["main_color"])
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual, holes, objects, assignment, leftover_hole, leftover_object = _solve_instance(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(holes) != len(objects):
        return False

    factors_used = {factor for _, factor in assignment.values()}
    if factors_used != {1, 2} and not ({1, 2} <= factors_used):
        return False

    hole_shapes = [tuple(map(int, hole["mask"].shape)) for hole in holes]
    if len(set(hole_shapes)) == len(hole_shapes):
        return False

    object_color_counts = []
    for obj in objects:
        fg = obj["patch"][obj["patch"] != 0]
        object_color_counts.append(Counter(map(int, fg)))
    for counts in object_color_counts:
        if len(counts) != 2 or sorted(counts.values()) != [1, sum(counts.values()) - 1]:
            return False

    if leftover_hole in assignment:
        return False
    if any(object_idx == leftover_object for object_idx, _ in assignment.values()):
        return False

    return True


def _generate_candidate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    n_objects = int(rng.choice([3, 4]))
    shape_set = _choose_shape_set(rng, n_objects)
    if shape_set is None:
        return None
    object_masks, matched_factors, unmatched_hole_mask, unmatched_hole_factor = shape_set

    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    template_color = palette[0]
    main_colors = palette[1 : 1 + n_objects]
    marker_colors = palette[1 + n_objects : 1 + 2 * n_objects]

    object_records = []
    for idx, mask in enumerate(object_masks):
        patch = _make_object_patch(mask, main_colors[idx], marker_colors[idx], rng)
        object_records.append(
            {
                "id": idx,
                "mask": mask.copy(),
                "patch": patch,
                "main_color": int(main_colors[idx]),
                "matched": idx < n_objects - 1,
                "factor": int(matched_factors[idx]) if idx < n_objects - 1 else None,
            }
        )

    hole_records = []
    for idx in range(n_objects - 1):
        factor = int(object_records[idx]["factor"])
        hole_mask = _scale_sprite(object_records[idx]["mask"].astype(int), factor).astype(bool)
        hole_records.append({"match_object_id": idx, "mask": hole_mask, "factor": factor})
    hole_records.append(
        {
            "match_object_id": None,
            "mask": _scale_sprite(unmatched_hole_mask.astype(int), unmatched_hole_factor).astype(bool),
            "factor": int(unmatched_hole_factor),
        }
    )

    hole_shapes = [tuple(map(int, rec["mask"].shape)) for rec in hole_records]
    if len(set(hole_shapes)) == len(hole_shapes):
        return None

    hole_order = list(rng.permutation(len(hole_records)))
    object_order = list(rng.permutation(len(object_records)))
    hole_records = [hole_records[idx] for idx in hole_order]
    object_records = [object_records[idx] for idx in object_order]

    hole_sizes = [tuple(map(int, rec["mask"].shape)) for rec in hole_records]
    object_sizes = [tuple(map(int, rec["patch"].shape)) for rec in object_records]
    min_panel_w = max(11, max(w for _, w in hole_sizes) + 4, max(w for _, w in object_sizes) + 2)
    min_h = max(11, max(h for h, _ in hole_sizes) + 4, max(h for h, _ in object_sizes) + 2)
    if min_panel_w > 14 or min_h > 18:
        return None

    for _ in range(80):
        panel_w = int(rng.integers(min_panel_w, 15))
        height = int(rng.integers(min_h, 19))
        hole_positions = _place_rectangles(height - 2, panel_w - 2, hole_sizes, rng, border=1, gap=1)
        if hole_positions is None:
            continue
        object_positions = _place_rectangles(height, panel_w, object_sizes, rng, border=1, gap=1)
        if object_positions is None:
            continue

        left = np.full((height, panel_w), bg, dtype=int)
        left[1:-1, 1:-1] = int(template_color)
        holes_for_expected = []
        for rec, (rr, cc) in zip(hole_records, hole_positions):
            r0 = 1 + rr
            c0 = 1 + cc
            mask = rec["mask"]
            left[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = bg
            coords = np.argwhere(mask) + np.array([r0, c0])
            holes_for_expected.append(
                {
                    "record": rec,
                    "coords": coords,
                    "bbox": (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1),
                    "mask": mask.copy(),
                }
            )

        right = np.full((height, panel_w), bg, dtype=int)
        objects_for_expected = []
        for rec, (r0, c0) in zip(object_records, object_positions):
            patch = rec["patch"]
            _paste(right, patch, r0, c0, transparent=bg)
            objects_for_expected.append(rec)

        assignment: dict[int, tuple[int, int]] = {}
        leftover_hole = -1
        object_index_by_id = {rec["id"]: idx for idx, rec in enumerate(objects_for_expected)}
        for hole_idx, hole in enumerate(holes_for_expected):
            match_object_id = hole["record"]["match_object_id"]
            if match_object_id is None:
                leftover_hole = hole_idx
                continue
            object_idx = object_index_by_id[match_object_id]
            assignment[hole_idx] = (object_idx, int(hole["record"]["factor"]))
        if leftover_hole < 0:
            return None

        leftover_object = object_index_by_id[n_objects - 1]
        expected = _compose_expected_left(left, holes_for_expected, objects_for_expected, assignment, leftover_hole, leftover_object, bg)
        inp = np.concatenate([left, np.zeros((height, 1), dtype=int), right], axis=1)
        if _validate_generated(inp, expected):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        generated = _generate_candidate(rng)
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid cross-scale stamping puzzle")
