from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FG = 5
NEIGHBORS_8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

GEN_ATTEMPTS = 500
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 24
INPUT_BBOX_GAP = 1
OUTPUT_PIXEL_GAP = 1
GROUP_PATTERNS = (
    (1, 2, 3),
    (1, 2, 3, 1),
    (1, 2, 3, 2),
    (1, 2, 3, 3),
    (1, 2, 4, 1),
    (1, 2, 4, 3),
    (1, 2, 3, 4),
)

BASE_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "L3": ((0, 0), (1, 0), (1, 1)),
    "L4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "T4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "P5": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    "F5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    "T5": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "U5": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "H6": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    "Z6": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0)),
    "G7": ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)),
}


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _round_half_up_from_doubled(value2: int) -> int:
    return (int(value2) + 1) // 2


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0
    arr = arr.copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_key(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = _normalize_coords(coords)
    return tuple((int(r), int(c)) for r, c in arr)


def _bbox_shape_from_key(key: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in key]
    cols = [c for _r, c in key]
    return max(rows) + 1, max(cols) + 1


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _bbox_shape_from_key(key)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[int(r), int(c)] = True
    return mask


def _mask_rotated_key(key: tuple[tuple[int, int], ...], steps_cw: int) -> tuple[tuple[int, int], ...]:
    mask = _mask_from_key(key)
    rotated = np.rot90(mask, -int(steps_cw) % 4)
    return _shape_key(np.argwhere(rotated))


def _rotation_canonical_key(key: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_mask_rotated_key(key, steps) for steps in range(4))


def _rotate_points(
    coords: np.ndarray | list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    steps_cw: int,
) -> np.ndarray:
    steps = int(steps_cw) % 4
    arr = np.asarray(coords, dtype=int)
    if steps == 0:
        unique = sorted({(int(r), int(c)) for r, c in arr})
        return np.asarray(unique, dtype=int)

    r0, c0, r1, c1 = map(int, bbox)
    cr2 = r0 + r1
    cc2 = c0 + c1
    out: list[tuple[int, int]] = []
    for r, c in arr:
        r = int(r)
        c = int(c)
        if steps == 1:
            nr2 = cr2 + 2 * c - cc2
            nc2 = cc2 - 2 * r + cr2
        elif steps == 2:
            nr2 = 2 * cr2 - 2 * r
            nc2 = 2 * cc2 - 2 * c
        else:
            nr2 = cr2 - 2 * c + cc2
            nc2 = cc2 + 2 * r - cr2
        out.append(
            (
                _round_half_up_from_doubled(nr2),
                _round_half_up_from_doubled(nc2),
            )
        )

    unique = sorted(set(out))
    return np.asarray(unique, dtype=int)


def _rounded_rotated_key(
    key: tuple[tuple[int, int], ...],
    steps_cw: int,
) -> tuple[tuple[int, int], ...]:
    h, w = _bbox_shape_from_key(key)
    coords = np.asarray(key, dtype=int)
    rotated = _rotate_points(coords, (0, 0, h - 1, w - 1), steps_cw)
    return _shape_key(rotated)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for start_r in range(h):
        for start_c in range(w):
            if seen[start_r, start_c] or int(g[start_r, start_c]) == int(bg):
                continue

            color = int(g[start_r, start_c])
            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS_8:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    assert int(g[nr, nc]) == color, "expected monochrome foreground components"
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            order = np.lexsort((arr[:, 1], arr[:, 0]))
            arr = arr[order]
            components.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "area": int(arr.shape[0]),
                }
            )

    components.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    return components


def _analyze_objects(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _dominant_color(g)
    fg_colors = sorted(int(v) for v in np.unique(g) if int(v) != int(bg))
    assert len(fg_colors) == 1, "expected a single foreground color"
    fg = int(fg_colors[0])

    objects = _find_components(g, bg)
    assert objects, "expected at least one foreground object"

    for obj in objects:
        key = _shape_key(obj["coords"])
        obj["shape_key"] = key
        obj["bbox_shape"] = _bbox_shape_from_key(key)

    return g, int(bg), fg, objects


def _render_from_objects(
    shape: tuple[int, int],
    objects: list[dict[str, object]],
    steps_by_shape: dict[tuple[tuple[int, int], ...], int],
    bg: int,
    fg: int,
    priority_counts: dict[tuple[tuple[int, int], ...], int] | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    if priority_counts is None:
        priority_counts = {key: 0 for key in steps_by_shape}

    ordered = sorted(
        objects,
        key=lambda obj: (
            int(priority_counts[obj["shape_key"]]),
            -int(obj["bbox"][0]),
            -int(obj["bbox"][1]),
        ),
    )

    h, w = shape
    for obj in ordered:
        steps = int(steps_by_shape[obj["shape_key"]]) % 4
        coords = _rotate_points(obj["coords"], obj["bbox"], steps)
        for r, c in coords:
            r = int(r)
            c = int(c)
            assert 0 <= r < h and 0 <= c < w, "rotated object must stay inside the canvas"
            out[r, c] = int(fg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each repeated foreground shape by a clockwise angle encoded by its exact shape frequency.

    Input structure:
    - A black background contains several same-color foreground objects.
    - Every foreground object is an 8-connected monochrome region, and objects are separated in the input.
    - Some objects share exactly the same normalized bbox mask after translation to the top-left of their own bbox.

    Rule:
    - Detect every 8-connected foreground object and normalize its pixel coordinates relative to its bbox top-left corner.
    - Group objects only by exact equality of that normalized mask, and let G be the size of each group.
    - Clear the canvas, then rotate every object clockwise by (G mod 4) x 90 degrees around the center of its own bbox.
    - When the rotated coordinates are half-integers, round each coordinate to the nearest grid cell with half-up rounding, then redraw the object on the clean background.
    - If two rotated objects overlap, keep the one from the larger group; if the group sizes tie, keep the object whose original bbox top-left comes first in reading order.

    Key insight:
    - The rotation amount is not marked locally on any object; it is encoded globally by how many times that exact normalized shape appears elsewhere in the grid.
    """
    g, bg, fg, objects = _analyze_objects(grid)
    group_counts = Counter(obj["shape_key"] for obj in objects)
    steps_by_shape = {key: int(count) % 4 for key, count in group_counts.items()}
    return _render_from_objects(g.shape, objects, steps_by_shape, bg, fg, priority_counts=group_counts)


def _build_shape_library() -> tuple[list[dict[str, object]], dict[str, list[tuple[tuple[int, int], ...]]]]:
    library: list[dict[str, object]] = []
    by_family: dict[str, list[tuple[tuple[int, int], ...]]] = {}

    for family, coords in BASE_SHAPES.items():
        base_key = _shape_key(np.asarray(coords, dtype=int))
        orbit: list[tuple[tuple[int, int], ...]] = []
        for steps in range(4):
            key = _mask_rotated_key(base_key, steps)
            if key not in orbit:
                orbit.append(key)
        by_family[family] = orbit

        for key in orbit:
            h, w = _bbox_shape_from_key(key)
            rounded_keys = {steps: _rounded_rotated_key(key, steps) for steps in range(4)}
            entry = {
                "family": family,
                "key": key,
                "area": len(key),
                "bbox_shape": (h, w),
                "mixed_parity": (h % 2) != (w % 2),
                "rotation_canonical": _rotation_canonical_key(key),
                "rounded_keys": rounded_keys,
            }
            library.append(entry)

    library.sort(key=lambda entry: (len(entry["key"]), entry["family"], entry["key"]))
    return library, by_family


SHAPE_LIBRARY, SHAPES_BY_FAMILY = _build_shape_library()
SHAPE_META_BY_KEY = {entry["key"]: entry for entry in SHAPE_LIBRARY}


def _sample_shape_key(
    count: int,
    used_keys: set[tuple[tuple[int, int], ...]],
    used_families: set[str],
    *,
    require_mixed: bool,
) -> tuple[tuple[int, int], ...]:
    candidates = [
        entry
        for entry in SHAPE_LIBRARY
        if entry["key"] not in used_keys
        and entry["family"] != "L3"
        and (count == 4 or entry["rounded_keys"][count] != entry["key"])
        and (not require_mixed or entry["mixed_parity"])
        and (count == 4 or len(entry["rounded_keys"][count]) == int(entry["area"]))
    ]
    assert candidates, "shape library unexpectedly exhausted"

    family_fresh = [entry for entry in candidates if entry["family"] not in used_families]
    if family_fresh:
        candidates = family_fresh

    if require_mixed:
        prefer = candidates
    elif count in (1, 3):
        prefer = [entry for entry in candidates if not entry["mixed_parity"]]
        if not prefer:
            prefer = candidates
    else:
        prefer = candidates

    entry = random.choice(prefer)
    return entry["key"]


def _sample_groups() -> list[dict[str, object]]:
    pattern = random.choice(GROUP_PATTERNS)
    groups: list[dict[str, object]] = []

    l3_orbit = list(SHAPES_BY_FAMILY["L3"])
    key_a, key_b = random.sample(l3_orbit, 2)
    pair = [(key_a, 1), (key_b, 2)]
    if random.random() < 0.5:
        pair[0], pair[1] = (pair[0][0], 2), (pair[1][0], 1)
    used_keys = {pair[0][0], pair[1][0]}
    used_families = {"L3"}
    groups.extend({"shape_key": key, "count": count} for key, count in pair)

    extras = list(pattern[2:])
    showcase_done = False
    for count in extras:
        require_mixed = (not showcase_done) and (count in (1, 3))
        key = _sample_shape_key(count, used_keys, used_families, require_mixed=require_mixed)
        groups.append({"shape_key": key, "count": count})
        used_keys.add(key)
        used_families.add(str(SHAPE_META_BY_KEY[key]["family"]))
        if require_mixed:
            showcase_done = True

    random.shuffle(groups)
    return groups


def _bbox_clear(
    bboxes: list[tuple[int, int, int, int]],
    candidate: tuple[int, int, int, int],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in bboxes:
        rows_overlap = not (r1 + gap < or0 or or1 + gap < r0)
        cols_overlap = not (c1 + gap < oc0 or oc1 + gap < c0)
        if rows_overlap and cols_overlap:
            return False
    return True


def _coords_clear(mask: np.ndarray, coords: np.ndarray, gap: int) -> bool:
    h, w = mask.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(h, r + gap + 1)
        c1 = min(w, c + gap + 1)
        if bool(mask[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_coords(mask: np.ndarray, coords: np.ndarray) -> None:
    for r, c in coords:
        mask[int(r), int(c)] = True


def _render_input(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in placed:
        for r, c in obj["input_coords"]:
            grid[int(r), int(c)] = FG
    return grid


def _expected_from_metadata(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    ordered = sorted(
        placed,
        key=lambda obj: (
            int(obj["group_size"]),
            -int(obj["bbox"][0]),
            -int(obj["bbox"][1]),
        ),
    )
    for obj in ordered:
        for r, c in obj["output_coords"]:
            out[int(r), int(c)] = FG
    return out


def _place_objects(
    shape: tuple[int, int],
    groups: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = shape
    specs: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        for _ in range(int(group["count"])):
            specs.append(
                {
                    "shape_key": group["shape_key"],
                    "group_size": int(group["count"]),
                    "group_index": group_index,
                }
            )

    specs.sort(
        key=lambda spec: (
            -max(_bbox_shape_from_key(spec["shape_key"])),
            -len(spec["shape_key"]),
            -int(spec["group_size"]),
            random.random(),
        )
    )

    input_bboxes: list[tuple[int, int, int, int]] = []
    output_used = np.zeros(shape, dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in specs:
        key = spec["shape_key"]
        count = int(spec["group_size"])
        h0, w0 = _bbox_shape_from_key(key)
        offsets = np.asarray(key, dtype=int)

        candidates = [(r0, c0) for r0 in range(h - h0 + 1) for c0 in range(w - w0 + 1)]
        random.shuffle(candidates)
        success = False

        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + h0 - 1, c0 + w0 - 1)
            if not _bbox_clear(input_bboxes, bbox, INPUT_BBOX_GAP):
                continue

            input_coords = offsets + np.array([r0, c0], dtype=int)
            output_coords = _rotate_points(input_coords, bbox, count % 4)
            if np.any(output_coords[:, 0] < 0) or np.any(output_coords[:, 0] >= h):
                continue
            if np.any(output_coords[:, 1] < 0) or np.any(output_coords[:, 1] >= w):
                continue
            if not _coords_clear(output_used, output_coords, OUTPUT_PIXEL_GAP):
                continue

            placed.append(
                {
                    "shape_key": key,
                    "group_size": count,
                    "bbox": bbox,
                    "input_coords": input_coords,
                    "output_coords": output_coords,
                }
            )
            input_bboxes.append(bbox)
            _mark_coords(output_used, output_coords)
            success = True
            break

        if not success:
            return None

    return placed


def _solve_with_group_key(
    grid: np.ndarray,
    key_func,
) -> np.ndarray | None:
    g, bg, fg, objects = _analyze_objects(grid)
    keys = [key_func(obj) for obj in objects]
    counts = Counter(keys)
    steps_by_shape = {obj["shape_key"]: int(counts[key]) % 4 for obj, key in zip(objects, keys)}
    priority = {obj["shape_key"]: int(counts[key]) for obj, key in zip(objects, keys)}
    try:
        return _render_from_objects(g.shape, objects, steps_by_shape, bg, fg, priority_counts=priority)
    except AssertionError:
        return None


def _solve_with_direct_steps(grid: np.ndarray, step_func) -> np.ndarray | None:
    g, bg, fg, objects = _analyze_objects(grid)
    steps_by_shape = {obj["shape_key"]: int(step_func(obj)) % 4 for obj in objects}
    try:
        return _render_from_objects(g.shape, objects, steps_by_shape, bg, fg)
    except AssertionError:
        return None


def _validate_candidate(
    inp: np.ndarray,
    expected: np.ndarray,
    groups: list[dict[str, object]],
) -> bool:
    try:
        g, _bg, _fg, objects = _analyze_objects(inp)
    except Exception:
        return False

    if g.shape != expected.shape or not (6 <= len(objects) <= 10):
        return False

    group_counts = Counter(obj["shape_key"] for obj in objects)
    if sorted(group_counts.values()) != sorted(int(group["count"]) for group in groups):
        return False
    if 1 not in group_counts.values() or 2 not in group_counts.values():
        return False
    if not any(count in (3, 4) for count in group_counts.values()):
        return False

    distinct_shapes = list(group_counts)
    if len(distinct_shapes) != len(groups):
        return False

    has_same_area_diff_shape = False
    has_same_bbox_diff_shape = False
    has_rot_equivalent_diff_shape = False
    for i, key_a in enumerate(distinct_shapes):
        meta_a = SHAPE_META_BY_KEY[key_a]
        for key_b in distinct_shapes[i + 1 :]:
            meta_b = SHAPE_META_BY_KEY[key_b]
            if meta_a["area"] == meta_b["area"]:
                has_same_area_diff_shape = True
            if meta_a["bbox_shape"] == meta_b["bbox_shape"]:
                has_same_bbox_diff_shape = True
            if meta_a["rotation_canonical"] == meta_b["rotation_canonical"]:
                has_rot_equivalent_diff_shape = True
    if not (has_same_area_diff_shape and has_same_bbox_diff_shape and has_rot_equivalent_diff_shape):
        return False

    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    wrong_outputs = [
        _solve_with_group_key(inp, lambda obj: int(obj["area"])),
        _solve_with_group_key(inp, lambda obj: tuple(obj["bbox_shape"])),
        _solve_with_group_key(inp, lambda obj: _rotation_canonical_key(obj["shape_key"])),
        _solve_with_direct_steps(inp, lambda obj: int(obj["area"]) % 4),
        _solve_with_direct_steps(inp, lambda _obj: 1),
    ]
    if any(wrong is not None and np.array_equal(wrong, expected) for wrong in wrong_outputs):
        return False

    return True


def _generate_candidate() -> np.ndarray | None:
    groups = _sample_groups()
    n_objects = sum(int(group["count"]) for group in groups)
    min_side = max(GRID_SIDE_MIN, 17 + (n_objects + 1) // 2)
    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)

    placed = _place_objects((h, w), groups)
    if placed is None:
        return None

    inp = _render_input((h, w), placed)
    expected = _expected_from_metadata((h, w), placed)
    return inp if _validate_candidate(inp, expected, groups) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid shape-frequency rotation puzzle")
