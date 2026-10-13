from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(coords)

    return components


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_repr(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    cropped = _crop_bool(mask)
    flat = tuple(int(x) for x in cropped.astype(np.uint8).ravel())
    return int(cropped.shape[0]), int(cropped.shape[1]), flat


def _canonical_mask(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(_mask_repr(rotated))
        variants.append(_mask_repr(np.fliplr(rotated)))
    return min(variants)


def _analyze_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    component_mask = np.zeros(patch.shape, dtype=bool)
    for r, c in coords:
        component_mask[r - r0, c - c0] = True

    colors = np.unique(patch[component_mask])
    if len(colors) != 2:
        return None

    boundary = np.zeros(patch.shape, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    boundary_colors = np.unique(patch[component_mask & boundary])
    if len(boundary_colors) != 1:
        return None

    outline_color = int(boundary_colors[0])
    fill_candidates = [int(color) for color in colors if int(color) != outline_color]
    if len(fill_candidates) != 1:
        return None
    fill_color = fill_candidates[0]

    outline = component_mask & (patch == outline_color)
    interior = component_mask & (patch == fill_color)
    if not bool(interior.any()) or not bool(outline.any()):
        return None
    if bool((interior & boundary).any()):
        return None

    interior_coords = [
        (int(r0 + rr), int(c0 + cc))
        for rr, cc in np.argwhere(interior)
    ]

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "outline_color": outline_color,
        "fill_color": fill_color,
        "shape_key": _canonical_mask(outline),
        "outline_mask": outline,
        "interior_mask": interior,
        "interior_coords": interior_coords,
    }


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []
    for coords in _components8(g != bg):
        obj = _analyze_component(g, coords)
        if obj is not None:
            objects.append(obj)
    objects.sort(key=lambda obj: obj["bbox"])
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair objects by outline shape up to rotation/reflection, then swap the fill colors inside each pair.

    Input structure:
    - The background is the dominant color.
    - Foreground objects are 8-connected two-color components: a one-pixel outline and a filled interior.
    - Matching is based only on the outline geometry, invariant to translation, rotation, and reflection.

    Rule:
    - Extract each object's outline mask.
    - Normalize that outline under the 8 symmetries of the square.
    - Pair objects with the same normalized outline and swap their interior fill colors.
    - Keep outlines, positions, and unpaired objects unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _bg, objects = _extract_objects(g)
    by_shape: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in objects:
        by_shape[obj["shape_key"]].append(obj)

    for group in by_shape.values():
        group.sort(key=lambda obj: obj["bbox"])
        for idx in range(0, len(group) - 1, 2):
            first = group[idx]
            second = group[idx + 1]
            for r, c in first["interior_coords"]:
                out[r, c] = int(second["fill_color"])
            for r, c in second["interior_coords"]:
                out[r, c] = int(first["fill_color"])

    return out


# --------------- generator ---------------

GRID_MIN_SIDE = 22
GRID_MAX_SIDE = 30
PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 3
UNPAIRED_PRESENT_PROB = 0.5
MAX_UNPAIRED = 2
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 240
GEN_RETRIES = 400
PLACEMENT_GAP = 1


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        out[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True
    return out


def _build_shape_from_interior(inner: np.ndarray) -> dict | None:
    canvas = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=bool)
    canvas[1 : 1 + inner.shape[0], 1 : 1 + inner.shape[1]] = inner
    outline = _dilate8(canvas) & (~canvas)
    full = outline | canvas

    pts = np.argwhere(full)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    outline = outline[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    canvas = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()

    patch = np.zeros(outline.shape, dtype=int)
    patch[outline] = 1
    patch[canvas] = 2
    obj = _analyze_component(patch, [tuple(map(int, xy)) for xy in np.argwhere(patch != 0)])
    if obj is None:
        return None
    if not bool(np.array_equal(obj["outline_mask"], outline)):
        return None
    if not bool(np.array_equal(obj["interior_mask"], canvas)):
        return None

    return {
        "outline_mask": outline,
        "interior_mask": canvas,
        "shape_key": obj["shape_key"],
    }


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr, cc = int(r), int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    return mask


def _random_rectangle_shape() -> dict | None:
    h = random.randint(2, 5)
    w = random.randint(2, 6)
    inner = np.ones((h, w), dtype=bool)
    return _build_shape_from_interior(inner)


def _random_irregular_shape() -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area_min = max(4, (h * w + 1) // 2)
        area_max = h * w - 1
        if area_min > area_max:
            continue
        target = random.randint(area_min, area_max)
        mask = _grow_connected_mask(h, w, target)
        cropped = _crop_bool(mask)
        if min(cropped.shape) < 2:
            continue
        if int(cropped.sum()) == int(cropped.shape[0] * cropped.shape[1]):
            continue
        shape = _build_shape_from_interior(cropped)
        if shape is not None:
            return shape
    return None


def _random_shape(existing: set[tuple[int, int, tuple[int, ...]]], kind: str) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        if kind == "rect":
            shape = _random_rectangle_shape()
        else:
            shape = _random_irregular_shape()
        if shape is None:
            continue
        if shape["shape_key"] in existing:
            continue
        return shape
    return None


def _transform_bool(mask: np.ndarray, variant: int) -> np.ndarray:
    transformed = np.rot90(mask, variant % 4)
    if variant >= 4:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _pair_colors(bg: int) -> tuple[int, int, int, int]:
    palette = [c for c in range(10) if c != bg]
    outline_a, outline_b, fill_a, fill_b = random.sample(palette, 4)
    return int(outline_a), int(outline_b), int(fill_a), int(fill_b)


def _single_colors(bg: int) -> tuple[int, int]:
    outline, fill = random.sample([c for c in range(10) if c != bg], 2)
    return int(outline), int(fill)


def _render_patch(
    outline_mask: np.ndarray,
    interior_mask: np.ndarray,
    bg: int,
    outline_color: int,
    fill_color: int,
) -> np.ndarray:
    patch = np.full(outline_mask.shape, bg, dtype=int)
    patch[outline_mask] = int(outline_color)
    patch[interior_mask] = int(fill_color)
    return patch


def _separated(
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


def _expected_from_records(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = grid.copy()
    by_shape: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for record in records:
        by_shape[record["shape_key"]].append(record)

    for group in by_shape.values():
        group.sort(key=lambda item: item["bbox"])
        for idx in range(0, len(group) - 1, 2):
            first = group[idx]
            second = group[idx + 1]
            for r, c in first["interior_coords"]:
                out[r, c] = int(second["fill_color"])
            for r, c in second["interior_coords"]:
                out[r, c] = int(first["fill_color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    if random.random() < UNPAIRED_PRESENT_PROB:
        unpaired_count = random.randint(1, MAX_UNPAIRED)
    else:
        unpaired_count = 0

    total_shapes = pair_count + unpaired_count
    shape_kinds = [random.choice(("rect", "irregular")) for _ in range(total_shapes)]

    seen_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    shapes: list[tuple[dict, int]] = []
    for kind in shape_kinds:
        shape = _random_shape(seen_keys, kind)
        if shape is None:
            return None
        seen_keys.add(shape["shape_key"])
        multiplicity = 2 if len(shapes) < pair_count else 1
        shapes.append((shape, multiplicity))

    instances: list[dict] = []
    for shape, multiplicity in shapes:
        outline_mask = shape["outline_mask"]
        interior_mask = shape["interior_mask"]
        if multiplicity == 2:
            outline_a, outline_b, fill_a, fill_b = _pair_colors(bg)
            variants = [random.randrange(8), random.randrange(8)]
            specs = [
                (outline_a, fill_a, variants[0]),
                (outline_b, fill_b, variants[1]),
            ]
        else:
            outline_color, fill_color = _single_colors(bg)
            specs = [(outline_color, fill_color, random.randrange(8))]

        for outline_color, fill_color, variant in specs:
            o_mask = _transform_bool(outline_mask, variant)
            i_mask = _transform_bool(interior_mask, variant)
            patch = _render_patch(o_mask, i_mask, bg, outline_color, fill_color)
            instances.append(
                {
                    "shape_key": shape["shape_key"],
                    "outline_mask": o_mask,
                    "interior_mask": i_mask,
                    "fill_color": int(fill_color),
                    "patch": patch,
                }
            )

    rows = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    cols = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    grid = np.full((rows, cols), bg, dtype=int)

    placed_bboxes: list[tuple[int, int, int, int]] = []
    placed_records: list[dict] = []
    order = sorted(
        range(len(instances)),
        key=lambda idx: int(instances[idx]["patch"].shape[0] * instances[idx]["patch"].shape[1]),
        reverse=True,
    )

    for idx in order:
        inst = instances[idx]
        patch = inst["patch"]
        ph, pw = patch.shape
        if ph > rows or pw > cols:
            return None

        bbox = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, rows - ph)
            c0 = random.randint(0, cols - pw)
            candidate = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_separated(candidate, prev, PLACEMENT_GAP) for prev in placed_bboxes):
                bbox = candidate
                break
        if bbox is None:
            return None

        r0, c0, _, _ = bbox
        non_bg = patch != bg
        grid[r0 : r0 + ph, c0 : c0 + pw][non_bg] = patch[non_bg]
        placed_bboxes.append(bbox)

        interior_coords = [
            (int(r0 + rr), int(c0 + cc))
            for rr, cc in np.argwhere(inst["interior_mask"])
        ]
        placed_records.append(
            {
                "shape_key": inst["shape_key"],
                "fill_color": inst["fill_color"],
                "bbox": bbox,
                "interior_coords": interior_coords,
            }
        )

    if _get_bg(grid) != bg:
        return None

    _bg, extracted = _extract_objects(grid)
    if len(extracted) != len(instances):
        return None

    expected = _expected_from_records(grid, placed_records)
    actual = main(grid)
    if not bool(np.array_equal(actual, expected)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
